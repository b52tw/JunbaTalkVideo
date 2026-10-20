package main

import (
	"fmt"
	"math"
	"os"
	"path/filepath"
	"sort"
	"strconv"
	"strings"
	"unicode/utf16"
)

type speechBoundary struct {
	pos     int
	seconds float64
}

func captionMode(s string) string {
	if s == "none" || s == "srt" {
		return s
	}
	return "burn"
}
func stamp(seconds float64) string {
	if seconds < 0 {
		seconds = 0
	}
	n := int(seconds*1000 + .5)
	return fmt.Sprintf("%02d:%02d:%02d,%03d", n/3600000, n/60000%60, n/1000%60, n%1000)
}
func writeCaptions(dir, text string, duration float64) (bool, string, error) {
	return writeCaptionsSecond(dir, text, "", duration)
}

// writeCaptionsSecond adds an optional second caption line (e.g. Tâi-lô under
// Taiwanese Han characters). With a second text, each script line becomes one
// two-line cue; the i-th second line goes under the i-th script line.
func writeCaptionsSecond(dir, text, second string, duration float64) (bool, string, error) {
	units := utf16.Encode([]rune(text))
	if len(units) == 0 {
		return false, "", nil
	}
	var boundaries []speechBoundary
	if b, e := os.ReadFile(filepath.Join(dir, "speech.wav.timing.tsv")); e == nil {
		for _, line := range strings.Split(strings.TrimSpace(string(b)), "\n") {
			f := strings.Fields(line)
			if len(f) != 2 {
				continue
			}
			p, e1 := strconv.Atoi(f[0])
			s, e2 := strconv.ParseFloat(f[1], 64)
			if e1 == nil && e2 == nil && p >= 0 && p < len(units) && s >= 0 && s < duration {
				boundaries = append(boundaries, speechBoundary{p, s})
			}
		}
	}
	// v0.8.7: voice activity of the actual speech. Without per-word events the
	// text is spread over SPOKEN time only (pauses excluded), and cue changes
	// snap to real pauses, so captions no longer drift behind the voice.
	spans := voicedSpans(filepath.Join(dir, "speech.wav"))
	voiced := voicedTimeMapper(spans)
	timing := "依講稿比例估時，請試看並視需要校正 SRT"
	if len(boundaries) > 1 {
		timing = "依 Windows 語音進度事件對齊，仍建議試看確認"
	} else if voiced != nil {
		timing = "依實際語音停頓對齊，仍建議試看確認"
	}
	timeAt := func(pos int) float64 {
		if pos >= len(units) {
			if voiced != nil && len(boundaries) <= 1 {
				return spans[len(spans)-1].b
			}
			return duration
		}
		if len(boundaries) > 1 {
			before := speechBoundary{0, 0}
			for _, b := range boundaries {
				if b.pos >= pos {
					if b.pos == before.pos {
						return b.seconds
					}
					return before.seconds + (b.seconds-before.seconds)*float64(pos-before.pos)/float64(b.pos-before.pos)
				}
				before = b
			}
			return before.seconds + (duration-before.seconds)*float64(pos-before.pos)/float64(len(units)-before.pos)
		}
		if voiced != nil {
			return voiced(float64(pos) / float64(len(units)))
		}
		return duration * float64(pos) / float64(len(units))
	}
	var cues []caption
	if strings.TrimSpace(second) != "" {
		seconds := nonEmptyLines(second)
		pos, n := 0, 0
		for _, line := range strings.Split(text, "\n") {
			width := len(utf16.Encode([]rune(line)))
			from, to := timeAt(pos), timeAt(pos+width)
			pos += width + 1
			body := strings.TrimSpace(line)
			if body == "" {
				continue
			}
			if n < len(seconds) {
				body += "\n" + seconds[n]
			}
			n++
			cues = append(cues, caption{from, to, body})
		}
		timing += "；雙行字幕"
	} else {
		start, visible := 0, 0
		for p, r := range units {
			if r >= 0xDC00 && r <= 0xDFFF {
				continue
			}
			visible++
			punct := strings.ContainsRune("，。！？；：、,.!?;:\n", rune(r))
			end := p + 1
			if r >= 0xD800 && r <= 0xDBFF {
				end++
			}
			if (punct && visible >= 6) || visible >= 12 || end >= len(units) {
				if line := strings.TrimSpace(string(utf16.Decode(units[start:end]))); line != "" {
					cues = append(cues, caption{timeAt(start), timeAt(end), line})
				}
				start = end
				visible = 0
			}
		}
		if start < len(units) {
			if line := strings.TrimSpace(string(utf16.Decode(units[start:]))); line != "" {
				cues = append(cues, caption{timeAt(start), duration, line})
			}
		}
	}
	if len(cues) == 0 {
		return false, "", nil
	}
	window := 1.5
	if len(boundaries) > 1 {
		window = .3
	}
	snapCaptions(cues, spans, window, duration)
	return true, timing, writeSRT(dir, cues, duration)
}

type caption struct {
	from, to float64
	text     string
}

type span struct{ a, b float64 }

func writeSRT(dir string, cues []caption, duration float64) error {
	var out strings.Builder
	for i, c := range cues {
		from, to := c.from, c.to
		if to > duration {
			to = duration
		}
		if to <= from {
			to = from + .02
		}
		fmt.Fprintf(&out, "%d\n%s --> %s\n%s\n\n", i+1, stamp(from), stamp(to), c.text)
	}
	return os.WriteFile(filepath.Join(dir, "subtitles.srt"), []byte(out.String()), 0600)
}

// voicedSpans finds spoken regions of a PCM WAV (20 ms frames; pauses shorter
// than 0.12 s are bridged). Returns nil when the audio cannot be analysed.
func voicedSpans(path string) []span {
	data, rate, e := readPCM16(path)
	if e != nil || rate == 0 {
		return nil
	}
	hop := int(rate) / 50
	n := len(data) / 2 / hop
	if n < 10 {
		return nil
	}
	rms := make([]float64, n)
	peak := 0.
	for i := 0; i < n; i++ {
		sum := 0.
		for k := 0; k < hop; k++ {
			o := (i*hop + k) * 2
			v := float64(int16(uint16(data[o]) | uint16(data[o+1])<<8))
			sum += v * v
		}
		rms[i] = math.Sqrt(sum / float64(hop))
		peak = math.Max(peak, rms[i])
	}
	sorted := append([]float64(nil), rms...)
	sort.Float64s(sorted)
	threshold := math.Max(sorted[n/5]*2.5, peak*.06)
	var out []span
	for i := 0; i < n; {
		if rms[i] < threshold {
			i++
			continue
		}
		j := i
		for j < n && rms[j] >= threshold {
			j++
		}
		a, b := float64(i)/50, float64(j)/50
		if len(out) > 0 && a-out[len(out)-1].b < .12 {
			out[len(out)-1].b = b
		} else if b-a >= .06 {
			out = append(out, span{a, b})
		}
		i = j
	}
	if len(out) == 0 {
		return nil
	}
	return out
}

// voicedTimeMapper maps a 0..1 fraction of the text onto spoken time only.
func voicedTimeMapper(spans []span) func(float64) float64 {
	if len(spans) == 0 {
		return nil
	}
	total := 0.
	for _, s := range spans {
		total += s.b - s.a
	}
	if total <= 0 {
		return nil
	}
	return func(frac float64) float64 {
		left := math.Max(0, math.Min(1, frac)) * total
		for _, s := range spans {
			if left <= s.b-s.a {
				return s.a + left
			}
			left -= s.b - s.a
		}
		return spans[len(spans)-1].b
	}
}

// snapCaptions moves each cue change to the nearest real pause (the next cue
// starts as its words start) within “window“ seconds; order is preserved.
func snapCaptions(cues []caption, spans []span, window, duration float64) {
	if len(spans) < 2 {
		return
	}
	cues[0].from = math.Min(cues[0].from, spans[0].a)
	for i := 1; i < len(cues); i++ {
		t := cues[i].from
		best, dist := t, window
		for k := 1; k < len(spans); k++ {
			onset := spans[k].a - .05
			if d := math.Abs(onset - t); d < dist && onset > cues[i-1].from+.3 {
				best, dist = onset, d
			}
		}
		cues[i].from = best
		cues[i-1].to = best
	}
	last := &cues[len(cues)-1]
	last.to = math.Min(duration, math.Max(last.to, spans[len(spans)-1].b+.3))
}

func addSubtitleFilter(args []string) []string {
	out := append([]string(nil), args...)
	sub := ",subtitles=filename=subtitles.srt:charenc=UTF-8:fontsdir=fonts:force_style='Fontname=Noto Sans TC,Fontsize=12,Outline=1.4,Shadow=0,MarginV=22,Alignment=2'"
	for i := 0; i < len(out)-1; i++ {
		if out[i] == "-vf" {
			out[i+1] += sub
			break
		}
		if out[i] == "-filter_complex" {
			out[i+1] = strings.Replace(out[i+1], ",fps=12", sub+",fps=12", 1)
			break
		}
	}
	return out
}

func nonEmptyLines(s string) []string {
	var out []string
	for _, l := range strings.Split(strings.ReplaceAll(s, "\r\n", "\n"), "\n") {
		if l = strings.TrimSpace(l); l != "" {
			out = append(out, l)
		}
	}
	return out
}
