package main

import (
	"fmt"
	"os"
	"path/filepath"
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
	timing := "依講稿比例估時，請試看並視需要校正 SRT"
	if len(boundaries) > 1 {
		timing = "依 Windows 語音進度事件對齊，仍建議試看確認"
	}
	timeAt := func(pos int) float64 {
		if pos >= len(units) {
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
		return duration * float64(pos) / float64(len(units))
	}
	if strings.TrimSpace(second) != "" {
		var out strings.Builder
		seconds := nonEmptyLines(second)
		pos, index, n := 0, 1, 0
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
			if to <= from {
				to = from + .02
			}
			if to > duration {
				to = duration
			}
			fmt.Fprintf(&out, "%d\n%s --> %s\n%s\n\n", index, stamp(from), stamp(to), body)
			index++
		}
		if out.Len() == 0 {
			return false, "", nil
		}
		return true, timing + "；雙行字幕", os.WriteFile(filepath.Join(dir, "subtitles.srt"), []byte(out.String()), 0600)
	}
	var out strings.Builder
	start, visible, index := 0, 0, 1
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
			line := strings.TrimSpace(string(utf16.Decode(units[start:end])))
			if line != "" {
				from, to := timeAt(start), timeAt(end)
				if to <= from {
					to = from + .02
				}
				if to > duration {
					to = duration
				}
				fmt.Fprintf(&out, "%d\n%s --> %s\n%s\n\n", index, stamp(from), stamp(to), line)
				index++
			}
			start = end
			visible = 0
		}
	}
	if start < len(units) {
		line := strings.TrimSpace(string(utf16.Decode(units[start:])))
		if line != "" {
			fmt.Fprintf(&out, "%d\n%s --> %s\n%s\n\n", index, stamp(timeAt(start)), stamp(duration), line)
		}
	}
	if out.Len() == 0 {
		return false, "", nil
	}
	return true, timing, os.WriteFile(filepath.Join(dir, "subtitles.srt"), []byte(out.String()), 0600)
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
