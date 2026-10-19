package main

// v0.8.5 multi-speaker dialogue (up to five speakers, A–E).
//
// Script lines start with a speaker letter, e.g. "A：大家好" / "B: 你好".
// The same speaker may appear many times (A opens, B…E reply, A closes).
// With Windows speech each line is synthesised with that speaker's voice and
// effect and concatenated, so the speaking timeline is exact. With an imported
// recording the user supplies the timeline ("A 0-3.2" per line).

import (
	"context"
	"encoding/binary"
	"encoding/json"
	"errors"
	"fmt"
	"os"
	"path/filepath"
	"regexp"
	"strconv"
	"strings"
	"unicode/utf8"
)

// DialogueLine is one spoken line. HanText and TailoText are independent so
// captions can render Han characters and Tâi-lô as separate layers/styles.
type DialogueLine struct {
	Speaker   string   `json:"speaker"`
	HanText   string   `json:"han_text"`
	TailoText string   `json:"tailo_text,omitempty"`
	Actions   []string `json:"actions,omitempty"`
	Start     float64  `json:"start"`
	End       float64  `json:"end"`
}

type Dialogue struct {
	Lines []DialogueLine    `json:"lines"`
	Names map[string]string `json:"names"`
}

type SpeakerBox struct {
	ID       string    `json:"id"`
	MouthBox []float64 `json:"mouth_box"`
	FaceBox  []float64 `json:"face_box,omitempty"`
}

const maxDialogueLines = 60

var (
	dialoguePrefix   = regexp.MustCompile(`^\s*([A-Ea-eＡ-Ｅａ-ｅ])\s*(?:[\[［]([^\]］]*)[\]］])?\s*[：:]\s*(.*)$`)
	dialogueTimeline = regexp.MustCompile(`^\s*([A-Ea-eＡ-Ｅａ-ｅ])\s+(\d+(?:\.\d+)?)\s*[-~～－]\s*(\d+(?:\.\d+)?)\s*(?:秒|s)?\s*$`)
)

func speakerLetter(s string) string {
	r, _ := utf8.DecodeRuneInString(s)
	switch {
	case r >= 'Ａ' && r <= 'Ｅ':
		r = 'A' + (r - 'Ａ')
	case r >= 'ａ' && r <= 'ｅ':
		r = 'A' + (r - 'ａ')
	case r >= 'a' && r <= 'e':
		r = r - 'a' + 'A'
	}
	return string(r)
}

// parseDialogueScript reads "A：text" lines. Every non-empty line needs a prefix.
func parseDialogueScript(text string) ([]DialogueLine, error) {
	var out []DialogueLine
	for n, raw := range strings.Split(strings.ReplaceAll(text, "\r\n", "\n"), "\n") {
		if strings.TrimSpace(raw) == "" {
			continue
		}
		m := dialoguePrefix.FindStringSubmatch(raw)
		if m == nil {
			return nil, fmt.Errorf("多人對話第 %d 行需以說話者開頭，例如「A：大家好」（A～E）", n+1)
		}
		body := strings.TrimSpace(m[3])
		if body == "" {
			return nil, fmt.Errorf("多人對話第 %d 行沒有台詞", n+1)
		}
		out = append(out, DialogueLine{Speaker: speakerLetter(m[1]), HanText: body, Actions: parseActionTags(m[2])})
	}
	if len(out) == 0 {
		return nil, errors.New("多人對話請至少輸入一行「A：台詞」")
	}
	if len(out) > maxDialogueLines {
		return nil, fmt.Errorf("多人對話最多 %d 行", maxDialogueLines)
	}
	return out, nil
}

// parseDialogueTimeline reads "A 0-3.2" lines for an imported recording.
func parseDialogueTimeline(s string, duration float64) ([]DialogueLine, error) {
	var out []DialogueLine
	for n, raw := range strings.Split(strings.ReplaceAll(s, "\r\n", "\n"), "\n") {
		if strings.TrimSpace(raw) == "" {
			continue
		}
		m := dialogueTimeline.FindStringSubmatch(raw)
		if m == nil {
			return nil, fmt.Errorf("說話時間第 %d 行格式應為「A 0-3.2」", n+1)
		}
		a, _ := strconv.ParseFloat(m[2], 64)
		b, _ := strconv.ParseFloat(m[3], 64)
		if b <= a || a < 0 || a > duration+.5 {
			return nil, fmt.Errorf("說話時間第 %d 行的起訖秒數不正確", n+1)
		}
		if b > duration {
			b = duration
		}
		out = append(out, DialogueLine{Speaker: speakerLetter(m[1]), Start: a, End: b})
	}
	if len(out) == 0 {
		return nil, errors.New("匯入錄音的多人對話，請填寫每段說話時間，例如「A 0-3.2」")
	}
	if len(out) > maxDialogueLines {
		return nil, fmt.Errorf("說話時間最多 %d 段", maxDialogueLines)
	}
	return out, nil
}

func readPCM16(path string) ([]byte, uint32, error) {
	b, e := os.ReadFile(path)
	if e != nil {
		return nil, 0, e
	}
	if len(b) < 44 || string(b[:4]) != "RIFF" || string(b[8:12]) != "WAVE" {
		return nil, 0, errors.New("語音片段不是 WAV")
	}
	var rate uint32
	var data []byte
	for i := 12; i+8 <= len(b); {
		n := int(binary.LittleEndian.Uint32(b[i+4 : i+8]))
		start := i + 8
		if n < 0 || n > len(b)-start {
			break
		}
		switch string(b[i : i+4]) {
		case "fmt ":
			if n >= 16 {
				rate = binary.LittleEndian.Uint32(b[start+4 : start+8])
			}
		case "data":
			data = append(data, b[start:start+n]...)
		}
		i = start + n + n%2
	}
	if rate == 0 || len(data) == 0 {
		return nil, 0, errors.New("語音片段沒有聲音資料")
	}
	return data, rate, nil
}

func writePCM16(path string, data []byte, rate uint32) error {
	h := make([]byte, 44)
	copy(h, "RIFF")
	binary.LittleEndian.PutUint32(h[4:], uint32(36+len(data)))
	copy(h[8:], "WAVEfmt ")
	binary.LittleEndian.PutUint32(h[16:], 16)
	binary.LittleEndian.PutUint16(h[20:], 1)
	binary.LittleEndian.PutUint16(h[22:], 1)
	binary.LittleEndian.PutUint32(h[24:], rate)
	binary.LittleEndian.PutUint32(h[28:], rate*2)
	binary.LittleEndian.PutUint16(h[32:], 2)
	binary.LittleEndian.PutUint16(h[34:], 16)
	copy(h[36:], "data")
	binary.LittleEndian.PutUint32(h[40:], uint32(len(data)))
	return os.WriteFile(path, append(h, data...), 0600)
}

// dialogueSpeech synthesises each line with its speaker's voice/effect and
// concatenates them (0.3 s pause between lines). Times are filled in.
func (a *App) dialogueSpeech(ctx context.Context, dir string, lines []DialogueLine, voices, effects map[string]string, rate int, out string) error {
	const gap = .3
	var all []byte
	var sampleRate uint32 = 24000
	t := 0.
	for i := range lines {
		raw := filepath.Join(dir, fmt.Sprintf("line-%02d-tts.wav", i))
		clip := filepath.Join(dir, fmt.Sprintf("line-%02d.wav", i))
		if e := a.tts(ctx, lines[i].HanText, voices[lines[i].Speaker], rate, raw); e != nil {
			return fmt.Errorf("第 %d 行（%s）朗讀失敗：%v", i+1, lines[i].Speaker, e)
		}
		effect := effects[lines[i].Speaker]
		if effect == "" {
			effect = "original"
		}
		if e := audioWAV(ctx, raw, clip, effect); e != nil {
			return e
		}
		data, r, e := readPCM16(clip)
		if e != nil {
			return e
		}
		sampleRate = r
		if i > 0 {
			silence := make([]byte, int(gap*float64(r))*2)
			all = append(all, silence...)
			t += gap
		}
		lines[i].Start = t
		t += float64(len(data)/2) / float64(r)
		lines[i].End = t
		all = append(all, data...)
		os.Remove(raw)
		os.Remove(clip)
	}
	return writePCM16(out, all, sampleRate)
}

func writeDialogueFile(dir string, d Dialogue) error {
	b, e := json.MarshalIndent(d, "", "  ")
	if e != nil {
		return e
	}
	return os.WriteFile(filepath.Join(dir, "dialogue.json"), b, 0600)
}

// splitCaption breaks one line into readable chunks (<= 14 visible characters).
func splitCaption(text string) []string {
	var parts []string
	var cur []rune
	for _, r := range text {
		cur = append(cur, r)
		punct := strings.ContainsRune("，。！？；：、,.!?;:", r)
		if (punct && len(cur) >= 6) || len(cur) >= 14 {
			if s := strings.TrimSpace(string(cur)); s != "" {
				parts = append(parts, s)
			}
			cur = nil
		}
	}
	if s := strings.TrimSpace(string(cur)); s != "" {
		parts = append(parts, s)
	}
	return parts
}

// writeDialogueCaptions writes speaker-labelled SRT cues inside each segment.
// A line with a second (e.g. Tâi-lô) caption is shown as one two-line cue.
func writeDialogueCaptions(dir string, d Dialogue, duration float64) (bool, error) {
	var out strings.Builder
	index := 1
	for _, l := range d.Lines {
		if strings.TrimSpace(l.HanText) == "" {
			continue
		}
		name := d.Names[l.Speaker]
		if name == "" {
			name = l.Speaker
		}
		end := l.End
		if end > duration {
			end = duration
		}
		if end <= l.Start {
			continue
		}
		chunks := []string{l.HanText}
		if l.TailoText == "" {
			chunks = splitCaption(l.HanText)
		}
		total := 0
		for _, c := range chunks {
			total += utf8.RuneCountInString(c)
		}
		at := l.Start
		for _, c := range chunks {
			span := (end - l.Start) * float64(utf8.RuneCountInString(c)) / float64(max(total, 1))
			body := name + "：" + c
			if l.TailoText != "" {
				body += "\n" + l.TailoText
			}
			fmt.Fprintf(&out, "%d\n%s --> %s\n%s\n\n", index, stamp(at), stamp(at+span), body)
			index++
			at += span
		}
	}
	if out.Len() == 0 {
		return false, nil
	}
	return true, os.WriteFile(filepath.Join(dir, "subtitles.srt"), []byte(out.String()), 0600)
}

func validateSpeakers(list []SpeakerBox) error {
	if len(list) > 5 {
		return errors.New("多人對話最多 5 位說話者")
	}
	seen := map[string]bool{}
	for _, s := range list {
		id := strings.ToUpper(s.ID)
		if len(id) != 1 || id[0] < 'A' || id[0] > 'E' || seen[id] {
			return errors.New("說話者代號需為 A～E 且不可重複")
		}
		seen[id] = true
		if !validRegion(s.MouthBox) {
			return fmt.Errorf("說話者 %s 的嘴巴框選無效，請重新框選", id)
		}
		if len(s.FaceBox) > 0 && !validRegion(s.FaceBox) {
			return fmt.Errorf("說話者 %s 的臉部框選無效，請重新框選", id)
		}
	}
	return nil
}

// Action tags bound to a line, e.g. "A[nod, sway]: 大家好". English or
// Chinese names; unknown tags are ignored.
var actionNames = map[string]string{"nod": "nod", "點頭": "nod", "tilt": "tilt", "側傾": "tilt", "歪頭": "tilt",
	"shake": "shake", "搖頭": "shake", "brow": "brow", "抬眉": "brow", "挑眉": "brow", "sway": "sway", "搖擺": "sway", "晃動": "sway"}

func parseActionTags(s string) []string {
	var out []string
	seen := map[string]bool{}
	for _, f := range strings.FieldsFunc(s, func(r rune) bool { return r == ',' || r == '，' || r == '、' || r == ' ' }) {
		if a, ok := actionNames[strings.ToLower(strings.TrimSpace(f))]; ok && !seen[a] {
			seen[a] = true
			out = append(out, a)
		}
	}
	return out
}
