package main

import (
	"math"
	"os"
	"path/filepath"
	"strings"
	"testing"
	"time"
)

// burstWAV writes 24 kHz mono speech-like bursts at the given [start,end) spans.
func burstWAV(t *testing.T, path string, total float64, spans [][2]float64) {
	t.Helper()
	rate := 24000
	n := int(total * float64(rate))
	data := make([]byte, n*2)
	for i := 0; i < n; i++ {
		ts := float64(i) / float64(rate)
		v := 0.
		for _, s := range spans {
			if ts >= s[0] && ts < s[1] {
				v = 9000 * math.Sin(2*math.Pi*220*ts) * (0.6 + 0.4*math.Sin(2*math.Pi*5*ts))
			}
		}
		x := int16(v)
		data[2*i], data[2*i+1] = byte(uint16(x)), byte(uint16(x)>>8)
	}
	if e := writePCM16(path, data, uint32(rate)); e != nil {
		t.Fatal(e)
	}
}

func TestCaptionsSnapToRealPauses(t *testing.T) {
	dir := t.TempDir()
	// Three phrases separated by long pauses; text length is NOT proportional
	// to the speaking time, which made proportional captions lag.
	burstWAV(t, filepath.Join(dir, "speech.wav"), 12, [][2]float64{{0.2, 1.6}, {4.0, 7.5}, {9.0, 11.0}})
	ok, timing, e := writeCaptionsSecond(dir, "大家好，我是峻爸！今天讓我們帶著好奇心，勇敢嘗試一小步。再見！", "", 12)
	if e != nil || !ok || !strings.Contains(timing, "停頓") {
		t.Fatal(e, timing)
	}
	b, _ := os.ReadFile(filepath.Join(dir, "subtitles.srt"))
	srt := string(b)
	// Second phrase must start at its voice onset (~3.95 s), not seconds later.
	if !strings.Contains(srt, "00:00:03,95") {
		t.Fatal(srt)
	}
	if !strings.Contains(srt, "00:00:08,95") {
		t.Fatal(srt)
	}
}

func TestConvertAppliesManualCaptionShift(t *testing.T) {
	a := newApp(t.TempDir())
	j := &Job{ID: randomID(), State: "done", Kind: "photo", Output: "video.mp4", HasCaptions: true, CaptionMode: "srt"}
	os.MkdirAll(a.dir(j.ID), 0700)
	os.WriteFile(filepath.Join(a.dir(j.ID), "subtitles.srt"), []byte("1\n00:00:02,000 --> 00:00:04,000\n你好\n\n"), 0600)
	a.jobs[j.ID] = j
	w := request(t, a, "POST", "/api/convert", strings.NewReader(`{"id":"`+j.ID+`","format":"mp4","caption_mode":"srt","caption_shift":-0.5}`), "application/json", true)
	if w.Code != 200 {
		t.Fatal(w.Code, w.Body.String())
	}
	b, _ := os.ReadFile(filepath.Join(a.dir(j.ID), "subtitles.srt"))
	if !strings.Contains(string(b), "00:00:01,500 --> 00:00:03,500") {
		t.Fatal(string(b))
	}
	waitConverted := func() {
		for i := 0; i < 200; i++ {
			a.mu.Lock()
			busy := j.busy
			a.mu.Unlock()
			if !busy {
				return
			}
			sleepMS(20)
		}
	}
	waitConverted()
	a.mu.Lock()
	j.State = "done"
	a.mu.Unlock()
	// Setting +0.5 later is absolute (relative to the generated captions).
	w = request(t, a, "POST", "/api/convert", strings.NewReader(`{"id":"`+j.ID+`","format":"mp4","caption_mode":"srt","caption_shift":0.5}`), "application/json", true)
	if w.Code != 200 {
		t.Fatal(w.Code, w.Body.String())
	}
	b, _ = os.ReadFile(filepath.Join(a.dir(j.ID), "subtitles.srt"))
	if !strings.Contains(string(b), "00:00:02,500 --> 00:00:04,500") {
		t.Fatal(string(b))
	}
	waitConverted()
}

func sleepMS(ms int) { time.Sleep(time.Duration(ms) * time.Millisecond) }
