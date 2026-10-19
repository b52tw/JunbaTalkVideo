package main

import (
	"encoding/json"
	"os"
	"path/filepath"
	"strings"
	"testing"
)

func TestQualityValidationAndCaptionShift(t *testing.T) {
	q := LocalRequest{Device: "AUTO", BBox: []float64{.2, .2, .5, .5}, MouthBox: []float64{.3, .5, .2, .1}, EyeBoxes: [][]float64{{.3, .3, .1, .04}}, FPS: 25, MaxSide: 1920, Restore: true}
	if e := validateLocal(&q); e != nil || q.MaxSide != 1920 {
		t.Fatal(q, e)
	}
	q.BodyMotion = "teleport"
	if validateLocal(&q) == nil {
		t.Fatal("accepted nonexistent pose model")
	}
	q.BodyMotion = "breathe"
	q.EyeBoxes = [][]float64{{.99, .3, .1, .04}}
	if validateLocal(&q) == nil {
		t.Fatal("accepted eye outside image")
	}
	dir := t.TempDir()
	orig := "1\n00:00:00,000 --> 00:00:02,340\n你好。\n\n"
	os.WriteFile(filepath.Join(dir, "subtitles.srt"), []byte(orig), 0600)
	if e := shiftCaptions(dir, .56); e != nil {
		t.Fatal(e)
	}
	b, _ := os.ReadFile(filepath.Join(dir, "subtitles.srt"))
	if !strings.Contains(string(b), "00:00:00,560 --> 00:00:02,900") {
		t.Fatal(string(b))
	}
	shiftCaptions(dir, -.56)
	b, _ = os.ReadFile(filepath.Join(dir, "subtitles.srt"))
	if string(b) != orig {
		t.Fatal(string(b))
	}
}
func TestRebuildPreservesOriginalAndResetsCaptionOffset(t *testing.T) {
	a := newApp(t.TempDir())
	j := prepared(t, a, "photo")
	j.State = "done"
	j.Engine = "local"
	j.LeadSeconds = .56
	j.Output = "video.mp4"
	j.Export = "export.mp4"
	j.Compatible = true
	srt := "1\n00:00:00,560 --> 00:00:01,560\n你好。\n\n"
	os.WriteFile(filepath.Join(a.dir(j.ID), "subtitles.srt"), []byte(srt), 0600)
	w := request(t, a, "POST", "/api/local/rebuild", strings.NewReader(`{"id":"`+j.ID+`"}`), "application/json", true)
	if w.Code != 200 {
		t.Fatal(w.Code, w.Body.String())
	}
	var got Job
	json.Unmarshal(w.Body.Bytes(), &got)
	if got.ID == j.ID || got.State != "ready" || got.Output != "" || got.Export != "" || got.LeadSeconds != 0 {
		t.Fatal(got)
	}
	b, _ := os.ReadFile(filepath.Join(a.dir(got.ID), "subtitles.srt"))
	if !strings.Contains(string(b), "00:00:00,000 --> 00:00:01,000") {
		t.Fatal(string(b))
	}
	b, _ = os.ReadFile(filepath.Join(a.dir(j.ID), "subtitles.srt"))
	if string(b) != srt || j.State != "done" || j.Output != "video.mp4" {
		t.Fatal("changed original")
	}
}
