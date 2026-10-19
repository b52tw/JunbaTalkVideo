package main

import (
	"bytes"
	"context"
	"encoding/json"
	"fmt"
	"io"
	"mime/multipart"
	"net/http"
	"os"
	"path/filepath"
	"strings"
	"testing"
	"time"
)

func TestLocalVoiceNoCloudAndRecovery(t *testing.T) {
	a := newApp(filepath.Join(t.TempDir(), "output"))
	w := request(t, a, "POST", "/api/local/voice-prepare", strings.NewReader(""), "application/json", true)
	if w.Code != 409 {
		t.Fatal(w.Code, w.Body.String())
	}
	j := prepared(t, a, "photo")
	j.Kind = "voice"
	j.Engine = "local_voice"
	j.State = "local_voice_generating"
	a.persist(j)
	b := newApp(a.root)
	b.load()
	if b.jobs[j.ID].State != "error" || b.jobs[j.ID].Queue.RequestID != "" {
		t.Fatal(b.jobs[j.ID])
	}
	if _, e := a.readVoiceProfile("../elsewhere"); e == nil {
		t.Fatal("invalid profile accepted")
	}
	if r := request(t, a, "GET", "/api/local/voice-profiles", nil, "", false); r.Code != http.StatusForbidden {
		t.Fatal("unauthorized profiles accepted")
	}
}
func voiceForm(t *testing.T, fields map[string]string, uploads map[string]string) (*bytes.Buffer, string) {
	t.Helper()
	var b bytes.Buffer
	m := multipart.NewWriter(&b)
	for k, v := range fields {
		m.WriteField(k, v)
	}
	for k, p := range uploads {
		src, e := os.Open(p)
		if e != nil {
			t.Fatal(e)
		}
		dst, e := m.CreateFormFile(k, filepath.Base(p))
		if e != nil {
			t.Fatal(e)
		}
		io.Copy(dst, src)
		src.Close()
	}
	m.Close()
	return &b, m.FormDataContentType()
}
func waitVoice(t *testing.T, a *App, id string) Job {
	t.Helper()
	until := time.Now().Add(2 * time.Minute)
	for time.Now().Before(until) {
		a.mu.Lock()
		j := *a.jobs[id]
		a.mu.Unlock()
		if !j.busy {
			if j.State != "voice_done" {
				t.Fatal(j.State, j.Message)
			}
			return j
		}
		time.Sleep(100 * time.Millisecond)
	}
	t.Fatal("voice timeout")
	return Job{}
}
func TestOfflineVoiceRealModels(t *testing.T) {
	py := os.Getenv("JUNBA_TEST_LOCAL_PYTHON")
	models := os.Getenv("JUNBA_TEST_VOICE_MODELS")
	audio := os.Getenv("JUNBA_TEST_LOCAL_AUDIO")
	reference := os.Getenv("JUNBA_TEST_VOICE_REFERENCE")
	if py == "" || models == "" || audio == "" || reference == "" {
		t.Skip("set local runtime, voice model, baseline and reference paths for real offline test")
	}
	a := newApp(filepath.Join(t.TempDir(), "output"))
	os.MkdirAll(a.root, 0700)
	os.MkdirAll(filepath.Join(a.localDir(), "python"), 0700)
	os.MkdirAll(filepath.Join(a.localDir(), "models"), 0700)
	os.WriteFile(a.localPython(), []byte("#!/bin/sh\nexec '"+strings.ReplaceAll(py, "'", "'\"'\"'")+"' \"$@\"\n"), 0700)
	for _, m := range voiceModels() {
		if e := os.Symlink(filepath.Join(models, m.Name), filepath.Join(a.localDir(), "models", m.Name)); e != nil {
			t.Fatal(e)
		}
	}
	for _, name := range []string{"runner.py", "voice_runner.py", "mel_filters.npz"} {
		b, _ := localFiles.ReadFile("local/" + name)
		os.WriteFile(filepath.Join(a.localDir(), name), b, 0600)
	}
	os.WriteFile(filepath.Join(a.localDir(), "voice-installed.txt"), []byte(voiceModelVersion), 0600)
	a.tts = func(ctx context.Context, text, voice string, rate int, path string) error {
		if !strings.HasPrefix(text, "New local sentence") {
			return fmt.Errorf("new text not passed: %s", text)
		}
		return copyLocalFile(audio, path)
	}
	fields := map[string]string{"text": "New local sentence one.", "base_mode": "tts", "device": "CPU", "profile_name": "本機測試音色"}
	body, ct := voiceForm(t, fields, map[string]string{"sample": reference})
	w := request(t, a, "POST", "/api/local/voice-prepare", body, ct, true)
	if w.Code != 200 {
		t.Fatal(w.Code, w.Body.String())
	}
	var first Job
	json.Unmarshal(w.Body.Bytes(), &first)
	first = waitVoice(t, a, first.ID)
	if first.Engine != "local_voice" || !first.HasCaptions || first.VoiceID == "" || first.Duration < 3 || len(first.ActualDevices) != 2 || first.Queue.RequestID != "" {
		t.Fatal(first)
	}
	p, e := a.readVoiceProfile(first.VoiceID)
	if e != nil || p.Name != "本機測試音色" {
		t.Fatal(p, e)
	}
	fields["text"] = "New local sentence two."
	fields["profile_id"] = first.VoiceID
	body, ct = voiceForm(t, fields, nil)
	w = request(t, a, "POST", "/api/local/voice-prepare", body, ct, true)
	if w.Code != 200 {
		t.Fatal(w.Code, w.Body.String())
	}
	var second Job
	json.Unmarshal(w.Body.Bytes(), &second)
	second = waitVoice(t, a, second.ID)
	if second.VoiceID != first.VoiceID {
		t.Fatal("profile not reused")
	}
	t.Logf("real offline clone + profile reuse passed; %.3fs, %v", second.Duration, second.ActualDevices)
	// Reusing the actual voice WAV in the photo preparation path must not call TTS.
	a.tts = func(context.Context, string, string, int, string) error {
		return fmt.Errorf("unexpected TTS on imported cloned WAV")
	}
	body, ct = voiceForm(t, map[string]string{"text": second.Text, "caption_mode": "burn", "format": "mp4"}, map[string]string{"audio": filepath.Join(a.dir(second.ID), second.Audio)})
	w = request(t, a, "POST", "/api/prepare", body, ct, true)
	if w.Code != 200 {
		t.Fatal(w.Code, w.Body.String())
	}
	var photo Job
	json.Unmarshal(w.Body.Bytes(), &photo)
	if photo.State != "ready" || photo.Duration < 3 || !photo.HasCaptions {
		t.Fatal(photo)
	}
	lipModels := os.Getenv("JUNBA_TEST_LOCAL_MODELS")
	if lipModels == "" {
		return
	}
	for _, m := range localModels() {
		if e := os.Symlink(filepath.Join(lipModels, m.Name), filepath.Join(a.localDir(), "models", m.Name)); e != nil {
			t.Fatal(e)
		}
	}
	os.WriteFile(filepath.Join(a.localDir(), "installed.txt"), []byte("openvino-2026.4.1-v2"), 0600)
	q := LocalRequest{ID: photo.ID, Device: "CPU", BBox: photo.FaceBox, FPS: 6, MaxSide: 512, Format: "mp4", Aspect: "original", Resolution: "720", CaptionMode: "burn"}
	payload, _ := json.Marshal(q)
	w = request(t, a, "POST", "/api/local/generate", bytes.NewReader(payload), "application/json", true)
	if w.Code != 200 {
		t.Fatal(w.Code, w.Body.String())
	}
	deadline := time.Now().Add(8 * time.Minute)
	for time.Now().Before(deadline) {
		a.mu.Lock()
		j := *a.jobs[photo.ID]
		a.mu.Unlock()
		if !j.busy {
			if j.State != "done" || j.Export != "export.mp4" || !j.HasCaptions {
				t.Fatal(j.State, j.Message)
			}
			t.Logf("full offline voice -> lip sync -> subtitles -> MP4 passed (%d frames)", j.LocalFrame)
			return
		}
		time.Sleep(time.Second)
	}
	t.Fatal("lip sync timeout")
}
