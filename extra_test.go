package main

import (
	"bytes"
	"context"
	"encoding/json"
	"fmt"
	"net/http"
	"net/http/httptest"
	"os"
	"os/exec"
	"path/filepath"
	"strings"
	"testing"
	"time"
)

func TestCaptionsTimingAndUnicode(t *testing.T) {
	dir := t.TempDir()
	os.WriteFile(filepath.Join(dir, "speech.wav.timing.tsv"), []byte("0\t0.0\n6\t2.0\n12\t4.0\n"), 0600)
	ok, timing, e := writeCaptions(dir, "大家好我是峻爸。今天一起嘗試😀，找到新的方向！", 8)
	if e != nil || !ok || !strings.Contains(timing, "語音進度") {
		t.Fatal(ok, timing, e)
	}
	b, _ := os.ReadFile(filepath.Join(dir, "subtitles.srt"))
	if !bytes.Contains(b, []byte("😀")) || !bytes.Contains(b, []byte("00:00:00,000")) {
		t.Fatalf("bad captions %s", b)
	}
	ok, timing, e = writeCaptions(t.TempDir(), "這是自行錄製的旁白字幕。", 5)
	if e != nil || !ok || !strings.Contains(timing, "估時") {
		t.Fatal("fallback not labelled")
	}
}
func TestBurnedCaptionsAndCartoonAudio(t *testing.T) {
	ff, e := ffmpegPath()
	if e != nil {
		t.Skip(e)
	}
	dir := t.TempDir()
	if e = prepareFont(dir); e != nil {
		t.Fatal(e)
	}
	writeCaptions(dir, "大家好，我是峻爸！字幕測試。", 1)
	src := filepath.Join(dir, "source.mp4")
	if b, e := exec.Command(ff, "-v", "error", "-f", "lavfi", "-i", "color=c=blue:s=320x240:r=12", "-t", "1", "-c:v", "libx264", "-y", src).CombinedOutput(); e != nil {
		t.Fatal(e, string(b))
	}
	out := filepath.Join(dir, "subbed.mp4")
	cmd := exec.Command(ff, addSubtitleFilter(convertArgs(src, out, "mp4", "original", "720"))...)
	cmd.Dir = dir
	if b, e := cmd.CombinedOutput(); e != nil {
		t.Fatal(e, string(b))
	}
	wav := filepath.Join(dir, "input.wav")
	testWAV(t, wav, 4)
	for _, effect := range []string{"original", "cartoon", "bright", "deep"} {
		dst := filepath.Join(dir, effect+".wav")
		if e := audioWAV(context.Background(), wav, dst, effect); e != nil {
			t.Fatal(effect, e)
		}
		d, e := wavDuration(dst)
		if e != nil || d < 3.8 || d > 4.2 {
			t.Fatal(effect, d, e)
		}
	}
}
func TestVoiceCloneAndReuse(t *testing.T) {
	for _, task := range []string{"clone", "speak"} {
		t.Run(task, func(t *testing.T) {
			a := newApp(t.TempDir())
			a.pollDelay = time.Millisecond
			id := randomID()
			os.MkdirAll(a.dir(id), 0700)
			j := &Job{ID: id, Kind: "voice", State: "ready", VoiceTask: task, VoiceID: "existing-voice", Text: "大家好，我是峻爸。", Audio: "speech.wav", Source: "sample.wav"}
			a.jobs[id] = j
			testWAV(t, filepath.Join(a.dir(id), j.Source), 12)
			audio := filepath.Join(t.TempDir(), "audio.wav")
			testWAV(t, audio, 4)
			audioBytes, _ := os.ReadFile(audio)
			var s *httptest.Server
			s = httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
				switch {
				case r.Method == "POST":
					expected := "/fal-ai/minimax/voice-clone"
					if task == "speak" {
						expected = "/fal-ai/minimax/speech-02-hd"
					}
					if r.URL.Path != expected {
						t.Error("wrong voice endpoint", r.URL.Path)
					}
					var input map[string]any
					json.NewDecoder(r.Body).Decode(&input)
					if task == "clone" && input["text"] != j.Text {
						t.Error("new script missing")
					}
					if task == "speak" && input["voice_setting"].(map[string]any)["voice_id"] != "existing-voice" {
						t.Error("voice ID lost")
					}
					respond(w, QueueInfo{RequestID: "voice-request", StatusURL: s.URL + "/fal-ai/test/status", ResponseURL: s.URL + "/fal-ai/test/result", CancelURL: s.URL + "/fal-ai/test/cancel"})
				case strings.HasSuffix(r.URL.Path, "status"):
					respond(w, map[string]string{"status": "COMPLETED"})
				case strings.HasSuffix(r.URL.Path, "result"):
					respond(w, map[string]any{"custom_voice_id": "new-voice-id", "audio": map[string]string{"url": s.URL + "/speech.wav"}})
				case r.URL.Path == "/speech.wav":
					if r.Header.Get("Authorization") != "" {
						t.Error("key leaked")
					}
					w.Write(audioBytes)
				}
			}))
			defer s.Close()
			a.queueBase = s.URL
			body := fmt.Sprintf(`{"id":%q,"key":"voice-secret"}`, id)
			w := request(t, a, "POST", "/api/generate", strings.NewReader(body), "application/json", true)
			if w.Code != 200 {
				t.Fatal(w.Body.String())
			}
			finished := false
			for n := 0; n < 500; n++ {
				a.mu.Lock()
				busy, state, msg := j.busy, j.State, j.Message
				a.mu.Unlock()
				if !busy {
					finished = true
					if state != "voice_done" {
						t.Fatalf("%s %s", state, msg)
					}
					break
				}
				time.Sleep(10 * time.Millisecond)
			}
			if !finished {
				t.Fatal("voice timeout")
			}
			if j.VoiceID != "new-voice-id" || !j.HasCaptions {
				t.Fatal("voice result incomplete")
			}
			b, _ := os.ReadFile(filepath.Join(a.dir(id), "project.json"))
			if bytes.Contains(b, []byte("voice-secret")) {
				t.Fatal("key stored")
			}
		})
	}
}
