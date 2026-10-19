package main

import (
	"bytes"
	"context"
	"encoding/binary"
	"encoding/json"
	"fmt"
	"io"
	"mime/multipart"
	"net/http"
	"net/http/httptest"
	"os"
	"os/exec"
	"path/filepath"
	"strings"
	"sync/atomic"
	"testing"
	"time"
)

func testWAV(t *testing.T, path string, seconds int) {
	t.Helper()
	samples := 8000 * seconds
	b := make([]byte, 44+samples*2)
	copy(b, "RIFF")
	binary.LittleEndian.PutUint32(b[4:], uint32(len(b)-8))
	copy(b[8:], "WAVEfmt ")
	binary.LittleEndian.PutUint32(b[16:], 16)
	binary.LittleEndian.PutUint16(b[20:], 1)
	binary.LittleEndian.PutUint16(b[22:], 1)
	binary.LittleEndian.PutUint32(b[24:], 8000)
	binary.LittleEndian.PutUint32(b[28:], 16000)
	binary.LittleEndian.PutUint16(b[32:], 2)
	binary.LittleEndian.PutUint16(b[34:], 16)
	copy(b[36:], "data")
	binary.LittleEndian.PutUint32(b[40:], uint32(samples*2))
	if e := os.WriteFile(path, b, 0600); e != nil {
		t.Fatal(e)
	}
}
func TestWAVValidation(t *testing.T) {
	p := filepath.Join(t.TempDir(), "voice.wav")
	testWAV(t, p, 4)
	d, e := wavDuration(p)
	if e != nil || d != 4 {
		t.Fatalf("%v %v", d, e)
	}
	b, _ := os.ReadFile(p)
	b[40] = 255
	b[41] = 255
	b[42] = 255
	b[43] = 127
	os.WriteFile(p, b, 0600)
	if _, e = wavDuration(p); e == nil {
		t.Fatal("accepted truncated WAV")
	}
}
func request(t *testing.T, a *App, method, path string, body io.Reader, content string, authorized bool) *httptest.ResponseRecorder {
	t.Helper()
	r := httptest.NewRequest(method, "http://127.0.0.1:8888"+path, body)
	r.Header.Set("Origin", "http://127.0.0.1:8888")
	r.Header.Set("Content-Type", content)
	if authorized {
		r.AddCookie(&http.Cookie{Name: a.cookieName(), Value: a.token})
	}
	w := httptest.NewRecorder()
	a.handler(w, r)
	return w
}
func prepared(t *testing.T, a *App, kind string) *Job {
	t.Helper()
	id := randomID()
	os.MkdirAll(a.dir(id), 0700)
	j := &Job{ID: id, Kind: kind, Created: time.Now().Format(time.RFC3339), Text: "測試", Source: "source.jpg", Audio: "speech.wav", State: "ready", Format: "mp4", Aspect: "original", Resolution: "720", Duration: 4}
	if kind == "video" {
		j.Source = "source.mp4"
	}
	os.WriteFile(filepath.Join(a.dir(id), j.Source), []byte("source-do-not-modify"), 0600)
	testWAV(t, filepath.Join(a.dir(id), j.Audio), 4)
	a.jobs[id] = j
	a.persist(j)
	return j
}
func TestPrepareAndSession(t *testing.T) {
	a := newApp(t.TempDir())
	a.tts = func(ctx context.Context, text, voice string, rate int, path string) error {
		if text != "你好！" {
			t.Error("text lost")
		}
		testWAV(t, path, 4)
		return nil
	}
	if r := request(t, a, "GET", "/api/jobs", nil, "", false); r.Code != 403 {
		t.Fatal("missing auth accepted")
	}
	var body bytes.Buffer
	m := multipart.NewWriter(&body)
	m.WriteField("text", "你好！")
	m.WriteField("format", "webm")
	m.Close()
	w := request(t, a, "POST", "/api/prepare", &body, m.FormDataContentType(), true)
	if w.Code != 200 {
		t.Fatal(w.Code, w.Body.String())
	}
	var j Job
	json.Unmarshal(w.Body.Bytes(), &j)
	if j.Format != "webm" || j.Duration != 4 {
		t.Fatalf("wrong prepared job %+v", j)
	}
	r := httptest.NewRequest("POST", "http://127.0.0.1:8888/api/generate", strings.NewReader(`{}`))
	r.AddCookie(&http.Cookie{Name: a.cookieName(), Value: a.token})
	r.Header.Set("Origin", "http://evil.example")
	wr := httptest.NewRecorder()
	a.handler(wr, r)
	if wr.Code != 403 {
		t.Fatal("cross origin accepted")
	}
}
func waitDone(t *testing.T, a *App, id string) {
	t.Helper()
	until := time.Now().Add(12 * time.Second)
	for time.Now().Before(until) {
		a.mu.Lock()
		j := a.jobs[id]
		busy, state, msg := j.busy, j.State, j.Message
		a.mu.Unlock()
		if !busy {
			if state != "done" {
				t.Fatalf("%s: %s", state, msg)
			}
			return
		}
		time.Sleep(10 * time.Millisecond)
	}
	t.Fatal("job timed out")
}
func TestFalPhotoAndVideoRoundTrip(t *testing.T) {
	for _, kind := range []string{"photo", "video"} {
		t.Run(kind, func(t *testing.T) {
			a := newApp(t.TempDir())
			a.pollDelay = time.Millisecond
			j := prepared(t, a, kind)
			var submits atomic.Int32
			var server *httptest.Server
			server = httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
				switch {
				case r.Method == "POST":
					submits.Add(1)
					if r.Header.Get("Authorization") != "Key secret-test-key" {
						t.Error("missing auth")
					}
					expected := "/" + photoModel
					if kind == "video" {
						expected = "/" + videoModel
					}
					if r.URL.Path != expected {
						t.Error("wrong model", r.URL.Path)
					}
					var in map[string]any
					json.NewDecoder(r.Body).Decode(&in)
					field := "image_url"
					if kind == "video" {
						field = "video_url"
						if in["sync_mode"] != "loop" {
							t.Error("wrong sync mode")
						}
					}
					if !strings.HasPrefix(in[field].(string), "data:") || !strings.HasPrefix(in["audio_url"].(string), "data:") {
						t.Error("expected embedded file inputs")
					}
					respond(w, QueueInfo{RequestID: "request-1", StatusURL: server.URL + "/fal-ai/test/status", ResponseURL: server.URL + "/fal-ai/test/result", CancelURL: server.URL + "/fal-ai/test/cancel"})
				case strings.HasSuffix(r.URL.Path, "status"):
					respond(w, map[string]string{"status": "COMPLETED"})
				case strings.HasSuffix(r.URL.Path, "result"):
					respond(w, map[string]any{"video": map[string]string{"url": server.URL + "/movie.mp4"}})
				case r.URL.Path == "/movie.mp4":
					if r.Header.Get("Authorization") != "" {
						t.Error("key leaked to download")
					}
					w.Write([]byte("\x00\x00\x00\x18ftypmp42-test-mp4-data"))
				default:
					http.NotFound(w, r)
				}
			}))
			defer server.Close()
			a.queueBase = server.URL
			body := fmt.Sprintf(`{"id":%q,"key":"secret-test-key","format":"mp4","aspect":"original"}`, j.ID)
			w := request(t, a, "POST", "/api/generate", strings.NewReader(body), "application/json", true)
			if w.Code != 200 {
				t.Fatal(w.Body.String())
			}
			waitDone(t, a, j.ID)
			b, _ := os.ReadFile(filepath.Join(a.dir(j.ID), j.Source))
			if string(b) != "source-do-not-modify" {
				t.Fatal("source modified")
			}
			meta, _ := os.ReadFile(filepath.Join(a.dir(j.ID), "project.json"))
			if bytes.Contains(meta, []byte("secret-test-key")) {
				t.Fatal("key persisted")
			}
			w = request(t, a, "POST", "/api/generate", strings.NewReader(body), "application/json", true)
			if w.Code != 409 || submits.Load() != 1 {
				t.Fatal("duplicate billed request")
			}
			fresh := newApp(a.root)
			fresh.load()
			if fresh.jobs[j.ID].State != "done" {
				t.Fatal("history not preserved")
			}
		})
	}
}
func TestUnknownSubmissionIsNotRetried(t *testing.T) {
	a := newApp(t.TempDir())
	j := prepared(t, a, "photo")
	var n atomic.Int32
	s := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) { n.Add(1); http.Error(w, "network error", 503) }))
	defer s.Close()
	a.queueBase = s.URL
	body := fmt.Sprintf(`{"id":%q,"key":"test"}`, j.ID)
	request(t, a, "POST", "/api/generate", strings.NewReader(body), "application/json", true)
	for i := 0; i < 200; i++ {
		a.mu.Lock()
		busy := j.busy
		a.mu.Unlock()
		if !busy {
			break
		}
		time.Sleep(5 * time.Millisecond)
	}
	a.mu.Lock()
	state := j.State
	a.mu.Unlock()
	if state != "uncertain" || n.Load() != 1 {
		t.Fatalf("state %s calls %d", state, n.Load())
	}
	w := request(t, a, "POST", "/api/generate", strings.NewReader(body), "application/json", true)
	if w.Code != 409 || n.Load() != 1 {
		t.Fatal("uncertain submission allowed retry")
	}
}
func TestResumeUsesOriginalRequest(t *testing.T) {
	a := newApp(t.TempDir())
	j := prepared(t, a, "photo")
	s := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if r.Method == "POST" {
			t.Error("resume resubmitted")
		}
		if strings.HasSuffix(r.URL.Path, "status") {
			respond(w, map[string]string{"status": "COMPLETED", "error": "test expected failure"})
		} else {
			http.NotFound(w, r)
		}
	}))
	defer s.Close()
	a.queueBase = s.URL
	j.Queue = QueueInfo{RequestID: "original", StatusURL: s.URL + "/fal-ai/test/status", ResponseURL: s.URL + "/fal-ai/test/result", CancelURL: s.URL + "/fal-ai/test/cancel"}
	j.State = "waiting_resume"
	a.persist(j)
	w := request(t, a, "POST", "/api/resume", strings.NewReader(fmt.Sprintf(`{"id":%q,"key":"test"}`, j.ID)), "application/json", true)
	if w.Code != 200 {
		t.Fatal(w.Body.String())
	}
	for i := 0; i < 200; i++ {
		a.mu.Lock()
		busy := j.busy
		a.mu.Unlock()
		if !busy {
			return
		}
		time.Sleep(5 * time.Millisecond)
	}
	t.Fatal("resume timed out")
}
func TestActualConversions(t *testing.T) {
	if _, err := exec.LookPath("ffprobe"); err != nil {
		t.Skip("ffprobe is required for stream inspection")
	}
	ff, e := ffmpegPath()
	if e != nil {
		t.Skip("ffmpeg unavailable")
	}
	dir := t.TempDir()
	source := filepath.Join(dir, "source.mp4")
	c := exec.Command(ff, "-v", "error", "-f", "lavfi", "-i", "color=c=blue:s=160x120:r=12", "-f", "lavfi", "-i", "sine=frequency=440:sample_rate=16000", "-t", "1", "-c:v", "libx264", "-pix_fmt", "yuv420p", "-c:a", "aac", "-y", source)
	if b, e := c.CombinedOutput(); e != nil {
		t.Fatal(e, string(b))
	}
	for _, format := range []string{"mp4", "wmv", "mov", "mkv", "webm", "avi", "m4v", "gif"} {
		t.Run(format, func(t *testing.T) {
			out := filepath.Join(dir, "output."+format)
			args := convertArgs(source, out, format, "original", "720")
			if b, e := exec.Command(ff, args...).CombinedOutput(); e != nil {
				t.Fatal(e, string(b))
			}
			b, e := exec.Command("ffprobe", "-v", "error", "-show_entries", "stream=codec_type,width,height", "-of", "json", out).Output()
			if e != nil {
				t.Fatal(e)
			}
			var probe struct {
				Streams []struct {
					Type string `json:"codec_type"`
					W    int    `json:"width"`
					H    int    `json:"height"`
				} `json:"streams"`
			}
			json.Unmarshal(b, &probe)
			video, audio := false, false
			for _, s := range probe.Streams {
				if s.Type == "video" {
					video = true
				}
				if s.Type == "audio" {
					audio = true
				}
			}
			if !video || audio != (format != "gif") {
				t.Fatalf("wrong streams %s", b)
			}
		})
	}
	out := filepath.Join(dir, "portrait.mp4")
	if b, e := exec.Command(ff, convertArgs(source, out, "mp4", "9:16", "720")...).CombinedOutput(); e != nil {
		t.Fatal(e, string(b))
	}
	b, _ := exec.Command("ffprobe", "-v", "error", "-select_streams", "v:0", "-show_entries", "stream=width,height", "-of", "csv=p=0", out).Output()
	if strings.TrimSpace(string(b)) != "720,1280" {
		t.Fatalf("wrong portrait dimensions: %s", b)
	}
}
