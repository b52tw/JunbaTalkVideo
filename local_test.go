package main

import (
	"archive/zip"
	"bytes"
	"context"
	"crypto/sha256"
	"encoding/hex"
	"encoding/json"
	"net/http"
	"net/http/httptest"
	"os"
	"path/filepath"
	"strings"
	"testing"
	"time"
)

func TestLocalDownloadResumeAndIntegrity(t *testing.T) {
	data := []byte("verified model bytes")
	h := sha256.Sum256(data)
	a := newApp(filepath.Join(t.TempDir(), "output"))
	os.MkdirAll(filepath.Join(a.localDir(), "models"), 0700)
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if r.Header.Get("Range") != "bytes=5-" {
			t.Error("missing resume range", r.Header.Get("Range"))
		}
		w.Header().Set("Content-Range", "bytes 5-19/20")
		w.WriteHeader(206)
		w.Write(data[5:])
	}))
	defer server.Close()
	m := LocalModel{Name: "test.onnx", Size: int64(len(data)), SHA: hex.EncodeToString(h[:]), URL: server.URL}
	p := filepath.Join(a.localDir(), "models", m.Name)
	os.WriteFile(p+".part", data[:5], 0600)
	if e := a.downloadLocalModel(context.Background(), m); e != nil {
		t.Fatal(e)
	}
	b, _ := os.ReadFile(p)
	if string(b) != string(data) {
		t.Fatal("corrupt result")
	}
	// A completed partial file left by an interrupted verification needs no GET.
	os.Rename(p, p+".part")
	if e := a.downloadLocalModel(context.Background(), m); e != nil {
		t.Fatal(e)
	}
	// A server returning incorrect bytes cannot create a trusted model.
	m.SHA = strings.Repeat("0", 64)
	os.Remove(p)
	os.WriteFile(p+".part", data[:5], 0600)
	if e := a.downloadLocalModel(context.Background(), m); e == nil {
		t.Fatal("hash mismatch accepted")
	}
	if _, e := os.Stat(p); e == nil {
		t.Fatal("published corrupt model")
	}
}

func TestLocalValidationAndNoCloudFallback(t *testing.T) {
	a := newApp(filepath.Join(t.TempDir(), "output"))
	q := LocalRequest{Device: "NPU", BBox: []float64{.2, .2, .4, .4}}
	if e := validateLocal(&q); e != nil || q.Device != "NPU" || q.FPS != 12 {
		t.Fatal(e, q)
	}
	for _, box := range [][]float64{{.9, .2, .4, .4}, {.2, .2, 0, .3}, {-1, .2, .3, .3}} {
		q.BBox = box
		if validateLocal(&q) == nil {
			t.Fatal("accepted invalid box", box)
		}
	}
	j := prepared(t, a, "photo")
	w := request(t, a, "POST", "/api/local/generate", strings.NewReader(`{"id":"`+j.ID+`","device":"GPU","bbox":[0.2,0.2,0.4,0.4]}`), "application/json", true)
	if w.Code != 409 || j.State != "ready" || j.Queue.RequestID != "" {
		t.Fatal(w.Code, w.Body.String(), j)
	}
	if r := request(t, a, "GET", "/api/local/status", nil, "", false); r.Code != 403 {
		t.Fatal("unauthed local access")
	}
}
func TestLocalResourceOptions(t *testing.T) {
	valid := func() LocalRequest { return LocalRequest{BBox: []float64{.2, .2, .4, .4}} }
	q := valid()
	if err := validateLocal(&q); err != nil || q.MemoryMode != "staged" || q.ResourceMode != "gentle" || q.DetailStrength != .5 {
		t.Fatal("resource defaults", err, q)
	}
	for _, change := range []func(*LocalRequest){
		func(q *LocalRequest) { q.MemoryMode = "concurrent" },
		func(q *LocalRequest) { q.ResourceMode = "unlimited" },
		func(q *LocalRequest) { q.RealDetail = true; q.AnimeDetail = true },
		func(q *LocalRequest) { q.DetailStrength = 1.1 },
	} {
		q = valid()
		change(&q)
		if validateLocal(&q) == nil {
			t.Fatal("invalid resource options accepted", q)
		}
	}
	q = valid()
	q.MemoryMode = "standard"
	q.ResourceMode = "normal"
	q.RealDetail = true
	q.DetailStrength = .35
	if err := validateLocal(&q); err != nil {
		t.Fatal("supported options", err)
	}
}

func TestLocalInterruptedRecovery(t *testing.T) {
	a := newApp(t.TempDir())
	j := prepared(t, a, "photo")
	j.Engine = "local"
	j.State = "local_generating"
	a.persist(j)
	b := newApp(a.root)
	b.load()
	if b.jobs[j.ID].State != "ready" {
		t.Fatal(b.jobs[j.ID])
	}
}
func TestSafeRuntimeExtraction(t *testing.T) {
	var b bytes.Buffer
	z := zip.NewWriter(&b)
	f, _ := z.Create("../escape")
	f.Write([]byte("x"))
	z.Close()
	if extractLocalArchive(b.Bytes(), t.TempDir()) == nil {
		t.Fatal("traversal accepted")
	}
}
func TestQSVArgumentMapping(t *testing.T) {
	args := qsvArgs(convertArgs("in.mp4", "out.mp4", "mp4", "original", "720"))
	s := strings.Join(args, " ")
	if !strings.Contains(s, "-c:v h264_qsv") || strings.Contains(s, "-crf") || !strings.Contains(s, "-global_quality 18") || !strings.Contains(s, "-pix_fmt nv12") {
		t.Fatal(s)
	}
}

// Opt-in integration uses the real downloaded weights, isolated Python runtime,
// user cartoon fixture, and existing conversion/subtitle/export HTTP endpoints.
func TestLocalModelEndToEnd(t *testing.T) {
	models := os.Getenv("JUNBA_TEST_LOCAL_MODELS")
	py := os.Getenv("JUNBA_TEST_LOCAL_PYTHON")
	audio := os.Getenv("JUNBA_TEST_LOCAL_AUDIO")
	if models == "" || py == "" || audio == "" {
		t.Skip("set JUNBA_TEST_LOCAL_MODELS, JUNBA_TEST_LOCAL_PYTHON, JUNBA_TEST_LOCAL_AUDIO for real-model test")
	}
	a := newApp(filepath.Join(t.TempDir(), "output"))
	os.MkdirAll(a.root, 0700)
	os.MkdirAll(filepath.Join(a.localDir(), "python"), 0700)
	os.MkdirAll(filepath.Join(a.localDir(), "models"), 0700)
	isolation := ""
	if os.Getenv("JUNBA_TEST_ISOLATED_PYTHON") == "1" {
		isolation = " -I"
	}
	if e := os.WriteFile(a.localPython(), []byte("#!/bin/sh\nexec '"+strings.ReplaceAll(py, "'", "'\"'\"'")+"'"+isolation+" \"$@\"\n"), 0700); e != nil {
		t.Fatal(e)
	}
	testModels := localModels()
	if os.Getenv("JUNBA_TEST_QUALITY") == "1" {
		testModels = append(testModels, qualityModels()...)
	}
	if os.Getenv("JUNBA_TEST_REAL_DETAIL") == "1" {
		testModels = append(testModels, realModels()...)
	}
	for _, m := range testModels {
		if e := os.Symlink(filepath.Join(models, m.Name), filepath.Join(a.localDir(), "models", m.Name)); e != nil {
			t.Fatal(e)
		}
	}
	for _, name := range []string{"runner.py", "mel_filters.npz"} {
		b, _ := localFiles.ReadFile("local/" + name)
		os.WriteFile(filepath.Join(a.localDir(), name), b, 0600)
	}
	os.WriteFile(filepath.Join(a.localDir(), "installed.txt"), []byte("openvino-2026.4.1-v2"), 0600)
	j := prepared(t, a, "photo")
	b, _ := files.ReadFile("assets/default.jpg")
	os.WriteFile(filepath.Join(a.dir(j.ID), j.Source), b, 0600)
	b, _ = os.ReadFile(audio)
	os.WriteFile(filepath.Join(a.dir(j.ID), j.Audio), b, 0600)
	j.Duration, _ = wavDuration(filepath.Join(a.dir(j.ID), j.Audio))
	j.HasCaptions = true
	j.CaptionMode = "burn"
	writeCaptions(a.dir(j.ID), "Hello. Welcome to my video.", j.Duration)
	q := LocalRequest{ID: j.ID, Device: "CPU", BBox: []float64{.28, .25, .47, .29}, FPS: 6, MaxSide: 512, Format: "mp4", Aspect: "original", Resolution: "720", CaptionMode: "burn"}
	if os.Getenv("JUNBA_TEST_QUALITY") == "1" {
		q.Restore = true
		q.FPS = 25
		q.MaxSide = 1920
		q.AnimeDetail = true
		q.MouthBox = []float64{.414, .409, .207, .060}
		j.Text = "大家好，我是峻爸！"
		writeCaptions(a.dir(j.ID), j.Text, j.Duration)
	}
	if isolation != "" {
		q.FPS = 6
		q.MaxSide = 720
		q.Restore = true
		q.AnimeDetail = true
		q.BodyMotion = "nod"
		q.Blink = true
		q.Gaze = true
		q.EyeBoxes = [][]float64{{.401, .345, .071, .024}, {.565, .348, .067, .023}}
	}
	if os.Getenv("JUNBA_TEST_REAL_DETAIL") == "1" {
		q.RealDetail = true
		q.AnimeDetail = false
		q.MemoryMode = "staged"
	}
	payload, _ := json.Marshal(q)
	w := request(t, a, "POST", "/api/local/generate", bytes.NewReader(payload), "application/json", true)
	if w.Code != 200 {
		t.Fatal(w.Code, w.Body.String())
	}
	deadline := time.Now().Add(15 * time.Minute)
	for time.Now().Before(deadline) {
		a.mu.Lock()
		snapshot := *a.jobs[j.ID]
		a.mu.Unlock()
		if !snapshot.busy {
			if snapshot.State != "done" {
				if output := os.Getenv("JUNBA_TEST_QUALITY_OUTPUT"); output != "" {
					os.MkdirAll(output, 0700)
					b, _ := os.ReadFile(filepath.Join(a.dir(j.ID), "local.log"))
					os.WriteFile(filepath.Join(output, "failed-local.log"), b, 0600)
				}
				t.Fatal(snapshot.Message)
			}
			if snapshot.Engine != "local" || snapshot.Export != "export.mp4" || len(snapshot.ActualDevices) == 0 || snapshot.LocalFrame != snapshot.LocalTotal {
				t.Fatal(snapshot)
			}
			if q.Restore && snapshot.LeadSeconds < .5 {
				t.Fatal("missing padded source holds")
			}
			if q.AnimeDetail && len(snapshot.ActualDevices["anime_mouth_detail"]) == 0 {
				t.Fatal("detail model did not execute")
			}
			if q.RealDetail && len(snapshot.ActualDevices["real_mouth_detail"]) == 0 {
				t.Fatal("real detail model did not execute")
			}
			if snapshot.LocalSettings.MemoryMode == "staged" {
				for _, phase := range []string{"audio", "encode", "infer", "decode", "compose"} {
					if snapshot.MemoryPeaks[phase] <= 0 {
						t.Fatal("stage RAM peak not recorded", phase, snapshot.MemoryPeaks)
					}
				}
			}

			if output := os.Getenv("JUNBA_TEST_QUALITY_OUTPUT"); output != "" {
				os.MkdirAll(output, 0700)
				snapshotJSON, _ := json.MarshalIndent(snapshot, "", "  ")
				os.WriteFile(filepath.Join(output, "job-snapshot.json"), snapshotJSON, 0600)
				for _, name := range []string{"video.mp4", "export.mp4", "subtitles.srt", "local.log", "conversion.log", "local-config.json"} {
					b, e := os.ReadFile(filepath.Join(a.dir(j.ID), name))
					if e != nil {
						t.Fatal(e)
					}
					if e = os.WriteFile(filepath.Join(output, name), b, 0600); e != nil {
						t.Fatal(e)
					}
				}
			}
			t.Logf("real MuseTalk completed %d frames; devices: %v", snapshot.LocalFrame, snapshot.ActualDevices)
			return
		}
		time.Sleep(time.Second)
	}
	t.Fatal("real-model test timed out")
}
