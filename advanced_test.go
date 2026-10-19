package main

import (
	"bytes"
	"context"
	"encoding/json"
	"fmt"
	"image"
	"image/color"
	"image/png"
	"os"
	"os/exec"
	"path/filepath"
	"runtime"
	"strings"
	"testing"
	"time"
)

func TestAdvancedValidationAndLiteralArguments(t *testing.T) {
	q := LocalRequest{BBox: []float64{.2, .2, .5, .5}}
	if e := validateLocal(&q); e != nil || q.Engine != "musetalk" || q.ExpressionScale != 1 || q.InferenceSteps != 20 {
		t.Fatal(e, q)
	}
	q.Engine = "online-free-api"
	if validateLocal(&q) == nil {
		t.Fatal("invalid engine accepted")
	}
	q.Engine = "sadtalker"
	q.PoseStyle = 46
	if validateLocal(&q) == nil {
		t.Fatal("bad pose")
	}
	q.PoseStyle = 2
	q.ExpressionScale = 1.2
	c := ExternalConfig{Engine: "sadtalker", Python: "/python", Root: "/path with spaces; literal", CPU: true}
	args := externalArgs(c, q, "/source ; &.png", "/speech ' quote.wav", "/out.mp4", "/results")
	if args[1] != filepath.Join(c.Root, "inference.py") || !strings.Contains(strings.Join(args, "\n"), "/speech ' quote.wav") || args[len(args)-1] != "--cpu" {
		t.Fatal(args)
	}
	// v0.8.3b: an unusable point count (e.g. 96 contour points) is dropped,
	// not fatal; bad coordinates are still rejected.
	q.Landmarks = make([][]float64, 96)
	for i := range q.Landmarks {
		q.Landmarks[i] = []float64{.5, .5}
	}
	if e := validateLocal(&q); e != nil || q.Landmarks != nil {
		t.Fatal("unusable point count should be dropped", e)
	}
	q.Landmarks = [][]float64{{.5, .5}, {.5, .5}, {.5, .5}, {.5, .5}, {1.5, .5}}
	if validateLocal(&q) == nil {
		t.Fatal("out-of-range landmark accepted")
	}
	q.Landmarks = nil
	a := newApp(t.TempDir())
	if r := request(t, a, "GET", "/api/local/engines", nil, "", false); r.Code != 403 {
		t.Fatal("external settings need session")
	}
	if r := request(t, a, "POST", "/api/local/face-detect", strings.NewReader(`{}`), "application/json", true); r.Code != 409 {
		t.Fatal(r.Code, r.Body.String())
	}
}

func TestExternalAdapterProcessAndExport(t *testing.T) {
	if runtime.GOOS == "windows" {
		t.Skip("POSIX fake process adapter test")
	}
	py, e := exec.LookPath("python3")
	if e != nil {
		t.Skip("Python unavailable")
	}
	ff, e := ffmpegPath()
	if e != nil {
		t.Skip("FFmpeg unavailable")
	}
	a := newApp(filepath.Join(t.TempDir(), "output"))
	os.MkdirAll(a.root, 0700)
	root := filepath.Join(t.TempDir(), "外部 引擎 ; literal")
	os.MkdirAll(filepath.Join(root, "checkpoints"), 0700)
	wrapper := filepath.Join(root, "python-adapter")
	script := "#!/bin/sh\nif [ \"$1\" = \"-c\" ]; then printf '%s\\n' '{\"cuda\":false,\"vram\":0}'; exit 0; fi\nexec '" + strings.ReplaceAll(py, "'", "'\\''") + "' \"$@\"\n"
	os.WriteFile(wrapper, []byte(script), 0700)
	for _, name := range []string{"SadTalker_V0.0.2_512.safetensors", "mapping_00109-model.pth.tar"} {
		os.WriteFile(filepath.Join(root, "checkpoints", name), []byte("adapter fixture, not AI weights"), 0600)
	}
	// The fixture exercises invocation and postprocessing only, not neural quality.
	inference := `import argparse,subprocess,os
p=argparse.ArgumentParser()
for k in ['source_image','driven_audio','checkpoint_dir','result_dir','preprocess','size','batch_size','pose_style','expression_scale']:p.add_argument('--'+k)
p.add_argument('--cpu',action='store_true');a=p.parse_args();assert a.size=='512' and a.cpu
subprocess.run([os.environ['JUNBA_ADAPTER_FFMPEG'],'-v','error','-nostdin','-y','-loop','1','-i',a.source_image,'-i',a.driven_audio,'-t','4','-r','6','-vf','scale=128:192','-c:v','libx264','-pix_fmt','yuv420p','-c:a','aac','-threads','2',os.path.join(a.result_dir,'fixture.mp4')],check=True)
`
	os.WriteFile(filepath.Join(root, "inference.py"), []byte(inference), 0600)
	t.Setenv("JUNBA_ADAPTER_FFMPEG", ff)
	c := ExternalConfig{Engine: "sadtalker", Python: wrapper, Root: root, CPU: true}
	payload, _ := json.Marshal(c)
	w := request(t, a, "POST", "/api/local/engines", bytes.NewReader(payload), "application/json", true)
	if w.Code != 200 {
		t.Fatal(w.Code, w.Body.String())
	}
	j := prepared(t, a, "photo")
	img := image.NewRGBA(image.Rect(0, 0, 128, 192))
	for y := 0; y < 192; y++ {
		for x := 0; x < 128; x++ {
			img.Set(x, y, color.RGBA{150, 180, 130, 255})
		}
	}
	f, _ := os.Create(filepath.Join(a.dir(j.ID), "source.jpg"))
	png.Encode(f, img)
	f.Close()
	q := LocalRequest{ID: j.ID, Engine: "sadtalker", BBox: []float64{.2, .2, .5, .5}, FPS: 6, MaxSide: 720, Format: "mp4", CaptionMode: "none", ExpressionScale: 1.2, PoseStyle: 2}
	payload, _ = json.Marshal(q)
	w = request(t, a, "POST", "/api/local/generate", bytes.NewReader(payload), "application/json", true)
	if w.Code != 200 {
		t.Fatal(w.Code, w.Body.String())
	}
	deadline := time.Now().Add(30 * time.Second)
	for time.Now().Before(deadline) {
		a.mu.Lock()
		done := !a.jobs[j.ID].busy
		state := a.jobs[j.ID].State
		msg := a.jobs[j.ID].Message
		out := a.jobs[j.ID].Export
		a.mu.Unlock()
		if done {
			if state != "done" || out != "export.mp4" {
				t.Fatal(state, msg, out)
			}
			return
		}
		time.Sleep(30 * time.Millisecond)
	}
	t.Fatal("adapter timed out")
}

func TestLatentSyncRejectsNonCUDAEvenWithModelPaths(t *testing.T) {
	if runtime.GOOS == "windows" {
		t.Skip("POSIX test fixture")
	}
	root := t.TempDir()
	for _, name := range []string{"scripts/inference.py", "configs/unet/stage2_512.yaml", "checkpoints/latentsync_unet.pt", "checkpoints/whisper/tiny.pt"} {
		path := filepath.Join(root, filepath.FromSlash(name))
		os.MkdirAll(filepath.Dir(path), 0700)
		os.WriteFile(path, []byte("adapter fixture"), 0600)
	}
	python := filepath.Join(root, "python-check")
	os.WriteFile(python, []byte("#!/bin/sh\nprintf '%s\\n' '{\"cuda\":false,\"vram\":0}'\n"), 0700)
	_, e := probeExternal(context.Background(), ExternalConfig{Engine: "latentsync", Python: python, Root: root})
	if e == nil || !strings.Contains(e.Error(), "CUDA") {
		t.Fatal("non-CUDA hardware accepted", e)
	}
}

func TestActualFaceDetectionAPI(t *testing.T) {
	models, source, py := os.Getenv("JUNBA_TEST_FACE_MODELS"), os.Getenv("JUNBA_TEST_FACE_SOURCE"), os.Getenv("JUNBA_TEST_LOCAL_PYTHON")
	if models == "" || source == "" || py == "" {
		t.Skip("actual face model paths not provided")
	}
	if runtime.GOOS == "windows" {
		t.Skip("test wrapper is POSIX")
	}
	a := newApp(filepath.Join(t.TempDir(), "output"))
	os.MkdirAll(a.root, 0700)
	os.MkdirAll(filepath.Join(a.localDir(), "models"), 0700)
	os.MkdirAll(filepath.Dir(a.localPython()), 0700)
	os.WriteFile(a.localPython(), []byte("#!/bin/sh\nexec '"+strings.ReplaceAll(py, "'", "'\\''")+"' \"$@\"\n"), 0700)
	for _, m := range faceModels() {
		if e := os.Symlink(filepath.Join(models, m.Name), filepath.Join(a.localDir(), "models", m.Name)); e != nil {
			t.Fatal(e)
		}
	}
	os.WriteFile(filepath.Join(a.localDir(), "face-installed.txt"), []byte(faceModelVersion), 0600)
	j := prepared(t, a, "photo")
	b, e := os.ReadFile(source)
	if e != nil {
		t.Fatal(e)
	}
	os.WriteFile(filepath.Join(a.dir(j.ID), j.Source), b, 0600)
	w := request(t, a, "POST", "/api/local/face-detect", strings.NewReader(fmt.Sprintf(`{"id":%q}`, j.ID)), "application/json", true)
	if w.Code != 200 {
		t.Fatal(w.Code, w.Body.String())
	}
	var result struct {
		Method    string      `json:"method"`
		Landmarks [][]float64 `json:"landmarks"`
		Mouth     []float64   `json:"mouth_box"`
		Eyes      [][]float64 `json:"eye_boxes"`
	}
	if e = json.Unmarshal(w.Body.Bytes(), &result); e != nil || result.Method != "landmarks98" || len(result.Landmarks) != 98 || len(result.Eyes) != 2 || !validRegion(result.Mouth) {
		t.Fatal(e, w.Body.String())
	}
	p := ExternalConfig{Engine: "latentsync", Python: "/missing", Root: "/missing"}
	if _, e = probeExternal(context.Background(), p); e == nil {
		t.Fatal("missing environment accepted")
	}
}
