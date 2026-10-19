package main

import (
	"context"
	"encoding/json"
	"errors"
	"io"
	"math"
	"net/http"
	"os"
	"path/filepath"
	"regexp"
	"strconv"
	"strings"
	"time"
)

func validRegion(box []float64) bool {
	if len(box) != 4 {
		return false
	}
	for _, v := range box {
		if math.IsNaN(v) || math.IsInf(v, 0) || v < 0 || v > 1 {
			return false
		}
	}
	return box[2] >= .005 && box[3] >= .003 && box[0]+box[2] <= 1.001 && box[1]+box[3] <= 1.001
}

var srtStamp = regexp.MustCompile(`\d{2}:\d{2}:\d{2},\d{3}`)

func shiftCaptions(dir string, delta float64) error {
	if math.Abs(delta) < .0001 {
		return nil
	}
	path := filepath.Join(dir, "subtitles.srt")
	b, e := os.ReadFile(path)
	if os.IsNotExist(e) {
		return nil
	}
	if e != nil {
		return e
	}
	shifted := srtStamp.ReplaceAllStringFunc(string(b), func(s string) string {
		fields := strings.FieldsFunc(s, func(r rune) bool { return r == ':' || r == ',' })
		vals := make([]int, 4)
		for i, v := range fields {
			vals[i], _ = strconv.Atoi(v)
		}
		return stamp(float64(vals[0]*3600+vals[1]*60+vals[2]) + float64(vals[3])/1000 + delta)
	})
	return os.WriteFile(path, []byte(shifted), 0600)
}

// Refresh bundled scripts at job start so an existing installation receives fixes
// without downloading the large model weights again. Caller owns localJob slot.
func (a *App) refreshLocalScripts() error {
	// v0.8.3b: include the Phase 1 contour modules; they were missing, so an
	// existing install kept its first-installed copies and never got fixes.
	for _, name := range []string{"semantic_mouth.py", "contour_mask.py", "contour_preflight.py", "cartoon_puppet.py", "stage_dispatch.py", "visual_polish.py", "body_motion.py", "micro_graph.py", "hardware_check.py", "detail_cache.py", "inference_backend.py", "backend_probe.py", "probe_matmul.onnx", "runner.py", "renderer.py", "quality.py", "model_cache.py", "stage_worker.py", "runtime_support.py", "face_worker.py", "face_geometry.py", "yunet.onnx", "voice_runner.py", "mel_filters.npz"} {
		b, e := localFiles.ReadFile("local/" + name)
		if e != nil {
			return e
		}
		if e = os.WriteFile(filepath.Join(a.localDir(), name), b, 0600); e != nil {
			return e
		}
	}
	return nil
}
func (a *App) rebuildLocal(w http.ResponseWriter, r *http.Request) {
	var q struct {
		ID string `json:"id"`
	}
	if json.NewDecoder(io.LimitReader(r.Body, 4096)).Decode(&q) != nil {
		fail(w, 400, errors.New("無效專案"))
		return
	}
	a.mu.Lock()
	defer a.mu.Unlock()
	old := a.jobs[q.ID]
	if old == nil || old.busy || old.State != "done" || (old.Kind != "photo" && old.Kind != "video") {
		fail(w, 409, errors.New("請選擇已完成的照片／影片專案"))
		return
	}
	j := *old
	j.ID = randomID()
	j.Created = time.Now().Format(time.RFC3339)
	j.State = "ready"
	j.Message = "已複製原始素材與語音，可調整嘴部、眼睛與畫質後重新生成。"
	j.Output = ""
	j.Export = ""
	j.Engine = ""
	j.Queue = QueueInfo{}
	j.Compatible = false
	j.ActualDevices = nil
	j.LocalSettings = nil
	j.MemoryPeaks = nil
	j.LocalStage = ""
	j.LocalFrame = 0
	j.LocalTotal = 0
	j.Encoder = ""
	j.key = ""
	j.CancelRequested = false
	j.LocalETA = 0
	j.LeadSeconds = 0
	dir := a.dir(j.ID)
	if e := os.MkdirAll(dir, 0700); e != nil {
		fail(w, 500, e)
		return
	}
	for _, name := range []string{old.Source, old.Audio, "subtitles.srt", "speech.wav.timing.tsv"} {
		if name == "" {
			continue
		}
		b, e := os.ReadFile(filepath.Join(a.dir(old.ID), name))
		if os.IsNotExist(e) && name != old.Source && name != old.Audio {
			continue
		}
		if e != nil {
			fail(w, 500, e)
			return
		}
		if e = os.WriteFile(filepath.Join(dir, name), b, 0600); e != nil {
			fail(w, 500, e)
			return
		}
	}
	if e := shiftCaptions(dir, -old.LeadSeconds); e != nil {
		fail(w, 500, e)
		return
	}
	a.jobs[j.ID] = &j
	a.persist(&j)
	respond(w, j)
}

func qualityModels() []LocalModel {
	b, _ := localFiles.ReadFile("local/quality_models.json")
	var v []LocalModel
	json.Unmarshal(b, &v)
	return v
}
func (a *App) qualityReady() bool {
	for _, m := range qualityModels() {
		s, e := os.Stat(filepath.Join(a.localDir(), "models", m.Name))
		if e != nil || s.Size() != m.Size {
			return false
		}
	}
	return len(qualityModels()) > 0
}
func (a *App) installQuality(w http.ResponseWriter, r *http.Request) {
	if !a.localReady() && !a.managedSadReady() {
		fail(w, 409, errors.New("請先安裝本機對嘴模型"))
		return
	}
	a.localMu.Lock()
	if a.localJob != "" || a.localState.Busy {
		a.localMu.Unlock()
		fail(w, 409, errors.New("請先等待目前工作完成"))
		return
	}
	a.localState.Busy = true
	a.localState.Message = "下載卡通嘴部細節模型（約 9 MB）"
	a.localMu.Unlock()
	go func() {
		ctx, cancel := context.WithTimeout(context.Background(), 30*time.Minute)
		defer cancel()
		var err error
		for _, m := range qualityModels() {
			if err = a.downloadLocalModel(ctx, m); err != nil {
				break
			}
		}
		a.localMu.Lock()
		a.localState.Busy = false
		if err != nil {
			a.localState.Message = "細節模型安裝失敗：" + err.Error()
		} else {
			a.localState.Message = "卡通嘴部細節模型安裝完成"
		}
		a.localMu.Unlock()
	}()
	respond(w, map[string]bool{"ok": true})
}

func realModels() []LocalModel {
	b, _ := localFiles.ReadFile("local/real_models.json")
	var v []LocalModel
	json.Unmarshal(b, &v)
	return v
}
func (a *App) realReady() bool {
	for _, m := range realModels() {
		s, e := os.Stat(filepath.Join(a.localDir(), "models", m.Name))
		if e != nil || s.Size() != m.Size {
			return false
		}
	}
	return len(realModels()) > 0
}
func (a *App) installReal(w http.ResponseWriter, r *http.Request) {
	if !a.localReady() && !a.managedSadReady() {
		fail(w, 409, errors.New("請先安裝本機對嘴模型"))
		return
	}
	a.localMu.Lock()
	if a.localJob != "" || a.localState.Busy {
		a.localMu.Unlock()
		fail(w, 409, errors.New("請先等待目前工作完成"))
		return
	}
	a.localState.Busy = true
	a.localState.Message = "下載真人嘴部修復模型（約 340 MB）"
	a.localMu.Unlock()
	go func() {
		ctx, cancel := context.WithTimeout(context.Background(), 30*time.Minute)
		defer cancel()
		var err error
		for _, m := range realModels() {
			if err = a.downloadLocalModel(ctx, m); err != nil {
				break
			}
		}
		a.localMu.Lock()
		a.localState.Busy = false
		if err != nil {
			a.localState.Message = "真人修復模型安裝失敗：" + err.Error()
		} else {
			a.localState.Message = "真人修復模型安裝完成，請先用短片比較唇齒形狀"
		}
		a.localMu.Unlock()
	}()
	respond(w, map[string]bool{"ok": true})
}
