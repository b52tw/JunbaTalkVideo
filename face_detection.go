package main

import (
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"net/http"
	"os"
	"os/exec"
	"path/filepath"
	"runtime"
	"strings"
	"time"
)

const faceModelVersion = "yunet98-v1"

func faceModels() []LocalModel {
	b, _ := localFiles.ReadFile("local/face_models.json")
	var m []LocalModel
	json.Unmarshal(b, &m)
	return m
}
func (a *App) faceReady() bool {
	b, e := os.ReadFile(filepath.Join(a.localDir(), "face-installed.txt"))
	if e != nil || string(b) != faceModelVersion {
		return false
	}
	if _, e = os.Stat(a.localPython()); e != nil {
		return false
	}
	for _, m := range faceModels() {
		st, e := os.Stat(filepath.Join(a.localDir(), "models", m.Name))
		if e != nil || st.Size() != m.Size {
			return false
		}
	}
	return len(faceModels()) == 2
}
func (a *App) installFace(w http.ResponseWriter, r *http.Request) {
	if runtime.GOOS != "windows" {
		fail(w, 400, errors.New("自動安裝適用 Windows x64"))
		return
	}
	a.localMu.Lock()
	if a.localState.Busy || a.localJob != "" {
		a.localMu.Unlock()
		fail(w, 409, errors.New("請先等待目前工作完成"))
		return
	}
	a.localState.Busy = true
	a.localState.Message = "安裝本機 AI 五官定位（模型約 21 MB，共用環境未安裝時另需約 185 MB）"
	a.localMu.Unlock()
	go func() {
		e := a.installModelSet(faceModels(), "face-installed.txt", faceModelVersion)
		a.localMu.Lock()
		defer a.localMu.Unlock()
		a.localState.Busy = false
		if e != nil {
			a.localState.Message = "五官定位安裝未完成：" + e.Error()
		} else {
			a.localState.Message = "AI 五官模型已安裝，可自動偵測臉部、嘴唇與眼睛。"
		}
	}()
	respond(w, map[string]bool{"ok": true})
}
func (a *App) detectFace(w http.ResponseWriter, r *http.Request) {
	if !a.faceReady() {
		fail(w, 409, errors.New("請先安裝「AI 五官定位模型」"))
		return
	}
	a.localMu.Lock()
	if a.localState.Busy || a.localJob != "" {
		a.localMu.Unlock()
		fail(w, 409, errors.New("請先等待目前工作完成"))
		return
	}
	a.localState.Busy = true
	a.localMu.Unlock()
	defer func() { a.localMu.Lock(); a.localState.Busy = false; a.localMu.Unlock() }()
	if e := os.MkdirAll(a.localDir(), 0700); e != nil {
		fail(w, 500, e)
		return
	}
	dir, e := os.MkdirTemp(a.localDir(), "face-check-")
	if e != nil {
		fail(w, 500, e)
		return
	}
	defer os.RemoveAll(dir)
	source, kind := "", "photo"
	var preservedEyes [][]float64
	if strings.HasPrefix(r.Header.Get("Content-Type"), "multipart/form-data") {
		r.Body = http.MaxBytesReader(w, r.Body, 22<<20)
		if e = r.ParseMultipartForm(1 << 20); e != nil {
			fail(w, 400, errors.New("素材無法讀取或超過 20 MB"))
			return
		}
		defer r.MultipartForm.RemoveAll()
		f, h, e := r.FormFile("source")
		if e != nil {
			fail(w, 400, errors.New("請選擇照片或影片"))
			return
		}
		f.Close()
		ext := strings.ToLower(filepath.Ext(h.Filename))
		switch ext {
		case ".jpg", ".jpeg", ".png", ".webp":
		case ".mp4", ".mov", ".webm":
			kind = "video"
		default:
			fail(w, 400, errors.New("素材格式不支援"))
			return
		}
		source = filepath.Join(dir, "source"+ext)
		if _, e = copyUpload(r, "source", source, maxMedia); e != nil {
			fail(w, 400, e)
			return
		}
	} else {
		var q struct {
			ID string `json:"id"`
		}
		if json.NewDecoder(io.LimitReader(r.Body, 4096)).Decode(&q) != nil {
			fail(w, 400, errors.New("無效專案"))
			return
		}
		a.mu.Lock()
		j := a.jobs[q.ID]
		if j != nil && !j.busy && (j.Kind == "photo" || j.Kind == "video") {
			source = filepath.Join(a.dir(j.ID), j.Source)
			kind = j.Kind
			if j.LocalSettings != nil {
				preservedEyes = j.LocalSettings.EyeBoxes
			}
		}
		a.mu.Unlock()
		if source == "" {
			fail(w, 409, errors.New("找不到可檢查的照片／影片專案"))
			return
		}
	}
	if e = a.refreshLocalScripts(); e != nil {
		fail(w, 500, e)
		return
	}
	ff, e := ffmpegPath()
	if e != nil {
		fail(w, 500, e)
		return
	}
	cfg := map[string]any{"source": source, "kind": kind, "models": filepath.Join(a.localDir(), "models"), "device": "AUTO", "ffmpeg": ff, "result": filepath.Join(dir, "result.json"), "low_impact": true}
	cfg["preserve_eye_boxes"] = preservedEyes
	b, _ := json.Marshal(cfg)
	config := filepath.Join(dir, "config.json")
	if e = os.WriteFile(config, b, 0600); e != nil {
		fail(w, 500, e)
		return
	}
	ctx, cancel := context.WithTimeout(r.Context(), 90*time.Second)
	defer cancel()
	c := exec.CommandContext(ctx, a.localPython(), "-I", "-u", filepath.Join(a.localDir(), "face_worker.py"), "--config", config)
	c.Env = localEnv()
	hideCommand(c)
	out, e := c.CombinedOutput()
	if e != nil {
		message := "AI 五官定位失敗"
		for _, line := range strings.Split(string(out), "\n") {
			var event struct{ Stage, Message string }
			if json.Unmarshal([]byte(line), &event) == nil && event.Stage == "error" {
				message = event.Message
			}
		}
		fail(w, 422, fmt.Errorf("%s；請改用手動框選", message))
		return
	}
	b, e = os.ReadFile(filepath.Join(dir, "result.json"))
	if e != nil {
		fail(w, 500, e)
		return
	}
	var result map[string]any
	if e = json.Unmarshal(b, &result); e != nil {
		fail(w, 500, e)
		return
	}
	respond(w, result)
}
