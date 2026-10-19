package main

import (
	"bytes"
	"context"
	"encoding/json"
	"errors"
	"io"
	"net/http"
	"os"
	"os/exec"
	"path/filepath"
	"strings"
	"time"
)

func portablePython() string {
	exe, e := os.Executable()
	if e != nil {
		return ""
	}
	base := filepath.Join(filepath.Dir(exe), "runtime", "base")
	b, e := os.ReadFile(filepath.Join(base, "bundle.json"))
	if e != nil {
		return ""
	}
	var v struct {
		ID string `json:"id"`
	}
	if json.Unmarshal(b, &v) != nil || v.ID != "junba-sdk-v081" {
		return ""
	}
	p := filepath.Join(base, "python", "python.exe")
	if st, e := os.Stat(p); e == nil && st.Mode().IsRegular() {
		return p
	}
	return ""
}
func (a *App) selfCheck(w http.ResponseWriter, r *http.Request) {
	if r.Method == "GET" {
		id := r.URL.Query().Get("id")
		if id == "" || filepath.Base(id) != id || strings.ContainsAny(id, "./\\") {
			fail(w, 400, errors.New("無效報告 ID"))
			return
		}
		http.ServeFile(w, r, filepath.Join(filepath.Dir(a.root), "diagnostics", id, "report.json"))
		return
	}
	var q struct {
		Engine, Mode, Device string
		FullHash             bool `json:"full_hash"`
	}
	if json.NewDecoder(io.LimitReader(r.Body, 4096)).Decode(&q) != nil {
		fail(w, 400, errors.New("無效資料"))
		return
	}
	if q.Mode != "enumerate" && q.Mode != "full" {
		fail(w, 400, errors.New("無效診斷模式"))
		return
	}
	if q.Device == "" {
		q.Device = "AUTO"
	}
	if q.Device != "AUTO" && q.Device != "CPU" && q.Device != "GPU" && q.Device != "GPU.0" && q.Device != "GPU.1" && q.Device != "NPU" {
		fail(w, 400, errors.New("無效裝置"))
		return
	}
	a.localMu.Lock()
	if a.localState.Busy || a.localJob != "" {
		a.localMu.Unlock()
		fail(w, 409, errors.New("請等安裝或生成結束再診斷"))
		return
	}
	a.localState.Busy = true
	a.localMu.Unlock()
	defer func() { a.localMu.Lock(); a.localState.Busy = false; a.localMu.Unlock() }()
	py, root, scripts := a.localPython(), a.localDir(), a.localDir()
	models := localModels()
	modelRoot := filepath.Join(root, "models")
	if q.Engine == "sadtalker" {
		root = filepath.Join(filepath.Dir(a.root), "engines", sadManagedDirectory)
		if c, e := a.loadExternal(q.Engine); e == nil {
			root = c.Root
			py = c.Python
		} else {
			py = filepath.Join(root, "python", "python.exe")
		}
		scripts = filepath.Join(root, "junba")
		models = nil
		b, _ := localFiles.ReadFile("local/sadtalker_models.json")
		json.Unmarshal(b, &models)
		modelRoot = root
	} else if q.Engine != "musetalk" && q.Engine != "" {
		fail(w, 400, errors.New("不支援的引擎"))
		return
	}
	if _, e := os.Stat(py); e != nil {
		respond(w, map[string]any{"runtime_available": false, "message": "尚無可用 Python 子模組。可下載 GitHub 的 EmbeddedPortable 包，或按程式內的一鍵安裝；不用自行安裝 Python。", "backends": []any{}})
		return
	}
	if e := os.MkdirAll(scripts, 0700); e != nil {
		fail(w, 500, e)
		return
	}
	for _, name := range []string{"hardware_check.py", "micro_graph.py", "inference_backend.py", "runtime_support.py"} {
		b, e := localFiles.ReadFile("local/" + name)
		if e != nil {
			fail(w, 500, e)
			return
		}
		if e = os.WriteFile(filepath.Join(scripts, name), b, 0600); e != nil {
			fail(w, 500, e)
			return
		}
	}
	id := time.Now().Format("20060102-150405") + "-" + randomID()
	dir := filepath.Join(filepath.Dir(a.root), "diagnostics", id)
	if e := os.MkdirAll(dir, 0700); e != nil {
		fail(w, 500, e)
		return
	}
	files := []map[string]any{}
	for _, m := range models {
		files = append(files, map[string]any{"name": m.Name, "path": filepath.Join(modelRoot, filepath.FromSlash(m.Name)), "size": m.Size, "sha256": m.SHA})
	}
	report := filepath.Join(dir, "report.json")
	cfg := map[string]any{"mode": q.Mode, "device": q.Device, "report": report, "files": files, "full_hash": q.FullHash}
	b, _ := json.Marshal(cfg)
	config := filepath.Join(dir, "config.json")
	if e := os.WriteFile(config, b, 0600); e != nil {
		fail(w, 500, e)
		return
	}
	ctx, cancel := context.WithTimeout(r.Context(), 3*time.Minute)
	defer cancel()
	c := exec.CommandContext(ctx, py, "-I", "-u", filepath.Join(scripts, "hardware_check.py"), "--config", config)
	c.Env = localEnv()
	hideCommand(c)
	var out bytes.Buffer
	c.Stdout = &out
	c.Stderr = &out
	e := c.Start()
	if e == nil {
		var release func()
		release, e = limitWorkerMemory(c, 200)
		if e != nil {
			c.Process.Kill()
			c.Wait()
		} else {
			e = c.Wait()
			release()
		}
	}
	os.WriteFile(filepath.Join(dir, "diagnostic.log"), out.Bytes(), 0600)
	var result map[string]any
	if data, re := os.ReadFile(report); re == nil {
		json.Unmarshal(data, &result)
	}
	if result == nil {
		result = map[string]any{"micro_only": true, "real_weights_loaded": false, "error": tailText(out.Bytes(), 5000), "message": "診斷子模組未完成。主程式仍可使用；請查看報告並修復 Python／DLL。"}
		if e != nil {
			result["process_error"] = e.Error()
		}
	}
	result["runtime_available"] = true
	result["report_url"] = "/api/local/self-check?id=" + id
	result["memory_budget_note"] = "Windows 子程序族私有提交預算 200 MiB；映射頁／GPU 驅動另計，峰值以報告為準。"
	result["memory"] = memoryInfo()
	b, _ = json.MarshalIndent(result, "", "  ")
	os.WriteFile(report, b, 0600)
	respond(w, result)
}
