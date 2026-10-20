package main

import (
	"context"
	"encoding/json"
	"errors"
	"io"
	"net/http"
	"os"
	"os/exec"
	"path/filepath"
	"runtime"
	"strings"
	"time"
)

const directMLVersion = "ort-directml-1.24.4-v08"

func directMLReady(root string) bool {
	b, e := os.ReadFile(filepath.Join(root, "directml-installed.txt"))
	if e == nil && string(b) == directMLVersion {
		return true
	}
	if filepath.Base(root) == "local-v0.2" {
		if _, e := os.Stat(filepath.Join(root, "python", "python.exe")); os.IsNotExist(e) {
			if p := portablePython(); p != "" {
				b, e := os.ReadFile(filepath.Join(filepath.Dir(filepath.Dir(p)), "directml-installed.txt"))
				return e == nil && string(b) == directMLVersion
			}
		}
	}
	return false
}
func v08Config(cfg map[string]any, q LocalRequest) {
	cfg["color_transfer"] = q.ColorTransfer
	cfg["lid_shadow"] = q.LidShadow
	cfg["peak_budget_mb"] = workerBudget(q.PeakBudgetMB, memoryInfo())
	cfg["runtime"] = q.Runtime
	cfg["dml_adapter"] = q.DMLAdapter
	cfg["npu_assist"] = q.NPUAssist
	cfg["blend_mode"] = q.BlendMode
	cfg["mouth_scope"] = q.MouthScope
	cfg["compose_style"] = q.ComposeStyle
	cfg["engine"] = q.Engine
	cfg["puppet_style"] = q.PuppetStyle
	cfg["puppet_open"] = q.PuppetOpen
	cfg["head_motion"] = q.HeadMotion
	cfg["body_amp"] = q.BodyAmp
	cfg["motion_timeline"] = q.MotionTimeline
	if len(q.Speakers) > 0 {
		cfg["speakers"] = q.Speakers
	}
	cfg["composition_mode"] = q.CompositionMode
	cfg["blink_profile"] = q.BlinkProfile
	cfg["brow_motion"] = q.BrowMotion
}
func (a *App) managedSadReady() bool {
	c, e := a.loadExternal("sadtalker")
	return e == nil && c.Managed && externalFiles(c) == nil
}
func (a *App) backendEnv(engine string) (py, root, scripts string, err error) {
	if engine == "sadtalker" {
		c, e := a.loadExternal(engine)
		if e != nil || !c.Managed {
			return "", "", "", errors.New("請先一鍵安裝／升級 SadTalker")
		}
		if e = externalFiles(c); e != nil {
			return "", "", "", e
		}
		e = writeManagedScripts(c.Root)
		return c.Python, c.Root, filepath.Join(c.Root, "junba"), e
	}
	if engine != "musetalk" && engine != "" {
		return "", "", "", errors.New("不支援的引擎")
	}
	if !a.localReady() {
		return "", "", "", errors.New("請先安裝本機模型")
	}
	e := a.refreshLocalScripts()
	return a.localPython(), a.localDir(), a.localDir(), e
}
func (a *App) v08Backend(w http.ResponseWriter, r *http.Request) {
	var q struct {
		Engine     string `json:"engine"`
		Runtime    string `json:"runtime"`
		Device     string `json:"device"`
		DMLAdapter int    `json:"dml_adapter"`
	}
	if json.NewDecoder(io.LimitReader(r.Body, 4096)).Decode(&q) != nil {
		fail(w, 400, errors.New("無效資料"))
		return
	}
	install := strings.HasSuffix(r.URL.Path, "directml-install")
	if install && runtime.GOOS != "windows" {
		fail(w, 400, errors.New("DirectML 自動安裝僅適用 Windows x64"))
		return
	}
	if !install {
		if (q.Runtime != "openvino" && q.Runtime != "directml") || q.DMLAdapter < 0 || q.DMLAdapter > 7 {
			fail(w, 400, errors.New("無效後端"))
			return
		}
		if q.Device != "CPU" && q.Device != "AUTO" && q.Device != "GPU" && q.Device != "GPU.0" && q.Device != "GPU.1" && q.Device != "NPU" {
			fail(w, 400, errors.New("無效裝置"))
			return
		}
		if q.Runtime == "directml" && q.Device == "NPU" {
			fail(w, 400, errors.New("DirectML 適用顯卡；NPU 請選 OpenVINO"))
			return
		}
	}
	a.localMu.Lock()
	if a.localState.Busy || a.localJob != "" {
		a.localMu.Unlock()
		fail(w, 409, errors.New("請等目前工作完成"))
		return
	}
	a.localState.Busy = true
	a.localMu.Unlock()
	unlock := func() { a.localMu.Lock(); a.localState.Busy = false; a.localMu.Unlock() }
	py, root, scripts, e := a.backendEnv(q.Engine)
	if e != nil {
		unlock()
		fail(w, 409, e)
		return
	}
	if install {
		go func() {
			e := a.installDirectML(py, root, scripts)
			a.localMu.Lock()
			a.localState.Busy = false
			a.localState.Downloaded = 0
			a.localState.Total = 0
			if e != nil {
				a.localState.Message = "DirectML 安裝未完成：" + e.Error()
			} else {
				a.localState.Message = "DirectML DLL／CPU 校驗完成；請測試所選後端，確認實際顯卡節點。"
			}
			a.localMu.Unlock()
		}()
		respond(w, map[string]bool{"ok": true})
		return
	}
	defer unlock()
	dir, e := os.MkdirTemp(filepath.Dir(a.root), "backend-probe-")
	if e != nil {
		fail(w, 500, e)
		return
	}
	defer os.RemoveAll(dir)
	cfg := map[string]any{"runtime": q.Runtime, "device": q.Device, "dml_adapter": q.DMLAdapter, "output": filepath.Join(dir, "test.mp4")}
	b, _ := json.Marshal(cfg)
	path := filepath.Join(dir, "config.json")
	if e = os.WriteFile(path, b, 0600); e != nil {
		fail(w, 500, e)
		return
	}
	ctx, cancel := context.WithTimeout(r.Context(), 3*time.Minute)
	defer cancel()
	c := exec.CommandContext(ctx, py, "-I", "-u", filepath.Join(scripts, "backend_probe.py"), "--config", path)
	c.Env = localEnv()
	hideCommand(c)
	out, e := c.CombinedOutput()
	if e != nil {
		fail(w, 422, errors.New(tailText(out, 1400)))
		return
	}
	for _, line := range strings.Split(string(out), "\n") {
		var v map[string]any
		if json.Unmarshal([]byte(line), &v) == nil && v["probe_ok"] == true {
			respond(w, v)
			return
		}
	}
	fail(w, 500, errors.New("後端未回報推論結果"))
}
func (a *App) installDirectML(py, root, scripts string) error {
	release, e := lockInstallFile(filepath.Join(root, "directml-install.lock"))
	if e != nil {
		return e
	}
	defer release()
	ctx, cancel := a.installContext(2 * time.Hour)
	defer cancel()
	var rows []LocalModel
	b, _ := localFiles.ReadFile("local/directml_wheels.json")
	if e = json.Unmarshal(b, &rows); e != nil {
		return e
	}
	folder := filepath.Join(root, "dml-wheels")
	if e = os.MkdirAll(folder, 0700); e != nil {
		return e
	}
	args := []string{"-m", "pip", "install", "--no-index", "--no-deps", "--no-compile", "--disable-pip-version-check"}
	for _, m := range rows {
		if e = a.downloadArtifact(ctx, m, folder); e != nil {
			return e
		}
		args = append(args, filepath.Join(folder, m.Name))
	}
	log, e := os.Create(filepath.Join(root, "directml-install.log"))
	if e != nil {
		return e
	}
	defer log.Close()
	c := exec.CommandContext(ctx, py, args...)
	c.Env = localEnv()
	hideCommand(c)
	c.Stdout = log
	c.Stderr = log
	if e = c.Run(); e != nil {
		return errors.New("套件安裝失敗，請查看 directml-install.log")
	}
	b, _ = json.Marshal(map[string]any{"runtime": "directml", "device": "CPU", "output": filepath.Join(root, "backend-probe", "test.mp4")})
	path := filepath.Join(root, "directml-probe.json")
	if e = os.WriteFile(path, b, 0600); e != nil {
		return e
	}
	c = exec.CommandContext(ctx, py, "-I", "-u", filepath.Join(scripts, "backend_probe.py"), "--config", path)
	c.Env = localEnv()
	hideCommand(c)
	c.Stdout = log
	c.Stderr = log
	if e = c.Run(); e != nil {
		return errors.New("DLL／CPU 推論校驗失敗，請查看 directml-install.log")
	}
	return os.WriteFile(filepath.Join(root, "directml-installed.txt"), []byte(directMLVersion), 0600)
}
