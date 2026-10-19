package main

import (
	"archive/zip"
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

const sadManagedVersion = "sad512-accelerated-v08"
const sadManagedDirectory = "sad512-cpu-staged-v07"

func (a *App) installEngineHandler(w http.ResponseWriter, r *http.Request) {
	var q struct {
		Engine string `json:"engine"`
	}
	if json.NewDecoder(io.LimitReader(r.Body, 4096)).Decode(&q) != nil || !validExternalName(q.Engine) {
		fail(w, 400, errors.New("請選擇可用引擎"))
		return
	}
	if q.Engine == "latentsync" {
		fail(w, 422, errors.New("LatentSync 1.6 需要 NVIDIA CUDA 與至少 18 GiB 專用顯存；Intel Arc／NPU 不相容。本版未提供經驗證的 Windows 自動安裝，不能將它標為可執行模式。"))
		return
	}
	if runtime.GOOS != "windows" {
		fail(w, 400, errors.New("此自動安裝器適用 Windows x64"))
		return
	}
	a.localMu.Lock()
	if a.localJob != "" || a.localState.Busy {
		a.localMu.Unlock()
		fail(w, 409, errors.New("請等目前工作完成"))
		return
	}
	a.localState.Busy = true
	a.localState.Message = "正在準備 SadTalker 獨立環境，不會更改系統 Python"
	a.localMu.Unlock()
	go func() {
		e := a.installManagedSad()
		a.localMu.Lock()
		defer a.localMu.Unlock()
		a.localState.Busy = false
		a.localState.Downloaded = 0
		a.localState.Total = 0
		if e != nil {
			a.localState.Message = "SadTalker 安裝未完成，按同一按鈕可續傳／修復：" + e.Error()
		} else {
			a.localState.Message = "SadTalker v0.8 已自動安裝並通過套件、模型完整性檢查；可選擇照片製作。逐階段、逐格生成，實際品質請試做比較。"
		}
	}()
	respond(w, map[string]bool{"ok": true})
}

// Extract only runtime code/config plus license from the pinned upstream ZIP.
// No examples, executable shell scripts, or absolute/traversal paths are executed.
func extractSadSource(archive, root string) error {
	z, e := zip.OpenReader(archive)
	if e != nil {
		return e
	}
	defer z.Close()
	for _, f := range z.File {
		parts := strings.SplitN(f.Name, "/", 2)
		if len(parts) != 2 {
			continue
		}
		name := parts[1]
		if name != "inference.py" && name != "LICENSE" && !strings.HasPrefix(name, "src/") {
			continue
		}
		clean := filepath.Clean(filepath.FromSlash(name))
		if filepath.IsAbs(clean) || clean == ".." || strings.HasPrefix(clean, ".."+string(os.PathSeparator)) {
			return errors.New("不安全的模型原始碼壓縮路徑")
		}
		target := filepath.Join(root, clean)
		if f.FileInfo().IsDir() {
			if e = os.MkdirAll(target, 0700); e != nil {
				return e
			}
			continue
		}
		if f.UncompressedSize64 > 64<<20 {
			return errors.New("模型程式檔案超出大小限制")
		}
		if e = os.MkdirAll(filepath.Dir(target), 0700); e != nil {
			return e
		}
		in, err := f.Open()
		if err != nil {
			return err
		}
		out, err := os.Create(target)
		if err != nil {
			in.Close()
			return err
		}
		_, err = io.Copy(out, in)
		in.Close()
		ce := out.Close()
		if err != nil {
			return err
		}
		if ce != nil {
			return ce
		}
	}
	return nil
}

func writeManagedScripts(root string) error {
	if e := os.MkdirAll(filepath.Join(root, "junba"), 0700); e != nil {
		return e
	}
	for _, name := range []string{"sad_worker.py", "sad_export.py", "torch_grid3d.py", "semantic_mouth.py", "contour_mask.py", "contour_preflight.py", "stage_dispatch.py", "visual_polish.py", "body_motion.py", "micro_graph.py", "hardware_check.py", "detail_cache.py", "inference_backend.py", "backend_probe.py", "probe_matmul.onnx", "renderer.py", "model_cache.py", "sad_pipeline.py", "quality.py", "runtime_support.py"} {
		b, e := localFiles.ReadFile("local/" + name)
		if e != nil {
			return e
		}
		if e = os.WriteFile(filepath.Join(root, "junba", name), b, 0600); e != nil {
			return e
		}
	}
	return nil
}
func (a *App) installManagedSad() error {
	root := filepath.Join(filepath.Dir(a.root), "engines", sadManagedDirectory)
	if e := os.MkdirAll(root, 0700); e != nil {
		return e
	}
	release, e := lockInstallFile(filepath.Join(root, "install.lock"))
	if e != nil {
		return e
	}
	defer release()
	ctx, cancel := context.WithTimeout(context.Background(), 8*time.Hour)
	defer cancel()
	log, e := os.Create(filepath.Join(root, "install.log"))
	if e != nil {
		return e
	}
	defer log.Close()
	pyDir := filepath.Join(root, "python")
	b, _ := localFiles.ReadFile("local/python-3.11.9-embed-amd64.zip")
	if e = extractLocalArchive(b, pyDir); e != nil {
		return e
	}
	if e = os.WriteFile(filepath.Join(pyDir, "python311._pth"), []byte("python311.zip\n.\nLib\\site-packages\nimport site\n"), 0600); e != nil {
		return e
	}
	b, _ = localFiles.ReadFile("local/pip-25.0.1-py3-none-any.whl")
	if e = extractLocalArchive(b, filepath.Join(pyDir, "Lib", "site-packages")); e != nil {
		return e
	}
	var models, wheels []LocalModel
	b, _ = localFiles.ReadFile("local/sadtalker_models.json")
	if e = json.Unmarshal(b, &models); e != nil {
		return e
	}
	b, _ = localFiles.ReadFile("local/sadtalker_wheels.json")
	if e = json.Unmarshal(b, &wheels); e != nil {
		return e
	}
	for _, m := range models {
		if e = a.downloadArtifact(ctx, m, root); e != nil {
			return e
		}
	}
	if e = extractSadSource(filepath.Join(root, "source.zip"), root); e != nil {
		return e
	}
	wheelDir := filepath.Join(root, "wheels")
	if e = os.MkdirAll(wheelDir, 0700); e != nil {
		return e
	}
	var args = []string{"-m", "pip", "install", "--disable-pip-version-check", "--no-warn-script-location", "--no-index", "--no-deps", "--no-compile"}
	for _, m := range wheels {
		if strings.HasPrefix(m.URL, "bundled:") {
			data, err := localFiles.ReadFile("local/" + m.Name)
			if err != nil {
				return err
			}
			if err = os.WriteFile(filepath.Join(wheelDir, m.Name), data, 0600); err != nil {
				return err
			}
			h, err := fileSHA(filepath.Join(wheelDir, m.Name))
			if err != nil || h != m.SHA {
				return errors.New("隨附套件校驗失敗")
			}
		} else if e = a.downloadArtifact(ctx, m, wheelDir); e != nil {
			return e
		}
		args = append(args, filepath.Join(wheelDir, m.Name))
	}
	a.setupMessage("正在自動安裝 SadTalker 專用套件；無需系統 Python 或編譯工具", 0, 0)
	c := exec.CommandContext(ctx, filepath.Join(pyDir, "python.exe"), args...)
	c.Env = localEnv()
	c.Stdout = log
	c.Stderr = log
	hideCommand(c)
	if e = c.Run(); e != nil {
		return fmt.Errorf("獨立套件安裝失敗，詳細記錄在 %s", filepath.Join(root, "install.log"))
	}
	// Resolve neither dependencies nor source builds on the user's computer.
	c = exec.CommandContext(ctx, filepath.Join(pyDir, "python.exe"), "-m", "pip", "check")
	c.Env = localEnv()
	c.Stdout = log
	c.Stderr = log
	hideCommand(c)
	if e = c.Run(); e != nil {
		return fmt.Errorf("套件相容性檢查失敗，請查看 %s", filepath.Join(root, "install.log"))
	}
	if e = writeManagedScripts(root); e != nil {
		return e
	}
	// Some Windows installations do not have the Microsoft C++ prerequisite.
	c = exec.CommandContext(ctx, filepath.Join(pyDir, "python.exe"), "-I", "-c", "import torch")
	c.Env = localEnv()
	hideCommand(c)
	check, checkErr := c.CombinedOutput()
	log.Write(check)
	if checkErr != nil {
		text := strings.ToLower(string(check))
		if !strings.Contains(text, "dll") && !strings.Contains(text, "winerror") {
			return fmt.Errorf("PyTorch 驗證失敗，請查看 %s", filepath.Join(root, "install.log"))
		}
		var vc LocalModel
		data, _ := localFiles.ReadFile("local/sad_runtime.json")
		if e = json.Unmarshal(data, &vc); e != nil {
			return e
		}
		a.setupMessage("準備 Microsoft 官方 C++ 執行階段；Windows 可能要求系統授權", 0, vc.Size)
		if e = a.downloadArtifact(ctx, vc, root); e != nil {
			return e
		}
		if e = installVCRuntime(ctx, filepath.Join(root, vc.Name)); e != nil {
			return e
		}
	}
	a.setupMessage("檢查 SadTalker 的完整套件與權重；此時尚未執行使用者影片", 0, 0)
	code := `import sys;sys.path.insert(0,sys.argv[1]);import torch,cv2,librosa,safetensors,onnx,openvino;from src.utils.preprocess import CropAndExtract;from src.test_audio2coeff import Audio2Coeff;from src.facerender.animate import AnimateFromCoeff;print('SAD_READY',torch.__version__)`
	c = exec.CommandContext(ctx, filepath.Join(pyDir, "python.exe"), "-I", "-c", code, root)
	c.Dir = root
	c.Env = localEnv()
	c.Stdout = log
	c.Stderr = log
	hideCommand(c)
	if e = c.Run(); e != nil {
		return fmt.Errorf("套件驗證失敗（可按安裝修復），詳細記錄在 %s", filepath.Join(root, "install.log"))
	}
	if e = os.WriteFile(filepath.Join(root, "installed.txt"), []byte(sadManagedVersion), 0600); e != nil {
		return e
	}
	cfg := ExternalConfig{Engine: "sadtalker", Root: root, Python: filepath.Join(pyDir, "python.exe"), CPU: true, Managed: true}
	b, _ = json.MarshalIndent(cfg, "", "  ")
	tmp := a.externalConfigPath("sadtalker") + ".tmp"
	if e = os.WriteFile(tmp, b, 0600); e != nil {
		return e
	}
	return os.Rename(tmp, a.externalConfigPath("sadtalker"))
}
