package main

import (
	"archive/zip"
	"bufio"
	"bytes"
	"context"
	"crypto/sha256"
	"embed"
	"encoding/hex"
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"math"
	"net/http"
	"os"
	"os/exec"
	"path/filepath"
	"runtime"
	"strings"
	"time"
)

//go:embed local/*.py local/*.npz local/*.json local/*.txt local/*.zip local/*.whl local/*.onnx
var localFiles embed.FS

type LocalModel struct {
	Name string `json:"name"`
	URL  string `json:"url"`
	SHA  string `json:"sha256"`
	Size int64  `json:"size"`
}
type LocalDevice struct {
	ID   string `json:"id"`
	Name string `json:"name"`
}
type LocalStatus struct {
	DirectMLReady bool          `json:"directml_ready"`
	FaceReady     bool          `json:"face_ready"`
	RealReady     bool          `json:"real_ready"`
	QualityReady  bool          `json:"quality_ready"`
	Ready         bool          `json:"ready"`
	VoiceReady    bool          `json:"voice_ready"`
	Busy          bool          `json:"busy"`
	Message       string        `json:"message"`
	Downloaded    int64         `json:"downloaded"`
	Total         int64         `json:"total"`
	Devices       []LocalDevice `json:"devices"`
	Runtime       string        `json:"runtime"`
	Memory        MemoryInfo    `json:"memory"`
}

func localModels() []LocalModel {
	b, _ := localFiles.ReadFile("local/models.json")
	var v []LocalModel
	json.Unmarshal(b, &v)
	return v
}
func (a *App) localDir() string { return filepath.Join(filepath.Dir(a.root), "local-v0.2") }
func (a *App) localPython() string {
	p := filepath.Join(a.localDir(), "python", "python.exe")
	if _, e := os.Stat(p); e == nil {
		return p
	}
	if sidecar := portablePython(); sidecar != "" {
		return sidecar
	}
	return p
}
func (a *App) localReady() bool {
	b, e := os.ReadFile(filepath.Join(a.localDir(), "installed.txt"))
	if e != nil || string(b) != "openvino-2026.4.1-v2" {
		return false
	}
	if _, e = os.Stat(a.localPython()); e != nil {
		return false
	}
	for _, m := range localModels() {
		s, e := os.Stat(filepath.Join(a.localDir(), "models", m.Name))
		if e != nil || s.Size() != m.Size {
			return false
		}
	}
	return true
}
func (a *App) setupMessage(msg string, n, total int64) {
	a.localMu.Lock()
	a.localState.Message = msg
	a.localState.Downloaded = n
	a.localState.Total = total
	a.localMu.Unlock()
}
func localEnv() []string {
	return append(os.Environ(), "PYTHONUTF8=1", "PYTHONUNBUFFERED=1", "OPENVINO_TELEMETRY=0", "OMP_NUM_THREADS=2", "OPENBLAS_NUM_THREADS=2", "MKL_NUM_THREADS=2", "NUMEXPR_NUM_THREADS=2")
}
func parseLocalProbe(out []byte) (LocalStatus, error) {
	lines := bytes.Split(bytes.TrimSpace(out), []byte("\n"))
	for i := len(lines) - 1; i >= 0; i-- {
		var s LocalStatus
		if json.Unmarshal(bytes.TrimSpace(lines[i]), &s) == nil && len(s.Devices) > 0 {
			return s, nil
		}
	}
	return LocalStatus{}, errors.New("裝置偵測沒有回傳可用裝置")
}
func (a *App) localHandler(w http.ResponseWriter, r *http.Request) {
	switch {
	case r.URL.Path == "/api/local/self-check" && (r.Method == "POST" || r.Method == "GET"):
		a.selfCheck(w, r)
	case r.URL.Path == "/api/local/error-report" && r.Method == "GET":
		a.errorReport(w, r)
	case (r.URL.Path == "/api/local/directml-install" || r.URL.Path == "/api/local/backend-probe") && r.Method == "POST":
		a.v08Backend(w, r)
	case r.URL.Path == "/api/local/engine-install" && r.Method == "POST":
		a.installEngineHandler(w, r)
	case r.URL.Path == "/api/local/engines":
		a.externalSettings(w, r)
	case r.URL.Path == "/api/local/status" && r.Method == "GET":
		a.localMu.Lock()
		s := a.localState
		a.localMu.Unlock()
		s.Ready = a.localReady()
		s.DirectMLReady = directMLReady(a.localDir())
		s.FaceReady = a.faceReady()
		s.QualityReady = a.qualityReady()
		s.RealReady = a.realReady()
		s.VoiceReady = a.voiceReady()
		s.Memory = memoryInfo()
		if s.Message == "" {
			if s.Ready || s.VoiceReady {
				s.Message = "本機模型已安裝，可按「偵測可用裝置」更新 GPU／NPU 清單。"
			} else {
				s.Message = "首次安裝約下載 2 GB；完成後可離線使用。"
			}
		}
		respond(w, s)
	case r.URL.Path == "/api/local/install" && r.Method == "POST":
		if runtime.GOOS != "windows" {
			fail(w, 400, errors.New("自動安裝適用 Windows 64 位元；其他平台請依原始碼說明安裝。"))
			return
		}
		a.localMu.Lock()
		if a.localState.Busy || a.localJob != "" {
			a.localMu.Unlock()
			fail(w, 409, errors.New("已有安裝或本機生成正在執行。"))
			return
		}
		a.localState.Busy = true
		a.localState.Message = "準備獨立的本機執行環境"
		a.localMu.Unlock()
		go func() {
			e := a.installLocal()
			a.localMu.Lock()
			a.localState.Busy = false
			if e != nil {
				a.localState.Message = "安裝未完成，可按安裝重試：" + e.Error()
			} else {
				a.localState.Message = "本機模型安裝完成。請按偵測裝置，再製作語音並生成。"
			}
			a.localMu.Unlock()
		}()
		respond(w, map[string]bool{"ok": true})
	case r.URL.Path == "/api/local/probe" && r.Method == "POST":
		if !a.localReady() && !a.voiceReady() {
			fail(w, 409, errors.New("請先安裝本機模型。"))
			return
		}
		a.localMu.Lock()
		if a.localState.Busy || a.localJob != "" {
			a.localMu.Unlock()
			fail(w, 409, errors.New("請等待目前工作完成。"))
			return
		}
		a.localState.Busy = true
		a.localMu.Unlock()
		defer func() { a.localMu.Lock(); a.localState.Busy = false; a.localMu.Unlock() }()
		if e := a.refreshLocalScripts(); e != nil {
			fail(w, 500, e)
			return
		}
		ctx, cancel := context.WithTimeout(r.Context(), 60*time.Second)
		defer cancel()
		c := exec.CommandContext(ctx, a.localPython(), "-u", filepath.Join(a.localDir(), "runner.py"), "--probe")
		c.Env = localEnv()
		hideCommand(c)
		out, e := c.CombinedOutput()
		if e != nil {
			fail(w, 500, fmt.Errorf("硬體偵測失敗，請確認 Intel 顯示與 NPU 驅動：%s", string(out)))
			return
		}
		s, e := parseLocalProbe(out)
		if e != nil {
			fail(w, 500, e)
			return
		}
		a.localMu.Lock()
		a.localState.Devices = s.Devices
		a.localState.Runtime = s.Runtime
		a.localMu.Unlock()
		s.Memory = memoryInfo()
		respond(w, s)
	case r.URL.Path == "/api/local/face-install" && r.Method == "POST":
		a.installFace(w, r)
	case r.URL.Path == "/api/local/face-detect" && r.Method == "POST":
		a.detectFace(w, r)
	case r.URL.Path == "/api/local/real-install" && r.Method == "POST":
		a.installReal(w, r)
	case r.URL.Path == "/api/local/quality-install" && r.Method == "POST":
		a.installQuality(w, r)
	case r.URL.Path == "/api/local/rebuild" && r.Method == "POST":
		a.rebuildLocal(w, r)
	case r.URL.Path == "/api/local/generate" && r.Method == "POST":
		a.localGenerate(w, r)
	case r.URL.Path == "/api/local/voice-install" && r.Method == "POST":
		a.startVoiceInstall(w, r)
	case r.URL.Path == "/api/local/voice-profiles" && r.Method == "GET":
		a.localVoiceProfiles(w, r)
	case r.URL.Path == "/api/local/voice-prepare" && r.Method == "POST":
		a.prepareLocalVoice(w, r)
	case r.URL.Path == "/api/local/cancel" && r.Method == "POST":
		var q struct {
			ID string `json:"id"`
		}
		if json.NewDecoder(io.LimitReader(r.Body, 4096)).Decode(&q) != nil {
			fail(w, 400, errors.New("無效資料"))
			return
		}
		a.localMu.Lock()
		if q.ID != a.localJob || a.localCancel == nil {
			a.localMu.Unlock()
			fail(w, 409, errors.New("找不到正在執行的本機生成。"))
			return
		}
		cancel := a.localCancel
		a.localMu.Unlock()
		cancel()
		respond(w, map[string]string{"message": "正在停止本機對嘴，已完成的素材與語音會保留。"})
	default:
		http.NotFound(w, r)
	}
}

// Extract trusted bundled archives with traversal and link checks.
func extractLocalArchive(data []byte, dir string) error {
	z, e := zip.NewReader(bytes.NewReader(data), int64(len(data)))
	if e != nil {
		return e
	}
	for _, f := range z.File {
		name := filepath.Clean(filepath.FromSlash(f.Name))
		if filepath.IsAbs(name) || strings.Contains(name, ":") || name == ".." || strings.HasPrefix(name, ".."+string(os.PathSeparator)) || f.Mode()&os.ModeSymlink != 0 {
			return errors.New("不安全的壓縮檔路徑")
		}
		p := filepath.Join(dir, name)
		if f.FileInfo().IsDir() {
			if e = os.MkdirAll(p, 0700); e != nil {
				return e
			}
			continue
		}
		if e = os.MkdirAll(filepath.Dir(p), 0700); e != nil {
			return e
		}
		src, e := f.Open()
		if e != nil {
			return e
		}
		dst, e := os.OpenFile(p, os.O_CREATE|os.O_TRUNC|os.O_WRONLY, 0600)
		if e != nil {
			src.Close()
			return e
		}
		_, e = io.Copy(dst, src)
		src.Close()
		ce := dst.Close()
		if e != nil {
			return e
		}
		if ce != nil {
			return ce
		}
	}
	return nil
}
func (a *App) installLocal() error {
	return a.installLocalModels(false)
}
func (a *App) installLocalModels(voice bool) error {
	models := localModels()
	marker, value := "installed.txt", "openvino-2026.4.1-v2"
	if voice {
		models = voiceModels()
		marker = "voice-installed.txt"
		value = voiceModelVersion
	}
	return a.installModelSet(models, marker, value)
}
func (a *App) installModelSet(models []LocalModel, marker, value string) error {
	dir := a.localDir()
	if e := os.MkdirAll(dir, 0700); e != nil {
		return e
	}
	release, e := lockInstallFile(filepath.Join(dir, "install.lock"))
	if e != nil {
		return e
	}
	defer release()
	usePortable := portablePython() != "" && a.localPython() == portablePython()
	if !usePortable {
		pyzip, _ := localFiles.ReadFile("local/python-3.11.9-embed-amd64.zip")
		if e := extractLocalArchive(pyzip, filepath.Join(dir, "python")); e != nil {
			return e
		}
		pth := []byte("python311.zip\n.\nLib\\site-packages\nimport site\n")
		if e := os.WriteFile(filepath.Join(dir, "python", "python311._pth"), pth, 0600); e != nil {
			return e
		}
		pip, _ := localFiles.ReadFile("local/pip-25.0.1-py3-none-any.whl")
		if e := extractLocalArchive(pip, filepath.Join(dir, "python", "Lib", "site-packages")); e != nil {
			return e
		}

	}
	for _, name := range []string{"semantic_mouth.py", "contour_mask.py", "contour_preflight.py", "cartoon_puppet.py", "stage_dispatch.py", "visual_polish.py", "body_motion.py", "micro_graph.py", "hardware_check.py", "detail_cache.py", "inference_backend.py", "backend_probe.py", "probe_matmul.onnx", "runner.py", "renderer.py", "quality.py", "model_cache.py", "stage_worker.py", "runtime_support.py", "face_worker.py", "face_geometry.py", "yunet.onnx", "voice_runner.py", "mel_filters.npz", "requirements.txt"} {
		b, _ := localFiles.ReadFile("local/" + name)
		if e := os.WriteFile(filepath.Join(dir, name), b, 0600); e != nil {
			return e
		}
	}
	a.setupMessage("正在安裝 OpenVINO、影像與音訊套件（約 185 MB）；請保持連網", 0, 0)
	ctx, cancel := context.WithTimeout(context.Background(), 8*time.Hour)
	defer cancel()
	log, e := os.Create(filepath.Join(dir, "install.log"))
	if e != nil {
		return e
	}
	defer log.Close()
	var c *exec.Cmd
	if !usePortable {
		c = exec.CommandContext(ctx, a.localPython(), "-m", "pip", "install", "--disable-pip-version-check", "--no-warn-script-location", "--only-binary=:all:", "--require-hashes", "-r", filepath.Join(dir, "requirements.txt"))
		c.Env = localEnv()
		c.Stdout = log
		c.Stderr = log
		hideCommand(c)
		if e = c.Run(); e != nil {
			return errors.New("套件安裝失敗，請查看 local-v0.2/install.log；可重試。")
		}

	}
	if e = os.MkdirAll(filepath.Join(dir, "models"), 0700); e != nil {
		return e
	}
	for _, m := range models {
		if e = a.downloadLocalModel(ctx, m); e != nil {
			return e
		}
	}
	// Import and actual device enumeration must succeed before marking ready.
	c = exec.CommandContext(ctx, a.localPython(), "-u", filepath.Join(dir, "runner.py"), "--probe")
	c.Env = localEnv()
	hideCommand(c)
	out, e := c.CombinedOutput()
	if e != nil {
		return fmt.Errorf("執行環境驗證失敗：%s", out)
	}
	s, e := parseLocalProbe(out)
	if e != nil {
		return e
	}
	a.localMu.Lock()
	a.localState.Devices = s.Devices
	a.localState.Runtime = s.Runtime
	a.localMu.Unlock()
	return os.WriteFile(filepath.Join(dir, marker), []byte(value), 0600)
}
func fileSHA(path string) (string, error) {
	f, e := os.Open(path)
	if e != nil {
		return "", e
	}
	defer f.Close()
	h := sha256.New()
	if _, e = io.Copy(h, f); e != nil {
		return "", e
	}
	return hex.EncodeToString(h.Sum(nil)), nil
}
func (a *App) downloadLocalModel(ctx context.Context, m LocalModel) error {
	return a.downloadArtifact(ctx, m, filepath.Join(a.localDir(), "models"))
}
func (a *App) downloadArtifact(ctx context.Context, m LocalModel, dir string) error {
	final := filepath.Join(dir, filepath.FromSlash(m.Name))
	if e := os.MkdirAll(filepath.Dir(final), 0700); e != nil {
		return e
	}
	if s, e := os.Stat(final); e == nil && s.Size() == m.Size {
		a.setupMessage("驗證模型 "+m.Name, 0, m.Size)
		h, e := fileSHA(final)
		if e == nil && h == m.SHA {
			return nil
		}
	}
	part := final + ".part"
	var offset int64
	if s, e := os.Stat(part); e == nil {
		offset = s.Size()
		if offset > m.Size {
			os.Remove(part)
			offset = 0
		}
	}
	if offset == m.Size {
		h, err := fileSHA(part)
		if err == nil && h == m.SHA {
			os.Remove(final)
			return os.Rename(part, final)
		}
		os.Remove(part)
		offset = 0
	}
	req, e := http.NewRequestWithContext(ctx, "GET", m.URL, nil)
	if e != nil {
		return e
	}
	if offset > 0 {
		req.Header.Set("Range", fmt.Sprintf("bytes=%d-", offset))
	}
	client := &http.Client{Timeout: 0}
	res, e := client.Do(req)
	if e != nil {
		return e
	}
	defer res.Body.Close()
	if res.StatusCode != 200 && res.StatusCode != 206 {
		return fmt.Errorf("模型下載失敗 %s：HTTP %d", m.Name, res.StatusCode)
	}
	flags := os.O_CREATE | os.O_WRONLY | os.O_APPEND
	if res.StatusCode == 200 {
		flags = os.O_CREATE | os.O_WRONLY | os.O_TRUNC
		offset = 0
	} else if !strings.HasPrefix(res.Header.Get("Content-Range"), fmt.Sprintf("bytes %d-", offset)) {
		return errors.New("下載續傳位置不符")
	}
	f, e := os.OpenFile(part, flags, 0600)
	if e != nil {
		return e
	}
	buf := make([]byte, 1<<20)
	last := time.Time{}
	for {
		n, re := res.Body.Read(buf)
		if n > 0 {
			wn, we := f.Write(buf[:n])
			offset += int64(wn)
			if we != nil {
				f.Close()
				return we
			}
			if offset > m.Size {
				f.Close()
				os.Remove(part)
				return errors.New("模型下載超出預期大小")
			}
			if time.Since(last) > 500*time.Millisecond {
				a.setupMessage("下載模型 "+m.Name, offset, m.Size)
				last = time.Now()
			}
		}
		if re != nil {
			f.Close()
			if re != io.EOF {
				return re
			}
			break
		}
	}
	a.setupMessage("驗證 SHA256："+m.Name, offset, m.Size)
	h, e := fileSHA(part)
	if e != nil {
		return e
	}
	if offset != m.Size || h != m.SHA {
		os.Remove(part)
		return errors.New("模型完整性驗證失敗，請重試下載：" + m.Name)
	}
	os.Remove(final)
	return os.Rename(part, final)
}

type LocalRequest struct {
	ColorTransfer   *bool   `json:"color_transfer"`
	LidShadow       *bool   `json:"lid_shadow"`
	PeakBudgetMB    int     `json:"peak_budget_mb"`
	Runtime         string  `json:"runtime"`
	DMLAdapter      int     `json:"dml_adapter"`
	NPUAssist       bool    `json:"npu_assist"`
	BlendMode       string  `json:"blend_mode"`
	MouthScope      string  `json:"mouth_scope"`
	ComposeStyle    string  `json:"compose_style"`
	PuppetStyle     string  `json:"puppet_style"`
	PuppetOpen      float64 `json:"puppet_open"`
	CompositionMode string  `json:"composition_mode"`
	BlinkProfile    string  `json:"blink_profile"`
	BrowMotion      bool    `json:"brow_motion"`

	Engine          string  `json:"engine"`
	PoseStyle       int     `json:"pose_style"`
	ExpressionScale float64 `json:"expression_scale"`
	InferenceSteps  int     `json:"inference_steps"`

	Landmarks      [][]float64 `json:"landmarks,omitempty"`
	MemoryMode     string      `json:"memory_mode"`
	ResourceMode   string      `json:"resource_mode"`
	RealDetail     bool        `json:"real_detail"`
	DetailStrength float64     `json:"detail_strength"`
	BodyMotion     string      `json:"body_motion"`
	HandBox        []float64   `json:"hand_box"`
	AnimeDetail    bool        `json:"anime_detail"`
	MouthBox       []float64   `json:"mouth_box"`
	EyeBoxes       [][]float64 `json:"eye_boxes"`
	Blink          bool        `json:"blink"`
	Gaze           bool        `json:"gaze"`
	MicroMotion    bool        `json:"micro_motion"`
	Restore        bool        `json:"restore"`
	ID             string      `json:"id"`
	Device         string      `json:"device"`
	BBox           []float64   `json:"bbox"`
	FPS            int         `json:"fps"`
	MaxSide        int         `json:"max_side"`
	Format         string      `json:"format"`
	Aspect         string      `json:"aspect"`
	Resolution     string      `json:"resolution"`
	CaptionMode    string      `json:"caption_mode"`
}

func validateLocal(q *LocalRequest) error {
	if q.ColorTransfer == nil {
		v := true
		q.ColorTransfer = &v
	}
	if q.LidShadow == nil {
		v := true
		q.LidShadow = &v
	}
	if q.PeakBudgetMB != 0 && q.PeakBudgetMB != 2560 && q.PeakBudgetMB != 3072 && q.PeakBudgetMB != 4096 && q.PeakBudgetMB != 5120 && q.PeakBudgetMB != 6144 {
		return errors.New("無效記憶體預算")
	}
	q.MemoryMode = "staged"

	if q.Runtime == "" {
		q.Runtime = "openvino"
	}
	if q.BlendMode == "" {
		q.BlendMode = "feather"
	}
	if q.MouthScope == "" {
		q.MouthScope = "lips"
	}
	if q.ComposeStyle == "" {
		q.ComposeStyle = "auto"
	}
	if q.CompositionMode == "" {
		q.CompositionMode = "mouth"
	}
	if q.BlinkProfile == "" {
		q.BlinkProfile = "bezier"
	}
	if q.Runtime != "openvino" && q.Runtime != "directml" && q.Runtime != "torch" {
		return errors.New("無效後端")
	}
	if q.Runtime == "torch" && q.Engine != "sadtalker" {
		return errors.New("PyTorch 僅適用 SadTalker")
	}
	if q.Runtime == "directml" && (q.Device == "NPU" || q.NPUAssist) {
		return errors.New("DirectML 適用顯卡；NPU 請選 OpenVINO")
	}
	if q.DMLAdapter < 0 || q.DMLAdapter > 7 {
		return errors.New("無效顯卡編號")
	}
	if q.BlendMode != "feather" && q.BlendMode != "poisson" {
		return errors.New("無效融合方式")
	}
	if q.MouthScope != "lips" && q.MouthScope != "jaw" {
		return errors.New("無效嘴部範圍")
	}
	if q.ComposeStyle != "auto" && q.ComposeStyle != "cartoon" && q.ComposeStyle != "real" {
		return errors.New("無效合成方式")
	}
	if q.PuppetStyle == "" {
		q.PuppetStyle = "smooth"
	}
	if q.PuppetStyle != "smooth" && q.PuppetStyle != "shapes" {
		return errors.New("無效唇形擺動方式")
	}
	if q.PuppetOpen == 0 {
		q.PuppetOpen = .55
	}
	if math.IsNaN(q.PuppetOpen) || q.PuppetOpen < .25 || q.PuppetOpen > .9 {
		return errors.New("張嘴幅度需介於 0.25 至 0.9")
	}
	if q.CompositionMode != "mouth" && q.CompositionMode != "full" {
		return errors.New("無效合成模式")
	}
	if q.BlinkProfile != "bezier" && q.BlinkProfile != "slow" {
		return errors.New("無效眨眼曲線")
	}
	if q.Engine != "sadtalker" {
		q.CompositionMode = "mouth"
	}
	if q.Runtime != "openvino" || q.NPUAssist {
		q.MemoryMode = "staged"
	}

	if e := validateAdvanced(q); e != nil {
		return e
	}
	if len(q.Landmarks) != 0 && len(q.Landmarks) != 5 && len(q.Landmarks) != 98 {
		// v0.8.3b: contour-fallback locating used to send 96 geometric points.
		// Only 98-point landmarks are consumed downstream, and the confirmed
		// mouth/eye boxes are re-measured anyway, so drop them instead of failing.
		q.Landmarks = nil
	}
	for _, p := range q.Landmarks {
		if len(p) != 2 || math.IsNaN(p[0]) || math.IsNaN(p[1]) || p[0] < 0 || p[0] > 1 || p[1] < 0 || p[1] > 1 {
			return errors.New("五官關鍵點座標無效")
		}
	}
	if q.MemoryMode == "" {
		q.MemoryMode = "staged"
	}
	if q.MemoryMode != "staged" && q.MemoryMode != "standard" {
		return errors.New("無效的記憶體模式")
	}
	if q.ResourceMode == "" {
		q.ResourceMode = "gentle"
	}
	if q.ResourceMode != "gentle" && q.ResourceMode != "normal" {
		return errors.New("無效的資源模式")
	}
	if q.RealDetail && q.AnimeDetail {
		return errors.New("請擇一使用真人修復或卡通補強")
	}
	if q.DetailStrength == 0 {
		q.DetailStrength = .5
	}
	if math.IsNaN(q.DetailStrength) || math.IsInf(q.DetailStrength, 0) || q.DetailStrength < .2 || q.DetailStrength > 1 {
		return errors.New("細節修復強度無效")
	}

	if q.Device == "" {
		q.Device = "AUTO"
	}
	if q.Device != "AUTO" && q.Device != "CPU" && q.Device != "NPU" && q.Device != "GPU" && q.Device != "GPU.0" && q.Device != "GPU.1" {
		return errors.New("無效的運算裝置")
	}
	if len(q.BBox) != 4 {
		return errors.New("請先框選臉部")
	}
	for _, v := range q.BBox {
		if v != v || v < 0 || v > 1 {
			return errors.New("臉部範圍無效")
		}
	}
	if q.BBox[2] < .03 || q.BBox[3] < .03 || q.BBox[0]+q.BBox[2] > 1.001 || q.BBox[1]+q.BBox[3] > 1.001 {
		return errors.New("臉部範圍超出畫面或太小")
	}
	if q.FPS != 6 && q.FPS != 12 && q.FPS != 25 {
		q.FPS = 12
	}
	switch q.MaxSide {
	case 512, 720, 1080, 1920, 2160:
	default:
		q.MaxSide = 1920
	}
	switch q.BodyMotion {
	case "", "breathe", "sway", "nod", "hand", "speech":
	default:
		return errors.New("無效的動作選項")
	}
	if len(q.HandBox) > 0 && !validRegion(q.HandBox) {
		return errors.New("手部範圍無效")
	}
	if len(q.MouthBox) > 0 && !validRegion(q.MouthBox) {
		return errors.New("嘴部框選無效")
	}
	if len(q.EyeBoxes) > 2 {
		return errors.New("最多框選兩隻眼睛")
	}
	for _, box := range q.EyeBoxes {
		if !validRegion(box) {
			return errors.New("眼睛框選無效")
		}
	}
	return nil
}
func (a *App) localGenerate(w http.ResponseWriter, r *http.Request) {
	var q LocalRequest
	if json.NewDecoder(io.LimitReader(r.Body, 8192)).Decode(&q) != nil {
		fail(w, 400, errors.New("無效資料"))
		return
	}
	if e := validateLocal(&q); e != nil {
		fail(w, 400, e)
		return
	}
	if q.Engine != "musetalk" && q.Engine != "puppet" {
		a.externalGenerate(w, r, q)
		return
	}
	if q.Runtime == "directml" && !directMLReady(a.localDir()) {
		fail(w, 409, errors.New("請先一鍵安裝 DirectML 後端"))
		return
	}
	if !a.localReady() {
		fail(w, 409, errors.New("請先完成本機模型安裝。"))
		return
	}
	if q.RealDetail && !a.realReady() {
		fail(w, 409, errors.New("請先安裝真人嘴部修復模型，或關閉真人修復"))
		return
	}
	if q.AnimeDetail && !a.qualityReady() {
		fail(w, 409, errors.New("請先安裝卡通嘴部細節模型，或關閉細節選項"))
		return
	}
	mem := memoryInfo()
	minimum := localMemoryMinimum(q.Device, q.MemoryMode)
	if mem.Available > 0 && mem.Available < minimum {
		hint := "請先關閉其他程式再試。"
		if q.MemoryMode != "staged" {
			hint = "可把「模型記憶體方式」改成「磁碟映射＋逐階段」，只需約 4.5 GB。"
		}
		fail(w, 409, fmt.Errorf("可用記憶體只有 %.1f GB，此模式至少需 %.1f GB。%s", float64(mem.Available)/(1<<30), float64(minimum)/(1<<30), hint))
		return
	}
	a.localMu.Lock()
	if a.localJob != "" || a.localState.Busy {
		a.localMu.Unlock()
		fail(w, 409, errors.New("已有本機生成或安裝正在執行。"))
		return
	}
	a.mu.Lock()
	j := a.jobs[q.ID]
	if j == nil || j.busy || (j.State != "ready" && !(j.State == "error" && j.Engine == "local")) || (j.Kind != "photo" && j.Kind != "video") {
		a.mu.Unlock()
		a.localMu.Unlock()
		fail(w, 409, errors.New("請先製作語音並確認素材。"))
		return
	}
	if q.Engine == "puppet" && j.Kind != "photo" {
		a.mu.Unlock()
		a.localMu.Unlock()
		fail(w, 409, errors.New("卡通原圖唇形只支援照片素材；影片請改用 MuseTalk。"))
		return
	}
	j.FaceBox = append([]float64{}, q.BBox...)
	if j.DefaultAvatar {
		if len(q.MouthBox) == 0 {
			q.MouthBox = []float64{.414, .409, .207, .060}
		}
		if len(q.HandBox) == 0 {
			q.HandBox = []float64{.25, .60, .23, .10}
		}
		if len(q.EyeBoxes) == 0 {
			q.EyeBoxes = [][]float64{{.401, .345, .071, .024}, {.565, .348, .067, .023}}
		}
	}
	if j.Kind == "photo" && q.BodyMotion == "hand" && len(q.HandBox) != 4 {
		a.mu.Unlock()
		a.localMu.Unlock()
		fail(w, 400, errors.New("請先框選手部範圍"))
		return
	}
	if j.Kind == "photo" && (q.Blink || q.Gaze) && len(q.EyeBoxes) != 2 {
		a.mu.Unlock()
		a.localMu.Unlock()
		fail(w, 400, errors.New("請分別框選兩隻眼睛，或關閉眨眼與視線選項"))
		return
	}
	if j.Kind == "video" {
		q.BodyMotion = ""
		q.Blink = false
		q.Gaze = false
		q.MicroMotion = false
	}
	lead := 0.0
	if q.Restore {
		lead = math.Round(.56*float64(q.FPS)) / float64(q.FPS)
	}
	if e := shiftCaptions(a.dir(j.ID), lead-j.LeadSeconds); e != nil {
		a.mu.Unlock()
		a.localMu.Unlock()
		fail(w, 500, e)
		return
	}
	ctx, cancel := context.WithCancel(context.Background())
	a.localCancel = cancel
	a.localJob = q.ID
	j.busy = true
	j.State = "local_generating"
	j.Engine = "local"
	j.LocalFrame = 0
	j.LocalTotal = 0
	j.ActualDevices = nil
	j.MemoryPeaks = nil
	j.LocalStage = ""
	j.LeadSeconds = lead
	j.LocalSettings = &q
	j.Message = "準備在本機執行 MuseTalk；首次編譯可能需要數分鐘。"
	if q.Engine == "puppet" {
		j.Message = "準備卡通原圖唇形擺動（不使用 AI 模型）。"
	}
	j.Format, j.Aspect, j.Resolution = normalizeExport(q.Format, q.Aspect, q.Resolution)
	j.CaptionMode = captionMode(q.CaptionMode)
	a.persist(j)
	snapshot := *j
	a.mu.Unlock()
	a.localMu.Unlock()
	go a.runLocal(ctx, q)
	respond(w, snapshot)
}
func (a *App) runLocal(ctx context.Context, q LocalRequest) {
	defer func() {
		a.localMu.Lock()
		a.localCancel = nil
		a.localJob = ""
		a.localMu.Unlock()
		a.mu.Lock()
		a.jobs[q.ID].busy = false
		a.mu.Unlock()
	}()
	a.mu.Lock()
	j := *a.jobs[q.ID]
	a.mu.Unlock()
	dir := a.dir(q.ID)
	if e := a.refreshLocalScripts(); e != nil {
		a.update(q.ID, "error", e.Error())
		return
	}
	engine, e := ffmpegPath()
	if e != nil {
		a.update(q.ID, "error", e.Error())
		return
	}
	cfg := map[string]any{"models": filepath.Join(a.localDir(), "models"), "device": q.Device, "audio": filepath.Join(dir, j.Audio), "source": filepath.Join(dir, j.Source), "output": filepath.Join(dir, "video.mp4"), "ffmpeg": engine, "kind": j.Kind, "bbox": q.BBox, "fps": q.FPS, "max_side": q.MaxSide, "mouth_box": q.MouthBox, "eye_boxes": q.EyeBoxes, "blink": q.Blink, "gaze": q.Gaze, "micro_motion": q.MicroMotion, "lead": j.LeadSeconds, "tail": j.LeadSeconds, "anime_detail": q.AnimeDetail, "body_motion": q.BodyMotion, "hand_box": q.HandBox, "run_id": randomID(), "low_impact": q.ResourceMode != "normal", "real_detail": q.RealDetail, "detail_strength": q.DetailStrength}
	v08Config(cfg, q)
	b, _ := json.MarshalIndent(cfg, "", "  ")
	if e = os.WriteFile(filepath.Join(dir, "local-config.json"), b, 0600); e != nil {
		a.update(q.ID, "error", e.Error())
		return
	}
	log, e := os.Create(filepath.Join(dir, "local.log"))
	if e != nil {
		a.update(q.ID, "error", e.Error())
		return
	}
	defer log.Close()
	type phase struct {
		script string
		args   []string
	}
	phases := []phase{{"runner.py", nil}, {"renderer.py", nil}}
	if q.MemoryMode == "staged" {
		phases = nil
		for _, name := range []string{"cache", "audio", "encode", "infer", "decode"} {
			phases = append(phases, phase{"stage_worker.py", []string{"--stage", name}})
		}
		phases = append(phases, phase{"renderer.py", nil})
	}
	if q.Engine == "puppet" {
		// Cartoon lip puppet warps the original mouth; no neural stages.
		phases = []phase{{"renderer.py", nil}}
	}
	for _, work := range phases {
		lastError, workerError := a.runLocalWorker(ctx, q.ID, work.script, filepath.Join(dir, "local-config.json"), log, work.args...)
		if ctx.Err() != nil {
			os.Remove(filepath.Join(dir, "local-render.mp4"))
			a.update(q.ID, "ready", "本機生成已停止，可重新生成；素材與語音已保留。")
			return
		}
		if workerError != nil {
			if lastError == "" {
				lastError = "推論子行程未回報例外，可能是 SDK／驅動或記憶體配置失敗"
			}
			a.update(q.ID, "error", "本機生成失敗："+lastError+"；"+workerError.Error()+"；請按「下載錯誤報告」。")
			return
		}
	}
	a.mu.Lock()
	a.jobs[q.ID].Output = "video.mp4"
	a.persist(a.jobs[q.ID])
	a.mu.Unlock()
	a.convertJob(q.ID, j.Format, j.Aspect, j.Resolution)
}

func (a *App) runLocalWorker(ctx context.Context, id, script, config string, log *os.File, extra ...string) (string, error) {
	gate := config + ".start-gate-" + randomID()
	defer os.Remove(gate)
	args := append([]string{"-I", "-u", filepath.Join(a.localDir(), "stage_dispatch.py"), "--script", script, "--config", config, "--start-gate", gate}, extra...)
	c := exec.CommandContext(ctx, a.localPython(), args...)
	c.Env = localEnv()
	return a.watchLocalWorker(id, c, log)
}
func (a *App) watchLocalWorker(id string, c *exec.Cmd, log *os.File) (string, error) {
	hideCommand(c)
	c.Cancel = func() error { return killLocalProcess(c) }
	c.WaitDelay = 10 * time.Second
	pipe, e := c.StdoutPipe()
	if e != nil {
		return "", e
	}
	c.Stderr = log
	if e = c.Start(); e != nil {
		return "", e
	}
	a.mu.Lock()
	budget := workerBudget(0, memoryInfo())
	if j := a.jobs[id]; j != nil && j.LocalSettings != nil && j.LocalSettings.PeakBudgetMB > 0 {
		budget = j.LocalSettings.PeakBudgetMB
	}
	a.mu.Unlock()
	// Use the exact effective budget recorded in this stage's config.
	for i, arg := range c.Args {
		if arg == "--config" && i+1 < len(c.Args) {
			var cfg struct {
				Budget int `json:"peak_budget_mb"`
			}
			if b, err := os.ReadFile(c.Args[i+1]); err == nil && len(b) < 1<<20 && json.Unmarshal(b, &cfg) == nil && cfg.Budget > 0 {
				budget = cfg.Budget
			}
		}
	}
	release, limitError := limitWorkerMemory(c, budget)
	if limitError != nil {
		c.Process.Kill()
		c.Wait()
		return "記憶體預算保護無法啟動", limitError
	}
	defer release()
	for i, arg := range c.Args {
		if arg == "--start-gate" && i+1 < len(c.Args) {
			if err := os.WriteFile(c.Args[i+1], []byte("ready"), 0600); err != nil {
				c.Process.Kill()
				c.Wait()
				return "無法啟用受限工作程序", err
			}
		}
	}
	fmt.Fprintf(log, "worker_pid=%d private_commit_budget_mib=%d\n", c.Process.Pid, budget)
	a.mu.Lock()
	if j := a.jobs[id]; j != nil {
		j.EffectiveBudgetMB = budget
		j.BudgetPolicy = "v082"
		a.persist(j)
	}
	a.mu.Unlock()
	scanner := bufio.NewScanner(pipe)
	scanner.Buffer(make([]byte, 4096), 1<<20)
	lastError := ""
	for scanner.Scan() {
		line := scanner.Bytes()
		log.Write(append(append([]byte{}, line...), '\n'))
		var p struct {
			Stage, Message    string
			StageName         string  `json:"stage_name"`
			MemoryPeakMB      float64 `json:"memory_peak_mb"`
			Frame, Total, ETA int
			Actual            map[string][]string
		}
		if json.Unmarshal(line, &p) != nil {
			continue
		}
		if p.Stage == "error" {
			lastError = p.Message
		}
		if p.Stage == "worker_start" {
			lastError = ""
		}
		a.mu.Lock()
		j := a.jobs[id]
		if p.Stage == "worker_start" || p.Stage == "compile" || p.Stage == "fallback" || p.Stage == "error" {
			j.LocalFrame, j.LocalTotal, j.LocalETA = 0, 0, 0
		}
		if p.StageName != "" {
			j.LocalStage = p.StageName
		}
		if p.MemoryPeakMB > 0 {
			if j.MemoryPeaks == nil {
				j.MemoryPeaks = map[string]float64{}
			}
			j.MemoryPeaks[p.StageName] = p.MemoryPeakMB
		}
		if p.Message != "" {
			j.Message = p.Message
		}
		if p.Actual != nil {
			j.ActualDevices = p.Actual
		}
		if p.Total > 0 {
			j.LocalFrame = p.Frame
			j.LocalTotal = p.Total
			j.LocalETA = p.ETA
		}
		a.persist(j)
		a.mu.Unlock()
	}
	e = c.Wait()
	if e == nil {
		e = scanner.Err()
	}
	if e != nil {
		a.mu.Lock()
		if j := a.jobs[id]; j != nil {
			j.LocalFrame, j.LocalTotal, j.LocalETA = 0, 0, 0
			a.persist(j)
		}
		a.mu.Unlock()
	}
	return lastError, e
}
