package main

import (
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"math"
	"net/http"
	"os"
	"os/exec"
	"path/filepath"
	"strconv"
	"strings"
	"time"
)

// These adapters call an independently installed official engine, never a shell.
// The built-in OpenVINO environment is not modified by configuring another engine.
type ExternalConfig struct {
	Managed bool   `json:"managed,omitempty"`
	Engine  string `json:"engine"`
	Python  string `json:"python"`
	Root    string `json:"root"`
	CPU     bool   `json:"cpu"`
}
type ExternalProbe struct {
	DirectMLReady bool           `json:"directml_ready"`
	Config        ExternalConfig `json:"config"`
	Configured    bool           `json:"configured"`
	Message       string         `json:"message"`
	Device        string         `json:"device,omitempty"`
	VRAM          float64        `json:"vram_gib,omitempty"`
}

func (a *App) externalConfigPath(engine string) string {
	return filepath.Join(filepath.Dir(a.root), "engine-"+engine+".json")
}
func validExternalName(s string) bool { return s == "sadtalker" || s == "latentsync" }
func (a *App) loadExternal(engine string) (ExternalConfig, error) {
	var c ExternalConfig
	if !validExternalName(engine) {
		return c, errors.New("不支援的外部引擎")
	}
	b, e := os.ReadFile(a.externalConfigPath(engine))
	if e != nil {
		return c, errors.New("尚未安裝所選引擎；請按「自動安裝所選引擎」")
	}
	e = json.Unmarshal(b, &c)
	return c, e
}
func externalFiles(c ExternalConfig) error {
	if !validExternalName(c.Engine) || !filepath.IsAbs(c.Python) || !filepath.IsAbs(c.Root) {
		return errors.New("請填入外部引擎 Python 和資料夾的完整路徑")
	}
	py, e := os.Stat(c.Python)
	if e != nil || !py.Mode().IsRegular() {
		return errors.New("找不到指定 Python 執行檔")
	}
	if c.Managed {
		marker, e := os.ReadFile(filepath.Join(c.Root, "installed.txt"))
		if e != nil || string(marker) != sadManagedVersion {
			return errors.New("自動安裝尚未完成；請按安裝修復")
		}
		var rows []LocalModel
		b, _ := localFiles.ReadFile("local/sadtalker_models.json")
		json.Unmarshal(b, &rows)
		for _, m := range rows {
			st, e := os.Stat(filepath.Join(c.Root, filepath.FromSlash(m.Name)))
			if e != nil || st.Size() != m.Size {
				return errors.New("模型不完整；按自動安裝修復")
			}
		}
	}
	files := []string{"inference.py", "checkpoints/SadTalker_V0.0.2_512.safetensors", "checkpoints/mapping_00109-model.pth.tar"}
	if c.Engine == "latentsync" {
		files = []string{"scripts/inference.py", "configs/unet/stage2_512.yaml", "checkpoints/latentsync_unet.pt", "checkpoints/whisper/tiny.pt"}
	}
	for _, name := range files {
		p, e := os.Stat(filepath.Join(c.Root, filepath.FromSlash(name)))
		if e != nil || !p.Mode().IsRegular() || p.Size() == 0 {
			return fmt.Errorf("缺少外部引擎檔案：%s；請先依官方說明安裝完整模型", name)
		}
	}
	return nil
}
func probeExternal(ctx context.Context, c ExternalConfig) (ExternalProbe, error) {
	p := ExternalProbe{Config: c}
	if e := externalFiles(c); e != nil {
		return p, e
	}
	code := `import json,torch;cuda=torch.cuda.is_available();print(json.dumps({'cuda':cuda,'vram':torch.cuda.get_device_properties(0).total_memory/(1024**3) if cuda else 0}))`
	cmd := exec.CommandContext(ctx, c.Python, "-c", code)
	cmd.Dir = c.Root
	cmd.Env = append(os.Environ(), "PYTHONUTF8=1", "HF_HUB_OFFLINE=1", "TRANSFORMERS_OFFLINE=1")
	hideCommand(cmd)
	out, e := cmd.CombinedOutput()
	if e != nil {
		return p, fmt.Errorf("外部 Python 無法匯入 PyTorch：%s", tailText(out, 800))
	}
	var v struct {
		CUDA bool    `json:"cuda"`
		VRAM float64 `json:"vram"`
	}
	found := false
	for _, line := range strings.Split(string(out), "\n") {
		if json.Unmarshal([]byte(line), &v) == nil && strings.Contains(line, "cuda") {
			found = true
		}
	}
	if !found {
		return p, errors.New("外部引擎裝置檢查未回傳有效資料")
	}
	p.VRAM = v.VRAM
	p.Device = "CPU"
	if v.CUDA && !c.CPU {
		p.Device = "CUDA"
	}
	if c.Engine == "latentsync" && (!v.CUDA || c.CPU || v.VRAM < 18) {
		return p, errors.New("LatentSync 1.6 目前需 CUDA 與至少 18 GiB NVIDIA 顯示記憶體；Intel GPU／NPU 不相容")
	}
	p.Configured = true
	p.Message = "路徑、主要權重與 PyTorch 檢查通過；尚不代表完整 AI 生成已驗證。"
	if p.Device == "CPU" {
		p.Message += " CPU 可嘗試，但可能非常慢。"
	}
	return p, nil
}
func tailText(b []byte, n int) string {
	if len(b) > n {
		b = b[len(b)-n:]
	}
	return strings.TrimSpace(string(b))
}
func (a *App) externalSettings(w http.ResponseWriter, r *http.Request) {
	if r.Method == "GET" {
		rows := []ExternalProbe{}
		for _, name := range []string{"sadtalker", "latentsync"} {
			c, e := a.loadExternal(name)
			p := ExternalProbe{Config: ExternalConfig{Engine: name}, Message: "尚未設定；不是已內建的模型"}
			if e == nil {
				p.Config = c
				p.DirectMLReady = directMLReady(c.Root)
				if e = externalFiles(c); e == nil {
					p.Message = "已安裝；執行前仍會檢查裝置與模型"
					if c.Managed {
						p.Message = "SadTalker 512 已由程式自動安裝；CPU 分階段、逐格輸出，可離線試做比較。"
					}
					p.Configured = true
				} else {
					p.Message = e.Error()
				}
			}
			rows = append(rows, p)
		}
		respond(w, rows)
		return
	}
	if r.Method != "POST" {
		http.NotFound(w, r)
		return
	}
	var c ExternalConfig
	if json.NewDecoder(io.LimitReader(r.Body, 8192)).Decode(&c) != nil {
		fail(w, 400, errors.New("無效設定"))
		return
	}
	c.Python = strings.TrimSpace(c.Python)
	c.Root = strings.TrimSpace(c.Root)
	a.localMu.Lock()
	if a.localJob != "" || a.localState.Busy {
		a.localMu.Unlock()
		fail(w, 409, errors.New("請先等待目前工作完成"))
		return
	}
	a.localState.Busy = true
	a.localMu.Unlock()
	defer func() { a.localMu.Lock(); a.localState.Busy = false; a.localMu.Unlock() }()
	ctx, cancel := context.WithTimeout(r.Context(), 40*time.Second)
	defer cancel()
	p, e := probeExternal(ctx, c)
	if e != nil {
		fail(w, 422, e)
		return
	}
	b, _ := json.MarshalIndent(c, "", "  ")
	if e = os.WriteFile(a.externalConfigPath(c.Engine), b, 0600); e != nil {
		fail(w, 500, e)
		return
	}
	respond(w, p)
}
func externalArgs(c ExternalConfig, q LocalRequest, source, audio, output, dir string) []string {
	if c.Engine == "sadtalker" {
		args := []string{"-u", filepath.Join(c.Root, "inference.py"), "--source_image", source, "--driven_audio", audio, "--checkpoint_dir", filepath.Join(c.Root, "checkpoints"), "--result_dir", dir, "--preprocess", "full", "--size", "512", "--batch_size", "1", "--pose_style", strconv.Itoa(q.PoseStyle), "--expression_scale", strconv.FormatFloat(q.ExpressionScale, 'f', 2, 64)}
		if c.CPU {
			args = append(args, "--cpu")
		}
		return args
	}
	return []string{"-u", "-m", "scripts.inference", "--unet_config_path", filepath.Join(c.Root, "configs", "unet", "stage2_512.yaml"), "--inference_ckpt_path", filepath.Join(c.Root, "checkpoints", "latentsync_unet.pt"), "--video_path", source, "--audio_path", audio, "--video_out_path", output, "--temp_dir", filepath.Join(dir, "temp"), "--inference_steps", strconv.Itoa(q.InferenceSteps), "--guidance_scale", "1.5", "--seed", "1247"}
}
func (a *App) externalGenerate(w http.ResponseWriter, r *http.Request, q LocalRequest) {
	c, e := a.loadExternal(q.Engine)
	if e != nil {
		fail(w, 409, e)
		return
	}
	if e = externalFiles(c); e != nil {
		fail(w, 409, e)
		return
	}
	if c.Managed && q.Runtime == "directml" && !directMLReady(c.Root) {
		fail(w, 409, errors.New("請先安裝此引擎 DirectML 後端"))
		return
	}
	if c.Managed && q.RealDetail && !a.realReady() {
		fail(w, 409, errors.New("請先安裝真人細節模型"))
		return
	}
	if c.Managed && q.AnimeDetail && !a.qualityReady() {
		fail(w, 409, errors.New("請先安裝卡通細節模型"))
		return
	}
	if c.Engine == "sadtalker" && c.CPU {
		m := memoryInfo()
		if m.Available > 0 && m.Available < 5<<30 {
			fail(w, 409, errors.New("外部 CPU 引擎請先預留至少 5 GB 可用 RAM；此門檻不保證模型一定足夠"))
			return
		}
	}
	if q.Restore && !c.Managed {
		fail(w, 400, errors.New("外部引擎目前保留自身首尾畫面；請關閉「片頭、片尾還原原圖」後生成"))
		return
	}
	a.localMu.Lock()
	if a.localJob != "" || a.localState.Busy {
		a.localMu.Unlock()
		fail(w, 409, errors.New("請先等待目前工作完成"))
		return
	}
	a.mu.Lock()
	j := a.jobs[q.ID]
	if j == nil || j.busy || (j.State != "ready" && !(j.State == "error" && j.Engine == "local")) || (j.Kind != "photo" && j.Kind != "video") {
		a.mu.Unlock()
		a.localMu.Unlock()
		fail(w, 409, errors.New("請先準備照片／影片與語音"))
		return
	}
	if q.Engine == "sadtalker" && j.Kind != "photo" {
		a.mu.Unlock()
		a.localMu.Unlock()
		fail(w, 400, errors.New("本版 SadTalker 接入僅接受照片；影片請選 LatentSync 或 MuseTalk"))
		return
	}
	lead := 0.0
	if c.Managed && q.Restore {
		lead = .56
	}
	if e = shiftCaptions(a.dir(j.ID), lead-j.LeadSeconds); e != nil {
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
	j.LocalSettings = &q
	j.LeadSeconds = lead
	j.MemoryPeaks = nil
	j.ActualDevices = nil
	j.LocalFrame = 0
	j.LocalTotal = 0
	j.LocalETA = 0
	j.Message = "準備外部 " + q.Engine + "；先檢查模型與裝置，編譯階段不提供假進度。"
	j.Format, j.Aspect, j.Resolution = normalizeExport(q.Format, q.Aspect, q.Resolution)
	j.CaptionMode = captionMode(q.CaptionMode)
	a.persist(j)
	snapshot := *j
	a.mu.Unlock()
	a.localMu.Unlock()
	go a.runExternal(ctx, c, q)
	respond(w, snapshot)
}
func (a *App) runExternal(ctx context.Context, c ExternalConfig, q LocalRequest) {
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
	runDir := filepath.Join(dir, "external-"+randomID())
	if e := os.MkdirAll(runDir, 0700); e != nil {
		a.update(q.ID, "error", e.Error())
		return
	}
	log, e := os.Create(filepath.Join(dir, "local.log"))
	if e != nil {
		a.update(q.ID, "error", e.Error())
		return
	}
	defer log.Close()
	probeCtx, cancel := context.WithTimeout(ctx, 40*time.Second)
	p, e := probeExternal(probeCtx, c)
	cancel()
	if e != nil {
		a.update(q.ID, "error", e.Error())
		return
	}
	a.mu.Lock()
	a.jobs[q.ID].ActualDevices = map[string][]string{q.Engine: {p.Device}}
	a.persist(a.jobs[q.ID])
	a.mu.Unlock()
	source := filepath.Join(dir, j.Source)
	audio := filepath.Join(dir, j.Audio)
	output := filepath.Join(runDir, "result.mp4")
	ff, e := ffmpegPath()
	if e != nil {
		a.update(q.ID, "error", e.Error())
		return
	}
	if q.Engine == "latentsync" && j.Kind == "photo" {
		input := filepath.Join(runDir, "input.mp4")
		cmd := exec.CommandContext(ctx, ff, "-v", "error", "-nostdin", "-y", "-loop", "1", "-i", source, "-t", strconv.FormatFloat(j.Duration, 'f', 3, 64), "-r", "25", "-vf", "scale=trunc(iw/2)*2:trunc(ih/2)*2", "-c:v", "libx264", "-crf", "16", "-pix_fmt", "yuv420p", "-threads", "2", input)
		hideCommand(cmd)
		cmd.Stdout = log
		cmd.Stderr = log
		if e = cmd.Run(); e != nil {
			a.update(q.ID, "error", "無法建立照片的 LatentSync 輸入影片")
			return
		}
		source = input
	}
	a.update(q.ID, "local_generating", "正在本機執行外部 "+q.Engine+"；實際裝置 "+p.Device+"。完整進度寫入 local.log，可停止工作。")
	args := externalArgs(c, q, source, audio, output, runDir)
	if c.Managed {
		if e = writeManagedScripts(c.Root); e != nil {
			a.update(q.ID, "error", e.Error())
			return
		}
		cfg := map[string]any{"root": c.Root, "work": runDir, "source": source, "audio": audio, "output": output, "ffmpeg": ff, "pose_style": q.PoseStyle, "expression_scale": q.ExpressionScale, "max_side": q.MaxSide}
		v08Config(cfg, q)
		cfg["device"] = q.Device
		cfg["low_impact"] = q.ResourceMode != "normal"
		cfg["threads"] = 2
		if q.ResourceMode == "normal" {
			cfg["threads"] = 4
		}
		cfg["lead"] = j.LeadSeconds
		cfg["tail"] = j.LeadSeconds
		cfg["mouth_box"] = q.MouthBox
		cfg["eye_boxes"] = q.EyeBoxes
		cfg["blink"] = q.Blink
		cfg["gaze"] = q.Gaze
		cfg["body_motion"] = q.BodyMotion
		cfg["hand_box"] = q.HandBox
		cfg["real_detail"] = q.RealDetail
		cfg["anime_detail"] = q.AnimeDetail
		cfg["detail_strength"] = q.DetailStrength
		cfg["detail_models"] = filepath.Join(a.localDir(), "models")
		data, _ := json.MarshalIndent(cfg, "", "  ")
		path := filepath.Join(runDir, "config.json")
		if e = os.WriteFile(path, data, 0600); e != nil {
			a.update(q.ID, "error", e.Error())
			return
		}
		args = []string{"-I", "-u", filepath.Join(c.Root, "junba", "sad_worker.py"), "--config", path}
	}
	cmd := exec.CommandContext(ctx, c.Python, args...)
	cmd.Dir = c.Root
	cmd.Env = append(localEnv(), "HF_HUB_OFFLINE=1", "TRANSFORMERS_OFFLINE=1")
	cmd.Env = append(cmd.Env, "PATH="+filepath.Dir(ff)+string(os.PathListSeparator)+os.Getenv("PATH"))
	hideCommand(cmd)
	cmd.Cancel = func() error { return killLocalProcess(cmd) }
	cmd.WaitDelay = 10 * time.Second
	if c.Managed {
		_, e = a.watchLocalWorker(q.ID, cmd, log)
	} else {
		cmd.Stdout = log
		cmd.Stderr = log
		e = cmd.Run()
	}
	if ctx.Err() != nil {
		a.update(q.ID, "ready", "外部引擎已停止；素材與語音保留，可重新生成。")
		return
	}
	if e != nil {
		a.update(q.ID, "error", "外部 "+q.Engine+" 執行失敗，請查看 local.log；可按「自動安裝所選引擎」修復套件與權重。")
		return
	}
	if q.Engine == "sadtalker" {
		matches, _ := filepath.Glob(filepath.Join(runDir, "*.mp4"))
		if len(matches) != 1 {
			a.update(q.ID, "error", "外部引擎沒有產生唯一的 MP4 成品，請查看 local.log")
			return
		}
		output = matches[0]
	}
	st, e := os.Stat(output)
	if e != nil || st.Size() < 100 {
		a.update(q.ID, "error", "外部引擎未產生有效影片")
		return
	}
	// Decode one frame to avoid claiming success for a mislabeled/broken result.
	check := exec.CommandContext(ctx, ff, "-v", "error", "-nostdin", "-i", output, "-frames:v", "1", "-f", "null", "-")
	hideCommand(check)
	check.Stdout = log
	check.Stderr = log
	if e = check.Run(); e != nil {
		a.update(q.ID, "error", "外部產出的影片無法解碼，請查看 local.log")
		return
	}
	b, e := os.Open(output)
	if e != nil {
		a.update(q.ID, "error", e.Error())
		return
	}
	dst, e := os.Create(filepath.Join(dir, "external-copy.mp4"))
	if e != nil {
		b.Close()
		a.update(q.ID, "error", e.Error())
		return
	}
	_, e = io.Copy(dst, b)
	b.Close()
	ce := dst.Close()
	if e == nil {
		e = ce
	}
	if e == nil {
		e = os.Rename(filepath.Join(dir, "external-copy.mp4"), filepath.Join(dir, "video.mp4"))
	}
	if e != nil {
		a.update(q.ID, "error", e.Error())
		return
	}
	a.mu.Lock()
	a.jobs[q.ID].Output = "video.mp4"
	a.persist(a.jobs[q.ID])
	a.mu.Unlock()
	a.convertJob(q.ID, j.Format, j.Aspect, j.Resolution)
}
func validateAdvanced(q *LocalRequest) error {
	if q.Engine == "" {
		q.Engine = "musetalk"
	}
	if q.Engine != "musetalk" && !validExternalName(q.Engine) {
		return errors.New("不支援的本機引擎")
	}
	if q.ExpressionScale == 0 {
		q.ExpressionScale = 1
	}
	if math.IsNaN(q.ExpressionScale) || math.IsInf(q.ExpressionScale, 0) || q.ExpressionScale < .5 || q.ExpressionScale > 1.5 {
		return errors.New("表情強度需介於 0.5 至 1.5")
	}
	if q.PoseStyle < 0 || q.PoseStyle > 45 {
		return errors.New("姿態樣式需介於 0 至 45")
	}
	if q.InferenceSteps == 0 {
		q.InferenceSteps = 20
	}
	if q.InferenceSteps != 20 && q.InferenceSteps != 30 {
		return errors.New("推論步數僅支援 20 或 30")
	}
	return nil
}
