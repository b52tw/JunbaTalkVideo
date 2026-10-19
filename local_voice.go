package main

import (
	"bufio"
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
	"sort"
	"strconv"
	"strings"
	"time"
	"unicode/utf8"
)

const voiceModelVersion = "openvoice-v2-34d010c-static256-v1"

type LocalVoiceProfile struct {
	ID       string  `json:"id"`
	Name     string  `json:"name"`
	Created  string  `json:"created"`
	Duration float64 `json:"duration"`
	Version  string  `json:"version"`
}

func voiceModels() []LocalModel {
	b, _ := localFiles.ReadFile("local/voice_models.json")
	var v []LocalModel
	json.Unmarshal(b, &v)
	return v
}
func (a *App) voiceReady() bool {
	b, e := os.ReadFile(filepath.Join(a.localDir(), "voice-installed.txt"))
	if e != nil || string(b) != voiceModelVersion {
		return false
	}
	for _, name := range []string{a.localPython(), filepath.Join(a.localDir(), "voice_runner.py")} {
		if _, e := os.Stat(name); e != nil {
			return false
		}
	}
	for _, m := range voiceModels() {
		s, e := os.Stat(filepath.Join(a.localDir(), "models", m.Name))
		if e != nil || s.Size() != m.Size {
			return false
		}
	}
	return true
}
func (a *App) voiceProfileDir(id string) string {
	return filepath.Join(filepath.Dir(a.root), "voice-profiles", id)
}
func (a *App) readVoiceProfile(id string) (LocalVoiceProfile, error) {
	var p LocalVoiceProfile
	if !validID(id) {
		return p, errors.New("無效的本機聲音 ID")
	}
	b, e := os.ReadFile(filepath.Join(a.voiceProfileDir(id), "profile.json"))
	if e != nil {
		return p, e
	}
	if json.Unmarshal(b, &p) != nil || p.ID != id || p.Version != voiceModelVersion {
		return p, errors.New("聲音樣本版本不符，請重新匯入")
	}
	return p, nil
}
func (a *App) localVoiceProfiles(w http.ResponseWriter, r *http.Request) {
	list := []LocalVoiceProfile{}
	entries, _ := os.ReadDir(filepath.Join(filepath.Dir(a.root), "voice-profiles"))
	for _, entry := range entries {
		if !entry.IsDir() {
			continue
		}
		p, e := a.readVoiceProfile(entry.Name())
		if e == nil {
			list = append(list, p)
		}
	}
	sort.Slice(list, func(i, j int) bool { return list[i].Created > list[j].Created })
	respond(w, list)
}
func (a *App) startVoiceInstall(w http.ResponseWriter, r *http.Request) {
	if runtime.GOOS != "windows" {
		fail(w, 400, errors.New("自動安裝適用 Windows x64；其他平台請依原始碼說明安裝。"))
		return
	}
	a.localMu.Lock()
	if a.localState.Busy || a.localJob != "" {
		a.localMu.Unlock()
		fail(w, 409, errors.New("已有本機工作正在執行。"))
		return
	}
	a.localState.Busy = true
	a.localState.Message = "準備本機聲音複製：下載約 131 MB 模型；首次環境另約 185 MB"
	a.localMu.Unlock()
	go func() {
		e := a.installLocalModels(true)
		a.localMu.Lock()
		defer a.localMu.Unlock()
		a.localState.Busy = false
		if e != nil {
			a.localState.Message = "本機聲音複製安裝未完成，可重試：" + e.Error()
		} else {
			a.localState.Message = "本機聲音複製已安裝。可錄音／匯入樣本，輸入新台詞後生成。"
		}
	}()
	respond(w, map[string]bool{"ok": true})
}
func copyLocalFile(from, to string) error {
	src, e := os.Open(from)
	if e != nil {
		return e
	}
	defer src.Close()
	dst, e := os.OpenFile(to, os.O_CREATE|os.O_TRUNC|os.O_WRONLY, 0600)
	if e != nil {
		return e
	}
	_, e = io.Copy(dst, src)
	ce := dst.Close()
	if e != nil {
		return e
	}
	return ce
}
func validVoiceDevice(device string) bool {
	return device == "AUTO" || device == "CPU" || device == "NPU" || device == "GPU" || device == "GPU.0" || device == "GPU.1"
}
func (a *App) prepareLocalVoice(w http.ResponseWriter, r *http.Request) {
	if !a.voiceReady() {
		fail(w, 409, errors.New("請先按「安裝本機聲音複製」。不需要 Fal API Key。"))
		return
	}
	r.Body = http.MaxBytesReader(w, r.Body, 33<<20)
	if e := r.ParseMultipartForm(2 << 20); e != nil {
		fail(w, 400, errors.New("樣本與基礎音檔各限 15 MB。"))
		return
	}
	defer r.MultipartForm.RemoveAll()
	device := r.FormValue("device")
	if device == "" {
		device = "AUTO"
	}
	if !validVoiceDevice(device) {
		fail(w, 400, errors.New("無效的聲音運算裝置"))
		return
	}
	text := strings.TrimSpace(r.FormValue("text"))
	mode := r.FormValue("base_mode")
	if mode != "audio" {
		mode = "tts"
	}
	if utf8.RuneCountInString(text) > 800 || (mode == "tts" && text == "") {
		fail(w, 400, errors.New("請輸入要用您的音色朗讀的新台詞，最多 800 字。"))
		return
	}
	profileID := r.FormValue("profile_id")
	var profile LocalVoiceProfile
	var e error
	if profileID != "" {
		profile, e = a.readVoiceProfile(profileID)
		if e != nil {
			fail(w, 400, errors.New("找不到可用的本機聲音，請重新匯入錄音樣本。"))
			return
		}
	} else if len(r.MultipartForm.File["sample"]) != 1 {
		fail(w, 400, errors.New("請先錄音或選擇 5～60 秒的音色樣本。"))
		return
	}
	if mode == "audio" && len(r.MultipartForm.File["baseline"]) != 1 {
		fail(w, 400, errors.New("請在匯入音檔欄位選擇要換音色的基礎語音。"))
		return
	}
	mem := memoryInfo()
	if mem.Available > 0 && mem.Available < 2<<30 {
		fail(w, 409, errors.New("本機聲音複製至少需 2 GB 可用記憶體，請先關閉其他程式。"))
		return
	}
	a.localMu.Lock()
	if a.localState.Busy || a.localJob != "" {
		a.localMu.Unlock()
		fail(w, 409, errors.New("已有本機安裝、聲音或影片生成正在執行。"))
		return
	}
	defer a.localMu.Unlock()
	id := randomID()
	dir := a.dir(id)
	if e = os.MkdirAll(dir, 0700); e != nil {
		fail(w, 500, e)
		return
	}
	keep := false
	defer func() {
		if !keep {
			os.RemoveAll(dir)
		}
	}()
	name := strings.TrimSpace(r.FormValue("profile_name"))
	if utf8.RuneCountInString(name) > 64 {
		fail(w, 400, errors.New("聲音名稱最多 64 字。"))
		return
	}
	if name == "" {
		name = "我的聲音 " + time.Now().Format("20060102-1504")
	}
	if profileID == "" {
		if _, e = copyUpload(r, "sample", filepath.Join(dir, "sample-upload.bin"), 15<<20); e != nil {
			fail(w, 400, e)
			return
		}
	} else {
		name = profile.Name
		if e = copyLocalFile(filepath.Join(a.voiceProfileDir(profileID), "sample.wav"), filepath.Join(dir, "sample.wav")); e != nil {
			fail(w, 400, e)
			return
		}
		if e = copyLocalFile(filepath.Join(a.voiceProfileDir(profileID), "embedding.npy"), filepath.Join(dir, "target-embedding.npy")); e != nil {
			fail(w, 400, e)
			return
		}
	}
	if mode == "audio" {
		if _, e = copyUpload(r, "baseline", filepath.Join(dir, "base-upload.bin"), 15<<20); e != nil {
			fail(w, 400, e)
			return
		}
	}
	rate, _ := strconv.Atoi(r.FormValue("rate"))
	if rate < -5 || rate > 5 {
		rate = 0
	}
	j := &Job{ID: id, Created: time.Now().Format(time.RFC3339), Kind: "voice", Text: text, Source: "sample.wav", Audio: "speech.wav", State: "local_voice_generating", Message: "正在準備本機聲音複製；素材不會上傳。", VoiceTask: "local_clone", VoiceID: profileID, VoiceName: name, VoiceBase: mode, Engine: "local_voice", busy: true}
	if e = a.persist(j); e != nil {
		fail(w, 500, e)
		return
	}
	a.mu.Lock()
	a.jobs[id] = j
	snapshot := *j
	a.mu.Unlock()
	ctx, cancel := context.WithCancel(context.Background())
	a.localJob = id
	a.localCancel = cancel
	keep = true
	go a.runLocalVoice(ctx, id, device, r.FormValue("voice"), rate, profileID != "")
	respond(w, snapshot)
}
func (a *App) runLocalVoice(ctx context.Context, id, device, voice string, rate int, reuse bool) {
	defer func() {
		a.localMu.Lock()
		a.localJob = ""
		a.localCancel = nil
		a.localMu.Unlock()
		a.mu.Lock()
		a.jobs[id].busy = false
		a.mu.Unlock()
	}()
	dir := a.dir(id)
	if e := a.refreshLocalScripts(); e != nil {
		a.update(id, "error", e.Error())
		return
	}
	failJob := func(e error) {
		if ctx.Err() != nil {
			a.update(id, "cancelled", "本機聲音複製已停止，錄音樣本已保留。")
		} else {
			a.update(id, "error", "本機聲音複製失敗："+e.Error())
		}
	}
	a.mu.Lock()
	j := *a.jobs[id]
	a.mu.Unlock()
	sampleDuration := float64(0)
	if !reuse {
		a.update(id, "local_voice_generating", "正在本機整理錄音樣本")
		if e := audioWAV(ctx, filepath.Join(dir, "sample-upload.bin"), filepath.Join(dir, "sample.wav"), "original"); e != nil {
			failJob(e)
			return
		}
		d, e := wavDuration(filepath.Join(dir, "sample.wav"))
		if e != nil || d < 5 || d > 60 {
			failJob(errors.New("請使用 5～60 秒清楚的單人樣本；背景音樂與多人聲會降低效果。"))
			return
		}
		sampleDuration = d
	}
	base := filepath.Join(dir, "base.wav")
	if j.VoiceBase == "audio" {
		if e := audioWAV(ctx, filepath.Join(dir, "base-upload.bin"), base, "original"); e != nil {
			failJob(e)
			return
		}
	} else {
		a.update(id, "local_voice_generating", "正在使用 Windows 本機語音朗讀新台詞")
		if !a.prep.TryLock() {
			failJob(errors.New("另一段基礎語音正在製作，請稍後重試。"))
			return
		}
		e := a.tts(ctx, j.Text, voice, rate, base)
		a.prep.Unlock()
		if e != nil {
			failJob(e)
			return
		}
	}
	duration, e := wavDuration(base)
	if e != nil || duration < 3 || duration > 60 {
		failJob(errors.New("基礎語音長度需為 3～60 秒；請調整台詞長度。"))
		return
	}
	a.mu.Lock()
	a.jobs[id].Duration = duration
	a.persist(a.jobs[id])
	a.mu.Unlock()
	cfg := map[string]any{"models": filepath.Join(a.localDir(), "models"), "device": device, "source": base, "reference": filepath.Join(dir, "sample.wav"), "output": filepath.Join(dir, "speech.wav"), "save_embedding": filepath.Join(dir, "target-embedding.npy")}
	if reuse {
		cfg["target_embedding"] = filepath.Join(dir, "target-embedding.npy")
	}
	b, _ := json.MarshalIndent(cfg, "", "  ")
	if e = os.WriteFile(filepath.Join(dir, "voice-config.json"), b, 0600); e != nil {
		failJob(e)
		return
	}
	c := exec.CommandContext(ctx, a.localPython(), "-u", filepath.Join(a.localDir(), "voice_runner.py"), "--config", filepath.Join(dir, "voice-config.json"))
	c.Env = localEnv()
	hideCommand(c)
	c.Cancel = func() error { return killLocalProcess(c) }
	c.WaitDelay = 10 * time.Second
	pipe, e := c.StdoutPipe()
	if e != nil {
		failJob(e)
		return
	}
	log, e := os.Create(filepath.Join(dir, "local-voice.log"))
	if e != nil {
		failJob(e)
		return
	}
	defer log.Close()
	c.Stderr = log
	if e = c.Start(); e != nil {
		failJob(e)
		return
	}
	scanner := bufio.NewScanner(pipe)
	scanner.Buffer(make([]byte, 4096), 1<<20)
	lastError := ""
	for scanner.Scan() {
		line := scanner.Bytes()
		log.Write(append(append([]byte{}, line...), '\n'))
		var p struct {
			Stage, Message    string
			Frame, Total, ETA int
			Actual            map[string][]string
			Fallbacks         []string
		}
		if json.Unmarshal(line, &p) != nil {
			continue
		}
		if p.Stage == "error" {
			lastError = p.Message
		}
		a.mu.Lock()
		jj := a.jobs[id]
		if p.Message != "" {
			jj.Message = p.Message
		}
		if p.Actual != nil {
			jj.ActualDevices = p.Actual
		}
		if p.Fallbacks != nil {
			jj.VoiceFallbacks = p.Fallbacks
		}
		if p.Total > 0 {
			jj.LocalFrame = p.Frame
			jj.LocalTotal = p.Total
			jj.LocalETA = p.ETA
		}
		a.persist(jj)
		a.mu.Unlock()
	}
	if e = c.Wait(); e != nil {
		if lastError == "" {
			lastError = "執行未完成，請查看 local-voice.log"
		}
		failJob(errors.New(lastError))
		return
	}
	if ctx.Err() != nil {
		failJob(ctx.Err())
		return
	}
	if e = scanner.Err(); e != nil {
		failJob(e)
		return
	}
	duration, e = wavDuration(filepath.Join(dir, "speech.wav"))
	if e != nil {
		failJob(e)
		return
	}
	var hasCaptions bool
	var captionTiming string
	if j.Text != "" {
		if j.VoiceBase == "tts" {
			copyLocalFile(base+".timing.tsv", filepath.Join(dir, "speech.wav.timing.tsv"))
		}
		hasCaptions, captionTiming, e = writeCaptions(dir, j.Text, duration)
		if e != nil {
			failJob(e)
			return
		}
	}
	if !reuse {
		profileID := randomID()
		pdir := a.voiceProfileDir(profileID)
		if e = os.MkdirAll(pdir, 0700); e != nil {
			failJob(e)
			return
		}
		for _, pair := range [][2]string{{"sample.wav", "sample.wav"}, {"target-embedding.npy", "embedding.npy"}} {
			if e = copyLocalFile(filepath.Join(dir, pair[0]), filepath.Join(pdir, pair[1])); e != nil {
				failJob(e)
				return
			}
		}
		profile := LocalVoiceProfile{ID: profileID, Name: j.VoiceName, Created: time.Now().Format(time.RFC3339), Duration: sampleDuration, Version: voiceModelVersion}
		b, _ = json.MarshalIndent(profile, "", "  ")
		if e = os.WriteFile(filepath.Join(pdir, "profile.json"), b, 0600); e != nil {
			failJob(e)
			return
		}
		j.VoiceID = profileID
	}
	a.mu.Lock()
	jj := a.jobs[id]
	jj.VoiceID = j.VoiceID
	jj.HasCaptions = hasCaptions
	jj.CaptionTiming = captionTiming
	jj.Duration = duration
	a.persist(jj)
	a.mu.Unlock()
	a.update(id, "voice_done", fmt.Sprintf("本機聲音複製完成（%.1f 秒）。請試聽後按「使用這段語音製作影片」；已保存音色樣本，之後可直接沿用。", duration))
}
