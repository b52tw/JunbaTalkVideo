package main

import (
	"context"
	"errors"
	"fmt"
	"io"
	"net/http"
	"net/url"
	"os"
	"os/exec"
	"path/filepath"
	"strings"
	"time"
	"unicode/utf8"
)

func (a *App) prepareVoice(w http.ResponseWriter, r *http.Request) {
	r.Body = http.MaxBytesReader(w, r.Body, 16<<20)
	if e := r.ParseMultipartForm(1 << 20); e != nil {
		fail(w, 400, errors.New("聲音樣本限 15 MB 的 PCM WAV。"))
		return
	}
	defer r.MultipartForm.RemoveAll()
	text := strings.TrimSpace(r.FormValue("text"))
	if text == "" || utf8.RuneCountInString(text) > 800 {
		fail(w, 400, errors.New("請先輸入要用您的聲音朗讀的新台詞（最多 800 字）。"))
		return
	}
	task := "clone"
	voiceID := strings.TrimSpace(r.FormValue("voice_id"))
	if voiceID != "" {
		task = "speak"
	}
	id := randomID()
	dir := a.dir(id)
	if e := os.MkdirAll(dir, 0700); e != nil {
		fail(w, 500, e)
		return
	}
	keep := false
	defer func() {
		if !keep {
			os.RemoveAll(dir)
		}
	}()
	j := &Job{ID: id, Created: time.Now().Format(time.RFC3339), Kind: "voice", Text: text, Audio: "speech.wav", State: "ready", Message: "已建立聲音任務，可送出付費生成。", VoiceTask: task, VoiceID: voiceID}
	if task == "clone" {
		j.Source = "sample.wav"
		if _, e := copyUpload(r, "sample", filepath.Join(dir, "original-sample.bin"), 15<<20); e != nil {
			fail(w, 400, errors.New("請先錄音或匯入您的 PCM WAV 樣本。"))
			return
		}
		ctx, cancel := context.WithTimeout(r.Context(), 3*time.Minute)
		defer cancel()
		if e := audioWAV(ctx, filepath.Join(dir, "original-sample.bin"), filepath.Join(dir, j.Source), "original"); e != nil {
			fail(w, 400, e)
			return
		}
		d, e := wavDuration(filepath.Join(dir, j.Source))
		if e != nil || d < 10 || d > 60 {
			fail(w, 400, errors.New("聲音複製樣本請使用 10～60 秒清楚的 PCM WAV。"))
			return
		}
	}
	var pitch int
	fmt.Sscanf(r.FormValue("pitch"), "%d", &pitch)
	if pitch < -12 || pitch > 12 {
		pitch = 0
	}
	j.VoicePitch = pitch
	if e := a.persist(j); e != nil {
		fail(w, 500, e)
		return
	}
	a.mu.Lock()
	a.jobs[id] = j
	a.mu.Unlock()
	keep = true
	respond(w, j)
}
func (a *App) runVoice(ctx context.Context, id, key string, resume bool) {
	a.mu.Lock()
	j := a.jobs[id]
	q := j.Queue
	task, text, voiceID, source, pitch := j.VoiceTask, j.Text, j.VoiceID, j.Source, j.VoicePitch
	a.mu.Unlock()
	if !resume {
		model := "fal-ai/minimax/voice-clone"
		var input map[string]any
		if task == "clone" {
			src, e := dataURI(filepath.Join(a.dir(id), source))
			if e != nil {
				a.update(id, "ready", e.Error())
				return
			}
			input = map[string]any{"audio_url": src, "text": text, "model": "speech-02-hd", "noise_reduction": true, "need_volume_normalization": true}
		} else {
			model = "fal-ai/minimax/speech-02-hd"
			input = map[string]any{"prompt": text, "voice_setting": map[string]any{"voice_id": voiceID, "speed": 1, "pitch": pitch}, "output_format": "url", "language_boost": "Chinese"}
		}
		if e := a.jsonRequest(ctx, "POST", a.queueBase+"/"+model, key, input, &q); e != nil {
			a.update(id, "uncertain", "聲音任務送出結果未確認；請先至 Fal 控制台確認，避免重複計費。"+e.Error())
			return
		}
		if q.RequestID == "" || !a.validQueueURL(q.StatusURL) || !a.validQueueURL(q.ResponseURL) || !a.validQueueURL(q.CancelURL) {
			a.update(id, "uncertain", "聲音任務回應格式異常，請查 Fal 控制台。任務 ID："+q.RequestID)
			return
		}
		a.mu.Lock()
		j.Queue = q
		e := a.persist(j)
		a.mu.Unlock()
		if e != nil {
			a.update(id, "waiting_resume", "任務已送出但記錄寫入失敗。任務 ID："+q.RequestID)
			return
		}
	}
	if !a.validQueueURL(q.StatusURL) || !a.validQueueURL(q.ResponseURL) {
		a.update(id, "error", "任務來源不正確，請至 Fal 控制台查看。")
		return
	}
	failures := 0
	for {
		var st struct {
			Status string `json:"status"`
			Error  string `json:"error"`
		}
		e := a.jsonRequest(ctx, "GET", q.StatusURL, key, nil, &st)
		if e != nil {
			a.mu.Lock()
			cancelling := j.CancelRequested
			a.mu.Unlock()
			if cancelling && strings.Contains(e.Error(), "HTTP 404") {
				a.update(id, "cancelled", "聲音任務已取消或移除，請以控制台狀態為準。")
				return
			}
			failures++
			if failures >= 6 {
				a.update(id, "waiting_resume", "聲音任務查詢中斷，可繼續查詢原任務。"+e.Error())
				return
			}
		} else {
			failures = 0
			switch st.Status {
			case "IN_QUEUE":
				a.update(id, "queued", "聲音生成正在排隊。")
			case "IN_PROGRESS":
				a.update(id, "generating", "AI 正在處理您的聲音。")
			case "COMPLETED":
				if st.Error != "" {
					a.update(id, "error", "聲音生成失敗："+st.Error)
					return
				}
				a.retrieveVoice(ctx, id, key, q.ResponseURL)
				return
			default:
				a.update(id, "waiting_resume", "聲音服務狀態尚待確認，可繼續查詢。")
				return
			}
		}
		select {
		case <-ctx.Done():
			a.update(id, "waiting_resume", "聲音任務查詢逾時，可繼續查詢。")
			return
		case <-time.After(a.pollDelay):
		}
	}
}
func (a *App) retrieveVoice(ctx context.Context, id, key, responseURL string) {
	var result struct {
		VoiceID string `json:"custom_voice_id"`
		Audio   struct {
			URL string `json:"url"`
		} `json:"audio"`
	}
	if e := a.jsonRequest(ctx, "GET", responseURL, key, nil, &result); e != nil {
		a.update(id, "waiting_resume", "聲音結果取回失敗，可繼續查詢。"+e.Error())
		return
	}
	a.mu.Lock()
	j := a.jobs[id]
	if result.VoiceID != "" {
		j.VoiceID = result.VoiceID
	}
	a.persist(j)
	a.mu.Unlock()
	if result.Audio.URL == "" {
		a.mu.Lock()
		j.Audio = ""
		a.mu.Unlock()
		a.update(id, "voice_done", "聲音 ID 已建立；請從聲音工作室選擇此聲音，再輸入文字生成語音。")
		return
	}
	u, e := url.Parse(result.Audio.URL)
	if e != nil || u.User != nil || u.Scheme != "https" {
		if !(e == nil && strings.HasPrefix(a.queueBase, "http://127.0.0.1:") && u.Scheme == "http") {
			a.update(id, "error", "沒有取得有效的語音下載網址。")
			return
		}
	}
	a.update(id, "downloading", "正在下載語音並轉成可用的 WAV。")
	req, e := http.NewRequestWithContext(ctx, "GET", result.Audio.URL, nil)
	if e != nil {
		a.update(id, "waiting_resume", e.Error())
		return
	}
	resp, e := a.client.Do(req)
	if e != nil {
		a.update(id, "waiting_resume", e.Error())
		return
	}
	defer resp.Body.Close()
	if resp.StatusCode != 200 {
		a.update(id, "waiting_resume", fmt.Sprintf("音檔下載回應 HTTP %d", resp.StatusCode))
		return
	}
	dir := a.dir(id)
	raw := filepath.Join(dir, "generated-audio.bin")
	f, e := os.Create(raw)
	if e != nil {
		a.update(id, "waiting_resume", e.Error())
		return
	}
	n, e := io.Copy(f, io.LimitReader(resp.Body, (20<<20)+1))
	ce := f.Close()
	if e == nil {
		e = ce
	}
	if e != nil || n > 20<<20 {
		a.update(id, "waiting_resume", "音檔下載失敗或檔案太大。")
		return
	}
	engine, e := ffmpegPath()
	if e != nil {
		a.update(id, "waiting_resume", e.Error())
		return
	}
	cmd := exec.CommandContext(ctx, engine, "-hide_banner", "-nostdin", "-y", "-i", raw, "-vn", "-acodec", "pcm_s16le", "-ar", "24000", "-ac", "1", filepath.Join(dir, "speech.wav"))
	hideCommand(cmd)
	if b, e := cmd.CombinedOutput(); e != nil {
		os.WriteFile(filepath.Join(dir, "voice-conversion.log"), b, 0600)
		a.update(id, "waiting_resume", "語音轉檔失敗，請查看 voice-conversion.log。")
		return
	}
	duration, e := wavDuration(filepath.Join(dir, "speech.wav"))
	if e != nil || duration < 3 || duration > 60 {
		a.update(id, "error", "生成語音須介於 3～60 秒；聲音 ID 已保存，可縮短台詞後使用原聲音 ID 再試。")
		return
	}
	a.mu.Lock()
	j.Audio = "speech.wav"
	j.Duration = duration
	j.HasCaptions, j.CaptionTiming, e = writeCaptions(dir, j.Text, duration)
	a.mu.Unlock()
	if e != nil {
		a.update(id, "error", e.Error())
		return
	}
	os.WriteFile(filepath.Join(dir, "script.txt"), []byte(j.Text), 0600)
	a.update(id, "voice_done", "語音完成！可下載 WAV，或按「使用這段語音製作影片」。這是雲端聲音複製／合成，不是本機持續自學。")
}
func (a *App) voiceProfiles(w http.ResponseWriter) {
	a.mu.Lock()
	defer a.mu.Unlock()
	var profiles []map[string]string
	seen := map[string]bool{}
	for _, j := range a.jobs {
		if j.VoiceID != "" && !seen[j.VoiceID] {
			profiles = append(profiles, map[string]string{"id": j.VoiceID, "created": j.Created})
			seen[j.VoiceID] = true
		}
	}
	respond(w, profiles)
}
