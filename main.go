package main

import (
	"bytes"
	"context"
	"crypto/rand"
	"crypto/subtle"
	"embed"
	"encoding/base64"
	"encoding/binary"
	"encoding/hex"
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"log"
	"mime"
	"net"
	"net/http"
	"net/url"
	"os"
	"path/filepath"
	"sort"
	"strconv"
	"strings"
	"sync"
	"time"
	"unicode/utf8"
)

//go:embed ui/index.html assets/default.jpg
var files embed.FS

const photoModel = "fal-ai/kling-video/ai-avatar/v2/standard"
const videoModel = "fal-ai/sync-lipsync/v2"
const maxMedia = 20 << 20

type Voice struct {
	Name string `json:"name"`
	Lang string `json:"lang"`
}
type QueueInfo struct {
	RequestID   string `json:"request_id"`
	StatusURL   string `json:"status_url"`
	ResponseURL string `json:"response_url"`
	CancelURL   string `json:"cancel_url"`
}
type Job struct {
	MemoryPeaks       map[string]float64  `json:"memory_peaks,omitempty"`
	LocalStage        string              `json:"local_stage,omitempty"`
	LeadSeconds       float64             `json:"lead_seconds,omitempty"`
	LocalSettings     *LocalRequest       `json:"local_settings,omitempty"`
	ID                string              `json:"id"`
	Created           string              `json:"created"`
	Kind              string              `json:"kind"`
	Text              string              `json:"text"`
	Source            string              `json:"source"`
	Audio             string              `json:"audio"`
	State             string              `json:"state"`
	Message           string              `json:"message"`
	Duration          float64             `json:"duration"`
	Output            string              `json:"output,omitempty"`
	Format            string              `json:"format"`
	Aspect            string              `json:"aspect"`
	Resolution        string              `json:"resolution"`
	Export            string              `json:"export,omitempty"`
	CaptionMode       string              `json:"caption_mode"`
	CaptionTiming     string              `json:"caption_timing,omitempty"`
	HasCaptions       bool                `json:"has_captions"`
	Dialogue          bool                `json:"dialogue,omitempty"`
	CaptionShift      float64             `json:"caption_shift,omitempty"`
	DialogueSpeakers  []string            `json:"dialogue_speakers,omitempty"`
	VoiceID           string              `json:"voice_id,omitempty"`
	VoiceTask         string              `json:"voice_task,omitempty"`
	VoicePitch        int                 `json:"voice_pitch,omitempty"`
	Compatible        bool                `json:"compatible"`
	Queue             QueueInfo           `json:"queue"`
	CancelRequested   bool                `json:"cancel_requested,omitempty"`
	Engine            string              `json:"engine,omitempty"`
	ActualDevices     map[string][]string `json:"actual_devices,omitempty"`
	LocalFrame        int                 `json:"local_frame,omitempty"`
	LocalTotal        int                 `json:"local_total,omitempty"`
	LocalETA          int                 `json:"local_eta"`
	EffectiveBudgetMB int                 `json:"effective_budget_mb,omitempty"`
	BudgetPolicy      string              `json:"budget_policy,omitempty"`
	Encoder           string              `json:"encoder,omitempty"`
	FaceBox           []float64           `json:"face_box,omitempty"`
	DefaultAvatar     bool                `json:"default_avatar,omitempty"`
	VoiceName         string              `json:"voice_name,omitempty"`
	VoiceBase         string              `json:"voice_base,omitempty"`
	VoiceFallbacks    []string            `json:"voice_fallbacks,omitempty"`
	busy              bool
	key               string
}
type App struct {
	mu                     sync.Mutex
	prep                   sync.Mutex
	conversion             sync.Mutex
	jobs                   map[string]*Job
	root, token, queueBase string
	client                 *http.Client
	pollDelay              time.Duration
	tts                    func(context.Context, string, string, int, string) error
	localMu                sync.Mutex
	localState             LocalStatus
	localCancel            context.CancelFunc
	installCancel          context.CancelFunc
	localJob               string
}

func randomID() string {
	b := make([]byte, 16)
	if _, e := rand.Read(b); e != nil {
		panic(e)
	}
	return hex.EncodeToString(b)
}
func newApp(root string) *App {
	return &App{jobs: map[string]*Job{}, root: root, token: randomID(), queueBase: "https://queue.fal.run", client: &http.Client{Timeout: 180 * time.Second, CheckRedirect: func(r *http.Request, via []*http.Request) error {
		if len(via) > 8 {
			return errors.New("重新導向過多")
		}
		if len(via) > 0 && via[0].Header.Get("Authorization") != "" && r.URL.Host != via[0].URL.Host {
			return errors.New("拒絕跨網站轉送 API 金鑰")
		}
		return nil
	}}, pollDelay: 4 * time.Second, tts: synthesize}
}
func (a *App) cookieName() string   { return "junba_session_" + a.token[:12] }
func (a *App) dir(id string) string { return filepath.Join(a.root, id) }
func (a *App) persist(j *Job) error {
	b, e := json.MarshalIndent(j, "", "  ")
	if e != nil {
		return e
	}
	p := filepath.Join(a.dir(j.ID), "project.json")
	tmp := p + ".tmp"
	if e = os.WriteFile(tmp, b, 0600); e != nil {
		return e
	}
	return os.Rename(tmp, p)
}
func (a *App) update(id, state, msg string) {
	a.mu.Lock()
	defer a.mu.Unlock()
	j := a.jobs[id]
	j.State = state
	j.Message = msg
	if e := a.persist(j); e != nil {
		log.Printf("儲存專案狀態失敗: %v", e)
	}
}
func (a *App) load() {
	entries, _ := os.ReadDir(a.root)
	for _, d := range entries {
		if !d.IsDir() {
			continue
		}
		b, e := os.ReadFile(filepath.Join(a.root, d.Name(), "project.json"))
		if e != nil {
			continue
		}
		var j Job
		if json.Unmarshal(b, &j) != nil || j.ID != d.Name() || !validID(j.ID) {
			continue
		}
		if j.State != "ready" && j.State != "done" && j.State != "error" && j.State != "cancelled" && j.State != "voice_done" {
			if _, err := os.Stat(filepath.Join(a.dir(j.ID), "video.mp4")); err == nil {
				j.State = "done"
				j.Output = "video.mp4"
				j.Message = "原始 MP4 已儲存；如轉檔中斷，可重新選擇格式轉出。"
			} else if j.Engine == "local_voice" {
				j.State = "error"
				j.Message = "上次本機聲音複製中斷，請重新按本機聲音複製；樣本與原始錄音已保留。"
			} else if j.Engine == "local" {
				j.State = "ready"
				j.Message = "上次本機生成中斷，語音與素材已保留，可以重新生成。"
			} else if j.Kind == "convert" {
				j.State = "error"
				j.Message = "上次轉檔中斷，請重新選擇素材轉檔。"
			} else if j.Queue.RequestID != "" {
				j.State = "waiting_resume"
				j.Message = "輸入金鑰後按「繼續查詢」，可取回原任務，不會重新付費送出。"
			} else {
				j.State = "uncertain"
				j.Message = "上次送出狀態不明，請先到 Fal 控制台確認，避免重複計費。"
			}
		}
		if _, e := os.Stat(filepath.Join(a.dir(j.ID), "export.mp4")); e == nil {
			j.Compatible = true
		}
		a.jobs[j.ID] = &j
	}
}
func validID(id string) bool {
	if len(id) != 32 {
		return false
	}
	_, e := hex.DecodeString(id)
	return e == nil
}
func respond(w http.ResponseWriter, v any) {
	w.Header().Set("Content-Type", "application/json; charset=utf-8")
	json.NewEncoder(w).Encode(v)
}
func fail(w http.ResponseWriter, code int, e error) {
	w.WriteHeader(code)
	respond(w, map[string]string{"error": e.Error()})
}
func (a *App) handler(w http.ResponseWriter, r *http.Request) {
	w.Header().Set("X-Content-Type-Options", "nosniff")
	w.Header().Set("Referrer-Policy", "no-referrer")
	w.Header().Set("Cache-Control", "no-store")
	w.Header().Set("Content-Security-Policy", "default-src 'self'; img-src 'self' blob:; media-src 'self' blob:; script-src 'self' 'unsafe-inline' blob:; worker-src 'self' blob:; style-src 'self' 'unsafe-inline'; connect-src 'self'; frame-ancestors 'none'; base-uri 'none'; form-action 'self'")
	host, _, e := net.SplitHostPort(r.Host)
	if e != nil || host != "127.0.0.1" {
		http.Error(w, "Forbidden", 403)
		return
	}
	if r.URL.Path == "/" && r.URL.Query().Get("session") != "" {
		if subtle.ConstantTimeCompare([]byte(r.URL.Query().Get("session")), []byte(a.token)) != 1 {
			http.Error(w, "Forbidden", 403)
			return
		}
		http.SetCookie(w, &http.Cookie{Name: a.cookieName(), Value: a.token, HttpOnly: true, SameSite: http.SameSiteStrictMode, Path: "/"})
		http.Redirect(w, r, "/", http.StatusSeeOther)
		return
	}
	cookie, e := r.Cookie(a.cookieName())
	if e != nil || subtle.ConstantTimeCompare([]byte(cookie.Value), []byte(a.token)) != 1 {
		http.Error(w, "請從程式啟動的視窗進入。", 403)
		return
	}
	if r.Method != "GET" {
		origin := r.Header.Get("Origin")
		if origin != "http://"+r.Host {
			http.Error(w, "Forbidden", 403)
			return
		}
	}
	switch {
	case r.URL.Path == "/api/local/install-cancel" && r.Method == "POST":
		a.cancelInstall(w)
	case strings.HasPrefix(r.URL.Path, "/api/local/"):
		a.localHandler(w, r)
	case r.URL.Path == "/" && r.Method == "GET":
		b, _ := files.ReadFile("ui/index.html")
		page := strings.ReplaceAll(string(b), "{{APP_VERSION}}", appVersion())
		w.Header().Set("Content-Type", "text/html; charset=utf-8")
		_, _ = w.Write([]byte(page))
	case r.URL.Path == "/default.jpg" && r.Method == "GET":
		b, _ := files.ReadFile("assets/default.jpg")
		w.Header().Set("Content-Type", "image/jpeg")
		w.Write(b)
	case r.URL.Path == "/api/voices" && r.Method == "GET":
		ctx, cancel := context.WithTimeout(r.Context(), 25*time.Second)
		defer cancel()
		v, e := installedVoices(ctx)
		if e != nil {
			respond(w, map[string]any{"voices": []Voice{}, "warning": e.Error()})
			return
		}
		respond(w, map[string]any{"voices": v})
	case r.URL.Path == "/api/jobs" && r.Method == "GET":
		a.mu.Lock()
		list := make([]Job, 0, len(a.jobs))
		for _, j := range a.jobs {
			if id := r.URL.Query().Get("id"); id == "" || id == j.ID {
				list = append(list, *j)
			}
		}
		a.mu.Unlock()
		sort.Slice(list, func(i, j int) bool { return list[i].Created > list[j].Created })
		respond(w, list)
	case r.URL.Path == "/api/prepare" && r.Method == "POST":
		a.prepare(w, r)
	case r.URL.Path == "/api/convert" && r.Method == "POST":
		a.convertHandler(w, r)
	case r.URL.Path == "/api/convert-file" && r.Method == "POST":
		a.convertFileHandler(w, r)
	case strings.HasPrefix(r.URL.Path, "/playback/") && r.Method == "GET":
		a.playbackPackage(w, r)
	case r.URL.Path == "/api/voice-prepare" && r.Method == "POST":
		a.prepareVoice(w, r)
	case r.URL.Path == "/api/voice-profiles" && r.Method == "GET":
		a.voiceProfiles(w)
	case r.URL.Path == "/api/generate" && r.Method == "POST":
		a.generate(w, r, false)
	case r.URL.Path == "/api/resume" && r.Method == "POST":
		a.generate(w, r, true)
	case r.URL.Path == "/api/cancel" && r.Method == "POST":
		a.cancel(w, r)
	case r.URL.Path == "/api/folder" && r.Method == "POST":
		var q struct {
			ID string `json:"id"`
		}
		if json.NewDecoder(io.LimitReader(r.Body, 4096)).Decode(&q) != nil {
			fail(w, 400, errors.New("無效資料"))
			return
		}
		a.mu.Lock()
		_, ok := a.jobs[q.ID]
		a.mu.Unlock()
		if !ok {
			fail(w, 404, errors.New("找不到專案"))
			return
		}
		if e := openFolder(a.dir(q.ID)); e != nil {
			fail(w, 500, e)
			return
		}
		respond(w, map[string]bool{"ok": true})
	case r.URL.Path == "/api/quit" && r.Method == "POST":
		respond(w, map[string]bool{"ok": true})
		go func() { time.Sleep(250 * time.Millisecond); os.Exit(0) }()
	case strings.HasPrefix(r.URL.Path, "/media/") && r.Method == "GET":
		a.serveMedia(w, r)
	default:
		http.NotFound(w, r)
	}
}
func (a *App) serveMedia(w http.ResponseWriter, r *http.Request) {
	p := strings.Split(strings.TrimPrefix(r.URL.Path, "/media/"), "/")
	if len(p) != 2 || !validID(p[0]) {
		http.NotFound(w, r)
		return
	}
	a.mu.Lock()
	j := a.jobs[p[0]]
	allowed := p[1] != "" && j != nil && (p[1] == j.Source || p[1] == j.Audio || p[1] == "video.mp4" || p[1] == j.Export || p[1] == "export.mp4" || p[1] == "subtitles.srt")
	a.mu.Unlock()
	if !allowed {
		http.NotFound(w, r)
		return
	}
	http.ServeFile(w, r, filepath.Join(a.dir(p[0]), p[1]))
}
func copyUpload(r *http.Request, field, dst string, limit int64) (string, error) {
	f, h, e := r.FormFile(field)
	if e != nil {
		return "", e
	}
	defer f.Close()
	if h.Size > limit {
		return "", fmt.Errorf("%s 太大，請縮小後重試", h.Filename)
	}
	out, e := os.Create(dst)
	if e != nil {
		return "", e
	}
	n, e := io.Copy(out, io.LimitReader(f, limit+1))
	ce := out.Close()
	if e == nil {
		e = ce
	}
	if n > limit {
		e = errors.New("檔案超過大小限制")
	}
	return h.Filename, e
}
func (a *App) prepare(w http.ResponseWriter, r *http.Request) {
	if !a.prep.TryLock() {
		fail(w, 409, errors.New("正在製作另一段語音，請稍候。"))
		return
	}
	defer a.prep.Unlock()
	r.Body = http.MaxBytesReader(w, r.Body, 38<<20)
	if e := r.ParseMultipartForm(2 << 20); e != nil {
		fail(w, 400, errors.New("檔案過大或格式不正確；照片／影片限 20 MB，WAV 限 15 MB。"))
		return
	}
	defer r.MultipartForm.RemoveAll()
	text := strings.TrimSpace(r.FormValue("text"))
	if utf8.RuneCountInString(text) > 800 {
		fail(w, 400, errors.New("測試版講稿最多 800 字，建議先用 1～2 句試做。"))
		return
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
	j := &Job{ID: id, Created: time.Now().Format(time.RFC3339), Kind: "photo", Audio: "speech.wav", State: "ready", Text: text, Message: "語音已備妥。確認後可送出 AI 對嘴。"}
	j.Format, j.Aspect, j.Resolution = normalizeExport(r.FormValue("format"), r.FormValue("aspect"), r.FormValue("resolution"))
	j.CaptionMode = captionMode(r.FormValue("caption_mode"))
	src := r.MultipartForm.File["source"]
	if len(src) > 0 {
		ext := strings.ToLower(filepath.Ext(src[0].Filename))
		switch ext {
		case ".jpg", ".jpeg", ".png", ".webp":
			j.Kind = "photo"
		case ".mp4", ".mov", ".webm":
			j.Kind = "video"
		default:
			fail(w, 400, errors.New("請選擇 JPG、PNG、WebP 照片或 MP4、MOV、WebM 影片。"))
			return
		}
		j.Source = "source" + ext
		if _, e := copyUpload(r, "source", filepath.Join(dir, j.Source), maxMedia); e != nil {
			fail(w, 400, e)
			return
		}
	} else {
		j.DefaultAvatar = true
		j.Source = "source.jpg"
		b, _ := files.ReadFile("assets/default.jpg")
		if e := os.WriteFile(filepath.Join(dir, j.Source), b, 0600); e != nil {
			fail(w, 500, e)
			return
		}
	}
	j.FaceBox = []float64{.25, .18, .5, .5}
	if j.DefaultAvatar {
		j.FaceBox = []float64{.28, .25, .47, .29}
	}
	if box := r.FormValue("face_box"); box != "" {
		var q LocalRequest
		if json.Unmarshal([]byte(box), &q.BBox) != nil || validateLocal(&q) != nil {
			fail(w, 400, errors.New("臉部框選範圍無效，請重新框選。"))
			return
		}
		j.FaceBox = q.BBox
	}
	dialogue := r.FormValue("dialogue") == "1"
	second := strings.TrimSpace(r.FormValue("caption_second"))
	if utf8.RuneCountInString(second) > 1600 {
		fail(w, 400, errors.New("第二行字幕最多 1600 字"))
		return
	}
	dlg := Dialogue{Names: map[string]string{}}
	var scriptLines []DialogueLine
	if dialogue {
		for _, id := range []string{"A", "B", "C", "D", "E"} {
			name := []rune(strings.TrimSpace(r.FormValue("speaker_name_" + id)))
			if len(name) > 12 {
				name = name[:12]
			}
			if len(name) > 0 {
				dlg.Names[id] = string(name)
			}
		}
		if text != "" {
			var e error
			if scriptLines, e = parseDialogueScript(text); e != nil && len(r.MultipartForm.File["audio"]) == 0 {
				fail(w, 400, e)
				return
			}
		}
	}
	if len(r.MultipartForm.File["audio"]) > 0 {
		ext := strings.ToLower(filepath.Ext(r.MultipartForm.File["audio"][0].Filename))
		switch ext {
		case ".wav", ".mp3", ".m4a", ".aac", ".ogg", ".flac", ".webm":
		default:
			fail(w, 400, errors.New("請匯入 WAV、MP3、M4A、AAC、OGG、FLAC 或 WebM 音檔。"))
			return
		}
		raw := filepath.Join(dir, "imported"+ext)
		if _, e := copyUpload(r, "audio", raw, 15<<20); e != nil {
			fail(w, 400, e)
			return
		}
		ctx, cancel := context.WithTimeout(r.Context(), 3*time.Minute)
		defer cancel()
		if e := audioWAV(ctx, raw, filepath.Join(dir, j.Audio), r.FormValue("effect")); e != nil {
			fail(w, 400, e)
			return
		}
	} else {
		if text == "" {
			fail(w, 400, errors.New("請輸入台詞，或匯入 WAV 語音。"))
			return
		}
		rate, _ := strconv.Atoi(r.FormValue("rate"))
		if rate < -5 || rate > 5 {
			rate = 0
		}
		ctx, cancel := context.WithTimeout(r.Context(), 6*time.Minute)
		defer cancel()
		if dialogue {
			voices, effects := map[string]string{}, map[string]string{}
			for _, id := range []string{"A", "B", "C", "D", "E"} {
				voices[id] = r.FormValue("voice_" + id)
				if voices[id] == "" {
					voices[id] = r.FormValue("voice")
				}
				effects[id] = r.FormValue("effect_" + id)
			}
			if e := a.dialogueSpeech(ctx, dir, scriptLines, voices, effects, rate, filepath.Join(dir, j.Audio)); e != nil {
				fail(w, 400, e)
				return
			}
			dlg.Lines = scriptLines
		} else if e := a.tts(ctx, text, r.FormValue("voice"), rate, filepath.Join(dir, j.Audio)); e != nil {
			fail(w, 400, e)
			return
		}
	}
	if !dialogue && len(r.MultipartForm.File["audio"]) == 0 && r.FormValue("effect") != "" && r.FormValue("effect") != "original" {
		ctx, cancel := context.WithTimeout(r.Context(), 3*time.Minute)
		defer cancel()
		altered := filepath.Join(dir, "altered.wav")
		if e := audioWAV(ctx, filepath.Join(dir, j.Audio), altered, r.FormValue("effect")); e != nil {
			fail(w, 400, e)
			return
		}
		if e := os.Rename(altered, filepath.Join(dir, j.Audio)); e != nil {
			fail(w, 500, e)
			return
		}
	}

	dur, e := wavDuration(filepath.Join(dir, j.Audio))
	if e != nil {
		fail(w, 400, e)
		return
	}
	limit := 60.
	if dialogue {
		limit = 120
	}
	if dur < 3 || dur > limit {
		fail(w, 400, fmt.Errorf("目前語音長 %.1f 秒；接受 3～%.0f 秒，請調整文字或 WAV。", dur, limit))
		return
	}
	j.Duration = dur
	if dialogue {
		if len(r.MultipartForm.File["audio"]) > 0 {
			timeline, e := parseDialogueTimeline(r.FormValue("dialogue_timeline"), dur)
			if e != nil {
				fail(w, 400, e)
				return
			}
			// Attach script texts in order when the counts match.
			if len(scriptLines) == len(timeline) {
				for i := range timeline {
					timeline[i].HanText = scriptLines[i].HanText
					timeline[i].Actions = scriptLines[i].Actions
				}
			}
			dlg.Lines = timeline
			j.CaptionTiming = "依您填寫的說話時間"
		} else {
			j.CaptionTiming = "依每位說話者的朗讀段落對齊"
		}
		seconds := nonEmptyLines(second)
		seen := map[string]bool{}
		for i := range dlg.Lines {
			if i < len(seconds) {
				dlg.Lines[i].TailoText = seconds[i]
			}
			if !seen[dlg.Lines[i].Speaker] {
				seen[dlg.Lines[i].Speaker] = true
				j.DialogueSpeakers = append(j.DialogueSpeakers, dlg.Lines[i].Speaker)
			}
		}
		if e = writeDialogueFile(dir, dlg); e != nil {
			fail(w, 500, e)
			return
		}
		j.Dialogue = true
		if j.HasCaptions, e = writeDialogueCaptions(dir, dlg, dur); e != nil {
			fail(w, 500, e)
			return
		}
	} else if text != "" {
		j.HasCaptions, j.CaptionTiming, e = writeCaptionsSecond(dir, text, second, dur)
		if e != nil {
			fail(w, 500, e)
			return
		}
	}
	if e := os.WriteFile(filepath.Join(dir, "script.txt"), []byte(text), 0600); e != nil {
		fail(w, 500, e)
		return
	}
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
func wavDuration(path string) (float64, error) {
	b, e := os.ReadFile(path)
	if e != nil {
		return 0, e
	}
	bad := errors.New("語音必須是有效的未壓縮 PCM WAV（可用 Windows 語音或錄音軟體匯出）。")
	if len(b) < 44 || string(b[:4]) != "RIFF" || string(b[8:12]) != "WAVE" {
		return 0, bad
	}
	var rate uint32
	var size uint64
	var pcm bool
	for i := 12; i+8 <= len(b); {
		n := int(binary.LittleEndian.Uint32(b[i+4 : i+8]))
		start := i + 8
		if n < 0 || n > len(b)-start {
			return 0, bad
		}
		switch string(b[i : i+4]) {
		case "fmt ":
			if n < 16 {
				return 0, bad
			}
			pcm = binary.LittleEndian.Uint16(b[start:start+2]) == 1
			rate = binary.LittleEndian.Uint32(b[start+8 : start+12])
		case "data":
			size += uint64(n)
		}
		i = start + n + n%2
	}
	if !pcm || rate == 0 || size == 0 {
		return 0, bad
	}
	return float64(size) / float64(rate), nil
}
func (a *App) generate(w http.ResponseWriter, r *http.Request, resume bool) {
	var q struct {
		ID          string `json:"id"`
		Key         string `json:"key"`
		Format      string `json:"format"`
		Aspect      string `json:"aspect"`
		Resolution  string `json:"resolution"`
		CaptionMode string `json:"caption_mode"`
	}
	if json.NewDecoder(io.LimitReader(r.Body, 8192)).Decode(&q) != nil || strings.TrimSpace(q.Key) == "" {
		fail(w, 400, errors.New("請輸入 Fal API Key。"))
		return
	}
	a.mu.Lock()
	j := a.jobs[q.ID]
	if j == nil {
		a.mu.Unlock()
		fail(w, 404, errors.New("找不到專案"))
		return
	}
	if j.busy {
		a.mu.Unlock()
		fail(w, 409, errors.New("這個任務正在處理，不必重複送出。"))
		return
	}
	if !resume && j.State != "ready" {
		a.mu.Unlock()
		fail(w, 409, errors.New("此專案已送過生成。請用「繼續查詢」取回原任務，避免重複計費。"))
		return
	}
	if resume && (j.Queue.RequestID == "" || (j.State == "done" || j.State == "voice_done")) {
		a.mu.Unlock()
		fail(w, 400, errors.New("此專案沒有可恢復的雲端任務。"))
		return
	}
	j.busy = true
	if !resume {
		j.Format, j.Aspect, j.Resolution = normalizeExport(q.Format, q.Aspect, q.Resolution)
		j.CaptionMode = captionMode(q.CaptionMode)
	}
	j.key = strings.TrimSpace(q.Key)
	j.State = "submitting"
	j.Message = "正在處理；請保持程式開啟。"
	e := a.persist(j)
	a.mu.Unlock()
	if e != nil {
		a.mu.Lock()
		j.busy = false
		j.key = ""
		j.State = "ready"
		a.mu.Unlock()
		fail(w, 500, e)
		return
	}
	go a.run(q.ID, resume)
	respond(w, map[string]bool{"ok": true})
}
func (a *App) jsonRequest(ctx context.Context, method, target, key string, input, output any) error {
	var reader io.Reader
	if input != nil {
		b, e := json.Marshal(input)
		if e != nil {
			return e
		}
		reader = bytes.NewReader(b)
	}
	req, e := http.NewRequestWithContext(ctx, method, target, reader)
	if e != nil {
		return e
	}
	if key != "" {
		req.Header.Set("Authorization", "Key "+key)
	}
	req.Header.Set("Content-Type", "application/json")
	req.Header.Set("X-Fal-No-Retry", "1")
	resp, e := a.client.Do(req)
	if e != nil {
		return e
	}
	defer resp.Body.Close()
	b, e := io.ReadAll(io.LimitReader(resp.Body, 2<<20))
	if e != nil {
		return e
	}
	if resp.StatusCode < 200 || resp.StatusCode >= 300 {
		msg := strings.ReplaceAll(string(b), key, "[隱藏金鑰]")
		if len(msg) > 1200 {
			msg = msg[:1200]
		}
		return fmt.Errorf("HTTP %d：%s", resp.StatusCode, msg)
	}
	if output != nil {
		return json.Unmarshal(b, output)
	}
	return nil
}
func dataURI(path string) (string, error) {
	b, e := os.ReadFile(path)
	if e != nil {
		return "", e
	}
	typ := mime.TypeByExtension(filepath.Ext(path))
	if typ == "" {
		typ = http.DetectContentType(b)
	}
	return "data:" + typ + ";base64," + base64.StdEncoding.EncodeToString(b), nil
}
func (a *App) validQueueURL(raw string) bool {
	u, e := url.Parse(raw)
	base, _ := url.Parse(a.queueBase)
	return e == nil && u.Scheme == base.Scheme && u.Host == base.Host && u.User == nil && strings.HasPrefix(u.Path, "/fal-ai/")
}
func (a *App) run(id string, resume bool) {
	a.mu.Lock()
	j := a.jobs[id]
	key := j.key
	dir := a.dir(id)
	q := j.Queue
	kind := j.Kind
	source := j.Source
	audio := j.Audio
	a.mu.Unlock()
	defer func() { a.mu.Lock(); j.busy = false; j.key = ""; a.mu.Unlock() }()
	ctx, cancel := context.WithTimeout(context.Background(), 90*time.Minute)
	defer cancel()
	if kind == "voice" {
		a.runVoice(ctx, id, key, resume)
		return
	}
	if !resume {
		src, e := dataURI(filepath.Join(dir, source))
		if e != nil {
			a.update(id, "ready", e.Error())
			return
		}
		aud, e := dataURI(filepath.Join(dir, audio))
		if e != nil {
			a.update(id, "ready", e.Error())
			return
		}
		input := map[string]any{"audio_url": aud}
		model := photoModel
		if kind == "video" {
			model = videoModel
			input["video_url"] = src
			input["sync_mode"] = "loop"
		} else {
			input["image_url"] = src
			input["prompt"] = "The character speaks naturally to camera, with gentle blinking, subtle head and body movements. Keep the same appearance, clothing, instrument and background. Keep the camera stable."
		}
		if e := a.jsonRequest(ctx, "POST", a.queueBase+"/"+model, key, input, &q); e != nil {
			a.update(id, "uncertain", "送出結果未確認，請先在 Fal 控制台檢查。程式不會自動重送，避免重複計費。詳細："+e.Error())
			return
		}
		if q.RequestID == "" || !a.validQueueURL(q.StatusURL) || !a.validQueueURL(q.ResponseURL) || !a.validQueueURL(q.CancelURL) {
			a.update(id, "uncertain", "雲端回傳格式異常。請到 Fal 控制台查詢，避免重新送出。任務 ID："+q.RequestID)
			return
		}
		a.mu.Lock()
		j.Queue = q
		e = a.persist(j)
		a.mu.Unlock()
		if e != nil {
			a.update(id, "waiting_resume", "雲端任務已送出但狀態儲存失敗。請記下任務 ID："+q.RequestID)
			return
		}
	}
	if !a.validQueueURL(q.StatusURL) || !a.validQueueURL(q.ResponseURL) {
		a.update(id, "error", "任務網址不符合服務來源，請到 Fal 控制台取回結果。")
		return
	}
	failures := 0
	for {
		var st struct {
			Status   string `json:"status"`
			Position int    `json:"queue_position"`
			Error    string `json:"error"`
		}
		e := a.jsonRequest(ctx, "GET", q.StatusURL, key, nil, &st)
		if e != nil {
			a.mu.Lock()
			cancelling := j.CancelRequested
			a.mu.Unlock()
			if cancelling && strings.Contains(e.Error(), "HTTP 404") {
				a.update(id, "cancelled", "服務回報任務不存在；取消或移除狀態請以 Fal 控制台為準。")
				return
			}
			failures++
			if failures >= 6 || ctx.Err() != nil {
				a.update(id, "waiting_resume", "查詢中斷；可按「繼續查詢」取回原任務，不必重新生成。"+e.Error())
				return
			}
			a.update(id, "retrying", "網路暫時不通，稍後重試查詢，不會重新送出生成。")
		} else {
			failures = 0
			switch st.Status {
			case "IN_QUEUE":
				a.update(id, "queued", fmt.Sprintf("雲端排隊中，前方約 %d 個任務。", st.Position))
			case "IN_PROGRESS":
				a.update(id, "generating", "AI 正在生成嘴型與畫面，剩餘時間由服務決定。")
			case "COMPLETED":
				if st.Error != "" {
					a.update(id, "error", "服務生成失敗："+st.Error)
					return
				}
				a.retrieve(ctx, id, key, q.ResponseURL)
				return
			default:
				a.update(id, "waiting_resume", "雲端回報未辨識狀態："+st.Status+"。請查看 Fal 控制台。")
				return
			}
		}
		select {
		case <-ctx.Done():
			a.update(id, "waiting_resume", "追蹤逾時；可按「繼續查詢」取回原任務。")
			return
		case <-time.After(a.pollDelay):
		}
	}
}
func (a *App) retrieve(ctx context.Context, id, key, responseURL string) {
	var result struct {
		Video struct {
			URL string `json:"url"`
		} `json:"video"`
		Error string `json:"error"`
	}
	if e := a.jsonRequest(ctx, "GET", responseURL, key, nil, &result); e != nil {
		a.update(id, "waiting_resume", "取回結果失敗，可繼續查詢原任務："+e.Error())
		return
	}
	u, e := url.Parse(result.Video.URL)
	if e != nil || u.User != nil || u.Scheme != "https" {
		if !(e == nil && strings.HasPrefix(a.queueBase, "http://127.0.0.1:") && u.Scheme == "http") {
			a.update(id, "error", "服務沒有回傳可下載的影片。"+result.Error)
			return
		}
	}
	a.update(id, "downloading", "生成完成，正在把 MP4 儲存到電腦。")
	req, e := http.NewRequestWithContext(ctx, "GET", result.Video.URL, nil)
	if e != nil {
		a.update(id, "waiting_resume", e.Error())
		return
	}
	resp, e := a.client.Do(req)
	if e != nil {
		a.update(id, "waiting_resume", "下載失敗，可繼續查詢取回。"+e.Error())
		return
	}
	defer resp.Body.Close()
	if resp.StatusCode != 200 {
		a.update(id, "waiting_resume", fmt.Sprintf("下載回應 HTTP %d，可繼續查詢重試。", resp.StatusCode))
		return
	}
	path := filepath.Join(a.dir(id), "video.mp4")
	f, e := os.Create(path + ".part")
	if e != nil {
		a.update(id, "waiting_resume", e.Error())
		return
	}
	n, e := io.Copy(f, io.LimitReader(resp.Body, (500<<20)+1))
	ce := f.Close()
	if e == nil {
		e = ce
	}
	if e == nil && (n < 16 || n > 500<<20) {
		e = errors.New("影片大小異常")
	}
	if e == nil {
		head := make([]byte, 12)
		rf, er := os.Open(path + ".part")
		if er == nil {
			_, er = io.ReadFull(rf, head)
			rf.Close()
		}
		if er != nil || string(head[4:8]) != "ftyp" {
			e = errors.New("回傳內容不是可辨識的 MP4，請至控制台確認。")
		}
	}
	if e != nil {
		os.Remove(path + ".part")
		a.update(id, "waiting_resume", e.Error())
		return
	}
	if e = os.Rename(path+".part", path); e != nil {
		a.update(id, "waiting_resume", e.Error())
		return
	}
	a.mu.Lock()
	a.jobs[id].Output = "video.mp4"
	a.mu.Unlock()
	a.update(id, "done", "完成！MP4 已自動儲存，可播放或另存影片。")
	a.mu.Lock()
	j := a.jobs[id]
	format, aspect, resolution := j.Format, j.Aspect, j.Resolution
	a.mu.Unlock()
	a.convertJob(id, "mp4", aspect, resolution)
	if format != "mp4" {
		a.convertJob(id, format, aspect, resolution)
	}
}
func (a *App) cancel(w http.ResponseWriter, r *http.Request) {
	var q struct {
		ID string `json:"id"`
	}
	if json.NewDecoder(io.LimitReader(r.Body, 4096)).Decode(&q) != nil {
		fail(w, 400, errors.New("無效資料"))
		return
	}
	a.mu.Lock()
	j := a.jobs[q.ID]
	if j == nil || !j.busy || j.Queue.CancelURL == "" || j.key == "" {
		a.mu.Unlock()
		fail(w, 400, errors.New("任務尚未可取消，或已結束。"))
		return
	}
	target, key := j.Queue.CancelURL, j.key
	a.mu.Unlock()
	if !a.validQueueURL(target) {
		fail(w, 400, errors.New("取消網址無效"))
		return
	}
	ctx, c := context.WithTimeout(r.Context(), 30*time.Second)
	defer c()
	if e := a.jsonRequest(ctx, "PUT", target, key, nil, nil); e != nil {
		fail(w, 400, e)
		return
	}
	a.mu.Lock()
	j.CancelRequested = true
	a.persist(j)
	a.mu.Unlock()
	respond(w, map[string]string{"message": "取消要求已送出。正在生成的任務可能仍會完成或計費；程式會繼續追蹤結果。"})
}
func main() {
	if runSelfTest(os.Args[1:]) {
		return
	}
	base, e := os.UserConfigDir()
	if e != nil {
		base = os.TempDir()
	}
	root := filepath.Join(base, "JunbaTalkVideo", "output")
	if e = os.MkdirAll(root, 0700); e != nil {
		return
	}
	lf, e := os.OpenFile(filepath.Join(filepath.Dir(root), "app.log"), os.O_CREATE|os.O_WRONLY|os.O_APPEND, 0600)
	if e == nil {
		defer lf.Close()
		log.SetOutput(lf)
	}
	a := newApp(root)
	a.load()
	listener, e := net.Listen("tcp4", "127.0.0.1:0")
	if e != nil {
		log.Print(e)
		return
	}
	address := "http://" + listener.Addr().String() + "/?session=" + a.token
	if os.Getenv("JUNBA_NO_BROWSER") != "1" {
		if e := openBrowser(address); e != nil {
			log.Print(e)
		}
	}
	fmt.Println(address)
	server := &http.Server{Handler: http.HandlerFunc(a.handler), ReadHeaderTimeout: 15 * time.Second, IdleTimeout: 120 * time.Second}
	if e = server.Serve(listener); e != nil {
		log.Print(e)
	}
}
