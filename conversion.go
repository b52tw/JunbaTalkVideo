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
	"sync"
	"time"
)

var qsvOnce sync.Once
var qsvAvailable bool

func hasQSV(engine string) bool {
	if runtime.GOOS != "windows" {
		return false
	}
	qsvOnce.Do(func() {
		ctx, cancel := context.WithTimeout(context.Background(), 12*time.Second)
		defer cancel()
		c := exec.CommandContext(ctx, engine, "-hide_banner", "-nostdin", "-f", "lavfi", "-i", "color=size=128x128:rate=1", "-frames:v", "2", "-vf", "format=nv12", "-c:v", "h264_qsv", "-f", "null", "-")
		hideCommand(c)
		qsvAvailable = c.Run() == nil
	})
	return qsvAvailable
}
func qsvArgs(args []string) []string {
	out := []string{}
	for i := 0; i < len(args); i++ {
		v := args[i]
		if v == "-c:v" && i+1 < len(args) && args[i+1] == "libx264" {
			out = append(out, v, "h264_qsv")
			i++
			continue
		}
		if v == "-crf" {
			out = append(out, "-global_quality", "18")
			i++
			continue
		}
		if v == "-preset" {
			out = append(out, v, "medium")
			i++
			continue
		}
		if v == "-pix_fmt" {
			out = append(out, v, "nv12")
			i++
			continue
		}
		out = append(out, v)
	}
	return out
}

func normalizeExport(format, aspect, resolution string) (string, string, string) {
	switch format {
	case "mp4", "mov", "mkv", "webm", "gif", "wmv", "avi", "m4v":
	default:
		format = "mp4"
	}
	switch aspect {
	case "original", "9:16", "16:9", "1:1":
	default:
		aspect = "original"
	}
	if resolution != "1080" {
		resolution = "720"
	}
	return format, aspect, resolution
}
func convertArgs(input, output, format, aspect, resolution string) []string {
	format, aspect, resolution = normalizeExport(format, aspect, resolution)
	filter := "scale=trunc(iw/2)*2:trunc(ih/2)*2,setsar=1"
	if aspect != "original" {
		w, h := 720, 1280
		if resolution == "1080" {
			w, h = 1080, 1920
		}
		if aspect == "16:9" {
			w, h = h, w
		}
		if aspect == "1:1" {
			h = w
		}
		filter = fmt.Sprintf("scale=%d:%d:force_original_aspect_ratio=decrease:force_divisible_by=2,pad=%d:%d:(ow-iw)/2:(oh-ih)/2:color=black,setsar=1", w, h, w, h)
	}
	args := []string{"-hide_banner", "-nostdin", "-y", "-i", input}
	switch format {
	case "wmv":
		args = append(args, "-vf", filter, "-c:v", "wmv2", "-q:v", "2", "-pix_fmt", "yuv420p", "-c:a", "wmav2", "-b:a", "128k", "-ar", "44100", "-ac", "2")
	case "avi":
		args = append(args, "-vf", filter, "-c:v", "mpeg4", "-q:v", "2", "-vtag", "XVID", "-pix_fmt", "yuv420p", "-c:a", "libmp3lame", "-b:a", "192k")
	case "webm":
		args = append(args, "-vf", filter, "-c:v", "libvpx-vp9", "-crf", "20", "-b:v", "0", "-deadline", "good", "-cpu-used", "4", "-c:a", "libopus", "-b:a", "128k")
	case "gif":
		graph := filter + ",fps=12,scale='min(720,iw)':-1:flags=lanczos,split[a][b];[a]palettegen=stats_mode=diff[p];[b][p]paletteuse=dither=bayer:bayer_scale=3"
		args = append(args, "-filter_complex", graph, "-an", "-loop", "0")
	default:
		args = append(args, "-vf", filter, "-c:v", "libx264", "-preset", "medium", "-crf", "16", "-pix_fmt", "yuv420p", "-c:a", "aac", "-b:a", "160k")
		if format == "mp4" || format == "mov" || format == "m4v" {
			args = append(args, "-movflags", "+faststart")
		}
	}
	if format == "m4v" {
		args = append(args, "-f", "mp4")
	}
	return append(args, "-threads", "2", output)
}
func (a *App) convertJob(id, format, aspect, resolution string) {
	a.conversion.Lock()
	defer a.conversion.Unlock()
	a.mu.Lock()
	j := a.jobs[id]
	source := j.Output
	if source == "" && j.Kind == "convert" {
		source = j.Source
	}
	a.mu.Unlock()
	if source == "" {
		a.update(id, "error", "找不到可轉檔的影片。")
		return
	}
	a.update(id, "converting", "正在本機轉檔為 "+strings.ToUpper(format)+"，不會呼叫雲端或產生生成費。首次使用會準備轉檔引擎。")
	engine, e := ffmpegPath()
	if e != nil {
		a.update(id, "done", "轉檔引擎無法啟動；原始影片已保留。"+e.Error())
		return
	}
	dir := a.dir(id)
	final := filepath.Join(dir, "export."+format)
	temp := filepath.Join(dir, "converting."+format)
	ctx, cancel := context.WithTimeout(context.Background(), 30*time.Minute)
	defer cancel()
	args := convertArgs(filepath.Join(dir, source), temp, format, aspect, resolution)
	if j.Kind != "convert" && j.Audio != "" && !(j.Engine == "local" && j.LeadSeconds > 0) {
		args = append(append(append([]string{}, args[:5]...), "-i", filepath.Join(dir, j.Audio), "-map", "0:v:0", "-map", "1:a:0", "-shortest"), args[5:]...)
	}

	if j.HasCaptions && j.CaptionMode == "burn" {
		if e := prepareFont(dir); e != nil {
			a.update(id, "done", "字幕字型準備失敗，原始影片已保留。"+e.Error())
			return
		}
		args = addSubtitleFilter(args)
	}
	logfile, e := os.Create(filepath.Join(dir, "conversion.log"))
	if e != nil {
		a.update(id, "done", "無法建立轉檔記錄，原始影片已保留。"+e.Error())
		return
	}
	run := func(argv []string) error {
		cmd := exec.CommandContext(ctx, engine, argv...)
		cmd.Dir = dir
		hideCommand(cmd)
		cmd.Stderr = logfile
		cmd.Stdout = logfile
		return cmd.Run()
	}
	encoder := "CPU 軟體編碼"
	useQSV := (format == "mp4" || format == "mov" || format == "m4v" || format == "mkv") && hasQSV(engine)
	if useQSV {
		e = run(qsvArgs(args))
		encoder = "Intel Quick Sync 硬體編碼"
		if e != nil {
			logfile.WriteString("\nIntel QSV failed; retrying with CPU encoder.\n")
			os.Remove(temp)
			e = run(args)
			encoder = "CPU 軟體編碼（Intel QSV 失敗後退回）"
		}
	} else {
		e = run(args)
	}
	logfile.Close()
	if e != nil {
		os.Remove(temp)
		a.update(id, "done", "轉檔未完成，原始影片已保留。請查看資料夾中的 conversion.log。"+e.Error())
		return
	}
	if e = os.Rename(temp, final); e != nil {
		a.update(id, "done", "轉檔儲存失敗，原始影片已保留。"+e.Error())
		return
	}
	a.mu.Lock()
	j.Export = filepath.Base(final)
	j.Encoder = encoder
	if format == "mp4" {
		j.Compatible = true
	}
	j.Format = format
	j.Aspect = aspect
	j.Resolution = resolution
	a.mu.Unlock()
	msg := "完成！已另存 " + strings.ToUpper(format) + "；" + encoder + "。原始影片也已保留。"
	if format == "gif" {
		msg += "GIF 為無聲動圖。"
	}
	if j.HasCaptions && j.CaptionMode == "burn" {
		msg += "字幕已直接嵌入畫面。"
	}
	a.update(id, "done", msg)
}
func (a *App) convertHandler(w http.ResponseWriter, r *http.Request) {
	var q struct {
		ID          string `json:"id"`
		Format      string `json:"format"`
		Aspect      string `json:"aspect"`
		Resolution  string `json:"resolution"`
		CaptionMode string `json:"caption_mode"`
	}
	if json.NewDecoder(io.LimitReader(r.Body, 4096)).Decode(&q) != nil {
		fail(w, 400, errors.New("無效資料"))
		return
	}
	f, ar, res := normalizeExport(q.Format, q.Aspect, q.Resolution)
	a.mu.Lock()
	j := a.jobs[q.ID]
	if j == nil || j.busy || j.State != "done" {
		a.mu.Unlock()
		fail(w, 409, errors.New("請等目前任務完成，再轉出其他格式。"))
		return
	}
	j.busy = true
	j.CaptionMode = captionMode(q.CaptionMode)
	j.State = "converting"
	a.persist(j)
	a.mu.Unlock()
	go func() { defer func() { a.mu.Lock(); j.busy = false; a.mu.Unlock() }(); a.convertJob(q.ID, f, ar, res) }()
	respond(w, map[string]bool{"ok": true})
}
func (a *App) convertFileHandler(w http.ResponseWriter, r *http.Request) {
	r.Body = http.MaxBytesReader(w, r.Body, 505<<20)
	if e := r.ParseMultipartForm(2 << 20); e != nil {
		fail(w, 400, errors.New("請選擇 500 MB 以內的影片。"))
		return
	}
	defer r.MultipartForm.RemoveAll()
	src := r.MultipartForm.File["video"]
	if len(src) != 1 {
		fail(w, 400, errors.New("請選擇要轉檔的影片。"))
		return
	}
	ext := strings.ToLower(filepath.Ext(src[0].Filename))
	switch ext {
	case ".mp4", ".mov", ".mkv", ".webm", ".avi", ".m4v", ".wmv":
	default:
		fail(w, 400, errors.New("請選擇 MP4、WMV、MOV、MKV、WebM、AVI 或 M4V。"))
		return
	}
	id := randomID()
	if e := os.MkdirAll(a.dir(id), 0700); e != nil {
		fail(w, 500, e)
		return
	}
	source := "source" + ext
	if _, e := copyUpload(r, "video", filepath.Join(a.dir(id), source), 500<<20); e != nil {
		os.RemoveAll(a.dir(id))
		fail(w, 400, e)
		return
	}
	f, ar, res := normalizeExport(r.FormValue("format"), r.FormValue("aspect"), r.FormValue("resolution"))
	j := &Job{ID: id, Created: time.Now().Format(time.RFC3339), Kind: "convert", Source: source, State: "converting", Message: "正在本機轉檔，不使用付費 AI。", Format: f, Aspect: ar, Resolution: res, busy: true}
	if e := a.persist(j); e != nil {
		fail(w, 500, e)
		return
	}
	a.mu.Lock()
	a.jobs[id] = j
	snapshot := *j
	a.mu.Unlock()
	go func() { defer func() { a.mu.Lock(); j.busy = false; a.mu.Unlock() }(); a.convertJob(id, f, ar, res) }()
	respond(w, snapshot)
}
