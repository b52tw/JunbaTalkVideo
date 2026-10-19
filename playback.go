package main

import (
	"archive/zip"
	"errors"
	"io"
	"net/http"
	"os"
	"path/filepath"
	"strings"
)

const playerHTML = `<!doctype html><html lang="zh-Hant"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1,viewport-fit=cover"><title>峻爸的短影音</title><style>*{box-sizing:border-box}body{margin:0;background:#141d19;color:#f4f1e6;font:15px/1.7 system-ui,'Microsoft JhengHei',sans-serif;min-height:100vh;display:grid;place-items:center;padding:20px}.wrap{width:min(100%,1100px)}video{display:block;width:100%;height:auto;max-height:82vh;max-height:82svh;object-fit:contain;background:#000;border-radius:14px}p{font-size:12px;opacity:.75}a{color:#ded4ac}header{font-size:14px;margin-bottom:13px;display:flex;justify-content:space-between;gap:15px}small{font-size:10px;opacity:.6}@media(max-width:600px){body{padding:12px}video{border-radius:9px}header{font-size:12px}}</style></head><body><div class="wrap"><header><span>峻爸的短影音</span><small>作者：峻爸</small></header><video controls playsinline preload="metadata" aria-label="短影音播放"><source src="video.mp4" type="video/mp4"></video><p>畫面會依螢幕寬度調整，可用電腦或手機觀看。<a href="video.mp4" download>儲存 MP4</a>　若檔案管理器無法開啟此頁，請直接用播放器開啟同資料夾的 video.mp4。</p></div></body></html>`

func (a *App) playbackPackage(w http.ResponseWriter, r *http.Request) {
	id := strings.TrimPrefix(r.URL.Path, "/playback/")
	if !validID(id) {
		http.NotFound(w, r)
		return
	}
	a.mu.Lock()
	j := a.jobs[id]
	ok := j != nil && j.State == "done"
	captioned := ok && j.HasCaptions
	export := ""
	if ok {
		export = j.Export
	}
	a.mu.Unlock()
	if !ok {
		fail(w, 400, errors.New("影片尚未完成"))
		return
	}
	dir := a.dir(id)
	source := filepath.Join(dir, "export.mp4")
	if _, e := os.Stat(source); e != nil {
		fail(w, 400, errors.New("尚未建立相容 MP4。請先選 MP4 並按本機轉檔。"))
		return
	}
	w.Header().Set("Content-Type", "application/zip")
	w.Header().Set("Content-Disposition", `attachment; filename="Junba_Playback_Package.zip"`)
	z := zip.NewWriter(w)
	defer z.Close()
	part, e := z.Create("player.html")
	if e != nil {
		return
	}
	io.WriteString(part, playerHTML)
	names := map[string]string{"video.mp4": "export.mp4"}
	if captioned {
		names["subtitles.srt"] = "subtitles.srt"
	}
	if export != "" && export != "export.mp4" {
		names[export] = export
	}
	for name, file := range names {
		f, e := os.Open(filepath.Join(dir, file))
		if e != nil {
			continue
		}
		part, e = z.CreateHeader(&zip.FileHeader{Name: name, Method: zip.Store})
		if e == nil {
			_, e = io.Copy(part, f)
		}
		f.Close()
		if e != nil {
			return
		}
	}
}
