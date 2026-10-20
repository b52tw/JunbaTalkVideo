package main

import (
	"archive/zip"
	"encoding/json"
	"errors"
	"io"
	"net/http"
	"os"
	"path/filepath"
	"runtime"
)

// Bounded, explicit log allowlist. Never include source media, speech text,
// API keys, the whole project file, or unrelated jobs.
func (a *App) errorReport(w http.ResponseWriter, r *http.Request) {
	id := r.URL.Query().Get("id")
	if !validID(id) {
		fail(w, 400, errors.New("無效專案 ID"))
		return
	}
	a.mu.Lock()
	j := a.jobs[id]
	if j == nil {
		a.mu.Unlock()
		fail(w, 404, errors.New("找不到專案"))
		return
	}
	if j.busy {
		a.mu.Unlock()
		fail(w, 409, errors.New("請等生成或停止完成後下載報告"))
		return
	}
	info := map[string]any{"version": "0.8.7", "os": runtime.GOOS, "arch": runtime.GOARCH,
		"id": j.ID, "state": j.State, "message": j.Message, "engine": j.Engine,
		"settings": j.LocalSettings, "actual_devices": j.ActualDevices, "memory_peaks": j.MemoryPeaks,
		"effective_budget_mb": j.EffectiveBudgetMB, "memory": memoryInfo()}
	b, _ := json.MarshalIndent(info, "", "  ")
	a.mu.Unlock()
	w.Header().Set("Content-Type", "application/zip")
	w.Header().Set("Content-Disposition", "attachment; filename=Junba_Error_"+id[:8]+".zip")
	z := zip.NewWriter(w)
	defer z.Close()
	f, e := z.Create("report.json")
	if e != nil {
		return
	}
	f.Write(b)
	for _, name := range []string{"local.log", "backend-faults.jsonl", "process-exits.jsonl", "stage-source-decode.log", "compose.log"} {
		p := filepath.Join(a.dir(id), name)
		st, err := os.Lstat(p)
		if err != nil || !st.Mode().IsRegular() {
			continue
		}
		src, err := os.Open(p)
		if err != nil {
			continue
		}
		// Tail at most 2 MiB per file; a long log must not consume application RAM.
		if st.Size() > 2<<20 {
			src.Seek(st.Size()-(2<<20), io.SeekStart)
		}
		dst, err := z.Create(name)
		if err == nil {
			io.Copy(dst, io.LimitReader(src, 2<<20))
		}
		src.Close()
	}
}
