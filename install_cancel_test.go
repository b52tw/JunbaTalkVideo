package main

import (
	"context"
	"net/http"
	"net/http/httptest"
	"os"
	"path/filepath"
	"strings"
	"testing"
	"time"
)

func TestDownloadStallAbortsAndKeepsPart(t *testing.T) {
	old := downloadStallLimit
	downloadStallLimit = 300 * time.Millisecond
	defer func() { downloadStallLimit = old }()
	release := make(chan struct{})
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.Header().Set("Content-Length", "1000")
		w.WriteHeader(200)
		w.Write(make([]byte, 100))
		w.(http.Flusher).Flush()
		<-release // then hang: no more data
	}))
	defer srv.Close()
	defer close(release)
	a := newApp(t.TempDir())
	dir := t.TempDir()
	start := time.Now()
	e := a.downloadArtifact(context.Background(), LocalModel{Name: "m.bin", URL: srv.URL, Size: 1000, SHA: "x"}, dir)
	if e == nil || !strings.Contains(e.Error(), "沒有資料") {
		t.Fatal("stall not reported", e)
	}
	if time.Since(start) > 5*time.Second {
		t.Fatal("stall detection too slow")
	}
	if s, err := os.Stat(filepath.Join(dir, "m.bin.part")); err != nil || s.Size() != 100 {
		t.Fatal("partial download not kept for resume", err)
	}
}

func TestInstallCancelEndpoint(t *testing.T) {
	a := newApp(t.TempDir())
	if w := request(t, a, "POST", "/api/local/install-cancel", nil, "", true); w.Code != 409 {
		t.Fatal("cancel without install should be 409", w.Code)
	}
	ctx, done := a.installContext(time.Hour)
	defer done()
	if w := request(t, a, "POST", "/api/local/install-cancel", nil, "", true); w.Code != 200 {
		t.Fatal(w.Code, w.Body.String())
	}
	select {
	case <-ctx.Done():
	case <-time.After(time.Second):
		t.Fatal("install context not cancelled")
	}
}
