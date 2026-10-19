package main

import (
	"archive/zip"
	"bytes"
	"encoding/json"
	"io"
	"net/http/httptest"
	"os"
	"path/filepath"
	"testing"
)

func TestWorkerBudgetPolicy(t *testing.T) {
	for _, tc := range []struct {
		available uint64
		want      int
	}{{0, 4096}, {6, 4096}, {10, 5120}, {12, 6144}, {32, 6144}} {
		if got := workerBudget(0, MemoryInfo{Available: tc.available << 30}); got != tc.want {
			t.Fatalf("available=%d got=%d want=%d", tc.available, got, tc.want)
		}
	}
	if workerBudget(3072, MemoryInfo{Available: 32 << 30}) != 3072 {
		t.Fatal("manual choice changed")
	}
	b, _ := json.Marshal(Job{})
	if !bytes.Contains(b, []byte(`"local_eta":0`)) {
		t.Fatalf("missing ETA default: %s", b)
	}
}

func TestErrorReportAllowlistAndBound(t *testing.T) {
	a := newApp(t.TempDir())
	id := "0123456789abcdef0123456789abcdef"
	os.MkdirAll(a.dir(id), 0700)
	a.jobs[id] = &Job{ID: id, State: "error", key: "SECRET_API_KEY", Text: "PRIVATE_SCRIPT"}
	os.WriteFile(filepath.Join(a.dir(id), "local.log"), bytes.Repeat([]byte("x"), 3<<20), 0600)
	os.WriteFile(filepath.Join(a.dir(id), "project.json"), []byte("SECRET_API_KEY"), 0600)
	os.WriteFile(filepath.Join(a.dir(id), "source.jpg"), []byte("PRIVATE_IMAGE"), 0600)
	w := httptest.NewRecorder()
	a.errorReport(w, httptest.NewRequest("GET", "/api/local/error-report?id="+id, nil))
	if w.Code != 200 {
		t.Fatal(w.Code)
	}
	z, e := zip.NewReader(bytes.NewReader(w.Body.Bytes()), int64(w.Body.Len()))
	if e != nil {
		t.Fatal(e)
	}
	if len(z.File) != 2 {
		t.Fatal("unexpected report entries", len(z.File))
	}
	for _, f := range z.File {
		if f.Name != "report.json" && f.Name != "local.log" {
			t.Fatal("unexpected file", f.Name)
		}
		if f.UncompressedSize64 > 2<<20 {
			t.Fatal("unbounded report")
		}
		r, _ := f.Open()
		b, _ := io.ReadAll(r)
		r.Close()
		if bytes.Contains(b, []byte("SECRET_API_KEY")) || bytes.Contains(b, []byte("PRIVATE_SCRIPT")) {
			t.Fatal("private data leaked")
		}
	}
	w = httptest.NewRecorder()
	a.errorReport(w, httptest.NewRequest("GET", "/api/local/error-report?id=../other", nil))
	if w.Code != 400 {
		t.Fatal("invalid ID accepted")
	}
}
