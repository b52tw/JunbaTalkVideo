package main

import (
	"archive/zip"
	"encoding/json"
	"os"
	"path/filepath"
	"strings"
	"testing"
)

func TestManagedSourceExtraction(t *testing.T) {
	for _, name := range []string{"upstream/src/config/model.yaml", "upstream/src/../../escape"} {
		dir := t.TempDir()
		archive := filepath.Join(dir, "source.zip")
		f, _ := os.Create(archive)
		w := zip.NewWriter(f)
		entry, _ := w.Create(name)
		entry.Write([]byte("pinned config"))
		w.Close()
		f.Close()
		e := extractSadSource(archive, filepath.Join(dir, "root"))
		if strings.Contains(name, "..") {
			if e == nil {
				t.Fatal("accepted traversal")
			}
		} else {
			if e != nil {
				t.Fatal(e)
			}
			b, e := os.ReadFile(filepath.Join(dir, "root", "src", "config", "model.yaml"))
			if e != nil || string(b) != "pinned config" {
				t.Fatal(e, string(b))
			}
		}
	}
}

func TestManagedWheelsArePinnedWindowsAndBundledSources(t *testing.T) {
	b, e := localFiles.ReadFile("local/sadtalker_wheels.json")
	if e != nil {
		t.Fatal(e)
	}
	var rows []LocalModel
	if e = json.Unmarshal(b, &rows); e != nil {
		t.Fatal(e)
	}
	if len(rows) < 60 {
		t.Fatal("incomplete dependency lock")
	}
	found := map[string]bool{}
	for _, m := range rows {
		if filepath.Base(m.Name) != m.Name || !strings.HasSuffix(m.Name, ".whl") || len(m.SHA) != 64 || m.Size <= 0 {
			t.Fatal(m)
		}
		if !strings.Contains(m.Name, "win_amd64") && !strings.Contains(m.Name, "none-any") {
			t.Fatal("non-Windows dependency", m.Name)
		}
		if strings.HasPrefix(m.URL, "bundled:") {
			b, e := localFiles.ReadFile("local/" + m.Name)
			if e != nil || int64(len(b)) != m.Size {
				t.Fatal("missing bundled pure Python wheel", m.Name, e)
			}
		} else if !strings.HasPrefix(m.URL, "https://files.pythonhosted.org/") && !strings.HasPrefix(m.URL, "https://download.pytorch.org/whl/cpu/") {
			t.Fatal("unexpected wheel origin", m.URL)
		}
		found[strings.Split(m.Name, "-")[0]] = true
	}
	for _, name := range []string{"torch", "torchvision", "numpy", "facexlib", "gfpgan", "basicsr", "filterpy", "librosa"} {
		if !found[name] {
			t.Fatal("dependency missing", name)
		}
	}
}

func TestInstallerLockSurvivesStaleFile(t *testing.T) {
	path := filepath.Join(t.TempDir(), "install.lock")
	os.WriteFile(path, []byte("previous interrupted install"), 0600)
	release, e := lockInstallFile(path)
	if e != nil {
		t.Fatal(e)
	}
	if r, e := lockInstallFile(path); e == nil {
		r()
		release()
		t.Fatal("allowed concurrent installers")
	}
	release()
	r, e := lockInstallFile(path)
	if e != nil {
		t.Fatal(e)
	}
	r()
}
