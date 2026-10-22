package main

import (
	"os"
	"path/filepath"
	"testing"
)

func TestLocalRuntimeReadyWithoutAIWeights(t *testing.T) {
	root := t.TempDir()
	app := &App{root: filepath.Join(root, "JunbaTalkVideo.exe")}
	dir := app.localDir()
	if err := os.MkdirAll(filepath.Join(dir, "python"), 0700); err != nil {
		t.Fatal(err)
	}
	if err := os.WriteFile(filepath.Join(dir, "python", "python.exe"), []byte("synthetic placeholder"), 0600); err != nil {
		t.Fatal(err)
	}
	if err := os.WriteFile(filepath.Join(dir, "runtime-installed.txt"), []byte(runtimeInstallMarker()), 0600); err != nil {
		t.Fatal(err)
	}
	if !app.localRuntimeReady() {
		t.Fatal("lightweight runtime should be ready without AI weights")
	}
	if app.localReady() {
		t.Fatal("full AI model readiness must remain false without model marker and weights")
	}
}
