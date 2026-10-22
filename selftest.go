package main

import (
	"context"
	"encoding/json"
	"fmt"
	"os"
	"os/exec"
	"path/filepath"
	"runtime"
	"strings"
	"time"
)

func runSelfTest(args []string) bool {
	if len(args) == 0 || args[0] != "--self-test-report" {
		return false
	}
	if len(args) != 2 {
		os.Exit(2)
	}
	report := map[string]any{"version": appVersion() + "-engineering", "os": runtime.GOOS, "arch": runtime.GOARCH, "ok": false}
	err := func() error {
		dir, e := os.MkdirTemp("", "Junba self test ")
		if e != nil {
			return e
		}
		defer os.RemoveAll(dir)
		ff, e := ffmpegPath()
		if e != nil {
			return e
		}
		ctx, cancel := context.WithTimeout(context.Background(), time.Minute)
		defer cancel()
		c := exec.CommandContext(ctx, ff, "-version")
		hideCommand(c)
		out, e := c.CombinedOutput()
		if e != nil {
			return fmt.Errorf("FFmpeg: %w", e)
		}
		report["ffmpeg"] = strings.SplitN(string(out), "\n", 2)[0]
		if e = prepareFont(dir); e != nil {
			return e
		}
		st, e := os.Stat(filepath.Join(dir, "fonts", "NotoSansTC.ttf"))
		if e != nil {
			return e
		}
		report["font_bytes"] = st.Size()
		report["ok"] = true
		return nil
	}()
	if err != nil {
		report["error"] = err.Error()
	}
	b, _ := json.MarshalIndent(report, "", "  ")
	if e := os.WriteFile(args[1], b, 0600); e != nil {
		os.Exit(1)
	}
	if err != nil {
		os.Exit(1)
	}
	return true
}
