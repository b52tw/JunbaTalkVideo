//go:build !windows

package main

import (
	"context"
	"errors"
	"os/exec"
)

func installedVoices(context.Context) ([]Voice, error) { return []Voice{}, nil }
func synthesize(context.Context, string, string, int, string) error {
	return errors.New("文字轉語音只支援 Windows；此平台可匯入 WAV 錄音測試。")
}
func openBrowser(url string) error { return exec.Command("xdg-open", url).Start() }
func openFolder(path string) error { return exec.Command("xdg-open", path).Start() }
func hideCommand(cmd *exec.Cmd)    {}
