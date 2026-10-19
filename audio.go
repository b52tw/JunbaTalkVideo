package main

import (
	"context"
	"fmt"
	"math"
	"os"
	"os/exec"
	"path/filepath"
)

func audioWAV(ctx context.Context, input, output, effect string) error {
	engine, e := ffmpegPath()
	if e != nil {
		return e
	}
	semi := 0.
	switch effect {
	case "cartoon":
		semi = 4
	case "bright":
		semi = 7
	case "deep":
		semi = -4
	}
	filter := "aresample=24000"
	if semi != 0 {
		factor := math.Pow(2, semi/12)
		filter = fmt.Sprintf("aresample=24000,asetrate=%.6f,aresample=24000,atempo=%.8f", 24000*factor, 1/factor)
	}
	cmd := exec.CommandContext(ctx, engine, "-hide_banner", "-nostdin", "-y", "-i", input, "-vn", "-af", filter, "-c:a", "pcm_s16le", "-ar", "24000", "-ac", "1", output)
	hideCommand(cmd)
	if b, e := cmd.CombinedOutput(); e != nil {
		os.WriteFile(filepath.Join(filepath.Dir(output), "audio-conversion.log"), b, 0600)
		return fmt.Errorf("音檔無法讀取或轉換，請確認檔案可正常播放。詳細記錄：audio-conversion.log")
	}
	return nil
}
