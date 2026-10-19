//go:build windows

package main

import (
	"bytes"
	"compress/gzip"
	"crypto/sha256"
	_ "embed"
	"errors"
	"fmt"
	"io"
	"os"
	"os/exec"
	"path/filepath"
	"sync"
)

//go:embed tools/ffmpeg.exe.gz
var packedFFmpeg []byte
var engineLock sync.Mutex

// externalFFmpeg is used when this build was produced without an embedded
// FFmpeg payload (offline build). It looks next to the EXE, then in tools/,
// then on PATH.
func externalFFmpeg() (string, error) {
	if exe, e := os.Executable(); e == nil {
		d := filepath.Dir(exe)
		for _, c := range []string{filepath.Join(d, "ffmpeg.exe"), filepath.Join(d, "tools", "ffmpeg.exe"), filepath.Join(d, "ffmpeg", "bin", "ffmpeg.exe")} {
			if st, e := os.Stat(c); e == nil && !st.IsDir() {
				return c, nil
			}
		}
	}
	if p, e := exec.LookPath("ffmpeg"); e == nil {
		return p, nil
	}
	return "", errors.New("找不到 ffmpeg.exe：請把 ffmpeg.exe 放在 JunbaTalkVideo.exe 同一資料夾（或 tools 子資料夾）")
}

func ffmpegPath() (string, error) {
	engineLock.Lock()
	defer engineLock.Unlock()
	if len(packedFFmpeg) < 1000000 {
		return externalFFmpeg()
	}
	base, e := os.UserCacheDir()
	if e != nil {
		return "", e
	}
	sum := sha256.Sum256(packedFFmpeg)
	dir := filepath.Join(base, "JunbaTalkVideo", "tools")
	if e = os.MkdirAll(dir, 0700); e != nil {
		return "", e
	}
	path := filepath.Join(dir, fmt.Sprintf("ffmpeg-%x.exe", sum[:8]))
	if st, e := os.Stat(path); e == nil && st.Size() > 1000000 {
		return path, nil
	}
	zr, e := gzip.NewReader(bytes.NewReader(packedFFmpeg))
	if e != nil {
		return "", e
	}
	defer zr.Close()
	tmp, e := os.CreateTemp(dir, "engine-*.tmp")
	if e != nil {
		return "", e
	}
	name := tmp.Name()
	defer os.Remove(name)
	_, e = io.Copy(tmp, zr)
	ce := tmp.Close()
	if e == nil {
		e = ce
	}
	if e != nil {
		return "", e
	}
	if e = os.Rename(name, path); e != nil {
		return "", e
	}
	return path, nil
}
