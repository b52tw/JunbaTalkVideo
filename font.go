package main

import (
	"bytes"
	"compress/gzip"
	"crypto/sha256"
	_ "embed"
	"fmt"
	"io"
	"os"
	"path/filepath"
	"sync"
)

//go:embed fonts/NotoSansTC.ttf.gz
var packedFont []byte
var fontLock sync.Mutex

// externalFontSource is used when this build has no embedded font payload
// (offline build): NotoSansTC.ttf next to the EXE, else a Windows CJK font.
func externalFontSource() (string, error) {
	var c []string
	if exe, e := os.Executable(); e == nil {
		d := filepath.Dir(exe)
		c = append(c, filepath.Join(d, "NotoSansTC.ttf"), filepath.Join(d, "fonts", "NotoSansTC.ttf"))
	}
	win := os.Getenv("WINDIR")
	if win == "" {
		win = `C:\Windows`
	}
	for _, n := range []string{"NotoSansTC-VF.ttf", "NotoSansTC-Regular.ttf", "msjh.ttc", "mingliu.ttc", "msyh.ttc"} {
		c = append(c, filepath.Join(win, "Fonts", n))
	}
	for _, p := range c {
		if st, e := os.Stat(p); e == nil && !st.IsDir() && st.Size() > 10000 {
			return p, nil
		}
	}
	return "", fmt.Errorf("找不到中文字型：請把 NotoSansTC.ttf 放在 JunbaTalkVideo.exe 同一資料夾")
}

func prepareFont(dir string) error {
	fontLock.Lock()
	defer fontLock.Unlock()
	if len(packedFont) < 10000 {
		src, e := externalFontSource()
		if e != nil {
			return e
		}
		targetDir := filepath.Join(dir, "fonts")
		if e = os.MkdirAll(targetDir, 0700); e != nil {
			return e
		}
		target := filepath.Join(targetDir, "NotoSansTC.ttf")
		if _, e = os.Stat(target); e == nil {
			return nil
		}
		in, e := os.Open(src)
		if e != nil {
			return e
		}
		defer in.Close()
		out, e := os.Create(target)
		if e != nil {
			return e
		}
		_, e = io.Copy(out, in)
		if ce := out.Close(); e == nil {
			e = ce
		}
		return e
	}
	cache, e := os.UserCacheDir()
	if e != nil {
		return e
	}
	hash := sha256.Sum256(packedFont)
	base := filepath.Join(cache, "JunbaTalkVideo", "fonts")
	if e = os.MkdirAll(base, 0700); e != nil {
		return e
	}
	source := filepath.Join(base, fmt.Sprintf("NotoSansTC-%x.ttf", hash[:6]))
	if _, e = os.Stat(source); e != nil {
		r, e := gzip.NewReader(bytes.NewReader(packedFont))
		if e != nil {
			return e
		}
		defer r.Close()
		tmp, e := os.CreateTemp(base, "font-*.tmp")
		if e != nil {
			return e
		}
		name := tmp.Name()
		defer os.Remove(name)
		_, e = io.Copy(tmp, r)
		ce := tmp.Close()
		if e == nil {
			e = ce
		}
		if e != nil {
			return e
		}
		if e = os.Rename(name, source); e != nil {
			return e
		}
	}
	targetDir := filepath.Join(dir, "fonts")
	if e = os.MkdirAll(targetDir, 0700); e != nil {
		return e
	}
	target := filepath.Join(targetDir, "NotoSansTC.ttf")
	if _, e = os.Stat(target); e == nil {
		return nil
	}
	if e = os.Link(source, target); e == nil {
		return nil
	}
	in, e := os.Open(source)
	if e != nil {
		return e
	}
	defer in.Close()
	out, e := os.Create(target)
	if e != nil {
		return e
	}
	_, e = io.Copy(out, in)
	ce := out.Close()
	if e != nil {
		return e
	}
	return ce
}
