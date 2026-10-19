//go:build !windows

package main

import "os/exec"

func ffmpegPath() (string, error) { return exec.LookPath("ffmpeg") }
