//go:build !windows

package main

import "os/exec"

type MemoryInfo struct {
	Total     uint64 `json:"total"`
	Available uint64 `json:"available"`
}

func memoryInfo() MemoryInfo { return MemoryInfo{} }
func killLocalProcess(c *exec.Cmd) error {
	if c.Process != nil {
		return c.Process.Kill()
	}
	return nil
}
