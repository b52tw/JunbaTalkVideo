//go:build !windows

package main

import "os/exec"

func limitWorkerMemory(c *exec.Cmd, mb int) (func(), error) { return func() {}, nil }
