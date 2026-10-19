//go:build windows

package main

import (
	"os/exec"
	"strconv"
	"syscall"
	"unsafe"
)

type MemoryInfo struct {
	Total     uint64 `json:"total"`
	Available uint64 `json:"available"`
}

func memoryInfo() MemoryInfo {
	var m struct {
		Length, Load                                                                          uint32
		TotalPhys, AvailPhys, TotalPage, AvailPage, TotalVirtual, AvailVirtual, AvailExtended uint64
	}
	m.Length = uint32(unsafe.Sizeof(m))
	p := syscall.NewLazyDLL("kernel32.dll").NewProc("GlobalMemoryStatusEx")
	ok, _, _ := p.Call(uintptr(unsafe.Pointer(&m)))
	if ok == 0 {
		return MemoryInfo{}
	}
	return MemoryInfo{m.TotalPhys, m.AvailPhys}
}
func killLocalProcess(c *exec.Cmd) error {
	if c.Process == nil {
		return nil
	}
	k := exec.Command("taskkill.exe", "/PID", strconv.Itoa(c.Process.Pid), "/T", "/F")
	hideCommand(k)
	if e := k.Run(); e != nil {
		return c.Process.Kill()
	}
	return nil
}
