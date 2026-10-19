//go:build windows

package main

import (
	"fmt"
	"os"
	"syscall"
	"unsafe"
)

func lockInstallFile(path string) (func(), error) {
	f, e := os.OpenFile(path, os.O_CREATE|os.O_RDWR, 0600)
	if e != nil {
		return nil, e
	}
	dll := syscall.NewLazyDLL("kernel32.dll")
	lock := dll.NewProc("LockFileEx")
	unlock := dll.NewProc("UnlockFileEx")
	over := &syscall.Overlapped{}
	ok, _, err := lock.Call(f.Fd(), 3, 0, 1, 0, uintptr(unsafe.Pointer(over)))
	if ok == 0 {
		f.Close()
		return nil, fmt.Errorf("另一個程式正在安裝，請等候或關閉重複視窗：%w", err)
	}
	return func() { unlock.Call(f.Fd(), 0, 1, 0, uintptr(unsafe.Pointer(over))); f.Close() }, nil
}
