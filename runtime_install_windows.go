//go:build windows

package main

import (
	"context"
	"fmt"
	"syscall"
	"unsafe"
)

// Used only when the managed Python reports missing C++ DLLs. Windows shows
// its own Microsoft installer/elevation UI; no system settings are bypassed.
func installVCRuntime(ctx context.Context, path string) error {
	type shellInfo struct {
		Size, Mask                        uint32
		Window                            uintptr
		Verb, File, Parameters, Directory *uint16
		Show                              int32
		Instance, IDList                  uintptr
		Class                             *uint16
		ClassKey                          uintptr
		HotKey                            uint32
		Icon                              uintptr
		Process                           syscall.Handle
	}
	verb, _ := syscall.UTF16PtrFromString("runas")
	file, e := syscall.UTF16PtrFromString(path)
	if e != nil {
		return e
	}
	params, _ := syscall.UTF16PtrFromString("/install /passive /norestart")
	s := shellInfo{Mask: 0x40 | 0x100, Verb: verb, File: file, Parameters: params, Show: 1}
	s.Size = uint32(unsafe.Sizeof(s))
	proc := syscall.NewLazyDLL("shell32.dll").NewProc("ShellExecuteExW")
	ok, _, err := proc.Call(uintptr(unsafe.Pointer(&s)))
	if ok == 0 {
		return fmt.Errorf("Windows 未允許 Microsoft 執行階段安裝，可按原按鈕重試：%w", err)
	}
	defer syscall.CloseHandle(s.Process)
	for {
		if ctx.Err() != nil {
			return ctx.Err()
		}
		result, e := syscall.WaitForSingleObject(s.Process, 1000)
		if e != nil {
			return e
		}
		if result == syscall.WAIT_TIMEOUT {
			continue
		}
		var code uint32
		if e = syscall.GetExitCodeProcess(s.Process, &code); e != nil {
			return e
		}
		if code == 0 || code == 3010 || code == 1638 {
			return nil
		}
		return fmt.Errorf("Microsoft 執行階段安裝未完成（代碼 %d）", code)
	}
}
