//go:build windows

package main

import (
	"fmt"
	"os/exec"
	"syscall"
	"unsafe"
)

type jobBasic struct {
	ProcessTime, JobTime int64
	Flags                uint32
	MinWS, MaxWS         uintptr
	Active               uint32
	Affinity             uintptr
	Priority, Schedule   uint32
}
type jobExtended struct {
	Basic                                        jobBasic
	IO                                           [6]uint64
	ProcessLimit, JobLimit, PeakProcess, PeakJob uintptr
}

// This is a private committed-memory budget, NOT a promise about RSS or GPU/shared memory.
func limitWorkerMemory(c *exec.Cmd, mb int) (func(), error) {
	if mb <= 0 {
		return func() {}, nil
	}
	dll := syscall.NewLazyDLL("kernel32.dll")
	create := dll.NewProc("CreateJobObjectW")
	set := dll.NewProc("SetInformationJobObject")
	assign := dll.NewProc("AssignProcessToJobObject")
	closeH := dll.NewProc("CloseHandle")
	open := dll.NewProc("OpenProcess")
	job, _, err := create.Call(0, 0)
	if job == 0 {
		return nil, fmt.Errorf("CreateJobObject: %w", err)
	}
	closeJob := func() { closeH.Call(job) }
	info := jobExtended{}
	info.Basic.Flags = 0x200 | 0x2000
	info.JobLimit = uintptr(mb) * 1024 * 1024
	ok, _, err := set.Call(job, 9, uintptr(unsafe.Pointer(&info)), unsafe.Sizeof(info))
	if ok == 0 {
		closeJob()
		return nil, fmt.Errorf("記憶體預算設定失敗：%w", err)
	}
	process, _, err := open.Call(0x100|0x1, 0, uintptr(c.Process.Pid))
	if process == 0 {
		closeJob()
		return nil, fmt.Errorf("OpenProcess: %w", err)
	}
	defer closeH.Call(process)
	ok, _, err = assign.Call(job, process)
	if ok == 0 {
		closeJob()
		return nil, fmt.Errorf("無法將工作納入記憶體預算：%w", err)
	}
	return closeJob, nil
}
