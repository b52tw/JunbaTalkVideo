package main

// localMemoryMinimum is the available-RAM gate before a local render starts.
//
// v0.8.3b: the staged (disk-mapped, one stage per process) mode runs stages
// sequentially and each previous process exits before the next starts. Field
// reports show the largest stage (mouth inference) peaks at ~3.9 GiB private
// commit while every other stage stays under 1 GiB, and each stage is capped by
// a 4 GiB job budget. Requiring 6 GiB for that mode forced users to close every
// other program, so staged mode now needs 4.5 GiB (4 GiB stage cap + headroom).
// The two-stage "standard" mode keeps the original, stricter gates.
func localMemoryMinimum(device, memoryMode string) uint64 {
	switch {
	case memoryMode == "staged":
		return 4608 << 20 // 4.5 GiB
	case device == "CPU":
		return 8 << 30
	default:
		return 6 << 30
	}
}
