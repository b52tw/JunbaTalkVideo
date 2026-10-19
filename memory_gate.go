package main

// localMemoryGate returns the available-RAM gates before a local render.
//
//   - hard: below this the render is refused (it would almost surely fail).
//   - recommended: below this the render still starts, with a warning.
//
// v0.8.3b: the staged (disk-mapped, one stage per process) MuseTalk mode runs
// stages sequentially; the largest stage (mouth inference) peaks at ~3.9 GiB
// private commit under a 4 GiB per-stage budget, so 4.5 GiB is recommended.
// v0.8.5: between 3 and 4.5 GiB it now starts with a warning instead of being
// refused, and the cartoon lip puppet (no neural model, measured peak ~0.4 GiB
// at 1920 px with head/body motion) only needs 1 GiB.
func localMemoryGate(engine, device, memoryMode string) (hard, recommended uint64) {
	switch {
	case engine == "puppet":
		return 1 << 30, 1 << 30
	case memoryMode == "staged":
		return 3 << 30, 4608 << 20
	case device == "CPU":
		return 8 << 30, 8 << 30
	default:
		return 6 << 30, 6 << 30
	}
}

// localMemoryMinimum is kept for callers/tests that need the recommended gate.
func localMemoryMinimum(device, memoryMode string) uint64 {
	_, rec := localMemoryGate("musetalk", device, memoryMode)
	return rec
}
