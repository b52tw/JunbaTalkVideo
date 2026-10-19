package main

// The user may select a fixed limit. Zero means choose once per stage from
// available physical memory: half available, clamped to 4–6 GiB.
// This is a private-commit limit, not an RSS/GPU/whole-system memory promise.
func workerBudget(requested int, m MemoryInfo) int {
	if requested > 0 {
		return requested
	}
	gib := int(m.Available / (1024 * 1024 * 1024) / 2)
	if gib < 4 {
		gib = 4
	}
	if gib > 6 {
		gib = 6
	}
	return gib * 1024
}
