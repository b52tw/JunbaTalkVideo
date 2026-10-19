package main

import "testing"

func TestLocalMemoryMinimum(t *testing.T) {
	if got := localMemoryMinimum("AUTO", "staged"); got != 4608<<20 {
		t.Fatalf("staged gate = %d", got)
	}
	if got := localMemoryMinimum("CPU", "staged"); got != 4608<<20 {
		t.Fatalf("staged CPU gate = %d", got)
	}
	if got := localMemoryMinimum("AUTO", "standard"); got != 6<<30 {
		t.Fatalf("standard GPU gate = %d", got)
	}
	if got := localMemoryMinimum("CPU", "standard"); got != 8<<30 {
		t.Fatalf("standard CPU gate = %d", got)
	}
	// Staged gate must stay above the per-stage 4 GiB budget floor.
	if localMemoryMinimum("AUTO", "staged") <= uint64(workerBudget(0, MemoryInfo{Available: 4608 << 20}))<<20 {
		t.Fatal("staged gate leaves no headroom over the stage budget")
	}
	if h, r := localMemoryGate("puppet", "AUTO", "staged"); h != 1<<30 || r != 1<<30 {
		t.Fatalf("puppet gate %d %d", h, r)
	}
	if h, r := localMemoryGate("musetalk", "AUTO", "staged"); h != 3<<30 || r != 4608<<20 {
		t.Fatalf("staged soft gate %d %d", h, r)
	}
}
