package main

import "testing"

func TestV08Options(t *testing.T) {
	base := func() LocalRequest { return LocalRequest{BBox: []float64{.2, .2, .4, .4}} }
	q := base()
	if e := validateLocal(&q); e != nil || q.Runtime != "openvino" || q.BlendMode != "feather" || q.BlinkProfile != "bezier" {
		t.Fatal(e, q)
	}
	for _, change := range []func(*LocalRequest){func(q *LocalRequest) { q.Runtime = "other" }, func(q *LocalRequest) { q.Runtime = "directml"; q.Device = "NPU" }, func(q *LocalRequest) { q.DMLAdapter = 8 }, func(q *LocalRequest) { q.MouthScope = "eyes" }, func(q *LocalRequest) { q.Runtime = "torch" }} {
		q = base()
		change(&q)
		if validateLocal(&q) == nil {
			t.Fatal("invalid option accepted", q)
		}
	}
	q = base()
	q.Runtime = "directml"
	q.MemoryMode = "standard"
	if e := validateLocal(&q); e != nil || q.MemoryMode != "staged" {
		t.Fatal(e, q)
	}
	q = base()
	q.Engine = "sadtalker"
	q.Runtime = "torch"
	q.CompositionMode = "full"
	if e := validateLocal(&q); e != nil {
		t.Fatal(e)
	}
}
