package main

import "testing"

func TestPuppetOptions(t *testing.T) {
	q := LocalRequest{BBox: []float64{.2, .2, .4, .4}, Engine: "puppet"}
	if e := validateLocal(&q); e != nil || q.PuppetStyle != "smooth" || q.PuppetOpen != .55 || q.ComposeStyle != "auto" {
		t.Fatal("puppet defaults", e, q.PuppetStyle, q.PuppetOpen)
	}
	q = LocalRequest{BBox: []float64{.2, .2, .4, .4}, Engine: "puppet", PuppetStyle: "shapes", PuppetOpen: .75}
	if e := validateLocal(&q); e != nil {
		t.Fatal(e)
	}
	for _, bad := range []LocalRequest{
		{BBox: []float64{.2, .2, .4, .4}, Engine: "puppet", PuppetStyle: "dance"},
		{BBox: []float64{.2, .2, .4, .4}, Engine: "puppet", PuppetOpen: 1.5},
		{BBox: []float64{.2, .2, .4, .4}, ComposeStyle: "anime"},
	} {
		b := bad
		if validateLocal(&b) == nil {
			t.Fatal("invalid puppet option accepted", b.PuppetStyle, b.PuppetOpen, b.ComposeStyle)
		}
	}
	cfg := map[string]any{}
	v08Config(cfg, q)
	if cfg["engine"] != "puppet" || cfg["puppet_style"] != "shapes" || cfg["puppet_open"] != .75 {
		t.Fatal(cfg)
	}
}
