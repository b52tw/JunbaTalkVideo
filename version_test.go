package main

import "testing"

func TestAppVersionIsSingleSource(t *testing.T) {
	if got, want := appVersion(), "0.9.0"; got != want {
		t.Fatalf("appVersion() = %q, want %q", got, want)
	}
	if got, want := portableSDKBundleID(), "junba-sdk-v090"; got != want {
		t.Fatalf("portableSDKBundleID() = %q, want %q", got, want)
	}
	if got, want := runtimeInstallMarker(), "junba-runtime-v090"; got != want {
		t.Fatalf("runtimeInstallMarker() = %q, want %q", got, want)
	}
}

func TestPortableSDKCompatibilityChecks(t *testing.T) {
	if !supportedPortableSDKBundle("junba-sdk-v081") {
		t.Fatal("existing v0.8.9 portable SDK should remain supported")
	}
	if !supportedPortableSDKBundle(portableSDKBundleID()) {
		t.Fatal("current SDK bundle should be supported")
	}
	if supportedPortableSDKBundle("junba-sdk-v091") {
		t.Fatal("unrecognized SDK bundle should be rejected")
	}
}
