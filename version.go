package main

import (
	_ "embed"
	"strings"
)

//go:embed VERSION
var embeddedAppVersion string

func appVersion() string { return strings.TrimSpace(embeddedAppVersion) }

func portableSDKBundleID() string {
	return "junba-sdk-v" + strings.ReplaceAll(appVersion(), ".", "")
}

func runtimeInstallMarker() string {
	return "junba-runtime-v" + strings.ReplaceAll(appVersion(), ".", "")
}

func supportedPortableSDKBundle(id string) bool {
	return id == "junba-sdk-v081" || id == portableSDKBundleID()
}
