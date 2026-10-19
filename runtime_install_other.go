//go:build !windows

package main

import (
	"context"
	"errors"
)

func installVCRuntime(ctx context.Context, path string) error {
	return errors.New("Microsoft 執行階段只適用 Windows")
}
