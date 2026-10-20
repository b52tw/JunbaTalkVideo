package main

import (
	"context"
	"errors"
	"net/http"
	"sync"
	"time"
)

// v0.8.8b: installs (SadTalker, models, DirectML) lock the whole app while
// they run. They used to allow up to 8 hours with no way out, so a stalled
// download froze every engine. Installs now get a cancellable context, and
// downloads abort after two minutes without data (resume later).

var installMu sync.Mutex

func (a *App) installContext(limit time.Duration) (context.Context, context.CancelFunc) {
	ctx, cancel := context.WithTimeout(context.Background(), limit)
	installMu.Lock()
	a.installCancel = cancel
	installMu.Unlock()
	return ctx, func() {
		installMu.Lock()
		a.installCancel = nil
		installMu.Unlock()
		cancel()
	}
}

func (a *App) cancelInstall(w http.ResponseWriter) {
	installMu.Lock()
	cancel := a.installCancel
	installMu.Unlock()
	if cancel == nil {
		fail(w, 409, errors.New("目前沒有正在進行的安裝"))
		return
	}
	cancel()
	a.setupMessage("正在取消安裝；已下載的部分會保留，之後按同一按鈕可續傳", 0, 0)
	respond(w, map[string]string{"message": "已要求取消安裝；稍候幾秒即可繼續使用其他功能。"})
}

// stallGuard cancels “cancel“ when “touch“ has not been called for “idle“.
type stallGuard struct {
	mu      sync.Mutex
	last    time.Time
	stalled bool
	done    chan struct{}
}

func newStallGuard(idle time.Duration, cancel context.CancelFunc) *stallGuard {
	g := &stallGuard{last: time.Now(), done: make(chan struct{})}
	go func() {
		t := time.NewTicker(idle / 8)
		defer t.Stop()
		for {
			select {
			case <-g.done:
				return
			case <-t.C:
				g.mu.Lock()
				quiet := time.Since(g.last) > idle
				if quiet {
					g.stalled = true
				}
				g.mu.Unlock()
				if quiet {
					cancel()
					return
				}
			}
		}
	}()
	return g
}

func (g *stallGuard) touch() { g.mu.Lock(); g.last = time.Now(); g.mu.Unlock() }
func (g *stallGuard) stop()  { close(g.done) }
func (g *stallGuard) isStalled() bool {
	g.mu.Lock()
	defer g.mu.Unlock()
	return g.stalled
}

var downloadStallLimit = 2 * time.Minute
