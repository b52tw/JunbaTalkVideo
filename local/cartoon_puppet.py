"""Cartoon lip puppet: open the ORIGINAL drawn mouth by audio, no neural redraw.

The upper lip lifts a little and the lower lip + jaw drop with the voice
envelope. The gap between the lips is filled with the mouth's own cavity
colours, so the original line art, teeth and colours stay crisp and there is
only ever one mouth. Corners stay pinned. Pure OpenCV/NumPy.

Mode A (``mouth_drive``): continuous opening. Mode B (``viseme_track``): five
auto-made mouth shapes switched like 2D limited animation.
"""
import cv2
import numpy as np


def _read_mono(audio_path):
    from scipy.io import wavfile
    sr, a = wavfile.read(audio_path)
    if np.issubdtype(a.dtype, np.integer):
        a = a.astype(np.float32) / max(abs(np.iinfo(a.dtype).min), np.iinfo(a.dtype).max)
    else:
        a = a.astype(np.float32)
    if a.ndim > 1:
        a = a.mean(1)
    return sr, a


def audio_features(audio_path, fps):
    """Per-video-frame speech level (0..1) and spectral brightness (0..1)."""
    sr, a = _read_mono(audio_path)
    hop = max(1, int(sr / fps))
    n = max(1, int(np.ceil(len(a) / hop)))
    size = 1 << int(np.ceil(np.log2(hop * 2)))
    freqs = np.fft.rfftfreq(size, 1 / sr)
    band = (freqs > 250) & (freqs < 3500)   # speech band: breaths/hum do not open the mouth
    win = np.hanning(size)
    energy = np.zeros(n, np.float32)
    centroid = np.zeros(n, np.float32)
    for i in range(n):
        seg = a[i * hop:i * hop + size]
        seg = np.pad(seg, (0, size - len(seg)))
        mag = np.abs(np.fft.rfft(seg * win))[band]
        energy[i] = np.sqrt(np.mean(mag ** 2))
        centroid[i] = float((mag * freqs[band]).sum() / max(mag.sum(), 1e-9))
    loud = np.percentile(energy, 95) if np.any(energy > 0) else 1.0
    gate = np.percentile(energy, 30)
    level = np.clip((energy - gate) / max(loud - gate, 1e-9), 0, 1) ** .7
    voiced = centroid[level > .15]
    if len(voiced) > 4:
        lo, hi = np.percentile(voiced, 15), np.percentile(voiced, 85)
        bright = np.clip((centroid - lo) / max(hi - lo, 1e-6), 0, 1)
    else:
        bright = np.full_like(level, .5)
    return level.astype(np.float32), bright.astype(np.float32)


def mouth_drive(audio_path, fps, attack=.45, release=.18):
    """Mode A: continuous mouth opening 0..1 following the voice envelope."""
    level, _ = audio_features(audio_path, fps)
    out = np.zeros_like(level)
    value = 0.0
    for i, target in enumerate(level):
        value += (attack if target > value else release) * (target - value)
        out[i] = value
    return out


# Mode B mouth shapes: (opening, width). width < 0 rounds (ㄨ/ㄛ), > 0 widens (ㄧ/ㄝ).
SHAPES = dict(rest=(0., 0.), half=(.40, 0.), open=(.85, 0.), round=(.50, -.16), wide=(.32, .10))


def viseme_track(audio_path, fps, hold=.12, blend=.08):
    """Mode B: classic 2D limited animation — pick one of five mouth shapes per
    beat from loudness and spectral brightness, hold it >= ``hold`` seconds and
    morph between shapes over ``blend`` seconds. Returns [(opening, width)]."""
    level, bright = audio_features(audio_path, fps)
    smooth = np.convolve(level, np.ones(3) / 3, mode='same')
    names = []
    for lv, br in zip(smooth, bright):
        if lv < .12:
            names.append('rest')
        elif br < .30 and lv > .25:
            names.append('round')
        elif br > .72:
            names.append('wide')
        elif lv < .50:
            names.append('half')
        else:
            names.append('open')
    min_hold = max(1, round(hold * fps))
    chosen, current, held = [], names[0] if names else 'rest', 0
    for name in names:
        if name != current and held >= min_hold:
            current, held = name, 0
        held += 1
        chosen.append(current)
    steps = max(1, round(blend * fps))
    out, prev = [], np.array(SHAPES['rest'], np.float32)
    for name in chosen:
        target = np.array(SHAPES[name], np.float32)
        prev = prev + (target - prev) / steps
        out.append((float(prev[0]), float(prev[1])))
    return out


def parting_line(image, outer):
    """Per-column y of the lip parting (darkest row inside the mouth)."""
    h, w = image.shape[:2]
    poly = np.rint(np.asarray(outer, np.float32) * [w, h]).astype(np.int32)
    mask = np.zeros((h, w), np.uint8)
    cv2.fillPoly(mask, [poly], 1)
    x1, y1, bw, bh = cv2.boundingRect(poly)
    gray = cv2.GaussianBlur(cv2.cvtColor(image, cv2.COLOR_BGR2GRAY), (0, 0), 1.5).astype(np.float32)
    xs, ys = [], []
    for x in range(x1, x1 + bw):
        col = np.nonzero(mask[:, x])[0]
        if len(col) < 3:
            continue
        ys.append(col[np.argmin(gray[col, x])])
        xs.append(x)
    if len(xs) < 5:
        return None
    # A smooth quadratic follows smiles/frowns and ignores single dark pixels.
    coef = np.polyfit(np.asarray(xs, np.float32), np.asarray(ys, np.float32), 2)
    return dict(coef=coef, box=(x1, y1, bw, bh))


class CartoonPuppet:
    def __init__(self, image, outer, max_open=.55, lift=.22, chin=1.6):
        self.base = image
        self.part = parting_line(image, outer)
        if self.part is None:
            raise ValueError('無法在原圖嘴部找到上下唇分界')
        x1, y1, bw, bh = self.part['box']
        h, w = image.shape[:2]
        self.max_open = max_open * bh
        self.lift = lift
        # Region of interest: mouth, a little above, and the jaw below.
        pad_x = int(bw * .35)
        self.roi = (max(0, x1 - pad_x), max(0, int(y1 - bh * 1.0)),
                    min(w, x1 + bw + pad_x), min(h, int(y1 + bh + bh * chin + 2)))
        rx1, ry1, rx2, ry2 = self.roi
        yy, xx = np.mgrid[ry1:ry2, rx1:rx2].astype(np.float32)
        self.yy = yy
        self.xx = xx
        self.yc = np.polyval(self.part['coef'], xx).astype(np.float32)
        cx = x1 + bw / 2
        u = np.clip(np.abs(xx - cx) / (bw * .44), 0, 1)  # zero before the corners
        horiz = np.clip(np.cos(u * np.pi / 2), 0, 1) ** 1.2  # corners stay pinned
        bottom = y1 + bh
        below = np.clip(1 - np.maximum(0, yy - bottom) / (bh * chin), 0, 1)
        self.w_down = (horiz * below * (yy >= self.yc)).astype(np.float32)
        above = np.clip(1 - (self.yc - yy) / (bh * .9), 0, 1)
        self.w_up = (horiz * above * (yy < self.yc)).astype(np.float32)
        # Horizontal squeeze/stretch field for round/wide shapes: full strength
        # around the mouth band, zero at the ROI border so there is no seam.
        self.cx = cx
        half = max(1.0, bw * .75)  # squeeze the mouth, not the cheeks
        band = np.clip(1 - np.abs(yy - self.yc) / (bh * 1.4), 0, 1)
        side = np.clip(1 - (np.abs(xx - cx) / half) ** 2, 0, 1)
        self.w_side = (band * side).astype(np.float32)
        # Cavity colours sampled from the original mouth interior.
        poly = np.rint(np.asarray(outer, np.float32) * [w, h]).astype(np.int32)
        inside = np.zeros((h, w), np.uint8)
        cv2.fillPoly(inside, [poly], 1)
        pix = image[inside > 0].astype(np.float32)
        lum = pix @ np.array([.114, .587, .299], np.float32)
        dark = pix[lum <= np.percentile(lum, 12)]
        mid = pix[(lum > np.percentile(lum, 12)) & (lum <= np.percentile(lum, 35))]
        self.cavity_top = (np.median(dark, 0) * .85 if len(dark) else np.array([30, 25, 60], np.float32)).astype(np.float32)
        self.cavity_bottom = (np.median(mid, 0) if len(mid) else self.cavity_top * 1.6).astype(np.float32)

    def frame(self, amount, image=None, width=0.0):
        image = self.base if image is None else image
        if amount <= 1e-3 and abs(width) < 1e-3:
            return image
        a = float(amount) * self.max_open
        src_x = self.xx
        if abs(width) >= 1e-3:
            src_x = (self.cx + (self.xx - self.cx) * (1 - float(width) * self.w_side)).astype(np.float32)
        src_y = self.yy.copy()
        down = self.w_down > 0
        src_y[down] = np.maximum(self.yy[down] - a * self.w_down[down], self.yc[down])
        up = self.w_up > 0
        src_y[up] = np.minimum(self.yy[up] + a * self.lift * self.w_up[up], self.yc[up] - 1)
        rx1, ry1, rx2, ry2 = self.roi
        warped = cv2.remap(image, src_x, src_y, cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE)
        # Fill the newly exposed gap with the mouth's own cavity colours (dark at
        # the top, toward the tongue tone at the bottom) instead of stretching a
        # pixel row, which smears tooth gaps into vertical streaks.
        shift = a * self.w_down
        # Only where the lips actually part by more than a pixel (not along the
        # whole parting curve past the corners).
        gap = (down & (shift > 1.0) & (self.yy - shift < self.yc)).astype(np.float32)
        if gap.any():
            depth = np.clip((self.yy - self.yc) / max(a, 1.0), 0, 1)[:, :, None]
            fill = self.cavity_top * (1 - depth) + self.cavity_bottom * depth
            soft = cv2.GaussianBlur(gap, (0, 0), 1.0)[:, :, None]
            warped = np.clip(warped * (1 - soft) + fill * soft, 0, 255).astype(np.uint8)
        out = image.copy()
        out[ry1:ry2, rx1:rx2] = warped
        return out
