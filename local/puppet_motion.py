"""Head, brow and multi-speaker timing for the cartoon lip puppet (2-D warps).

All motion is a smooth displacement field applied with cv2.remap, so no holes
open and no new pixels are invented. Angles stay small (a few degrees): this
adds life to a still picture; it is not a 3-D head or body generator.

Timeline syntax (one action per line, seconds):
    1-3 點頭        nod repeatedly between 1 s and 3 s
    4.5 點頭        one nod at 4.5 s
    6-8 側傾        tilt the head
    9 搖頭          one head shake
    10-11 抬眉      raise the brows
Optional speaker prefix for dialogue:  B 2-3 點頭
"""
import math
import re

import cv2
import numpy as np

AMPLITUDE = dict(off=0.0, small=1.2, medium=2.2, large=3.4)
ACTIONS = {'點頭': 'nod', 'nod': 'nod', '側傾': 'tilt', '歪頭': 'tilt', 'tilt': 'tilt',
           '搖頭': 'shake', 'shake': 'shake', '抬眉': 'brow', '挑眉': 'brow', 'brow': 'brow',
           '搖擺': 'sway', '晃動': 'sway', 'sway': 'sway'}
BODY_DEG = dict(small=1.0, medium=1.8, large=2.8)


def parse_timeline(text):
    """Return [(speaker or None, start, end, action)]; unknown lines are skipped."""
    items = []
    for raw in (text or '').splitlines():
        line = raw.strip().replace('～', '-').replace('~', '-').replace('－', '-')
        m = re.match(r'^(?:([A-Ea-e])\s+)?(\d+(?:\.\d+)?)(?:\s*-\s*(\d+(?:\.\d+)?))?\s*(?:秒)?\s*(\S+)', line)
        if not m:
            continue
        action = ACTIONS.get(m.group(4).lower() if m.group(4).isascii() else m.group(4))
        if not action:
            continue
        start = float(m.group(2))
        end = float(m.group(3)) if m.group(3) else start + (.6 if action != 'brow' else .8)
        if end <= start:
            continue
        items.append(((m.group(1) or '').upper() or None, start, end, action))
    return items


def _bump(t, center, width):
    """Smooth 0..1..0 pulse of total length ``width`` centred at ``center``."""
    x = (t - center) / max(width / 2, 1e-3)
    return 0.5 * (1 + np.cos(np.clip(x, -1, 1) * np.pi))


def motion_curves(level, fps, amplitude='small', timeline=(), active=None, speaker=None, continuous=True):
    """Per-frame (angle_deg, nod_px_fraction, shake_fraction, brow) curves.

    ``level``: speech level 0..1 per frame. ``active``: optional 0/1 mask for
    the frames this speaker talks (dialogue). Continuous mode adds accent nods
    and a slow idle sway; timeline items add explicit one-shot/ranged actions.
    """
    n = len(level)
    t = np.arange(n) / float(fps)
    k = AMPLITUDE.get(amplitude, 0.0)
    angle = np.zeros(n, np.float32)
    nod = np.zeros(n, np.float32)
    shake = np.zeros(n, np.float32)
    brow = np.zeros(n, np.float32)
    talk = level if active is None else level * active
    if continuous and k > 0:
        # Slow idle sway (always) + stronger while talking.
        angle += k * (.35 * np.sin(2 * np.pi * t / 5.3) + .25 * np.sin(2 * np.pi * t / 3.1 + 1.3))
        # Accent nods: rising edges of the speech envelope.
        rise = np.diff(np.r_[0, np.convolve(talk, np.ones(3) / 3, 'same')])
        last = -10.0
        for i in np.nonzero(rise > .12)[0]:
            if t[i] - last < .55:
                continue
            last = t[i]
            strength = float(min(1, talk[i] + .3))
            nod += k * .006 * strength * _bump(t, t[i] + .12, .35)
            brow += .45 * strength * min(1, k) * _bump(t, t[i] + .1, .45)
        # A small settling nod after each phrase (speech -> silence > 0.4 s).
        speaking = np.convolve(talk > .12, np.ones(max(1, int(.4 * fps))), 'same') > 0
        ends = np.nonzero(np.diff(speaking.astype(np.int8)) == -1)[0]
        for i in ends:
            nod += k * .008 * _bump(t, t[i] + .1, .5)
    for who, start, end, action in timeline:
        if who and speaker and who != speaker:
            continue
        if who and not speaker and who != 'A':
            continue
        kk = max(k, 1.0)
        span = end - start
        if action == 'nod':
            beats = max(1, int(round(span / .6)))
            for b in range(beats):
                nod += min(kk * .011, .03) * _bump(t, start + (b + .5) * span / beats, min(.6, span / beats))
        elif action == 'tilt':
            env = _bump(t, (start + end) / 2, span) if span < 1.2 else np.clip(
                np.minimum((t - start) / .3, (end - t) / .3), 0, 1)
            angle += min(kk * 2.0, 6.0) * env
        elif action == 'shake':
            env = np.clip(np.minimum((t - start) / .15, (end - t) / .15), 0, 1)
            shake += kk * .010 * env * np.sin(2 * np.pi * (t - start) / .35)
        elif action == 'brow':
            env = np.clip(np.minimum((t - start) / .15, (end - t) / .2), 0, 1)
            brow += env
    return angle, nod, shake, np.clip(brow, 0, 1)


class HeadRig:
    """Rotate/nod/shake one head about the neck with a soft falloff mask."""

    def __init__(self, shape, face_box):
        h, w = shape[:2]
        x, y, bw, bh = face_box
        fx, fy, fw, fh = x * w, y * h, bw * w, bh * h
        self.pivot = np.array([fx + fw / 2, fy + fh * 1.02], np.float32)
        self.fh = fh
        # Head ellipse: face box widened a little and extended up for hair.
        cx, cy = fx + fw / 2, fy + fh * .38
        ax, ay = fw * .72, fh * .95
        pad = int(max(fw, fh) * .45)
        self.roi = (max(0, int(fx - pad)), max(0, int(fy - fh * .75)),
                    min(w, int(fx + fw + pad)), min(h, int(fy + fh * 1.35)))
        rx1, ry1, rx2, ry2 = self.roi
        yy, xx = np.mgrid[ry1:ry2, rx1:rx2].astype(np.float32)
        r = np.sqrt(((xx - cx) / ax) ** 2 + ((yy - cy) / ay) ** 2)
        inside = np.clip((1.35 - r) / .35, 0, 1)
        # Neck/shoulders below the pivot stay put.
        below = np.clip(1 - (yy - self.pivot[1]) / (fh * .25), 0, 1)
        weight = inside * below
        self.weight = (weight * weight * (3 - 2 * weight)).astype(np.float32)
        self.xx, self.yy = xx, yy

    def apply(self, image, angle_deg, nod, shake):
        if abs(angle_deg) < .02 and abs(nod) < 1e-4 and abs(shake) < 1e-4:
            return image
        a = math.radians(angle_deg)
        dx, dy = shake * self.fh, nod * self.fh
        px, py = self.pivot
        # Inverse map: destination -> source (rotate by -a about the pivot).
        ux, uy = self.xx - px - dx, self.yy - py - dy
        sx = px + ux * math.cos(a) + uy * math.sin(a)
        sy = py - ux * math.sin(a) + uy * math.cos(a)
        mx = (self.xx + (sx - self.xx) * self.weight).astype(np.float32)
        my = (self.yy + (sy - self.yy) * self.weight).astype(np.float32)
        rx1, ry1, rx2, ry2 = self.roi
        out = image.copy()
        out[ry1:ry2, rx1:rx2] = cv2.remap(image, mx, my, cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE)
        return out


class BrowRig:
    """Lift the eyebrows above each eye box; the eyes themselves never move.

    The whole brow band moves rigidly (same lift top and bottom, so the brow
    keeps its shape); only the skin between eye and brow and the forehead
    above absorb the stretch.
    """

    def __init__(self, shape, eye_boxes):
        h, w = shape[:2]
        self.parts = []
        for x, y, bw, bh in eye_boxes or []:
            ex1, ey1, ew, eh = x * w, y * h, bw * w, bh * h
            lift = max(2.0, eh * .5)
            top = ey1 - eh * 6.0
            rx1, rx2 = int(max(0, ex1 - ew * .45)), int(min(w, ex1 + ew * 1.45))
            ry1, ry2 = int(max(0, top)), int(min(h, ey1))
            if rx2 - rx1 < 4 or ry2 - ry1 < 4:
                continue
            yy, xx = np.mgrid[ry1:ry2, rx1:rx2].astype(np.float32)
            u = np.clip(np.abs(xx - (ex1 + ew / 2)) / (ew * .95), 0, 1)
            horiz = np.clip(np.cos(u * np.pi / 2), 0, 1) ** 1.2
            d = ey1 - yy                                  # distance above the eye
            rise = np.clip(d / (eh * .9), 0, 1)           # eye -> brow gap stretches
            fall = np.clip((ey1 - eh * 6.0 - yy + eh * 2.2) / (eh * 2.2), 0, 1)  # forehead compresses
            vert = rise * fall
            vert = vert * vert * (3 - 2 * vert)
            self.parts.append(((rx1, ry1, rx2, ry2), xx, yy, (horiz * vert).astype(np.float32), lift))

    def apply(self, image, amount):
        if amount < .02 or not self.parts:
            return image
        out = image.copy()
        for (rx1, ry1, rx2, ry2), xx, yy, wgt, lift in self.parts:
            my = (yy + lift * amount * wgt).astype(np.float32)
            out[ry1:ry2, rx1:rx2] = cv2.remap(out, xx, my, cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE)
        return out


def dialogue_actions(dialogue):
    """Action tags bound to dialogue lines -> timeline items (audio time)."""
    items = []
    for seg in dialogue:
        for act in seg.get('actions') or []:
            if act in ('nod', 'tilt', 'shake', 'brow', 'sway'):
                items.append((str(seg.get('speaker', 'A')).upper(), float(seg['start']), float(seg['end']), act))
    return items


def body_sway_curve(level, fps, amplitude='small', timeline=(), active=None, speaker=None, continuous=True):
    """Low-frequency body rotation (degrees) while talking: weight shift, not a gesture."""
    n = len(level)
    t = np.arange(n) / float(fps)
    deg = BODY_DEG.get(amplitude, .8)
    talk = level if active is None else level * active
    env = np.convolve((talk > .1).astype(np.float32), np.ones(max(1, int(.8 * fps))) / max(1, int(.8 * fps)), 'same')
    angle = np.zeros(n, np.float32)
    if continuous:
        angle += deg * np.clip(env, 0, 1) * np.sin(2 * np.pi * t / 2.8)
    for who, start, end, action in timeline:
        if action != 'sway' or (who and speaker and who != speaker):
            continue
        e = np.clip(np.minimum((t - start) / .35, (end - t) / .35), 0, 1)
        angle += max(deg, 1.2) * e * np.sin(2 * np.pi * (t - start) / 2.2)
    return angle


class BodyRig(HeadRig):
    """Rotate head + torso a little about the chest (weight shift)."""

    def __init__(self, shape, face_box):
        h, w = shape[:2]
        x, y, bw, bh = face_box
        fx, fy, fw, fh = x * w, y * h, bw * w, bh * h
        self.pivot = np.array([fx + fw / 2, fy + fh * 2.4], np.float32)
        self.fh = fh
        cx, cy = fx + fw / 2, fy + fh * 1.2
        ax, ay = fw * 1.25, fh * 2.0
        self.roi = (max(0, int(cx - ax * 1.4)), max(0, int(fy - fh * .9)),
                    min(w, int(cx + ax * 1.4)), min(h, int(fy + fh * 3.6)))
        rx1, ry1, rx2, ry2 = self.roi
        yy, xx = np.mgrid[ry1:ry2, rx1:rx2].astype(np.float32)
        r = np.sqrt(((xx - cx) / ax) ** 2 + ((yy - cy) / ay) ** 2)
        weight = np.clip((1.3 - r) / .45, 0, 1)
        self.weight = (weight * weight * (3 - 2 * weight)).astype(np.float32)
        self.xx, self.yy = xx, yy


def speaker_masks(dialogue, n, fps, fade=.06):
    """Per-speaker 0..1 activity per frame from [{speaker,start,end}] segments."""
    t = np.arange(n) / float(fps)
    masks = {}
    for seg in dialogue:
        who = str(seg.get('speaker', 'A')).upper()
        start, end = float(seg['start']), float(seg['end'])
        m = np.clip(np.minimum((t - start + fade) / fade, (end + fade - t) / fade), 0, 1)
        masks[who] = np.maximum(masks.get(who, np.zeros(n, np.float32)), m)
    return masks
