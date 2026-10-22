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
    """Conservative foreground-only head motion with a fixed, inpainted background.

    The mask must pass GrabCut and connected-component checks. If not, motion is
    disabled instead of warping background pixels along with the face.
    """

    def __init__(self, shape, face_box, image=None, style='auto'):
        self.ready = False
        self.last_union = None
        self.roi = (0, 0, 0, 0)
        if image is None or image.shape[:2] != tuple(shape[:2]):
            return
        h, w = shape[:2]
        x, y, bw, bh = face_box
        fx, fy, fw, fh = x * w, y * h, bw * w, bh * h
        if fw < 12 or fh < 12:
            return
        x1, y1 = max(0, int(fx - .70 * fw)), max(0, int(fy - .55 * fh))
        x2, y2 = min(w, int(fx + 1.70 * fw)), min(h, int(fy + 1.35 * fh))
        if x2 - x1 < 24 or y2 - y1 < 24:
            return
        self.roi = (x1, y1, x2, y2)
        source = image[y1:y2, x1:x2].copy()
        rh, rw = source.shape[:2]
        scale = min(1.0, 384.0 / max(rh, rw))
        sw, sh = max(16, int(rw * scale)), max(16, int(rh * scale))
        small = cv2.resize(source, (sw, sh), interpolation=cv2.INTER_AREA if scale < 1 else cv2.INTER_LINEAR)
        # A flat/uniform crop carries no reliable foreground boundary; fail closed.
        if float(np.std(small)) < 1.5:
            return
        cx = int((fx + fw * .5 - x1) * scale)
        cy = int((fy + fh * .40 - y1) * scale)
        ax, ay = max(5, int(fw * .66 * scale)), max(5, int(fh * .80 * scale))
        mask = np.full((sh, sw), cv2.GC_BGD, np.uint8)
        cv2.ellipse(mask, (cx, cy), (ax, ay), 0, 0, 360, cv2.GC_PR_FGD, -1)
        core_x1 = max(0, int((fx + fw * .20 - x1) * scale))
        core_x2 = min(sw - 1, int((fx + fw * .80 - x1) * scale))
        core_y1 = max(0, int((fy + fh * .14 - y1) * scale))
        core_y2 = min(sh - 1, int((fy + fh * .78 - y1) * scale))
        if core_x2 <= core_x1 or core_y2 <= core_y1:
            return
        cv2.rectangle(mask, (core_x1, core_y1), (core_x2, core_y2), cv2.GC_FGD, -1)
        bgd = np.zeros((1, 65), np.float64)
        fgd = np.zeros((1, 65), np.float64)
        try:
            cv2.grabCut(small, mask, None, bgd, fgd, 3, cv2.GC_INIT_WITH_MASK)
        except cv2.error:
            return
        hard = ((mask == cv2.GC_FGD) | (mask == cv2.GC_PR_FGD)).astype(np.uint8)
        ratio = float(hard.mean())
        if not .015 < ratio < .78:
            return
        count, labels = cv2.connectedComponents(hard)
        if count < 2:
            return
        sx = int(np.clip(cx, 0, sw - 1))
        sy = int(np.clip(cy, 0, sh - 1))
        label = int(labels[sy, sx])
        if label == 0:
            core = labels[core_y1:core_y2 + 1, core_x1:core_x2 + 1]
            candidates = core[core > 0]
            if candidates.size == 0:
                return
            label = int(np.bincount(candidates).argmax())
        hard = (labels == label).astype(np.uint8)
        if float(hard.mean()) < .01:
            return
        hard = cv2.morphologyEx(hard, cv2.MORPH_CLOSE, np.ones((3, 3), np.uint8))
        hard = cv2.resize(hard, (rw, rh), interpolation=cv2.INTER_NEAREST)
        hard = cv2.dilate(hard, np.ones((3, 3), np.uint8), iterations=1)
        self.alpha_roi = cv2.GaussianBlur(hard.astype(np.float32), (0, 0), .8).clip(0, 1)
        original_mask = (self.alpha_roi > .06).astype(np.uint8)
        inpaint_mask = cv2.dilate(original_mask, np.ones((5, 5), np.uint8), iterations=1)
        try:
            self.background_roi = cv2.inpaint(source, inpaint_mask, 3, cv2.INPAINT_TELEA)
        except cv2.error:
            return
        self.inpaint_union = inpaint_mask > 0
        self.last_union = self.inpaint_union.copy()
        self.fh = float(fh)
        self.local_pivot = (float(fx + fw * .5 - x1), float(fy + fh * .96 - y1))
        self.ready = True

    def apply(self, image, angle_deg, nod, shake):
        if not self.ready:
            return image
        if abs(angle_deg) < .02 and abs(nod) < 1e-4 and abs(shake) < 1e-4:
            self.last_union = self.inpaint_union.copy()
            return image
        x1, y1, x2, y2 = self.roi
        rw, rh = x2 - x1, y2 - y1
        px, py = self.local_pivot
        angle = float(np.clip(angle_deg, -1.5, 1.5))
        dx = float(np.clip(shake * self.fh, -2.0, 2.0))
        dy = float(np.clip(nod * self.fh, -2.0, 2.0))
        matrix = cv2.getRotationMatrix2D((px, py), angle, 1.0)
        matrix[0, 2] += dx
        matrix[1, 2] += dy
        foreground = cv2.warpAffine(image[y1:y2, x1:x2], matrix, (rw, rh),
                                    flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE)
        alpha = cv2.warpAffine(self.alpha_roi, matrix, (rw, rh), flags=cv2.INTER_LINEAR,
                               borderMode=cv2.BORDER_CONSTANT, borderValue=0)
        alpha = cv2.GaussianBlur(np.clip(alpha, 0, 1), (0, 0), .55)
        union = self.inpaint_union | (alpha > .025)
        bg = self.background_roi.astype(np.float32)
        fg = foreground.astype(np.float32)
        composed = fg * alpha[:, :, None] + bg * (1.0 - alpha[:, :, None])
        out = image.copy()
        roi = out[y1:y2, x1:x2]
        roi[union] = np.clip(composed[union], 0, 255).astype(np.uint8)
        self.last_union = union
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
