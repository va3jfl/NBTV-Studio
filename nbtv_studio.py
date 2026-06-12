#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
NBTV Studio - a narrow-band television experimenter's suite
============================================================
Transmit and receive mechanical-television style video over a sound card,
a WAV file, or a pure software loopback.

 * Classic club-style modes (32-line NBTV, Baird 30-line, 60/90/120-line eras)
 * Experimental wideband modes that exploit 24-bit / 96k / 192k sound cards
 * Mono, frame-sequential colour, line-sequential colour, and a stereo Y/C
   colour system (luma on the left channel, alternating U/V chroma on the
   right channel - a mono receiver still gets a clean black & white picture)
 * Sources: built-in test cards, still images, animated GIFs, video files,
   webcam, and live screen-capture of a region
 * Selectable output bandwidth: authentic narrow-band low-pass filters or a
   "direct cable" full-bandwidth mode
 * TX monitor and RX monitor, WAV record / WAV decode, software loopback

Required:   numpy, Pillow            (pip install numpy pillow)
For audio:  sounddevice              (pip install sounddevice)
Optional:   opencv-python  (video files & webcam),  mss (faster capture)

Run the GUI:        python nbtv_studio.py
Codec self-test:    python nbtv_studio.py --selftest
List audio devices: python nbtv_studio.py --list-devices

Signal convention used by this suite (self-consistent TX<->RX, adjustable):
  video units 0..1:  sync tip = 0.00, black = 0.30, peak white = 1.00
  mapped to audio as  a = 2*v - 1   (sync = -1.0, black = -0.4, white = +1.0)
  Each line starts with a sync pulse (default 12% of the line period).
  The first line of each frame carries a broad (50%) sync pulse.
  In frame-sequential colour the RED field carries broad pulses on its first
  TWO lines so the receiver can identify the field sequence.
"""

import argparse
import base64
import json
import math
import os
import queue
import struct
import sys
import threading
import time
import wave
import zlib

import numpy as np
from PIL import Image, ImageDraw, ImageFont

APP_NAME = "NBTV Studio By VA3JFL"
APP_VERSION = "1.0"

# ----------------------------------------------------------------------------
# Timing constants (fractions of one line period). The custom-mode editor can
# override sync_f per mode; the rest are shared by encoder and decoder.
# ----------------------------------------------------------------------------
DEF_SYNC_F   = 0.12   # line sync pulse width
DEF_BPORCH_F = 0.05   # back porch (black) after sync
DEF_FPORCH_F = 0.03   # front porch (black) before next sync
BROAD_F      = 0.50   # broad (frame) sync pulse width
BLACK_LEVEL  = 0.30   # black level in video units (sync = 0.0, white = 1.0)

SAMPLE_RATES = [22050, 44100, 48000, 96000, 192000]

OUTPUT_FILTERS = [
    ("Direct cable (full bandwidth)", None),
    ("20 kHz low-pass", 20000.0),
    ("15 kHz low-pass", 15000.0),
    ("10 kHz low-pass (classic NBTV)", 10000.0),
    ("7 kHz low-pass", 7000.0),
    ("5 kHz low-pass", 5000.0),
    ("3.4 kHz low-pass (comms radio)", 3400.0),
]

COLOR_SYSTEMS = [
    ("Monochrome", "mono"),
    ("Frame-sequential colour (R,G,B fields)", "fsc"),
    ("Line-sequential colour (R,G,B lines)", "lsc"),
    ("Stereo Y/C colour (L=luma, R=chroma)", "yc"),
]

# ----------------------------------------------------------------------------
# Mode table.
#   lines  : number of scan lines per frame
#   fps    : frames per second
#   aspect : (width, height) of the displayed picture
#   scan   : 'V' = vertical lines (classic NBTV, bottom-to-top, left-to-right)
#            'H' = horizontal lines (like later electronic TV / SSTV)
#   sync_f : optional per-mode sync width override
# Horizontal resolution is NOT fixed by the mode - it follows from the sample
# rate:  pixels per line ~ 0.80 * sample_rate / (lines * fps).
# ----------------------------------------------------------------------------
MODES = [
    dict(name="NBTV Club 32-line  (32 / 12.5 fps, 2:3, vertical)",
         lines=32, fps=12.5, aspect=(2, 3), scan='V'),
    dict(name="Baird 30-line  (30 / 12.5 fps, 3:7, vertical)",
         lines=30, fps=12.5, aspect=(3, 7), scan='V'),
    dict(name="Experimental 24-line  (24 / 12.5 fps, 2:3, vertical)",
         lines=24, fps=12.5, aspect=(2, 3), scan='V'),
    dict(name="Club 48-line  (48 / 12.5 fps, 4:3, horizontal)",
         lines=48, fps=12.5, aspect=(4, 3), scan='H'),
    dict(name="1931-era 60-line  (60 / 20 fps, 4:3, horizontal)",
         lines=60, fps=20.0, aspect=(4, 3), scan='H'),
    dict(name="90-line  (90 / 12.5 fps, 4:3, horizontal)",
         lines=90, fps=12.5, aspect=(4, 3), scan='H'),
    dict(name="Mid-30s 120-line  (120 / 12.5 fps, 4:3, horizontal)",
         lines=120, fps=12.5, aspect=(4, 3), scan='H'),
    dict(name="X96 wideband  (96 / 12.5 fps, 4:3)  [96k+]",
         lines=96, fps=12.5, aspect=(4, 3), scan='H'),
    dict(name="X120 wideband  (120 / 25 fps, 4:3)  [192k]",
         lines=120, fps=25.0, aspect=(4, 3), scan='H'),
    dict(name="X160 wideband  (160 / 12.5 fps, 4:3)  [192k]",
         lines=160, fps=12.5, aspect=(4, 3), scan='H'),
    dict(name="X240 hi-def slow  (240 / 6.25 fps, 4:3)  [192k]",
         lines=240, fps=6.25, aspect=(4, 3), scan='H'),
    dict(name="X288 hi-def slow  (288 / 6.25 fps, 4:3)  [192k]",
         lines=288, fps=6.25, aspect=(4, 3), scan='H'),
    dict(name="X360 photo-scan  (360 / 3.125 fps, 4:3)  [192k]",
         lines=360, fps=3.125, aspect=(4, 3), scan='H'),
    dict(name="X480 photo-scan  (480 / 2 fps, 4:3)  [192k]",
         lines=480, fps=2.0, aspect=(4, 3), scan='H'),
    dict(name="UltraWide 32  (32 / 12.5 fps, 2:3) - max horiz. detail",
         lines=32, fps=12.5, aspect=(2, 3), scan='V'),
]

TEST_PATTERNS = [
    "Colour bars",
    "Grey staircase",
    "Horizontal gradient",
    "Crosshatch + circle test card",
    "Resolution wedges",
    "Motion demo (bouncing block + clock)",
    "Black & white split",
]


def clamp(v, lo, hi):
    return lo if v < lo else (hi if v > hi else v)


# ----------------------------------------------------------------------------
# Geometry: everything the encoder/decoder needs to know about a mode at a
# given sample rate.
# ----------------------------------------------------------------------------
class Geometry:
    def __init__(self, mode, rate, sync_f=None):
        self.mode = mode
        self.rate = float(rate)
        self.lines = int(mode["lines"])
        self.fps = float(mode["fps"])
        self.aspect = tuple(mode["aspect"])
        self.scan = mode.get("scan", "H")
        self.sync_f = float(sync_f if sync_f is not None
                            else mode.get("sync_f", DEF_SYNC_F))
        self.bporch_f = DEF_BPORCH_F
        self.fporch_f = DEF_FPORCH_F
        self.line_rate = self.lines * self.fps              # Hz
        self.spl = self.rate / self.line_rate               # samples per line (float)
        self.active_f = 1.0 - self.sync_f - self.bporch_f - self.fporch_f
        self.act_start_f = self.sync_f + self.bporch_f
        self.act_end_f = 1.0 - self.fporch_f
        self.n_px = int(clamp(round(self.spl * self.active_f), 12, 800))
        # frame buffer pixel grid (n_lines x n_px); display orientation depends
        # on scan direction.
        self.usable = self.spl >= 24

    def grid_size(self):
        """PIL (width, height) of the source grid the encoder samples."""
        if self.scan == 'V':
            return (self.lines, self.n_px)   # vertical lines = columns
        return (self.n_px, self.lines)

    def describe(self):
        bw = 0.5 * self.rate * self.active_f  # crude max video bandwidth
        return ("%d lines @ %g fps | line rate %.1f Hz | %.1f samples/line | "
                "~%d px/line | video BW up to ~%.1f kHz"
                % (self.lines, self.fps, self.line_rate, self.spl,
                   self.n_px, bw / 1000.0))


# ----------------------------------------------------------------------------
# Image fitting helpers
# ----------------------------------------------------------------------------
def fit_to_aspect(img, aw, ah, fill=True):
    """Crop (fill=True) or letterbox (fill=False) a PIL image to aspect aw:ah."""
    w, h = img.size
    target = aw / ah
    cur = w / h
    if abs(cur - target) < 1e-6:
        return img
    if fill:
        if cur > target:        # too wide -> crop sides
            nw = int(round(h * target))
            x0 = (w - nw) // 2
            return img.crop((x0, 0, x0 + nw, h))
        else:                   # too tall -> crop top/bottom
            nh = int(round(w / target))
            y0 = (h - nh) // 2
            return img.crop((0, y0, w, y0 + nh))
    else:
        if cur > target:
            nh = int(round(w / target))
            canvas = Image.new("RGB", (w, nh), "black")
            canvas.paste(img, (0, (nh - h) // 2))
        else:
            nw = int(round(h * target))
            canvas = Image.new("RGB", (nw, h), "black")
            canvas.paste(img, ((nw - w) // 2, 0))
        return canvas


# Default strength of the optional "Detail boost" aperture correction that
# restores apparent crispness after the big downscale to the scan grid.
DETAIL_SHARPEN = 0.7

_SRGB_LUT = None


def _srgb_to_linear_lut():
    """256-entry uint8 sRGB -> float32 linear-light lookup table."""
    global _SRGB_LUT
    if _SRGB_LUT is None:
        x = np.arange(256, dtype=np.float32) / 255.0
        _SRGB_LUT = np.where(x <= 0.04045, x / 12.92,
                             ((x + 0.055) / 1.055) ** 2.4).astype(np.float32)
    return _SRGB_LUT


def _linear_to_srgb(a):
    a = np.clip(a, 0.0, 1.0)
    return np.where(a <= 0.0031308, a * 12.92,
                    1.055 * np.power(a, 1.0 / 2.4) - 0.055).astype(np.float32)


def _blur3(a, axis):
    """Separable [0.25, 0.5, 0.25] blur along one axis (edge-padded)."""
    pad = [(0, 0)] * a.ndim
    pad[axis] = (1, 1)
    e = np.pad(a, pad, mode="edge")
    s0 = [slice(None)] * a.ndim
    s1 = list(s0)
    s2 = list(s0)
    s0[axis] = slice(0, -2)
    s1[axis] = slice(1, -1)
    s2[axis] = slice(2, None)
    return (0.25 * e[tuple(s0)] + 0.5 * e[tuple(s1)]
            + 0.25 * e[tuple(s2)]).astype(np.float32)


def _aperture_sharpen(arr, amount):
    """Mild unsharp mask ('aperture correction', as classic TV cameras did)
    applied after downscaling, so the few scan lines we do have carry as
    much apparent detail as possible."""
    blur = _blur3(_blur3(arr, 0), 1)
    return np.clip(arr + amount * (arr - blur), 0.0, 1.0)


def source_to_grid(img, geom, fill=True, sharpen=0.0):
    """Fit a PIL RGB image to the mode aspect, then resize to the scan grid.
    Returns a float32 array shaped (lines, n_px, 3) in 0..1 where row i is
    scan line i in transmission order.  Images already at the exact grid
    size (e.g. machine-generated data frames) bypass fitting and resampling
    so they reach the encoder pixel-perfect.

    Resampled sources (photos, video, screen grabs) are downscaled in two
    stages: a Lanczos pre-shrink to a 4x supersampled grid, then a Lanczos
    finish in *linear light* so fine detail averages to the correct
    brightness instead of going muddy (gamma-space averaging darkens busy
    areas).  `sharpen` > 0 adds aperture correction afterwards."""
    img = img.convert("RGB")
    gw, gh = geom.grid_size()
    if img.size != (gw, gh):
        aw, ah = geom.aspect
        img = fit_to_aspect(img, aw, ah, fill)
        # stage 1: fast sRGB pre-shrink to a 4x supersampled grid
        ss = 4
        if img.size[0] > gw * ss or img.size[1] > gh * ss:
            img = img.resize((gw * ss, gh * ss), Image.LANCZOS)
        # stage 2: finish the resize per channel in linear light
        lin = _srgb_to_linear_lut()[np.asarray(img, dtype=np.uint8)]
        chans = [np.asarray(Image.fromarray(lin[..., c], mode="F")
                            .resize((gw, gh), Image.LANCZOS),
                            dtype=np.float32) for c in range(3)]
        arr = _linear_to_srgb(np.stack(chans, axis=-1))   # (gh, gw, 3)
        if sharpen > 0.0:
            arr = _aperture_sharpen(arr, float(sharpen))
        if geom.scan == 'V':
            lines = np.transpose(arr, (1, 0, 2))[:, ::-1, :]
        else:
            lines = arr
        return np.ascontiguousarray(lines)
    arr = np.asarray(img, dtype=np.float32) / 255.0   # (gh, gw, 3)
    if geom.scan == 'V':
        # line j is image column j scanned bottom-to-top
        lines = np.transpose(arr, (1, 0, 2))[:, ::-1, :]
    else:
        lines = arr
    return np.ascontiguousarray(lines)                # (lines, n_px, 3)


def grid_to_display(lines_arr, geom):
    """Inverse of source_to_grid orientation: (lines, n_px[,3]) -> display
    array with natural up/right orientation."""
    if geom.scan == 'V':
        if lines_arr.ndim == 3:
            return np.transpose(lines_arr[:, ::-1, :], (1, 0, 2))
        return lines_arr[:, ::-1].T
    return lines_arr


# ----------------------------------------------------------------------------
# Test pattern generator (drawn at a comfortable size, then fitted per mode)
# ----------------------------------------------------------------------------
def make_test_pattern(name, aspect, t=0.0):
    aw, ah = aspect
    H = 480
    W = int(round(H * aw / ah))
    img = Image.new("RGB", (W, H), "black")
    d = ImageDraw.Draw(img)
    if name == "Colour bars":
        cols = [(255, 255, 255), (255, 255, 0), (0, 255, 255), (0, 255, 0),
                (255, 0, 255), (255, 0, 0), (0, 0, 255), (0, 0, 0)]
        bw = W / len(cols)
        for i, c in enumerate(cols):
            d.rectangle([i * bw, 0, (i + 1) * bw, H * 0.75], fill=c)
        # lower quarter: -I/white/Q style strip simplified to grey steps
        for i in range(8):
            g = int(255 * i / 7)
            d.rectangle([i * bw, H * 0.75, (i + 1) * bw, H], fill=(g, g, g))
    elif name == "Grey staircase":
        steps = 10
        bw = W / steps
        for i in range(steps):
            g = int(255 * i / (steps - 1))
            d.rectangle([i * bw, 0, (i + 1) * bw, H], fill=(g, g, g))
    elif name == "Horizontal gradient":
        g = np.tile(np.linspace(0, 255, W, dtype=np.uint8), (H, 1))
        img = Image.fromarray(np.stack([g, g, g], axis=-1))
    elif name == "Crosshatch + circle test card":
        d.rectangle([0, 0, W, H], fill=(40, 40, 40))
        n = 8
        for i in range(n + 1):
            x = int(W * i / n)
            d.line([x, 0, x, H], fill="white", width=2)
        for j in range(n + 1):
            y = int(H * j / n)
            d.line([0, y, W, y], fill="white", width=2)
        r = int(min(W, H) * 0.45)
        d.ellipse([W // 2 - r, H // 2 - r, W // 2 + r, H // 2 + r],
                  outline="white", width=3)
        d.line([W // 2 - r, H // 2, W // 2 + r, H // 2], fill="white", width=2)
        d.line([W // 2, H // 2 - r, W // 2, H // 2 + r], fill="white", width=2)
        try:
            f = ImageFont.load_default()
            d.text((W // 2, H // 2 - r // 2), "NBTV", fill="white",
                   font=f, anchor="mm")
        except Exception:
            pass
    elif name == "Resolution wedges":
        d.rectangle([0, 0, W, H], fill="black")
        # vertical bars of increasing frequency (tests horizontal resolution)
        x = 10
        wbar = 24
        while x < W - 10 and wbar >= 1:
            for k in range(4):
                d.rectangle([x, 10, x + wbar, H // 2 - 10], fill="white")
                x += wbar * 2
            wbar = max(1, int(wbar * 0.6))
            x += 12
        # horizontal bars (tests line count)
        y = H // 2 + 10
        hbar = 24
        while y < H - 10 and hbar >= 1:
            for k in range(4):
                d.rectangle([10, y, W - 10, y + hbar], fill="white")
                y += hbar * 2
            hbar = max(1, int(hbar * 0.6))
            y += 12
    elif name == "Motion demo (bouncing block + clock)":
        d.rectangle([0, 0, W, H], fill=(20, 20, 60))
        for i in range(6):
            g = int(255 * i / 5)
            d.rectangle([i * W / 6, 0, (i + 1) * W / 6, H * 0.18],
                        fill=(g, g, g))
        # bouncing block
        px = 0.5 + 0.42 * math.sin(t * 2.0)
        py = 0.55 + 0.30 * abs(math.sin(t * 3.1))
        s = int(min(W, H) * 0.16)
        cx, cy = int(px * W), int(py * H)
        d.rectangle([cx - s, cy - s, cx + s, cy + s], fill=(255, 80, 80))
        d.ellipse([W * 0.1, H * 0.65, W * 0.1 + s * 1.6, H * 0.65 + s * 1.6],
                  fill=(80, 255, 120))
        try:
            f = ImageFont.load_default()
            d.text((W // 2, int(H * 0.32)), time.strftime("%H:%M:%S"),
                   fill="white", font=f, anchor="mm")
        except Exception:
            pass
    elif name == "Black & white split":
        d.rectangle([0, 0, W // 2, H], fill="white")
        d.rectangle([W // 2, 0, W, H], fill="black")
        d.ellipse([W * 0.25, H * 0.4, W * 0.45, H * 0.6], fill="black")
        d.ellipse([W * 0.55, H * 0.4, W * 0.75, H * 0.6], fill="white")
    else:
        d.rectangle([0, 0, W, H], fill=(128, 128, 128))
    return img


# ----------------------------------------------------------------------------
# Streaming FIR low-pass (windowed sinc), keeps state across blocks.
# ----------------------------------------------------------------------------
class FIRLowpass:
    def __init__(self, rate, cutoff, taps=127, channels=2):
        self.enabled = cutoff is not None and cutoff < rate * 0.49
        self.channels = channels
        if not self.enabled:
            self.h = None
            return
        n = np.arange(taps) - (taps - 1) / 2.0
        h = np.sinc(2.0 * cutoff / rate * n) * np.hamming(taps)
        self.h = (h / np.sum(h)).astype(np.float32)
        self.state = np.zeros((taps - 1, channels), dtype=np.float32)

    def process(self, block):
        """block: float32 (n, channels) -> filtered same shape."""
        if not self.enabled:
            return block
        x = np.vstack([self.state, block])
        out = np.empty_like(block)
        for c in range(self.channels):
            out[:, c] = np.convolve(x[:, c], self.h, mode="valid")
        self.state = x[-(len(self.h) - 1):, :]
        return out


# ----------------------------------------------------------------------------
# Thread-safe float32 ring buffer, shape (capacity, channels)
# ----------------------------------------------------------------------------
class Ring:
    def __init__(self, capacity, channels=2):
        self.buf = np.zeros((capacity, channels), dtype=np.float32)
        self.cap = capacity
        self.channels = channels
        self.r = 0
        self.w = 0
        self.count = 0
        self.lock = threading.Lock()
        self.underruns = 0
        self.overruns = 0

    def clear(self):
        with self.lock:
            self.r = self.w = self.count = 0
            self.underruns = self.overruns = 0

    def write(self, data):
        with self.lock:
            n = len(data)
            if n > self.cap - self.count:
                drop = n - (self.cap - self.count)
                self.r = (self.r + drop) % self.cap
                self.count -= drop
                self.overruns += 1
            first = min(n, self.cap - self.w)
            self.buf[self.w:self.w + first] = data[:first]
            if n > first:
                self.buf[:n - first] = data[first:]
            self.w = (self.w + n) % self.cap
            self.count += n

    def read(self, n):
        out = np.zeros((n, self.channels), dtype=np.float32)
        with self.lock:
            avail = min(n, self.count)
            if avail < n:
                self.underruns += 1
            first = min(avail, self.cap - self.r)
            out[:first] = self.buf[self.r:self.r + first]
            if avail > first:
                out[first:avail] = self.buf[:avail - first]
            self.r = (self.r + avail) % self.cap
            self.count -= avail
        return out

    def fill_fraction(self):
        with self.lock:
            return self.count / self.cap


# ----------------------------------------------------------------------------
# Colour helpers (BT.601-ish)
# ----------------------------------------------------------------------------
U_MAX = 0.886   # range of (B - Y) for RGB in 0..1
V_MAX = 0.701   # range of (R - Y)


def rgb_to_y(arr):
    return (0.299 * arr[..., 0] + 0.587 * arr[..., 1]
            + 0.114 * arr[..., 2]).astype(np.float32)


# ----------------------------------------------------------------------------
# Encoder: turns source frames into the audio signal.
# ----------------------------------------------------------------------------
class Encoder:
    """Stateful encoder. Call set_frame() with a PIL image whenever a new
    source frame is wanted, then encode_frame() to get one full NBTV frame of
    float32 audio shaped (n_samples, 2). Channel 0 carries the composite
    video; channel 1 duplicates it, except in Y/C mode where it carries
    chroma."""

    def __init__(self, geom, color_sys="mono", gain=0.9, fill=True,
                 sharpen=0.0):
        self.geom = geom
        self.color_sys = color_sys
        self.gain = float(gain)
        self.fill = fill
        self.sharpen = float(sharpen)
        self.phase = 0.0          # running fractional sample position
        self.frame_no = 0
        self.grid = None          # (lines, n_px, 3) float
        self.preview = None       # PIL image of what is being sent
        self._grid_src = None     # source object the current grid came from
        self._grid_params = None  # (fill, sharpen) the grid was built with

    def set_frame(self, pil_img):
        params = (self.fill, float(self.sharpen))
        if (self.grid is not None and pil_img is self._grid_src
                and params == self._grid_params):
            return                # still image unchanged: keep cached grid
        self.grid = source_to_grid(pil_img, self.geom, self.fill,
                                   self.sharpen)
        self._grid_src = pil_img
        self._grid_params = params

    # -- per-line signal builder ------------------------------------------
    def _line_signal(self, nsamp, row_vid, broad):
        g = self.geom
        sync_n = int(round(nsamp * (BROAD_F if broad else g.sync_f)))
        sync_n = clamp(sync_n, 1, nsamp - 2)
        a0 = int(round(nsamp * g.act_start_f))
        a1 = int(round(nsamp * g.act_end_f))
        a0 = clamp(max(a0, sync_n), 1, nsamp - 1)
        a1 = clamp(a1, a0 + 1, nsamp)
        line = np.full(nsamp, BLACK_LEVEL, dtype=np.float32)
        line[:sync_n] = 0.0
        if not broad and row_vid is not None:
            m = a1 - a0
            xp = np.arange(len(row_vid), dtype=np.float32)
            x = np.linspace(0.0, len(row_vid) - 1.0, m, dtype=np.float32)
            vid = np.interp(x, xp, row_vid).astype(np.float32)
            line[a0:a1] = BLACK_LEVEL + (1.0 - BLACK_LEVEL) * vid
        return line, a0, a1

    def _chroma_line(self, nsamp, a0, a1, row_c, cmax):
        """Chroma channel line: 0.5 baseline, active region carries
        0.5 + 0.5*C/cmax."""
        line = np.full(nsamp, 0.5, dtype=np.float32)
        if row_c is not None:
            m = a1 - a0
            xp = np.arange(len(row_c), dtype=np.float32)
            x = np.linspace(0.0, len(row_c) - 1.0, m, dtype=np.float32)
            c = np.interp(x, xp, row_c).astype(np.float32)
            line[a0:a1] = 0.5 + 0.5 * np.clip(c / cmax, -1.0, 1.0)
        return line

    def encode_frame(self):
        """Returns (audio float32 (n,2), preview PIL image)."""
        g = self.geom
        if self.grid is None:
            self.grid = np.zeros((g.lines, g.n_px, 3), dtype=np.float32)
        grid = self.grid
        cs = self.color_sys
        field = self.frame_no % 3

        if cs == "mono":
            vid_rows = rgb_to_y(grid)
            prev = vid_rows
        elif cs == "fsc":
            vid_rows = grid[..., field]
            prev = None  # preview handled below
        elif cs == "lsc":
            planes = (np.arange(g.lines) + field) % 3
            vid_rows = grid[np.arange(g.lines), :, planes]
            prev = None
        elif cs == "yc":
            y = rgb_to_y(grid)
            u = grid[..., 2] - y          # B - Y
            v = grid[..., 0] - y          # R - Y
            vid_rows = y
            prev = y
        else:
            raise ValueError("unknown colour system " + cs)

        chunks = []
        cchunks = []
        for li in range(g.lines):
            start = self.phase
            self.phase += g.spl
            nsamp = int(round(self.phase)) - int(round(start))
            if cs in ("fsc", "lsc"):
                broad = (li == 0) or (li == 1 and field == 0)
            else:
                broad = (li == 0)
            row = None if broad else np.ascontiguousarray(vid_rows[li])
            line, a0, a1 = self._line_signal(nsamp, row, broad)
            chunks.append(line)
            if cs == "yc":
                if broad:
                    cchunks.append(np.full(nsamp, 0.5, dtype=np.float32))
                else:
                    comp = u[li] if (li % 2 == 0) else v[li]
                    cmax = U_MAX if (li % 2 == 0) else V_MAX
                    cchunks.append(self._chroma_line(nsamp, a0, a1,
                                                     comp, cmax))
        comp_sig = np.concatenate(chunks)
        audio = np.empty((len(comp_sig), 2), dtype=np.float32)
        audio[:, 0] = (comp_sig * 2.0 - 1.0) * self.gain
        if cs == "yc":
            csig = np.concatenate(cchunks)
            audio[:, 1] = (csig * 2.0 - 1.0) * self.gain
        else:
            audio[:, 1] = audio[:, 0]

        # preview image (what an ideal receiver would show for this frame)
        if cs in ("mono", "yc"):
            disp = grid_to_display(prev, g)
            pim = Image.fromarray((np.clip(disp, 0, 1) * 255).astype(np.uint8))
        else:
            disp = grid_to_display(grid, g)
            pim = Image.fromarray((np.clip(disp, 0, 1) * 255).astype(np.uint8))
        self.preview = pim
        self.frame_no += 1
        # keep phase from growing unbounded
        if self.phase > 1e9:
            self.phase -= int(self.phase)
        return audio, pim


# ----------------------------------------------------------------------------
# Decoder: turns incoming audio back into pictures.
# Edge-driven flywheel sync: falling edges below the sync threshold start
# lines; pulse width separates line syncs from broad frame syncs; missing
# pulses are coasted through at the estimated line period.
# ----------------------------------------------------------------------------
class Decoder:
    def __init__(self, geom, color_sys="mono"):
        self.geom = geom
        self.color_sys = color_sys
        g = geom
        self.buf = np.zeros(0, dtype=np.float32)
        self.cbuf = np.zeros(0, dtype=np.float32)
        self.base = 0                      # absolute index of buf[0]
        self.lo = None
        self.hi = None
        self.sync_level = 0.15             # threshold in normalized units
        self.saturation = 1.0              # Y/C chroma gain
        self.spl_nom = g.spl
        self.spl_est = g.spl
        self.locked = False
        self.synced = False                # have we seen a frame pulse
        self.cur_start = 0.0
        self.cur_broad = False
        self.last_was_broad = False
        self.line_idx = 0
        self.last_handled_f = -1           # absolute idx of last handled edge
        # frame stores
        L, P = g.lines, g.n_px
        self.yb = np.zeros((L, P), dtype=np.float32)
        self.rgbb = np.zeros((L, P, 3), dtype=np.float32)
        self.urows = np.zeros((L, P), dtype=np.float32)
        self.vrows = np.zeros((L, P), dtype=np.float32)
        self.umask = np.zeros(L, dtype=bool)
        self.vmask = np.zeros(L, dtype=bool)
        # colour-phase tracking (fsc field / lsc plane rotation)
        self.cur_phase = 0
        self.phase_anchored = False
        # stats
        self.lines_count = 0
        self.frames_count = 0
        self.coasted = 0
        self._out = []
        self._n01 = None
        self._c01 = None

    # ------------------------------------------------------------------
    def set_sync_level(self, v):
        self.sync_level = clamp(float(v), 0.03, 0.45)

    def set_saturation(self, v):
        self.saturation = clamp(float(v), 0.0, 3.0)

    # ------------------------------------------------------------------
    def feed(self, block):
        """block: float32 (n,) or (n, ch). Returns list of
        (display_array, info) tuples for each completed frame."""
        if block.ndim == 1:
            comp = block.astype(np.float32)
            chrm = comp
        else:
            comp = np.ascontiguousarray(block[:, 0], dtype=np.float32)
            chrm = np.ascontiguousarray(
                block[:, 1] if block.shape[1] > 1 else block[:, 0],
                dtype=np.float32)
        if len(comp) == 0:
            return []
        self.buf = np.concatenate([self.buf, comp])
        self.cbuf = np.concatenate([self.cbuf, chrm])

        # --- normalization tracking -----------------------------------
        bm = float(comp.min())
        bM = float(comp.max())
        if self.lo is None:
            self.lo, self.hi = bm, bM
        else:
            span = max(self.hi - self.lo, 1e-6)
            self.lo = bm if bm < self.lo else self.lo + 0.02 * (bm - self.lo)
            self.hi = bM if bM > self.hi else self.hi - 0.0008 * span
        span = max(self.hi - self.lo, 1e-6)
        self._n01 = (self.buf - self.lo) / span
        self._c01 = (self.cbuf - self.lo) / span

        # --- find sync edges -------------------------------------------
        mask = self._n01 < self.sync_level
        m = mask.view(np.int8)
        d = np.diff(m)
        fall = np.nonzero(d == 1)[0] + 1     # index where low region begins
        rise = np.nonzero(d == -1)[0] + 1    # index where it ends
        self._out = []
        unmatched_fall_abs = None
        for fi in fall:
            f_abs = self.base + int(fi)
            if f_abs <= self.last_handled_f:
                continue
            j = np.searchsorted(rise, fi, side="right")
            if j >= len(rise):
                unmatched_fall_abs = f_abs
                break
            width = int(rise[j]) - int(fi)
            self.last_handled_f = f_abs
            if width < max(2, 0.03 * self.spl_est):
                continue                      # noise blip
            broad = width >= 0.30 * self.spl_est
            self._handle_sync(float(f_abs), broad)

        # --- trim buffer ------------------------------------------------
        keep_abs = self.base + len(self.buf) - 8
        if self.locked:
            keep_abs = min(keep_abs, int(self.cur_start) - 8)
        if unmatched_fall_abs is not None:
            keep_abs = min(keep_abs, unmatched_fall_abs - 8)
        keep_abs = max(keep_abs, self.base)
        cut = keep_abs - self.base
        if cut > 0:
            self.buf = self.buf[cut:]
            self.cbuf = self.cbuf[cut:]
            self.base = keep_abs
        # cap runaway buffer (no syncs found in garbage input)
        maxlen = int(self.spl_nom * (self.geom.lines + 8))
        if len(self.buf) > maxlen:
            cut = len(self.buf) - maxlen
            self.buf = self.buf[cut:]
            self.cbuf = self.cbuf[cut:]
            self.base += cut
            if self.locked and self.cur_start < self.base:
                self.locked = False
        return self._out

    # ------------------------------------------------------------------
    def _handle_sync(self, e_abs, broad):
        if not self.locked:
            self.locked = True
            self.cur_start = e_abs
            self.cur_broad = broad
            self.last_was_broad = broad
            self.line_idx = 0
            if broad:
                self._frame_begin(anchor_possible=True)
            return
        gap = e_abs - self.cur_start
        if gap < 0.45 * self.spl_est:
            return                            # too soon - ignore
        coasted = False
        while gap > 1.6 * self.spl_est:
            self._emit_line(self.cur_start, self.cur_start + self.spl_est)
            self.cur_start += self.spl_est
            gap = e_abs - self.cur_start
            self.coasted += 1
            coasted = True
        prev_broad = self.cur_broad and not coasted
        self._emit_line(self.cur_start, e_abs)
        if 0.8 * self.spl_est < gap < 1.2 * self.spl_est:
            self.spl_est += 0.03 * (gap - self.spl_est)
            limit = 0.06 * self.spl_nom
            self.spl_est = clamp(self.spl_est, self.spl_nom - limit,
                                 self.spl_nom + limit)
        # the new line beginning at e_abs
        if broad and not prev_broad:
            self._finish_frame()
            self.line_idx = 0
            self._frame_begin(anchor_possible=True)
            self.synced = True
        elif broad and prev_broad:
            # second consecutive broad pulse: colour phase anchor
            self.cur_phase = 0
            self.phase_anchored = True
        self.cur_start = e_abs
        self.cur_broad = broad
        self.last_was_broad = broad

    def _frame_begin(self, anchor_possible):
        # provisional colour phase: advance from previous frame; a second
        # broad pulse (handled above) will pin it to 0.
        if self.frames_count > 0 or not anchor_possible:
            self.cur_phase = (self.cur_phase + 1) % 3
        else:
            self.cur_phase = 0
        # phase_anchored persists once any anchor has been seen
        return

    # ------------------------------------------------------------------
    def _emit_line(self, s, e):
        g = self.geom
        if not self.cur_broad:
            L = e - s
            i0 = int(round(s - self.base + L * g.act_start_f))
            i1 = int(round(s - self.base + L * g.act_end_f))
            i0 = clamp(i0, 0, len(self.buf) - 2)
            i1 = clamp(i1, i0 + 2, len(self.buf))
            # porch window: just after the sync pulse, before active video
            p0 = int(round(s - self.base + L * g.sync_f * 1.05))
            p1 = int(round(s - self.base + L * g.act_start_f * 0.98))
            p0 = clamp(p0, 0, len(self.buf) - 2)
            p1 = clamp(p1, p0 + 1, len(self.buf))
            porch = float(np.mean(self._n01[p0:p1]))
            seg = self._n01[i0:i1]
            # per-line black clamp removes AC-coupling droop
            vid = (seg - porch) / (1.0 - BLACK_LEVEL)
            x = np.linspace(0.0, len(vid) - 1.0, g.n_px, dtype=np.float32)
            row = np.interp(x, np.arange(len(vid), dtype=np.float32),
                            vid).astype(np.float32)
            row = np.clip(row, 0.0, 1.0)
            li = self.line_idx % g.lines
            cs = self.color_sys
            if cs == "mono" or cs == "fsc" or cs == "yc":
                self.yb[li] = row
            if cs == "lsc":
                plane = (li + self.cur_phase) % 3
                self.rgbb[li, :, plane] = row
                self.yb[li] = row
            if cs == "yc":
                cref = float(np.mean(self._c01[p0:p1]))
                cseg = self._c01[i0:i1] - cref
                crow = np.interp(x, np.arange(len(cseg), dtype=np.float32),
                                 cseg).astype(np.float32)
                if li % 2 == 0:
                    self.urows[li] = crow * 2.0 * U_MAX * self.saturation
                    self.umask[li] = True
                else:
                    self.vrows[li] = crow * 2.0 * V_MAX * self.saturation
                    self.vmask[li] = True
        self.line_idx += 1
        self.lines_count += 1
        self.cur_broad = False
        if self.line_idx >= g.lines and not self.synced:
            # free-running frame wrap when no frame pulses are seen
            self._finish_frame()
            self.line_idx = 0
            self._frame_begin(anchor_possible=False)
        elif self.line_idx >= 2 * g.lines:
            # frame pulses lost mid-stream: keep the display alive anyway
            self._finish_frame()
            self.line_idx = 0
            self._frame_begin(anchor_possible=False)

    # ------------------------------------------------------------------
    def _fill_rows(self, rows, mask):
        """Nearest-row fill for the chroma lines that this parity skipped."""
        out = rows.copy()
        idx = np.nonzero(mask)[0]
        if len(idx) == 0:
            return out
        all_i = np.arange(len(rows))
        nearest = idx[np.clip(np.searchsorted(idx, all_i), 0, len(idx) - 1)]
        prev = idx[np.clip(np.searchsorted(idx, all_i) - 1, 0, len(idx) - 1)]
        use_prev = np.abs(prev - all_i) < np.abs(nearest - all_i)
        pick = np.where(use_prev, prev, nearest)
        return rows[pick]

    def _finish_frame(self):
        g = self.geom
        cs = self.color_sys
        if cs == "mono":
            frame = self.yb.copy()
        elif cs == "fsc":
            self.rgbb[..., self.cur_phase] = self.yb
            frame = self.rgbb.copy()
        elif cs == "lsc":
            frame = self.rgbb.copy()
        elif cs == "yc":
            u = self._fill_rows(self.urows, self.umask)
            v = self._fill_rows(self.vrows, self.vmask)
            y = self.yb
            r = y + v
            b = y + u
            gch = (y - 0.299 * r - 0.114 * b) / 0.587
            frame = np.clip(np.stack([r, gch, b], axis=-1), 0.0, 1.0)
        else:
            frame = self.yb.copy()
        disp = grid_to_display(np.clip(frame, 0.0, 1.0), g)
        self.frames_count += 1
        info = dict(frame=self.frames_count, lines=self.lines_count,
                    locked=self.locked, synced=self.synced,
                    spl=self.spl_est, coasted=self.coasted)
        self._out.append((disp, info))


# ----------------------------------------------------------------------------
# WAV helpers (16 / 24 / 32-bit PCM and 32-bit float)
# ----------------------------------------------------------------------------
class WavWriter:
    def __init__(self, path, rate, channels=2, bits=24):
        self.bits = bits
        self.channels = channels
        self.wf = wave.open(path, "wb")
        self.wf.setnchannels(channels)
        self.wf.setsampwidth(bits // 8)
        self.wf.setframerate(int(rate))
        self.lock = threading.Lock()

    def write(self, data):
        """data: float32 (n, channels) in -1..1"""
        a = np.clip(data, -1.0, 1.0)
        if self.bits == 16:
            pcm = (a * 32767.0).astype("<i2").tobytes()
        elif self.bits == 24:
            v = (a * 8388607.0).astype("<i4")
            b = v.astype("<i4").tobytes()
            arr = np.frombuffer(b, dtype=np.uint8).reshape(-1, 4)
            pcm = np.ascontiguousarray(arr[:, :3]).tobytes()
        else:
            pcm = (a * 2147483647.0).astype("<i4").tobytes()
        with self.lock:
            self.wf.writeframes(pcm)

    def close(self):
        with self.lock:
            try:
                self.wf.close()
            except Exception:
                pass


def read_wav(path):
    """Returns (rate, float32 array (n, channels))."""
    wf = wave.open(path, "rb")
    rate = wf.getframerate()
    ch = wf.getnchannels()
    sw = wf.getsampwidth()
    raw = wf.readframes(wf.getnframes())
    wf.close()
    if sw == 1:
        a = (np.frombuffer(raw, dtype=np.uint8).astype(np.float32)
             - 128.0) / 128.0
    elif sw == 2:
        a = np.frombuffer(raw, dtype="<i2").astype(np.float32) / 32768.0
    elif sw == 3:
        b = np.frombuffer(raw, dtype=np.uint8).reshape(-1, 3)
        v = (b[:, 0].astype(np.int32)
             | (b[:, 1].astype(np.int32) << 8)
             | (b[:, 2].astype(np.int32) << 16))
        v = np.where(v >= (1 << 23), v - (1 << 24), v)
        a = v.astype(np.float32) / 8388608.0
    elif sw == 4:
        a = np.frombuffer(raw, dtype="<i4").astype(np.float32) / 2147483648.0
    else:
        raise ValueError("unsupported WAV sample width: %d" % sw)
    a = a.reshape(-1, ch)
    return rate, a


# ----------------------------------------------------------------------------
# Video sources
# ----------------------------------------------------------------------------
class SourceManager:
    """Provides PIL RGB frames from: test patterns, still images, animated
    GIFs, video files (OpenCV), a webcam (OpenCV), or screen capture."""

    def __init__(self):
        self.kind = "test"
        self.pattern = TEST_PATTERNS[0]
        self.image = None
        self.gif_frames = []
        self.gif_times = []
        self.gif_total = 0.0
        self.cap = None              # cv2 capture (video file / webcam)
        self.cap_fps = 25.0
        self.cap_frames = 0
        self.cap_pos = -1
        self.bbox = None             # screen capture region (l, t, r, b)
        self.ft_frames = []          # pre-rendered file-transfer frames
        self.ft_period = 0.2         # seconds each file frame stays up
        self.ft_sig = None           # geometry signature the frames were
                                     # built for (see App.build_filetx)
        self.lock = threading.Lock()
        self.error = None

    # -- loaders -----------------------------------------------------------
    def use_test(self, pattern):
        with self.lock:
            self._release()
            self.kind = "test"
            self.pattern = pattern

    def use_image(self, path):
        img = Image.open(path)
        img.load()
        with self.lock:
            self._release()
            self.kind = "image"
            self.image = img.convert("RGB")

    def use_gif(self, path):
        im = Image.open(path)
        frames, times = [], []
        t = 0.0
        try:
            i = 0
            while True:
                im.seek(i)
                fr = im.convert("RGB").copy()
                dur = im.info.get("duration", 100) / 1000.0
                dur = max(dur, 0.02)
                frames.append(fr)
                times.append(t)
                t += dur
                i += 1
        except EOFError:
            pass
        if not frames:
            raise ValueError("no frames in GIF")
        with self.lock:
            self._release()
            self.kind = "gif"
            self.gif_frames = frames
            self.gif_times = times
            self.gif_total = t

    def use_video(self, path):
        import cv2
        cap = cv2.VideoCapture(path)
        if not cap.isOpened():
            raise ValueError("could not open video: " + path)
        with self.lock:
            self._release()
            self.kind = "video"
            self.cap = cap
            self.cap_fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
            if not (1.0 <= self.cap_fps <= 240.0):
                self.cap_fps = 25.0
            self.cap_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
            self.cap_pos = -1

    def use_webcam(self, index=0):
        import cv2
        cap = cv2.VideoCapture(int(index))
        if not cap.isOpened():
            raise ValueError("could not open webcam %d" % index)
        with self.lock:
            self._release()
            self.kind = "webcam"
            self.cap = cap

    def use_capture(self, bbox):
        with self.lock:
            self._release()
            self.kind = "capture"
            self.bbox = tuple(int(v) for v in bbox)

    def use_file_frames(self, frames, period, sig):
        """Pre-rendered file-transfer (QR) frames, cycled every `period` s."""
        with self.lock:
            self._release()
            self.kind = "filetx"
            self.ft_frames = list(frames)
            self.ft_period = max(float(period), 0.01)
            self.ft_sig = sig

    def _release(self):
        self.ft_frames = []
        self.ft_sig = None
        if self.cap is not None:
            try:
                self.cap.release()
            except Exception:
                pass
            self.cap = None

    def close(self):
        with self.lock:
            self._release()

    # -- frame fetch ---------------------------------------------------------
    def get_frame(self, t, aspect):
        with self.lock:
            try:
                if self.kind == "test":
                    return make_test_pattern(self.pattern, aspect, t)
                if self.kind == "image" and self.image is not None:
                    return self.image
                if self.kind == "gif" and self.gif_frames:
                    tt = t % self.gif_total
                    idx = np.searchsorted(self.gif_times, tt, side="right") - 1
                    return self.gif_frames[int(clamp(idx, 0,
                                            len(self.gif_frames) - 1))]
                if self.kind == "filetx" and self.ft_frames:
                    idx = int(t / self.ft_period) % len(self.ft_frames)
                    return self.ft_frames[idx]
                if self.kind == "video" and self.cap is not None:
                    return self._video_frame(t)
                if self.kind == "webcam" and self.cap is not None:
                    ok, fr = self.cap.read()
                    if ok:
                        import cv2
                        return Image.fromarray(
                            cv2.cvtColor(fr, cv2.COLOR_BGR2RGB))
                if self.kind == "capture" and self.bbox is not None:
                    return self._grab_screen()
            except Exception as e:
                self.error = str(e)
        # fallback: black
        aw, ah = aspect
        return Image.new("RGB", (aw * 80, ah * 80), "black")

    def _video_frame(self, t):
        import cv2
        target = int(t * self.cap_fps)
        if self.cap_frames > 0:
            target = target % max(self.cap_frames, 1)
        if target < self.cap_pos:
            self.cap.set(cv2.CAP_PROP_POS_FRAMES, target)
            self.cap_pos = target - 1
        fr = None
        guard = 0
        while self.cap_pos < target and guard < 300:
            ok, fr2 = self.cap.read()
            if not ok:
                self.cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                self.cap_pos = -1
                target = 0
                guard += 1
                continue
            fr = fr2
            self.cap_pos += 1
            guard += 1
        if fr is None:
            ok, fr = self.cap.read()
            if not ok:
                raise ValueError("video read failed")
            self.cap_pos += 1
        return Image.fromarray(cv2.cvtColor(fr, cv2.COLOR_BGR2RGB))

    def _grab_screen(self):
        l, t_, r, b = self.bbox
        try:
            import mss
            with mss.mss() as sct:
                shot = sct.grab({"left": l, "top": t_,
                                 "width": r - l, "height": b - t_})
                return Image.frombytes("RGB", shot.size, shot.bgra,
                                       "raw", "BGRX")
        except Exception:
            from PIL import ImageGrab
            return ImageGrab.grab(bbox=(l, t_, r, b)).convert("RGB")


# ----------------------------------------------------------------------------
# Ring extension: read only what is available (no zero padding)
# ----------------------------------------------------------------------------
def _ring_read_upto(self, n):
    with self.lock:
        avail = min(n, self.count)
        if avail == 0:
            return np.zeros((0, self.channels), dtype=np.float32)
        out = np.empty((avail, self.channels), dtype=np.float32)
        first = min(avail, self.cap - self.r)
        out[:first] = self.buf[self.r:self.r + first]
        if avail > first:
            out[first:] = self.buf[:avail - first]
        self.r = (self.r + avail) % self.cap
        self.count -= avail
        return out


Ring.read_upto = _ring_read_upto


# ----------------------------------------------------------------------------
# TX engine thread
# ----------------------------------------------------------------------------
class TXEngine(threading.Thread):
    def __init__(self, geom, color_sys, gain, lpf_cutoff, source, fill,
                 tx_ring=None, loop_ring=None, wav_writer=None,
                 preview_q=None, status_cb=None, sharpen=0.0):
        super().__init__(daemon=True)
        self.geom = geom
        self.enc = Encoder(geom, color_sys, gain, fill, sharpen)
        self.lpf = FIRLowpass(geom.rate, lpf_cutoff, channels=2)
        self.source = source
        self.tx_ring = tx_ring
        self.loop_ring = loop_ring
        self.wav = wav_writer
        self.preview_q = preview_q
        self.status_cb = status_cb
        self.stop_evt = threading.Event()
        self.frames_sent = 0

    def stop(self):
        self.stop_evt.set()

    def run(self):
        g = self.geom
        fdur = 1.0 / g.fps
        t0 = time.monotonic()
        next_t = t0
        while not self.stop_evt.is_set():
            tsrc = time.monotonic() - t0
            img = self.source.get_frame(tsrc, g.aspect)
            self.enc.set_frame(img)
            audio, prev = self.enc.encode_frame()
            audio = self.lpf.process(audio)
            if self.tx_ring is not None:
                self.tx_ring.write(audio)
            if self.loop_ring is not None:
                self.loop_ring.write(audio)
            if self.wav is not None:
                self.wav.write(audio)
            if self.preview_q is not None:
                try:
                    self.preview_q.put_nowait(prev)
                except queue.Full:
                    pass
            self.frames_sent += 1
            next_t += fdur
            delay = next_t - time.monotonic()
            if delay > 0:
                if self.stop_evt.wait(delay):
                    break
            elif delay < -2.0:
                next_t = time.monotonic()   # fell behind badly: resync clock
        if self.wav is not None:
            self.wav.close()


def render_wav_offline(geom, color_sys, gain, lpf_cutoff, source, fill,
                       path, seconds, bits=24, progress_cb=None,
                       stop_evt=None, sharpen=0.0):
    """Render `seconds` of signal straight to a WAV file (faster than real
    time; no audio hardware needed)."""
    enc = Encoder(geom, color_sys, gain, fill, sharpen)
    lpf = FIRLowpass(geom.rate, lpf_cutoff, channels=2)
    wav = WavWriter(path, geom.rate, 2, bits)
    nframes = max(1, int(math.ceil(seconds * geom.fps)))
    try:
        for i in range(nframes):
            if stop_evt is not None and stop_evt.is_set():
                break
            t = i / geom.fps
            enc.set_frame(source.get_frame(t, geom.aspect))
            audio, _ = enc.encode_frame()
            wav.write(lpf.process(audio))
            if progress_cb and (i % 5 == 0 or i == nframes - 1):
                progress_cb((i + 1) / nframes)
    finally:
        wav.close()
    return nframes


# ----------------------------------------------------------------------------
# RX engine thread
# ----------------------------------------------------------------------------
class RXEngine(threading.Thread):
    def __init__(self, geom, color_sys, rx_ring, frame_q, status_cb=None):
        super().__init__(daemon=True)
        self.geom = geom
        self.dec = Decoder(geom, color_sys)
        self.rx_ring = rx_ring
        self.frame_q = frame_q
        self.status_cb = status_cb
        self.stop_evt = threading.Event()
        self.integrate = 1
        self._acc = None
        self.last_rms = 0.0

    def stop(self):
        self.stop_evt.set()

    def set_integrate(self, n):
        self.integrate = int(clamp(n, 1, 64))

    def run(self):
        while not self.stop_evt.is_set():
            data = self.rx_ring.read_upto(8192)
            if len(data) == 0:
                time.sleep(0.01)
                continue
            self.last_rms = float(np.sqrt(np.mean(data[:, 0] ** 2)))
            try:
                frames = self.dec.feed(data)
            except Exception as e:
                if self.status_cb:
                    self.status_cb("RX decode error: %s" % e)
                frames = []
            for disp, info in frames:
                if self.integrate > 1:
                    a = 1.0 / self.integrate
                    if self._acc is None or self._acc.shape != disp.shape:
                        self._acc = disp.astype(np.float32).copy()
                    else:
                        self._acc = (1.0 - a) * self._acc + a * disp
                    out = self._acc
                else:
                    self._acc = None
                    out = disp
                try:
                    self.frame_q.put_nowait((out.copy(), info))
                except queue.Full:
                    try:
                        self.frame_q.get_nowait()
                        self.frame_q.put_nowait((out.copy(), info))
                    except Exception:
                        pass


class WavFeeder(threading.Thread):
    """Streams a WAV file into the RX ring at a selectable speed."""

    def __init__(self, path, rx_ring, speed=4.0, status_cb=None):
        super().__init__(daemon=True)
        self.path = path
        self.rx_ring = rx_ring
        self.speed = speed
        self.status_cb = status_cb
        self.stop_evt = threading.Event()
        self.rate = None

    def stop(self):
        self.stop_evt.set()

    def run(self):
        try:
            rate, data = read_wav(self.path)
        except Exception as e:
            if self.status_cb:
                self.status_cb("WAV read failed: %s" % e)
            return
        self.rate = rate
        if data.shape[1] == 1:
            data = np.repeat(data, 2, axis=1)
        elif data.shape[1] > 2:
            data = data[:, :2]
        chunk = 8192
        n = len(data)
        i = 0
        while i < n and not self.stop_evt.is_set():
            j = min(i + chunk, n)
            self.rx_ring.write(np.ascontiguousarray(data[i:j]))
            if self.speed > 0:
                time.sleep((j - i) / rate / self.speed)
            else:
                while (self.rx_ring.fill_fraction() > 0.5
                       and not self.stop_evt.is_set()):
                    time.sleep(0.005)
            i = j
        if self.status_cb:
            self.status_cb("WAV playback finished (%s)"
                           % os.path.basename(self.path))


# ----------------------------------------------------------------------------
# File-over-NBTV: stream a small file as a carousel of QR-coded video frames.
#
# Each frame is one QR code carrying either file metadata or a chunk of the
# file (base64).  The transmitter cycles through all chunks repeatedly; the
# receiver collects them in any order until the set is complete, then checks
# the CRC and writes the file.  QR error correction plus the carousel makes
# the link tolerant of noise and dropped frames with no back-channel at all.
#
# Frame payloads (ASCII, fits QR byte mode):
#   'M' + b36(nchunks,3) + b36(filesize,6) + b36(crc32,7)     metadata, 17 ch
#   'N' + base64(filename)                                    optional name
#   'D' + b36(seq,3) + base64(chunk)                          data, seq 1..n
# ----------------------------------------------------------------------------
_CV2_UNSET = object()
_CV2 = _CV2_UNSET


def get_cv2():
    """Lazy import of OpenCV; returns module or None."""
    global _CV2
    if _CV2 is _CV2_UNSET:
        try:
            import cv2
            _CV2 = cv2
        except Exception:
            _CV2 = None
    return _CV2


_B36 = "0123456789abcdefghijklmnopqrstuvwxyz"


def b36e(n, width):
    n = int(n)
    s = ""
    for _ in range(width):
        s = _B36[n % 36] + s
        n //= 36
    return s


def b36d(s):
    v = 0
    for c in s:
        v = v * 36 + _B36.index(c)
    return v


QR_SKIP_LINES = 2        # scan lines 0..1 may carry broad sync: keep clear
QR_FILTER_RATIO = 0.55   # along-scan module width vs filter blur, minimum

QR_FINDER = np.array([[1, 1, 1, 1, 1, 1, 1],
                      [1, 0, 0, 0, 0, 0, 1],
                      [1, 0, 1, 1, 1, 0, 1],
                      [1, 0, 1, 1, 1, 0, 1],
                      [1, 0, 1, 1, 1, 0, 1],
                      [1, 0, 0, 0, 0, 0, 1],
                      [1, 1, 1, 1, 1, 1, 1]], dtype=np.float32)

_QR_CAP_CACHE = {}


def qr_encoder(version, ec):
    cv2 = get_cv2()
    p = cv2.QRCodeEncoder_Params()
    p.version = int(version)
    p.correction_level = int(ec)        # 0=L 1=M 2=Q 3=H
    return cv2.QRCodeEncoder_create(p)


def qr_capacity(version, ec):
    """Byte-mode capacity, measured empirically (cached)."""
    key = (version, ec)
    if key in _QR_CAP_CACHE:
        return _QR_CAP_CACHE[key]
    cv2 = get_cv2()
    enc = qr_encoder(version, ec)
    want = 17 + 4 * version + 4         # modules incl. 2-module quiet zone
    lo, hi, best = 1, 3000, 0
    while lo <= hi:
        mid = (lo + hi) // 2
        try:
            img = enc.encode("a" * mid)
            if img.shape[0] == want:
                best, lo = mid, mid + 1
            else:
                hi = mid - 1
        except cv2.error:
            hi = mid - 1
    _QR_CAP_CACHE[key] = best
    return best


def _qr_fit(geom, need):
    """Module sizes and placement for a `need`-module QR in this geometry.
    Returns (ph, pv, x0, y0) in frame-array coords, or None if it can't fit.
    Frame array is (rows, cols); along-scan axis is rows for V, cols for H.
    The first QR_SKIP_LINES scan lines are kept clear of the code."""
    gw, gh = geom.grid_size()           # PIL (W, H); array is (gh, gw)
    if geom.scan == 'V':
        along_total, cross_total = gh, gw - QR_SKIP_LINES
    else:
        along_total, cross_total = gw, gh - QR_SKIP_LINES
    p_along = along_total // need
    p_cross = cross_total // need
    if p_along < 1 or p_cross < 1:
        return None
    along0 = (along_total - need * p_along) // 2
    cross0 = QR_SKIP_LINES + (cross_total - need * p_cross) // 2
    if geom.scan == 'V':
        pv, ph, y0, x0 = p_along, p_cross, along0, cross0
    else:
        ph, pv, x0, y0 = p_along, p_cross, along0, cross0
    return ph, pv, x0, y0, p_along, p_cross


def qr_plan(geom, cutoff, ec):
    """Pick the largest QR version this mode can carry.  Returns
    (plan_dict, None) or (None, reason)."""
    if get_cv2() is None:
        return None, ("File transfer needs OpenCV:  pip install "
                      "opencv-python")
    spp = geom.spl * geom.active_f / geom.n_px   # samples per px along line
    blur = (geom.rate / cutoff) if cutoff else 0.0
    for trim in (0, 1, 2):
        for version in range(40, 0, -1):
            need = 17 + 4 * version + 4 - 2 * trim
            fit = _qr_fit(geom, need)
            if fit is None:
                continue
            ph, pv, x0, y0, p_along, p_cross = fit
            if blur:
                if p_along < 2 or p_along * spp < QR_FILTER_RATIO * blur:
                    continue
            cap = qr_capacity(version, ec)
            if cap < 17:
                continue
            marginal = []
            if p_cross == 1:
                marginal.append("1 scan line per module")
            if not blur and p_along == 1:
                marginal.append("1 sample per module")
            return dict(version=version, trim=trim, need=need, ph=ph,
                        pv=pv, x0=x0, y0=y0, cap=cap, p_along=p_along,
                        p_cross=p_cross,
                        marginal=", ".join(marginal)), None
    if blur:
        return None, ("This mode + filter cannot carry a QR code: the "
                      "low-pass smears modules along the scan line.  Use "
                      "'Direct cable', a wider filter, or a mode with "
                      "fewer lines.")
    return None, "Mode too small for even the smallest QR code."


def qr_render_frame(geom, plan, ec, payload):
    """One file-transfer frame as a PIL RGB image at exact grid size."""
    qr = qr_encoder(plan["version"], ec).encode(payload)
    t = plan["trim"]
    if t:
        qr = qr[t:-t, t:-t]
    big = np.kron(qr, np.ones((plan["pv"], plan["ph"]), np.uint8))
    gw, gh = geom.grid_size()
    frame = np.full((gh, gw), 255, np.uint8)
    y0, x0 = plan["y0"], plan["x0"]
    frame[y0:y0 + big.shape[0], x0:x0 + big.shape[1]] = big
    return Image.fromarray(frame, "L").convert("RGB"), qr


class FileSender:
    """Holds a file and renders its QR carousel for a given geometry."""

    MAX_CHUNKS = 36 ** 3 - 1

    def __init__(self):
        self.path = None
        self.name = ""
        self.data = b""
        self.frames = []
        self.stats = None

    def load(self, path):
        with open(path, "rb") as f:
            self.data = f.read()
        self.path = path
        self.name = os.path.basename(path)
        self.frames = []
        self.stats = None

    def build(self, geom, cutoff, ec, repeat):
        """Render all frames.  Returns stats dict; raises ValueError with a
        human-readable reason on failure."""
        if not self.data:
            raise ValueError("No file loaded (or the file is empty).")
        plan, err = qr_plan(geom, cutoff, ec)
        if plan is None:
            raise ValueError(err)
        raw = ((plan["cap"] - 4) // 4) * 3
        if raw < 1:
            raise ValueError("QR too small for any payload in this mode.")
        n = (len(self.data) + raw - 1) // raw
        if n > self.MAX_CHUNKS:
            raise ValueError("File too big for this mode: needs %d chunks "
                             "(max %d).  Use a bigger mode or a smaller "
                             "file." % (n, self.MAX_CHUNKS))
        crc = zlib.crc32(self.data) & 0xffffffff
        payloads = ["M" + b36e(n, 3) + b36e(len(self.data), 6)
                    + b36e(crc, 7)]
        nb64 = base64.b64encode(self.name.encode("utf-8",
                                                 "ignore")).decode()
        if plan["cap"] >= 1 + len(nb64):
            payloads.append("N" + nb64)
        for i in range(n):
            chunk = self.data[i * raw:(i + 1) * raw]
            payloads.append("D" + b36e(i + 1, 3)
                            + base64.b64encode(chunk).decode())
        frames = []
        for p in payloads:
            img, _ = qr_render_frame(geom, plan, ec, p)
            frames.append(img)
        self.frames = frames
        period = repeat / geom.fps
        pass_s = len(frames) * period
        self.stats = dict(plan=plan, chunks=n, raw=raw,
                          total_frames=len(frames), period=period,
                          pass_seconds=pass_s,
                          rate=len(self.data) / pass_s if pass_s else 0.0)
        return self.stats


class FileReceiver:
    """Stateless-per-frame QR receiver: tries every QR version that fits the
    geometry, scores candidates by their finder patterns, re-rasterises the
    best one and decodes it.  Collects chunks until the file is complete."""

    def __init__(self, geom, ec_hint=1):
        self.geom = geom
        self.cands = []
        for version in range(40, 0, -1):
            for trim in (0, 1, 2):
                need = 17 + 4 * version + 4 - 2 * trim
                fit = _qr_fit(geom, need)
                if fit is not None:
                    ph, pv, x0, y0, _pa, _pc = fit
                    self.cands.append(dict(version=version, trim=trim,
                                           need=need, ph=ph, pv=pv,
                                           x0=x0, y0=y0))
                    break               # one trim per version is enough
        cv2 = get_cv2()
        self.det = None
        self.det_fb = None
        if cv2 is not None:
            try:                      # Aruco-based detector (OpenCV >= 4.8)
                self.det = cv2.QRCodeDetectorAruco()   # far more reliable
            except AttributeError:
                pass
            try:
                fb = cv2.QRCodeDetector()
                if self.det is None:
                    self.det = fb
                else:
                    self.det_fb = fb
            except Exception:
                pass
        self.last_cand = None
        self.reset()

    def reset(self):
        self.n = None
        self.size = None
        self.crc = None
        self.name = ""
        self.chunks = {}
        self.done = False
        self.blob = None
        self.frames_seen = 0
        self.frames_decoded = 0

    # -- internals -----------------------------------------------------
    def _cells(self, a, c, dy, dx):
        bh, bw = c["need"] * c["pv"], c["need"] * c["ph"]
        yy, xx = c["y0"] + dy, c["x0"] + dx
        if yy < 0 or xx < 0 or yy + bh > a.shape[0] or xx + bw > a.shape[1]:
            return None
        sub = a[yy:yy + bh, xx:xx + bw]
        return sub.reshape(c["need"], c["pv"], c["need"],
                           c["ph"]).mean(axis=(1, 3))

    def _finder_score(self, cells, trim):
        """Mean Pearson correlation of the three corner finder patterns
        against the ideal: ~1 for a real QR, ~0 for flat or unrelated."""
        q = 2 - trim
        m = cells.shape[0] - 2 * q
        if m < 7:
            return -1e18
        dark = 1.0 - cells
        fz = QR_FINDER - QR_FINDER.mean()
        fzn = float(np.sqrt((fz * fz).sum()))
        s = 0.0
        for (r, cc) in [(q, q), (q, q + m - 7), (q + m - 7, q)]:
            blk = dark[r:r + 7, cc:cc + 7]
            bz = blk - blk.mean()
            den = float(np.sqrt((bz * bz).sum())) * fzn + 1e-9
            s += float((bz * fz).sum()) / den
        return s / 3.0

    def _try_decode(self, cells):
        lo, hi = float(cells.min()), float(cells.max())
        if hi - lo < 0.05:
            return ""
        mods = (cells > 0.5 * (lo + hi)).astype(np.uint8) * 255
        crisp = np.kron(mods, np.ones((8, 8), np.uint8))
        crisp = np.pad(crisp, 32, constant_values=255)
        for det in (self.det, self.det_fb):
            if det is None:
                continue
            try:
                data, _pts, _st = det.detectAndDecode(crisp)
            except Exception:
                data = ""
            if data:
                return data
        return ""

    # -- public ----------------------------------------------------------
    def offer(self, arr):
        """Feed one decoded display frame (float 0..1).  Returns an event
        string when something notable happened, else None."""
        if self.done or self.det is None or not self.cands:
            return None
        self.frames_seen += 1
        a = arr if arr.ndim == 2 else arr.mean(axis=2)
        a = np.ascontiguousarray(a, dtype=np.float32)
        # coarse pass: every candidate at zero shift
        scored = []
        for c in self.cands:
            cells = self._cells(a, c, 0, 0)
            if cells is not None:
                scored.append((self._finder_score(cells, c["trim"]), c))
        if not scored:
            return None
        scored.sort(key=lambda t: -t[0])
        short = [c for _s, c in scored[:2]]
        if self.last_cand is not None and self.last_cand not in short:
            short.insert(0, self.last_cand)
        # fine pass: small shift search on the shortlist
        best = (-1e18, None, None)
        for c in short:
            S = max(2, c["ph"] + c["pv"])
            for dy in range(-S, S + 1):
                for dx in range(-S, S + 1):
                    cells = self._cells(a, c, dy, dx)
                    if cells is None:
                        continue
                    sc = self._finder_score(cells, c["trim"])
                    if sc > best[0]:
                        best = (sc, cells, c)
        if best[1] is None:
            return None
        data = self._try_decode(best[1])
        if not data:
            return None
        self.last_cand = best[2]
        self.frames_decoded += 1
        return self._ingest(data)

    def _ingest(self, s):
        try:
            kind = s[0]
            if kind == "M" and len(s) >= 17:
                self.n = b36d(s[1:4])
                self.size = b36d(s[4:10])
                self.crc = b36d(s[10:17])
                return self._maybe_finish() or "meta"
            if kind == "N":
                name = base64.b64decode(s[1:]).decode("utf-8", "ignore")
                self.name = os.path.basename(name)[:120]
                return "name"
            if kind == "D" and len(s) > 4:
                seq = b36d(s[1:4])
                if seq not in self.chunks:
                    self.chunks[seq] = base64.b64decode(s[4:])
                    return self._maybe_finish() or "chunk"
                return None
        except Exception:
            return None
        return None

    def _maybe_finish(self):
        if (self.n is None or self.done
                or len(self.chunks) < self.n):
            return None
        blob = b"".join(self.chunks[i] for i in range(1, self.n + 1)
                        if i in self.chunks)
        if len(blob) != self.size or \
                (zlib.crc32(blob) & 0xffffffff) != self.crc:
            # a wrong chunk slipped through: start over
            self.chunks = {}
            return "crc-restart"
        self.blob = blob
        self.done = True
        return "complete"

    def progress(self):
        return (len(self.chunks), self.n)

    def save(self, folder):
        name = self.name or "received.bin"
        base, ext = os.path.splitext(name)
        path = os.path.join(folder, name)
        k = 1
        while os.path.exists(path):
            path = os.path.join(folder, "%s (%d)%s" % (base, k, ext))
            k += 1
        with open(path, "wb") as f:
            f.write(self.blob)
        return path


class FileRXWorker(threading.Thread):
    """Pulls decoded frames from a queue and feeds the FileReceiver, so QR
    scanning never blocks the UI thread."""

    def __init__(self, receiver, save_dir, status_cb=None, done_cb=None):
        super().__init__(daemon=True)
        self.rxr = receiver
        self.save_dir = save_dir
        self.status_cb = status_cb
        self.done_cb = done_cb
        self.q = queue.Queue(maxsize=3)
        self.stop_evt = threading.Event()

    def stop(self):
        self.stop_evt.set()

    def feed(self, arr):
        try:
            self.q.put_nowait(arr)
        except queue.Full:
            try:
                self.q.get_nowait()
                self.q.put_nowait(arr)
            except Exception:
                pass

    def run(self):
        while not self.stop_evt.is_set():
            try:
                arr = self.q.get(timeout=0.2)
            except queue.Empty:
                continue
            try:
                ev = self.rxr.offer(arr)
            except Exception as e:
                if self.status_cb:
                    self.status_cb("File RX error: %s" % e)
                continue
            if ev in ("chunk", "meta", "name", "crc-restart"):
                got, n = self.rxr.progress()
                if self.status_cb:
                    extra = " (bad CRC - restarting)" \
                        if ev == "crc-restart" else ""
                    self.status_cb("File RX: %d/%s chunks%s%s"
                                   % (got, n if n else "?",
                                      (" - " + self.rxr.name)
                                      if self.rxr.name else "", extra))
            elif ev == "complete":
                try:
                    path = self.rxr.save(self.save_dir)
                    if self.status_cb:
                        self.status_cb("File received OK (%d bytes, CRC "
                                       "good): %s"
                                       % (len(self.rxr.blob), path))
                    if self.done_cb:
                        self.done_cb(path)
                except Exception as e:
                    if self.status_cb:
                        self.status_cb("File received but save failed: %s"
                                       % e)
                return


# ----------------------------------------------------------------------------
# GUI (tkinter).  Imported defensively so --selftest works headless.
# ----------------------------------------------------------------------------
try:
    import tkinter as tk
    from tkinter import ttk, filedialog, messagebox, simpledialog
    from tkinter import font as tkfont
    HAVE_TK = True
except Exception:
    HAVE_TK = False

try:
    from PIL import ImageTk
    HAVE_IMAGETK = True
except Exception:
    HAVE_IMAGETK = False

_SD_UNSET = object()
_SD = _SD_UNSET


def get_sd():
    """Lazy import of sounddevice; returns module or None."""
    global _SD
    if _SD is _SD_UNSET:
        try:
            import sounddevice as sd
            _SD = sd
        except Exception:
            _SD = None
    return _SD


SETTINGS_PATH = os.path.join(os.path.expanduser("~"), ".nbtv_studio.json")


def load_settings():
    try:
        with open(SETTINGS_PATH, "r") as f:
            return json.load(f)
    except Exception:
        return {}


def save_settings(d):
    try:
        with open(SETTINGS_PATH, "w") as f:
            json.dump(d, f, indent=1)
    except Exception:
        pass


SOURCES = ["Test pattern", "Image file", "Animated GIF", "Video file",
           "Webcam", "Screen capture", "File over NBTV (QR)"]
TINTS = [("White", (1.0, 1.0, 1.0)),
         ("Green CRT", (0.25, 1.0, 0.35)),
         ("Amber CRT", (1.0, 0.72, 0.18))]
WAV_SPEEDS = [("1x (real time)", 1.0), ("4x", 4.0), ("Max speed", 0.0)]


class App:
    def __init__(self, root):
        self.root = root
        root.title("NBTV Studio - narrow-band television experimenter")
        root.minsize(1180, 760)

        self._setup_fonts_and_style()

        s = load_settings()

        # --- shared plumbing -------------------------------------------
        cap = 192000 * 4
        self.tx_ring = Ring(cap, 2)
        self.rx_ring = Ring(cap, 2)
        self.preview_q = queue.Queue(maxsize=2)
        self.frame_q = queue.Queue(maxsize=4)
        self.status_q = queue.Queue(maxsize=64)
        self.source = SourceManager()

        self.tx = None
        self.rx = None
        self.feeder = None
        self.out_stream = None
        self.in_stream = None
        self.render_thread = None
        self.render_stop = threading.Event()

        self.file_sender = FileSender()
        self.file_worker = None
        self.file_ec = int(s.get("file_ec", 1))          # 0=L 1=M 2=Q
        self.file_repeat = int(s.get("file_repeat", 2))
        self.file_save_dir = s.get("file_save_dir",
                                   os.path.expanduser("~"))

        self.in_dev = s.get("in_dev")    # sounddevice indices or None
        self.out_dev = s.get("out_dev")
        self.record_path = s.get("record_path", "")
        self.custom_modes_loaded(s)

        self.last_tx_pil = None
        self.last_rx = None              # (array, info)
        self._photos = {}
        self._stat_tick = 0

        # --- tk variables ----------------------------------------------
        self.v_mode = tk.StringVar(value=s.get("mode", MODES[0]["name"]))
        self.v_rate = tk.StringVar(value=str(s.get("rate", 48000)))
        self.v_color = tk.StringVar(value=s.get("color", COLOR_SYSTEMS[0][0]))
        self.v_filter = tk.StringVar(value=s.get("filter",
                                                 OUTPUT_FILTERS[3][0]))
        self.v_source = tk.StringVar(value="Test pattern")
        self.v_pattern = tk.StringVar(value=s.get("pattern",
                                                  TEST_PATTERNS[0]))
        self.v_fit = tk.StringVar(value=s.get("fit", "Crop to fill"))
        self.v_gain = tk.DoubleVar(value=s.get("gain", 0.9))
        self.v_sharp = tk.BooleanVar(value=bool(s.get("sharpen_src", True)))
        self.v_record = tk.BooleanVar(value=False)
        self.v_bits = tk.StringVar(value=s.get("bits", "24"))
        self.v_loop = tk.BooleanVar(value=s.get("loop", True))
        self.v_speed = tk.StringVar(value=WAV_SPEEDS[1][0])
        self.v_sync = tk.DoubleVar(value=s.get("sync", 0.15))
        self.v_bright = tk.DoubleVar(value=s.get("bright", 0.0))
        self.v_contrast = tk.DoubleVar(value=s.get("contrast", 1.0))
        self.v_sat = tk.DoubleVar(value=s.get("sat", 1.0))
        self.v_integrate = tk.IntVar(value=s.get("integrate", 1))
        self.v_smooth = tk.BooleanVar(value=s.get("smooth", True))
        self.v_hflip = tk.BooleanVar(value=False)
        self.v_vflip = tk.BooleanVar(value=False)
        self.v_tint = tk.StringVar(value=s.get("tint", TINTS[0][0]))
        self.v_capture = [tk.IntVar(value=v) for v in
                          s.get("capture", [100, 100, 640, 480])]

        self._build_layout()
        self._mode_info_refresh()
        self.set_status("Ready.  Tip: tick 'Soft loopback', press Start TX, "
                        "then Start RX - no cables needed.")
        root.protocol("WM_DELETE_WINDOW", self.on_close)
        root.after(40, self._pump)

    # ------------------------------------------------------------------
    def custom_modes_loaded(self, s):
        for m in s.get("custom_modes", []):
            try:
                d = dict(name=str(m["name"]), lines=int(m["lines"]),
                         fps=float(m["fps"]),
                         aspect=(int(m["aspect"][0]), int(m["aspect"][1])),
                         scan=str(m.get("scan", "H")))
                if "sync_f" in m:
                    d["sync_f"] = float(m["sync_f"])
                if not any(x["name"] == d["name"] for x in MODES):
                    MODES.append(d)
            except Exception:
                pass

    def _setup_fonts_and_style(self):
        for name in ("TkDefaultFont", "TkTextFont", "TkMenuFont",
                     "TkHeadingFont"):
            try:
                tkfont.nametofont(name).configure(size=12)
            except Exception:
                pass
        try:
            tkfont.nametofont("TkFixedFont").configure(size=11)
        except Exception:
            pass
        st = ttk.Style(self.root)
        try:
            st.configure("Big.TButton", padding=(12, 8),
                         font=("TkDefaultFont", 12, "bold"))
            st.configure("TButton", padding=(8, 5))
            st.configure("TLabelframe.Label",
                         font=("TkDefaultFont", 12, "bold"))
            st.configure("TCheckbutton", padding=3)
        except Exception:
            pass

    # ------------------------------------------------------------------
    def _build_layout(self):
        root = self.root
        root.columnconfigure(0, weight=1)
        root.rowconfigure(1, weight=1)

        # ---------- top bar ----------
        top = ttk.Frame(root, padding=(8, 6))
        top.grid(row=0, column=0, sticky="ew")
        for c in range(12):
            top.columnconfigure(c, weight=0)
        top.columnconfigure(11, weight=1)

        ttk.Label(top, text="Mode:").grid(row=0, column=0, sticky="w")
        self.cb_mode = ttk.Combobox(top, textvariable=self.v_mode,
                                    state="readonly", width=44,
                                    values=[m["name"] for m in MODES])
        self.cb_mode.grid(row=0, column=1, padx=(4, 14), sticky="w")
        self.cb_mode.bind("<<ComboboxSelected>>", self._settings_changed)

        ttk.Label(top, text="Sample rate:").grid(row=0, column=2, sticky="w")
        self.cb_rate = ttk.Combobox(top, textvariable=self.v_rate,
                                    state="readonly", width=8,
                                    values=[str(r) for r in SAMPLE_RATES])
        self.cb_rate.grid(row=0, column=3, padx=(4, 14))
        self.cb_rate.bind("<<ComboboxSelected>>", self._settings_changed)

        ttk.Label(top, text="Colour:").grid(row=0, column=4, sticky="w")
        self.cb_color = ttk.Combobox(top, textvariable=self.v_color,
                                     state="readonly", width=34,
                                     values=[c[0] for c in COLOR_SYSTEMS])
        self.cb_color.grid(row=0, column=5, padx=(4, 14))
        self.cb_color.bind("<<ComboboxSelected>>", self._settings_changed)

        ttk.Label(top, text="TX filter:").grid(row=1, column=0, sticky="w",
                                               pady=(6, 0))
        self.cb_filter = ttk.Combobox(top, textvariable=self.v_filter,
                                      state="readonly", width=30,
                                      values=[f[0] for f in OUTPUT_FILTERS])
        self.cb_filter.grid(row=1, column=1, padx=(4, 14), sticky="w",
                            pady=(6, 0))
        self.cb_filter.bind("<<ComboboxSelected>>", self._filter_changed)

        bbar = ttk.Frame(top)
        bbar.grid(row=1, column=2, columnspan=10, sticky="w", pady=(6, 0))
        ttk.Button(bbar, text="Audio setup...",
                   command=self.dlg_audio).pack(side="left", padx=3)
        ttk.Button(bbar, text="Capture area...",
                   command=self.dlg_capture).pack(side="left", padx=3)
        ttk.Button(bbar, text="Custom mode...",
                   command=self.dlg_custom_mode).pack(side="left", padx=3)
        ttk.Button(bbar, text="File transfer...",
                   command=self.dlg_file).pack(side="left", padx=3)
        ttk.Button(bbar, text="Help / About...",
                   command=self.dlg_help).pack(side="left", padx=3)

        self.lbl_mode_info = ttk.Label(top, text="", foreground="#555")
        self.lbl_mode_info.grid(row=2, column=0, columnspan=12, sticky="w",
                                pady=(6, 0))

        # ---------- middle: TX | RX ----------
        mid = ttk.Frame(root, padding=(8, 2))
        mid.grid(row=1, column=0, sticky="nsew")
        mid.columnconfigure(0, weight=1, uniform="panes")
        mid.columnconfigure(1, weight=1, uniform="panes")
        mid.rowconfigure(0, weight=1)

        self._build_tx_panel(mid)
        self._build_rx_panel(mid)

        # ---------- status bar ----------
        bar = ttk.Frame(root, padding=(8, 4))
        bar.grid(row=2, column=0, sticky="ew")
        bar.columnconfigure(0, weight=1)
        self.lbl_status = ttk.Label(bar, text="", anchor="w")
        self.lbl_status.grid(row=0, column=0, sticky="ew")
        self.lbl_stats = ttk.Label(bar, text="", anchor="e",
                                   foreground="#555")
        self.lbl_stats.grid(row=0, column=1, sticky="e")

    # ------------------------------------------------------------------
    def _build_tx_panel(self, parent):
        f = ttk.Labelframe(parent, text="  TRANSMIT  ", padding=8)
        f.grid(row=0, column=0, sticky="nsew", padx=(0, 5))
        f.columnconfigure(1, weight=1)
        r = 0

        ttk.Label(f, text="Source:").grid(row=r, column=0, sticky="w")
        row = ttk.Frame(f)
        row.grid(row=r, column=1, sticky="ew")
        cb = ttk.Combobox(row, textvariable=self.v_source, state="readonly",
                          values=SOURCES, width=16)
        cb.pack(side="left")
        cb.bind("<<ComboboxSelected>>", self._source_selected)
        ttk.Button(row, text="Load / configure...",
                   command=self._source_configure).pack(side="left", padx=6)
        r += 1

        ttk.Label(f, text="Test pattern:").grid(row=r, column=0, sticky="w",
                                                pady=(4, 0))
        self.cb_pattern = ttk.Combobox(f, textvariable=self.v_pattern,
                                       state="readonly",
                                       values=TEST_PATTERNS, width=34)
        self.cb_pattern.grid(row=r, column=1, sticky="w", pady=(4, 0))
        self.cb_pattern.bind("<<ComboboxSelected>>", self._pattern_selected)
        r += 1

        ttk.Label(f, text="Fit:").grid(row=r, column=0, sticky="w",
                                       pady=(4, 0))
        rowf = ttk.Frame(f)
        rowf.grid(row=r, column=1, sticky="ew", pady=(4, 0))
        ttk.Combobox(rowf, textvariable=self.v_fit, state="readonly",
                     values=["Crop to fill", "Pad with black"],
                     width=14).pack(side="left")
        ttk.Label(rowf, text="   Gain:").pack(side="left")
        tk.Scale(rowf, variable=self.v_gain, from_=0.1, to=1.0,
                 resolution=0.05, orient="horizontal", length=150,
                 showvalue=True, command=self._gain_changed).pack(side="left")
        ttk.Checkbutton(rowf, text="Detail boost",
                        variable=self.v_sharp,
                        command=self._sharpen_changed).pack(side="left",
                                                            padx=(10, 0))
        r += 1

        rowr = ttk.Frame(f)
        rowr.grid(row=r, column=0, columnspan=2, sticky="ew", pady=(4, 0))
        ttk.Checkbutton(rowr, text="Record TX to WAV",
                        variable=self.v_record).pack(side="left")
        ttk.Combobox(rowr, textvariable=self.v_bits, state="readonly",
                     values=["16", "24"], width=4).pack(side="left", padx=4)
        ttk.Label(rowr, text="bit").pack(side="left")
        ttk.Button(rowr, text="File...",
                   command=self._pick_record_file).pack(side="left", padx=6)
        r += 1

        rowb = ttk.Frame(f)
        rowb.grid(row=r, column=0, columnspan=2, sticky="ew", pady=(8, 4))
        self.btn_tx = ttk.Button(rowb, text="START TX", style="Big.TButton",
                                 command=self.start_tx)
        self.btn_tx.pack(side="left", padx=(0, 6))
        ttk.Button(rowb, text="Stop TX", style="Big.TButton",
                   command=self.stop_tx).pack(side="left", padx=6)
        ttk.Button(rowb, text="Render WAV (offline)...",
                   command=self.render_wav_dialog).pack(side="left", padx=6)
        r += 1

        self.tx_canvas = tk.Canvas(f, bg="black", height=320,
                                   highlightthickness=1,
                                   highlightbackground="#555")
        self.tx_canvas.grid(row=r, column=0, columnspan=2, sticky="nsew",
                            pady=(4, 0))
        f.rowconfigure(r, weight=1)
        self.tx_canvas.bind("<Configure>",
                            lambda e: self._redraw_tx())
        r += 1
        self.lbl_tx_info = ttk.Label(f, text="TX idle", foreground="#555")
        self.lbl_tx_info.grid(row=r, column=0, columnspan=2, sticky="w")

    # ------------------------------------------------------------------
    def _build_rx_panel(self, parent):
        f = ttk.Labelframe(parent, text="  RECEIVE  ", padding=8)
        f.grid(row=0, column=1, sticky="nsew", padx=(5, 0))
        f.columnconfigure(0, weight=1)
        r = 0

        rowb = ttk.Frame(f)
        rowb.grid(row=r, column=0, sticky="ew")
        self.btn_rx = ttk.Button(rowb, text="START RX", style="Big.TButton",
                                 command=self.start_rx_audio)
        self.btn_rx.pack(side="left", padx=(0, 6))
        ttk.Button(rowb, text="Stop RX", style="Big.TButton",
                   command=self.stop_rx).pack(side="left", padx=6)
        ttk.Button(rowb, text="Decode WAV...",
                   command=self.decode_wav_dialog).pack(side="left", padx=6)
        ttk.Combobox(rowb, textvariable=self.v_speed, state="readonly",
                     values=[w[0] for w in WAV_SPEEDS],
                     width=13).pack(side="left", padx=4)
        r += 1

        ttk.Checkbutton(f, text="Soft loopback (feed TX straight into RX - "
                                "no cable needed)",
                        variable=self.v_loop).grid(row=r, column=0,
                                                   sticky="w", pady=(4, 0))
        r += 1

        rows = ttk.Frame(f)
        rows.grid(row=r, column=0, sticky="ew", pady=(2, 0))
        for i, (label, var, lo, hi, res, cmd) in enumerate([
                ("Sync level", self.v_sync, 0.05, 0.40, 0.01,
                 self._sync_changed),
                ("Brightness", self.v_bright, -0.5, 0.5, 0.02, None),
                ("Contrast", self.v_contrast, 0.2, 3.0, 0.05, None),
                ("Saturation", self.v_sat, 0.0, 3.0, 0.05,
                 self._sat_changed)]):
            fr = ttk.Frame(rows)
            fr.grid(row=i // 2, column=i % 2, sticky="w", padx=(0, 12))
            ttk.Label(fr, text=label + ":", width=10).pack(side="left")
            kw = dict(variable=var, from_=lo, to=hi, resolution=res,
                      orient="horizontal", length=170, showvalue=True)
            if cmd:
                kw["command"] = cmd
            tk.Scale(fr, **kw).pack(side="left")
        r += 1

        rowo = ttk.Frame(f)
        rowo.grid(row=r, column=0, sticky="ew", pady=(4, 0))
        ttk.Label(rowo, text="Integrate frames:").pack(side="left")
        sp = tk.Spinbox(rowo, from_=1, to=64, width=4,
                        textvariable=self.v_integrate,
                        command=self._integrate_changed,
                        font=("TkDefaultFont", 12))
        sp.pack(side="left", padx=(4, 12))
        ttk.Checkbutton(rowo, text="Smooth", variable=self.v_smooth,
                        command=self._redraw_rx).pack(side="left", padx=4)
        ttk.Checkbutton(rowo, text="H flip", variable=self.v_hflip,
                        command=self._redraw_rx).pack(side="left", padx=4)
        ttk.Checkbutton(rowo, text="V flip", variable=self.v_vflip,
                        command=self._redraw_rx).pack(side="left", padx=4)
        ttk.Label(rowo, text="  Tint:").pack(side="left")
        cbt = ttk.Combobox(rowo, textvariable=self.v_tint, state="readonly",
                           values=[t[0] for t in TINTS], width=10)
        cbt.pack(side="left", padx=4)
        cbt.bind("<<ComboboxSelected>>", lambda e: self._redraw_rx())
        ttk.Button(rowo, text="Save frame...",
                   command=self.save_rx_frame).pack(side="left", padx=8)
        r += 1

        self.rx_canvas = tk.Canvas(f, bg="black", height=320,
                                   highlightthickness=1,
                                   highlightbackground="#555")
        self.rx_canvas.grid(row=r, column=0, sticky="nsew", pady=(4, 0))
        f.rowconfigure(r, weight=1)
        self.rx_canvas.bind("<Configure>", lambda e: self._redraw_rx())
        r += 1
        self.lbl_rx_info = ttk.Label(f, text="RX idle", foreground="#555")
        self.lbl_rx_info.grid(row=r, column=0, sticky="w")

    # ------------------------------------------------------------------
    # Current configuration helpers
    # ------------------------------------------------------------------
    def current_mode(self):
        name = self.v_mode.get()
        for m in MODES:
            if m["name"] == name:
                return m
        return MODES[0]

    def current_geom(self, rate=None):
        return Geometry(self.current_mode(),
                        rate if rate else int(self.v_rate.get()))

    def current_color(self):
        name = self.v_color.get()
        for label, key in COLOR_SYSTEMS:
            if label == name:
                return key
        return "mono"

    def current_filter(self):
        name = self.v_filter.get()
        for label, cut in OUTPUT_FILTERS:
            if label == name:
                return cut
        return None

    def current_fill(self):
        return self.v_fit.get() == "Crop to fill"

    def current_sharpen(self):
        return DETAIL_SHARPEN if self.v_sharp.get() else 0.0

    def set_status(self, msg):
        self.lbl_status.configure(text=msg)

    def status_threadsafe(self, msg):
        try:
            self.status_q.put_nowait(msg)
        except queue.Full:
            pass

    def _mode_info_refresh(self):
        g = self.current_geom()
        txt = g.describe()
        warn = ""
        if not g.usable:
            warn = ("  !! Too few samples per line at this sample rate - "
                    "raise the rate or pick fewer lines !!")
        if g.rate > 48000:
            txt += "  (needs a 96/192 kHz capable sound device)"
        self.lbl_mode_info.configure(
            text=txt + warn, foreground="#b00" if warn else "#555")

    # ------------------------------------------------------------------
    # Settings-change handlers
    # ------------------------------------------------------------------
    def _settings_changed(self, _evt=None):
        self._mode_info_refresh()
        if self.tx or self.rx or self.feeder:
            self.stop_tx()
            self.stop_rx()
            self.set_status("Mode / rate / colour changed - engines stopped. "
                            "Press Start again.")

    def _filter_changed(self, _evt=None):
        if self.tx:
            try:
                self.tx.lpf = FIRLowpass(self.tx.geom.rate,
                                         self.current_filter(), channels=2)
                self.set_status("TX filter switched live: %s"
                                % self.v_filter.get())
            except Exception as e:
                self.set_status("Filter change failed: %s" % e)

    def _gain_changed(self, _v=None):
        if self.tx:
            self.tx.enc.gain = float(self.v_gain.get())

    def _sharpen_changed(self):
        if self.tx:
            self.tx.enc.sharpen = self.current_sharpen()
        self.set_status("Detail boost (aperture correction) %s."
                        % ("on" if self.v_sharp.get() else "off"))

    def _sync_changed(self, _v=None):
        if self.rx:
            self.rx.dec.set_sync_level(self.v_sync.get())

    def _sat_changed(self, _v=None):
        if self.rx:
            self.rx.dec.set_saturation(self.v_sat.get())

    def _integrate_changed(self):
        if self.rx:
            try:
                self.rx.set_integrate(int(self.v_integrate.get()))
            except Exception:
                pass

    def _pattern_selected(self, _evt=None):
        if self.v_source.get() == "Test pattern":
            self.source.use_test(self.v_pattern.get())
            self.set_status("Test pattern: %s" % self.v_pattern.get())

    def _source_selected(self, _evt=None):
        kind = self.v_source.get()
        if kind == "Test pattern":
            self.source.use_test(self.v_pattern.get())
            self.set_status("Source: test pattern (%s)"
                            % self.v_pattern.get())
        elif kind == "File over NBTV (QR)":
            self.set_status("Source: file transfer - press 'Load / "
                            "configure...' or 'File transfer...' to pick "
                            "the file.")
        else:
            self.set_status("Source: %s - press 'Load / configure...' "
                            "to pick it." % kind)

    def _source_configure(self):
        kind = self.v_source.get()
        try:
            if kind == "Test pattern":
                self.source.use_test(self.v_pattern.get())
                self.set_status("Test pattern ready.")
            elif kind == "Image file":
                p = filedialog.askopenfilename(
                    title="Choose image",
                    filetypes=[("Images", "*.png *.jpg *.jpeg *.bmp *.gif "
                                "*.tif *.tiff *.webp"), ("All files", "*")])
                if p:
                    self.source.use_image(p)
                    self.set_status("Image loaded: %s"
                                    % os.path.basename(p))
            elif kind == "Animated GIF":
                p = filedialog.askopenfilename(
                    title="Choose animated GIF",
                    filetypes=[("GIF", "*.gif"), ("All files", "*")])
                if p:
                    self.source.use_gif(p)
                    self.set_status("GIF loaded: %s (%d frames)"
                                    % (os.path.basename(p),
                                       len(self.source.gif_frames)))
            elif kind == "Video file":
                p = filedialog.askopenfilename(
                    title="Choose video file",
                    filetypes=[("Video", "*.mp4 *.avi *.mov *.mkv *.mpg "
                                "*.mpeg *.wmv"), ("All files", "*")])
                if p:
                    self.source.use_video(p)
                    self.set_status("Video loaded: %s"
                                    % os.path.basename(p))
            elif kind == "Webcam":
                idx = simpledialog.askinteger(
                    "Webcam", "Camera index (0 = first camera):",
                    parent=self.root, initialvalue=0, minvalue=0, maxvalue=8)
                if idx is not None:
                    self.source.use_webcam(idx)
                    self.set_status("Webcam %d opened." % idx)
            elif kind == "Screen capture":
                self.dlg_capture()
            elif kind == "File over NBTV (QR)":
                self.dlg_file()
        except Exception as e:
            messagebox.showerror("Source error", str(e), parent=self.root)
            self.set_status("Source error: %s" % e)

    def _pick_record_file(self):
        p = filedialog.asksaveasfilename(
            title="Record TX to WAV", defaultextension=".wav",
            filetypes=[("WAV", "*.wav")],
            initialfile="nbtv_tx.wav")
        if p:
            self.record_path = p
            self.v_record.set(True)
            self.set_status("Will record to: %s" % p)

    # ------------------------------------------------------------------
    # Audio stream helpers
    # ------------------------------------------------------------------
    def _open_output_stream(self, rate):
        sd = get_sd()
        if sd is None:
            return None, ("sounddevice not installed - audio output off "
                          "(loopback / WAV still work). "
                          "Install with: pip install sounddevice")

        def cb2(outdata, frames, t, status):
            outdata[:] = self.tx_ring.read(frames)

        def cb1(outdata, frames, t, status):
            outdata[:, 0] = self.tx_ring.read(frames).mean(axis=1)

        for ch, cb in ((2, cb2), (1, cb1)):
            try:
                st = sd.OutputStream(samplerate=rate, channels=ch,
                                     dtype="float32", device=self.out_dev,
                                     callback=cb)
                st.start()
                return st, None
            except Exception as e:
                err = e
        return None, "audio output failed: %s" % err

    def _open_input_stream(self, rate):
        sd = get_sd()
        if sd is None:
            return None, ("sounddevice not installed - audio input off "
                          "(loopback / WAV decode still work).")

        def cb2(indata, frames, t, status):
            self.rx_ring.write(indata.copy())

        def cb1(indata, frames, t, status):
            self.rx_ring.write(np.repeat(indata, 2, axis=1))

        for ch, cb in ((2, cb2), (1, cb1)):
            try:
                st = sd.InputStream(samplerate=rate, channels=ch,
                                    dtype="float32", device=self.in_dev,
                                    callback=cb)
                st.start()
                return st, None
            except Exception as e:
                err = e
        return None, "audio input failed: %s" % err

    # ------------------------------------------------------------------
    # TX / RX engine control
    # ------------------------------------------------------------------
    def start_tx(self):
        self.stop_tx()
        g = self.current_geom()
        if not g.usable:
            if not messagebox.askyesno(
                    "Very low resolution",
                    "This mode has under 24 samples per line at the chosen "
                    "sample rate, so the picture will be mush.\n\n"
                    "Start anyway?", parent=self.root):
                return
        cs = self.current_color()
        if self.source.kind == "filetx":
            sig = self._file_sig(g)
            if self.source.ft_sig != sig:
                if not self.build_filetx(g):
                    return
        wav = None
        if self.v_record.get():
            if not self.record_path:
                self._pick_record_file()
            if self.record_path:
                try:
                    wav = WavWriter(self.record_path, g.rate, 2,
                                    int(self.v_bits.get()))
                except Exception as e:
                    messagebox.showerror("Record", str(e), parent=self.root)
                    wav = None
        loop_ring = None
        if self.v_loop.get():
            loop_ring = self.rx_ring
            if self.rx is None:
                self._start_rx_engine(g, cs, clear=True)
        self.tx_ring.clear()
        self.out_stream, err = self._open_output_stream(g.rate)
        if err:
            self.status_threadsafe(err)
        tx_ring = self.tx_ring if self.out_stream else None
        if tx_ring is None and loop_ring is None and wav is None:
            self.set_status("Nothing to send to: no audio out, loopback off, "
                            "not recording. Tick one of those first.")
            return
        self.tx = TXEngine(g, cs, float(self.v_gain.get()),
                           self.current_filter(), self.source,
                           self.current_fill(), tx_ring=tx_ring,
                           loop_ring=loop_ring, wav_writer=wav,
                           preview_q=self.preview_q,
                           status_cb=self.status_threadsafe,
                           sharpen=self.current_sharpen())
        self.tx.start()
        sinks = [s for s, on in (("audio out", tx_ring is not None),
                                 ("loopback", loop_ring is not None),
                                 ("WAV", wav is not None)) if on]
        self.set_status("TX running: %d lines @ %g fps -> %s"
                        % (g.lines, g.fps, " + ".join(sinks)))

    def stop_tx(self):
        if self.tx:
            self.tx.stop()
            self.tx.join(timeout=2.0)
            self.tx = None
        if self.out_stream:
            try:
                self.out_stream.stop()
                self.out_stream.close()
            except Exception:
                pass
            self.out_stream = None
        self.lbl_tx_info.configure(text="TX idle")

    def _start_rx_engine(self, geom, cs, clear=True):
        if clear:
            self.rx_ring.clear()
        self.rx = RXEngine(geom, cs, self.rx_ring, self.frame_q,
                           status_cb=self.status_threadsafe)
        self.rx.dec.set_sync_level(self.v_sync.get())
        self.rx.dec.set_saturation(self.v_sat.get())
        self.rx.set_integrate(int(self.v_integrate.get()))
        self.rx.start()

    def start_rx_audio(self):
        self.stop_rx()
        g = self.current_geom()
        cs = self.current_color()
        self.in_stream, err = self._open_input_stream(g.rate)
        self._start_rx_engine(g, cs, clear=True)
        if err:
            self.set_status(err + "  (RX engine running for loopback/WAV.)")
        else:
            self.set_status("RX listening on audio input: %d lines @ %g fps"
                            % (g.lines, g.fps))

    def stop_rx(self):
        if self.feeder:
            self.feeder.stop()
            self.feeder = None
        if self.rx:
            self.rx.stop()
            self.rx.join(timeout=2.0)
            self.rx = None
        if self.in_stream:
            try:
                self.in_stream.stop()
                self.in_stream.close()
            except Exception:
                pass
            self.in_stream = None
        self.lbl_rx_info.configure(text="RX idle")

    def decode_wav_dialog(self):
        p = filedialog.askopenfilename(
            title="Decode WAV recording",
            filetypes=[("WAV", "*.wav"), ("All files", "*")])
        if not p:
            return
        try:
            wf = wave.open(p, "rb")
            file_rate = wf.getframerate()
            wf.close()
        except Exception as e:
            messagebox.showerror("WAV", "Cannot read file: %s" % e,
                                 parent=self.root)
            return
        self.stop_rx()
        g = Geometry(self.current_mode(), file_rate)
        speed = dict(WAV_SPEEDS).get(self.v_speed.get(), 4.0)
        self._start_rx_engine(g, self.current_color(), clear=True)
        self.feeder = WavFeeder(p, self.rx_ring, speed=speed,
                                status_cb=self.status_threadsafe)
        self.feeder.start()
        self.set_status("Decoding %s at %d Hz (%s) with mode '%s'..."
                        % (os.path.basename(p), file_rate,
                           self.v_speed.get(), self.current_mode()["name"]))

    # ------------------------------------------------------------------
    # Offline WAV render
    # ------------------------------------------------------------------
    def render_wav_dialog(self):
        if self.render_thread and self.render_thread.is_alive():
            self.render_stop.set()
            self.set_status("Stopping current render...")
            return
        secs = simpledialog.askfloat(
            "Render WAV", "Seconds of signal to render:",
            parent=self.root, initialvalue=10.0, minvalue=0.5,
            maxvalue=3600.0)
        if not secs:
            return
        p = filedialog.asksaveasfilename(
            title="Save rendered WAV", defaultextension=".wav",
            filetypes=[("WAV", "*.wav")], initialfile="nbtv_render.wav")
        if not p:
            return
        g = self.current_geom()
        cs = self.current_color()
        cut = self.current_filter()
        gain = float(self.v_gain.get())
        fill = self.current_fill()
        sharp = self.current_sharpen()
        bits = int(self.v_bits.get())
        self.render_stop = threading.Event()
        stop = self.render_stop

        def job():
            try:
                n = render_wav_offline(
                    g, cs, gain, cut, self.source, fill, p, secs, bits,
                    progress_cb=lambda fr: self.status_threadsafe(
                        "Rendering WAV... %d%%" % int(fr * 100)),
                    stop_evt=stop, sharpen=sharp)
                self.status_threadsafe(
                    "Rendered %d frames (%.1f s) to %s"
                    % (n, n / g.fps, os.path.basename(p)))
            except Exception as e:
                self.status_threadsafe("Render failed: %s" % e)

        self.render_thread = threading.Thread(target=job, daemon=True)
        self.render_thread.start()
        self.set_status("Rendering... (press the button again to cancel)")

    # ------------------------------------------------------------------
    # Aux dialogs
    # ------------------------------------------------------------------
    def dlg_audio(self):
        sd = get_sd()
        top = tk.Toplevel(self.root)
        top.title("Audio setup")
        top.transient(self.root)
        frm = ttk.Frame(top, padding=12)
        frm.pack(fill="both", expand=True)
        if sd is None:
            ttk.Label(frm, text="The 'sounddevice' module is not installed,"
                      "\nso live audio in/out is unavailable.\n\n"
                      "Install it with:\n    pip install sounddevice\n\n"
                      "Soft loopback and WAV render / decode work fine "
                      "without it.").pack()
            ttk.Button(frm, text="Close",
                       command=top.destroy).pack(pady=8)
            return
        try:
            devs = sd.query_devices()
        except Exception as e:
            ttk.Label(frm, text="Could not query devices: %s" % e).pack()
            ttk.Button(frm, text="Close", command=top.destroy).pack(pady=8)
            return
        ins, outs = ["(system default)"], ["(system default)"]
        in_map, out_map = [None], [None]
        for i, d in enumerate(devs):
            label = "%d: %s (%s)" % (i, d["name"],
                                     d.get("hostapi_name",
                                           d.get("hostapi", "")))
            if d.get("max_input_channels", 0) > 0:
                ins.append(label + "  [%d ch, %.0f Hz]"
                           % (d["max_input_channels"],
                              d.get("default_samplerate", 0)))
                in_map.append(i)
            if d.get("max_output_channels", 0) > 0:
                outs.append(label + "  [%d ch, %.0f Hz]"
                            % (d["max_output_channels"],
                               d.get("default_samplerate", 0)))
                out_map.append(i)

        ttk.Label(frm, text="Output device (TX):").grid(row=0, column=0,
                                                        sticky="w")
        cbo = ttk.Combobox(frm, state="readonly", values=outs, width=64)
        cbo.grid(row=1, column=0, sticky="ew", pady=(2, 8))
        cbo.current(out_map.index(self.out_dev)
                    if self.out_dev in out_map else 0)

        ttk.Label(frm, text="Input device (RX):").grid(row=2, column=0,
                                                       sticky="w")
        cbi = ttk.Combobox(frm, state="readonly", values=ins, width=64)
        cbi.grid(row=3, column=0, sticky="ew", pady=(2, 8))
        cbi.current(in_map.index(self.in_dev)
                    if self.in_dev in in_map else 0)

        def apply():
            self.out_dev = out_map[cbo.current()]
            self.in_dev = in_map[cbi.current()]
            self.set_status("Audio devices set. They take effect on the "
                            "next Start TX / Start RX.")
            top.destroy()

        def test_tone():
            try:
                rate = int(self.v_rate.get())
                t = np.arange(int(rate * 0.6)) / rate
                tone = (0.4 * np.sin(2 * np.pi * 440.0 * t)
                        ).astype(np.float32)
                sd.play(np.column_stack([tone, tone]), rate,
                        device=out_map[cbo.current()])
            except Exception as e:
                messagebox.showerror("Test tone", str(e), parent=top)

        btns = ttk.Frame(frm)
        btns.grid(row=4, column=0, sticky="e", pady=(6, 0))
        ttk.Button(btns, text="Test tone",
                   command=test_tone).pack(side="left", padx=4)
        ttk.Button(btns, text="OK", style="Big.TButton",
                   command=apply).pack(side="left", padx=4)
        ttk.Button(btns, text="Cancel",
                   command=top.destroy).pack(side="left", padx=4)

    def dlg_capture(self):
        top = tk.Toplevel(self.root)
        top.title("Screen capture area")
        top.transient(self.root)
        frm = ttk.Frame(top, padding=12)
        frm.pack(fill="both", expand=True)
        labels = ["Left", "Top", "Width", "Height"]
        for i, (lab, var) in enumerate(zip(labels, self.v_capture)):
            ttk.Label(frm, text=lab + ":").grid(row=i, column=0, sticky="w",
                                                pady=2)
            tk.Spinbox(frm, from_=0, to=10000, increment=10, width=7,
                       textvariable=var,
                       font=("TkDefaultFont", 12)).grid(row=i, column=1,
                                                        sticky="w", pady=2)

        def apply():
            l, t_, w, h = [v.get() for v in self.v_capture]
            if w < 8 or h < 8:
                messagebox.showerror("Capture", "Width/height too small.",
                                     parent=top)
                return
            try:
                self.source.use_capture((l, t_, l + w, t_ + h))
                self.v_source.set("Screen capture")
                self.set_status("Capturing screen region %dx%d at (%d,%d)"
                                % (w, h, l, t_))
                top.destroy()
            except Exception as e:
                messagebox.showerror("Capture", str(e), parent=top)

        def drag_pick():
            top.withdraw()
            self.root.withdraw()
            self.root.after(150, lambda: self._drag_select(top))

        btns = ttk.Frame(frm)
        btns.grid(row=4, column=0, columnspan=2, pady=(8, 0))
        ttk.Button(btns, text="Drag to select on screen...",
                   command=drag_pick).pack(side="left", padx=4)
        ttk.Button(btns, text="OK", style="Big.TButton",
                   command=apply).pack(side="left", padx=4)
        ttk.Button(btns, text="Cancel",
                   command=top.destroy).pack(side="left", padx=4)

    def _drag_select(self, owner):
        try:
            ov = tk.Toplevel(self.root)
            ov.attributes("-fullscreen", True)
            try:
                ov.attributes("-alpha", 0.30)
            except Exception:
                pass
            ov.configure(bg="black", cursor="crosshair")
            cv = tk.Canvas(ov, bg="black", highlightthickness=0)
            cv.pack(fill="both", expand=True)
            cv.create_text(ov.winfo_screenwidth() // 2, 40,
                           text="Drag a box around the area to capture.  "
                                "Esc = cancel.",
                           fill="#00ff66", font=("TkDefaultFont", 16, "bold"))
            st = {"x0": 0, "y0": 0, "id": None}

            def done(cancel=False):
                try:
                    ov.destroy()
                except Exception:
                    pass
                self.root.deiconify()
                try:
                    owner.deiconify()
                except Exception:
                    pass

            def press(e):
                st["x0"], st["y0"] = e.x, e.y
                st["id"] = cv.create_rectangle(e.x, e.y, e.x, e.y,
                                               outline="#00ff66", width=3)

            def drag(e):
                if st["id"]:
                    cv.coords(st["id"], st["x0"], st["y0"], e.x, e.y)

            def release(e):
                ox, oy = ov.winfo_rootx(), ov.winfo_rooty()
                l = min(st["x0"], e.x) + ox
                t_ = min(st["y0"], e.y) + oy
                w = abs(e.x - st["x0"])
                h = abs(e.y - st["y0"])
                done()
                if w >= 8 and h >= 8:
                    for var, val in zip(self.v_capture, (l, t_, w, h)):
                        var.set(int(val))
                    try:
                        self.source.use_capture((l, t_, l + w, t_ + h))
                        self.v_source.set("Screen capture")
                        self.set_status(
                            "Capturing screen region %dx%d at (%d,%d)"
                            % (w, h, l, t_))
                    except Exception as ex:
                        self.set_status("Capture error: %s" % ex)

            cv.bind("<ButtonPress-1>", press)
            cv.bind("<B1-Motion>", drag)
            cv.bind("<ButtonRelease-1>", release)
            ov.bind("<Escape>", lambda e: done(True))
            ov.focus_force()
        except Exception as e:
            self.root.deiconify()
            try:
                owner.deiconify()
            except Exception:
                pass
            self.set_status("Drag-select unavailable here (%s) - type the "
                            "numbers in instead." % e)

    def dlg_custom_mode(self):
        top = tk.Toplevel(self.root)
        top.title("Custom mode")
        top.transient(self.root)
        frm = ttk.Frame(top, padding=12)
        frm.pack(fill="both", expand=True)
        vs = dict(lines=tk.IntVar(value=32), fps=tk.DoubleVar(value=12.5),
                  aw=tk.IntVar(value=4), ah=tk.IntVar(value=3),
                  sync=tk.DoubleVar(value=DEF_SYNC_F * 100))
        scan = tk.StringVar(value="H (horizontal lines)")
        rows = [("Lines per frame", vs["lines"], 4, 1000, 1),
                ("Frames per second", vs["fps"], 0.5, 60, 0.125),
                ("Aspect width", vs["aw"], 1, 32, 1),
                ("Aspect height", vs["ah"], 1, 32, 1),
                ("Sync width (% of line)", vs["sync"], 4, 30, 1)]
        for i, (lab, var, lo, hi, inc) in enumerate(rows):
            ttk.Label(frm, text=lab + ":").grid(row=i, column=0, sticky="w",
                                                pady=2)
            tk.Spinbox(frm, from_=lo, to=hi, increment=inc, width=8,
                       textvariable=var,
                       font=("TkDefaultFont", 12)).grid(row=i, column=1,
                                                        sticky="w", pady=2)
        ttk.Label(frm, text="Scan direction:").grid(row=5, column=0,
                                                    sticky="w", pady=2)
        ttk.Combobox(frm, textvariable=scan, state="readonly", width=22,
                     values=["H (horizontal lines)",
                             "V (vertical, classic NBTV)"]
                     ).grid(row=5, column=1, sticky="w", pady=2)
        info = ttk.Label(frm, text="", foreground="#555")
        info.grid(row=6, column=0, columnspan=2, sticky="w", pady=(6, 0))

        def preview(*_a):
            try:
                m = build()
                g = Geometry(m, int(self.v_rate.get()))
                info.configure(text=g.describe())
            except Exception:
                pass

        def build():
            return dict(
                name="Custom %dL/%gfps %s" % (vs["lines"].get(),
                                              vs["fps"].get(),
                                              scan.get()[0]),
                lines=int(vs["lines"].get()), fps=float(vs["fps"].get()),
                aspect=(int(vs["aw"].get()), int(vs["ah"].get())),
                scan=scan.get()[0], sync_f=float(vs["sync"].get()) / 100.0)

        def apply():
            try:
                m = build()
            except Exception as e:
                messagebox.showerror("Custom mode", str(e), parent=top)
                return
            MODES[:] = [x for x in MODES if x["name"] != m["name"]]
            MODES.append(m)
            self.cb_mode.configure(values=[x["name"] for x in MODES])
            self.v_mode.set(m["name"])
            self._settings_changed()
            top.destroy()

        for v in vs.values():
            try:
                v.trace_add("write", preview)
            except Exception:
                pass
        preview()
        btns = ttk.Frame(frm)
        btns.grid(row=7, column=0, columnspan=2, pady=(8, 0))
        ttk.Button(btns, text="Add mode", style="Big.TButton",
                   command=apply).pack(side="left", padx=4)
        ttk.Button(btns, text="Cancel",
                   command=top.destroy).pack(side="left", padx=4)

    # ------------------------------------------------------------------
    # File over NBTV (QR)
    # ------------------------------------------------------------------
    def _file_sig(self, g):
        return (g.lines, g.fps, g.scan, g.rate, self.current_filter(),
                self.file_ec, self.file_repeat)

    def build_filetx(self, g):
        """(Re)build the QR frame carousel for the current settings."""
        try:
            stats = self.file_sender.build(g, self.current_filter(),
                                           self.file_ec, self.file_repeat)
        except ValueError as e:
            messagebox.showerror("File transfer", str(e), parent=self.root)
            self.set_status("File TX: %s" % e)
            return False
        self.source.use_file_frames(self.file_sender.frames,
                                    stats["period"], self._file_sig(g))
        p = stats["plan"]
        self.set_status("File TX ready: %s, QR v%d, %d B/frame, %d frames, "
                        "~%.1f s per pass (~%.0f B/s)"
                        % (self.file_sender.name, p["version"],
                           stats["raw"], stats["total_frames"],
                           stats["pass_seconds"], stats["rate"]))
        return True

    def _file_plan_text(self):
        g = self.current_geom()
        cutoff = self.current_filter()
        plan, err = qr_plan(g, cutoff, self.file_ec)
        if plan is None:
            return "Plan: " + err
        raw = ((plan["cap"] - 4) // 4) * 3
        txt = ("Plan for current mode: QR v%d (%dx%d modules), %d bytes per "
               "frame, module %dx%d px"
               % (plan["version"], plan["need"], plan["need"], raw,
                  plan["p_along"], plan["p_cross"]))
        if self.file_sender.data:
            n = (len(self.file_sender.data) + raw - 1) // raw
            secs = (n + 2) * self.file_repeat / g.fps
            txt += ("\n%s: %d bytes -> %d chunks, ~%.1f s per pass "
                    "(~%.0f B/s)"
                    % (self.file_sender.name, len(self.file_sender.data),
                       n, secs,
                       len(self.file_sender.data) / secs if secs else 0))
        if plan["marginal"]:
            txt += "\nMarginal: %s - use Direct cable." % plan["marginal"]
        return txt

    def _file_arm_toggle(self):
        if self.file_worker is not None and self.file_worker.is_alive():
            self.file_worker.stop()
            self.file_worker = None
            self.set_status("File RX disarmed.")
            return
        if get_cv2() is None:
            messagebox.showerror(
                "File transfer", "File transfer needs OpenCV:\n\n"
                "pip install opencv-python", parent=self.root)
            return
        rxr = FileReceiver(self.current_geom())
        if not rxr.cands:
            messagebox.showerror("File transfer",
                                 "Mode too small to carry a QR code.",
                                 parent=self.root)
            return
        self.file_worker = FileRXWorker(rxr, self.file_save_dir,
                                        status_cb=self.status_threadsafe)
        self.file_worker.start()
        if int(self.v_integrate.get()) != 1:
            self.v_integrate.set(1)
            self._integrate_changed()
        if self.rx is None and not self.v_loop.get():
            self.set_status("File RX armed - now press START RX (or tick "
                            "Loopback and start TX).")
        else:
            self.set_status("File RX armed (frame integration set to 1).")

    def dlg_file(self):
        top = tk.Toplevel(self.root)
        top.title("File over NBTV (QR)")
        top.transient(self.root)
        frm = ttk.Frame(top, padding=10)
        frm.pack(fill="both", expand=True)

        # ---- send side ----
        txf = ttk.LabelFrame(frm, text=" Send a file ", padding=8)
        txf.pack(fill="x")
        lbl_file = ttk.Label(txf, text=self.file_sender.name
                             or "(no file chosen)")

        def choose():
            p = filedialog.askopenfilename(title="Choose file to send",
                                           parent=top)
            if p:
                try:
                    self.file_sender.load(p)
                except Exception as e:
                    messagebox.showerror("File transfer", str(e),
                                         parent=top)
                    return
                kb = len(self.file_sender.data) / 1024.0
                lbl_file.configure(text="%s  (%.1f KB)"
                                   % (self.file_sender.name, kb))
                if kb > 64:
                    self.set_status("Heads up: %.0f KB is a lot for NBTV - "
                                    "expect a long transfer." % kb)

        ttk.Button(txf, text="Choose file...",
                   command=choose).grid(row=0, column=0, padx=3, pady=3,
                                        sticky="w")
        lbl_file.grid(row=0, column=1, columnspan=3, sticky="w", padx=6)

        ttk.Label(txf, text="Error correction:").grid(row=1, column=0,
                                                      sticky="w", padx=3)
        v_ec = tk.StringVar(value=["L (most data)", "M (recommended)",
                                   "Q (most robust)"][self.file_ec])
        cb_ec = ttk.Combobox(txf, textvariable=v_ec, state="readonly",
                             width=16, values=["L (most data)",
                                               "M (recommended)",
                                               "Q (most robust)"])
        cb_ec.grid(row=1, column=1, sticky="w", padx=3)
        ttk.Label(txf, text="Repeat each frame:").grid(row=1, column=2,
                                                       sticky="e", padx=3)
        v_rep = tk.IntVar(value=self.file_repeat)
        ttk.Spinbox(txf, from_=1, to=5, textvariable=v_rep, width=4
                    ).grid(row=1, column=3, sticky="w", padx=3)

        lbl_plan = ttk.Label(txf, text="", justify="left", wraplength=520,
                             foreground="#444")
        lbl_plan.grid(row=2, column=0, columnspan=4, sticky="w", padx=3,
                      pady=(6, 2))

        def pull_opts():
            try:
                self.file_ec = ["L", "M", "Q"].index(v_ec.get()[0])
                self.file_repeat = int(clamp(v_rep.get(), 1, 5))
            except Exception:
                pass                  # mid-edit spinbox etc.

        def send():
            pull_opts()
            if not self.file_sender.data:
                choose()
                if not self.file_sender.data:
                    return
            self.v_source.set("File over NBTV (QR)")
            if self.build_filetx(self.current_geom()):
                self.start_tx()

        ttk.Button(txf, text="Send this file (start TX)",
                   style="Big.TButton", command=send
                   ).grid(row=3, column=0, columnspan=2, padx=3,
                          pady=(4, 2), sticky="w")
        ttk.Label(txf, text="Tip: Monochrome + Direct cable is the "
                            "reliable combination.",
                  foreground="#666").grid(row=3, column=2, columnspan=2,
                                          sticky="e", padx=3)

        # ---- receive side ----
        rxf = ttk.LabelFrame(frm, text=" Receive a file ", padding=8)
        rxf.pack(fill="x", pady=(10, 0))
        lbl_dir = ttk.Label(rxf, text=self.file_save_dir)

        def pick_dir():
            d = filedialog.askdirectory(title="Save received files to",
                                        parent=top,
                                        initialdir=self.file_save_dir)
            if d:
                self.file_save_dir = d
                lbl_dir.configure(text=d)
                if self.file_worker is not None:
                    self.file_worker.save_dir = d

        ttk.Button(rxf, text="Save folder...",
                   command=pick_dir).grid(row=0, column=0, padx=3, pady=3,
                                          sticky="w")
        lbl_dir.grid(row=0, column=1, columnspan=2, sticky="w", padx=6)

        btn_arm = ttk.Button(rxf, style="Big.TButton",
                             command=self._file_arm_toggle)
        btn_arm.grid(row=1, column=0, padx=3, pady=(4, 2), sticky="w")

        def reset_rx():
            was = self.file_worker is not None and \
                self.file_worker.is_alive()
            if self.file_worker is not None:
                self.file_worker.stop()
                self.file_worker = None
            if was:
                self._file_arm_toggle()
            self.set_status("File RX reset.")

        ttk.Button(rxf, text="Reset",
                   command=reset_rx).grid(row=1, column=1, padx=3,
                                          pady=(4, 2), sticky="w")
        lbl_prog = ttk.Label(rxf, text="", foreground="#444")
        lbl_prog.grid(row=2, column=0, columnspan=3, sticky="w", padx=3)

        def poll():
            try:
                if not top.winfo_exists():
                    return
                pull_opts()
                lbl_plan.configure(text=self._file_plan_text())
                armed = self.file_worker is not None and \
                    self.file_worker.is_alive()
                btn_arm.configure(text="Disarm receive" if armed
                                  else "Arm receive")
                if self.file_worker is not None:
                    r = self.file_worker.rxr
                    if r.done:
                        lbl_prog.configure(
                            text="Complete: %s (%d bytes, CRC good)"
                            % (r.name or "received.bin",
                               len(r.blob or b"")))
                    else:
                        got, n = r.progress()
                        lbl_prog.configure(
                            text="Armed: %d/%s chunks  |  %d frames seen, "
                                 "%d QR decodes"
                            % (got, n if n else "?", r.frames_seen,
                               r.frames_decoded))
                else:
                    lbl_prog.configure(text="Not armed.")
                top.after(400, poll)
            except tk.TclError:
                return                # window went away mid-update

        poll()

    def dlg_help(self):
        top = tk.Toplevel(self.root)
        top.title("Help / About")
        top.transient(self.root)
        txt = tk.Text(top, wrap="word", width=78, height=28,
                      font=("TkDefaultFont", 11), padx=10, pady=10)
        txt.pack(fill="both", expand=True)
        txt.insert("1.0", HELP_TEXT)
        txt.configure(state="disabled")
        ttk.Button(top, text="Close", command=top.destroy).pack(pady=6)

    # ------------------------------------------------------------------
    # Save decoded frame
    # ------------------------------------------------------------------
    def save_rx_frame(self):
        if self.last_rx is None:
            self.set_status("No decoded frame to save yet.")
            return
        p = filedialog.asksaveasfilename(
            title="Save decoded frame", defaultextension=".png",
            filetypes=[("PNG", "*.png")], initialfile="nbtv_frame.png")
        if not p:
            return
        try:
            img = self._rx_to_pil(self.last_rx[0])
            aw, ah = self.current_mode()["aspect"]
            h = 480
            w = max(1, int(round(h * aw / ah)))
            img = img.resize((w, h), Image.LANCZOS if self.v_smooth.get()
                             else Image.NEAREST)
            img.save(p)
            self.set_status("Saved %s (%dx%d)" % (os.path.basename(p),
                                                  *img.size))
        except Exception as e:
            messagebox.showerror("Save frame", str(e), parent=self.root)

    # ------------------------------------------------------------------
    # Display plumbing
    # ------------------------------------------------------------------
    def _rx_to_pil(self, arr):
        a = arr.astype(np.float32)
        c = float(self.v_contrast.get())
        b = float(self.v_bright.get())
        a = np.clip((a - 0.5) * c + 0.5 + b, 0.0, 1.0)
        if self.v_hflip.get():
            a = a[:, ::-1]
        if self.v_vflip.get():
            a = a[::-1, :]
        if a.ndim == 2:
            tint = dict(TINTS).get(self.v_tint.get(), (1, 1, 1))
            if tint != (1.0, 1.0, 1.0):
                a = np.stack([a * tint[0], a * tint[1], a * tint[2]],
                             axis=-1)
                return Image.fromarray((a * 255).astype(np.uint8), "RGB")
            return Image.fromarray((a * 255).astype(np.uint8), "L")
        return Image.fromarray((a * 255).astype(np.uint8), "RGB")

    def _draw_on_canvas(self, canvas, pil_img, key, aspect=None):
        if not HAVE_IMAGETK or pil_img is None:
            return
        cw = max(canvas.winfo_width(), 32)
        ch = max(canvas.winfo_height(), 32)
        iw, ih = pil_img.size
        if aspect:
            ar = aspect[0] / aspect[1]      # shown at the mode aspect,
            sc = min(cw / ar, ch)           # not the raw pixel ratio
            nw, nh = max(1, int(sc * ar)), max(1, int(sc))
        else:
            sc = min(cw / iw, ch / ih)
            nw, nh = max(1, int(iw * sc)), max(1, int(ih * sc))
        # Smooth = proper Lanczos reconstruction (closer to what a real
        # monitor's spot/diffuser shows); off = honest hard pixels.
        resample = (Image.LANCZOS if self.v_smooth.get()
                    else Image.NEAREST)
        img = pil_img.resize((nw, nh), resample)
        photo = ImageTk.PhotoImage(img)
        self._photos[key] = photo
        canvas.delete("all")
        canvas.create_image(cw // 2, ch // 2, image=photo)

    def _redraw_tx(self):
        if self.last_tx_pil is not None:
            self._draw_on_canvas(self.tx_canvas, self.last_tx_pil, "tx",
                                 aspect=self.current_mode()["aspect"])

    def _redraw_rx(self):
        if self.last_rx is not None:
            self._draw_on_canvas(self.rx_canvas,
                                 self._rx_to_pil(self.last_rx[0]), "rx",
                                 aspect=self.current_mode()["aspect"])

    # ------------------------------------------------------------------
    # Periodic UI pump
    # ------------------------------------------------------------------
    def _pump(self):
        got_tx = False
        try:
            while True:
                self.last_tx_pil = self.preview_q.get_nowait()
                got_tx = True
        except queue.Empty:
            pass
        got_rx = False
        try:
            while True:
                self.last_rx = self.frame_q.get_nowait()
                got_rx = True
        except queue.Empty:
            pass
        try:
            while True:
                self.set_status(self.status_q.get_nowait())
        except queue.Empty:
            pass

        if got_tx:
            self._redraw_tx()
        if got_rx:
            self._redraw_rx()
            if self.file_worker is not None and self.file_worker.is_alive():
                self.file_worker.feed(self.last_rx[0])
            info = self.last_rx[1]
            self.lbl_rx_info.configure(
                text="frame %d | %s | measured %.1f samples/line | "
                     "coasted lines %d"
                % (info.get("frame", 0),
                   "LOCKED" if info.get("locked") else
                   ("sync" if info.get("synced") else "searching"),
                   info.get("spl", 0.0), info.get("coasted", 0)))

        self._stat_tick += 1
        if self._stat_tick % 6 == 0:
            bits = []
            if self.tx:
                bits.append("TX %d frames" % self.tx.frames_sent)
                self.lbl_tx_info.configure(
                    text="TX live - frame %d" % self.tx.frames_sent)
            if self.rx:
                bits.append("rms %.2f" % self.rx.last_rms)
            bits.append("rings tx %2d%% rx %2d%%"
                        % (self.tx_ring.fill_fraction() * 100,
                           self.rx_ring.fill_fraction() * 100))
            bits.append("over/under %d/%d"
                        % (self.tx_ring.overruns + self.rx_ring.overruns,
                           self.tx_ring.underruns + self.rx_ring.underruns))
            self.lbl_stats.configure(text="  |  ".join(bits))
        self.root.after(40, self._pump)

    # ------------------------------------------------------------------
    def on_close(self):
        try:
            self.render_stop.set()
            if self.file_worker is not None:
                self.file_worker.stop()
            self.stop_tx()
            self.stop_rx()
        except Exception:
            pass
        s = dict(
            mode=self.v_mode.get(), rate=int(self.v_rate.get()),
            color=self.v_color.get(), filter=self.v_filter.get(),
            pattern=self.v_pattern.get(), fit=self.v_fit.get(),
            gain=float(self.v_gain.get()), bits=self.v_bits.get(),
            sharpen_src=bool(self.v_sharp.get()),
            loop=bool(self.v_loop.get()), sync=float(self.v_sync.get()),
            bright=float(self.v_bright.get()),
            contrast=float(self.v_contrast.get()),
            sat=float(self.v_sat.get()),
            integrate=int(self.v_integrate.get()),
            smooth=bool(self.v_smooth.get()), tint=self.v_tint.get(),
            capture=[v.get() for v in self.v_capture],
            in_dev=self.in_dev, out_dev=self.out_dev,
            record_path=self.record_path,
            file_ec=self.file_ec, file_repeat=self.file_repeat,
            file_save_dir=self.file_save_dir,
            custom_modes=[m for m in MODES
                          if m["name"].startswith("Custom ")])
        save_settings(s)
        self.root.destroy()


HELP_TEXT = """NBTV STUDIO - quick guide

WHAT IT IS
A transmit + receive workbench for narrow-band television over audio:
classic 30/32-line mechanical-TV style modes, plus experimental wideband
modes that use the full 24-bit / 192 kHz bandwidth of a modern sound card.

FIVE-MINUTE START (no cables, no hardware)
 1. Tick "Soft loopback".
 2. Press START TX.  The test card appears in the TX window.
 3. Press START RX.  The decoded picture appears in the RX window.
 4. Play: change test patterns, colour systems, modes, sync level.

REAL AUDIO
Use "Audio setup..." to pick sound devices.  Run a cable from line-out to
line-in (or use a virtual audio cable program) and choose
"Direct cable (full bandwidth)" as the TX filter for best quality.
For radio-style narrow band, pick a low-pass filter and a low-line mode.

SIGNAL FORMAT (honest note)
Levels and timing are NBTV-club inspired and fully self-consistent between
this program's TX and RX, with sync at -1.0, black at -0.4, peak white at
+1.0, ~12% line sync and a half-line broad frame pulse.  If you interface
with other NBTV hardware/software, expect to trim sync level, gain and
polarity - that is half the fun.

COLOUR SYSTEMS
 - Monochrome: one composite signal (both channels identical).
 - Frame-sequential: R, G, B sent as successive fields (double broad pulse
   marks the red field).  Needs a steady source; great for stills.
 - Line-sequential: line colours rotate R,G,B and precess each frame.
 - Stereo Y/C: left channel is a normal mono-compatible picture, right
   channel carries chroma.  A mono receiver still gets a proper B&W image.

TIPS
 - "Render WAV (offline)" makes perfect recordings with no audio hardware;
   "Decode WAV..." plays them back into the receiver at up to max speed.
 - Frame integration (2-16) cleans up noisy stills beautifully.
 - If the picture tears sideways: nudge "Sync level" or check that nothing
   is resampling your audio (Windows: set the device's default format to
   the same rate you chose here, e.g. 24-bit 192000 Hz, in Sound settings).
 - If the picture is upside-down or mirrored on real hardware, use the
   H/V flip boxes - scan conventions varied between builders.
 - Picture too dark/washy: Brightness/Contrast act on the display only;
   TX Gain changes the actual signal.

FILE TRANSFER OVER NBTV (QR carousel)
Yes, really: "File transfer..." sends a small file as a stream of QR-coded
video frames.  Each frame is one QR code carrying a chunk of the file; the
transmitter cycles through all chunks until the receiver has collected the
whole set, then the file is CRC-checked and saved.  QR error correction
plus endless repetition means no back-channel is needed - it is one-way,
like RTTY with pictures.  Quite possibly a first for NBTV.

How to use it:
 - Sender: "File transfer..." -> Choose file -> Send this file.  The QR
   size (and so the speed) is picked automatically from the current mode
   and filter; the dialog shows bytes-per-frame and seconds-per-pass.
 - Receiver: same dialog -> pick a save folder -> Arm receive -> START RX.
   Progress shows as chunks arrive; the file saves itself when complete.
 - Loopback works too: tick "Soft loopback", arm receive, send the file.

What to expect (measured, not guessed):
 - Direct cable works on EVERY mode, even Baird 30 and 32-line - at about
   15 bytes/frame there, up to ~1.3 kB/frame on X480 at 192 kHz.
 - Through a 10 kHz low-pass, the classic 32-line vertical-scan mode still
   transfers files (~125 B/s): genuine narrowband digital over NBTV.
 - Heavily filtered high-line modes are refused with an honest message:
   the low-pass physically smears the QR modules along the scan line.
 - Throughput is hundreds of bytes per second, not kilobytes.  This is a
   proof of concept and a party trick, not a modem.

Rules of thumb: Monochrome colour system, Direct cable when possible,
frame integration 1 (arming sets this for you), and keep files small -
a few KB transfers in seconds on wideband modes, a 50 KB file on a
32-line mode is a cup of tea.  Needs opencv-python on both ends.

This is hobby software for experimenting.  Have fun, break modes, invent
new ones with "Custom mode..." - 480 lines over a patch cable at 192 kHz
is a picture Baird would have given an arm for.

WHY DO QR FRAMES LOOK RAZOR SHARP BUT PHOTOS LOOK CHUNKY?
A 48-line mode carries exactly 48 picture rows - that is the medium, not a
bug.  QR data frames are generated natively on that scan grid (block
graphics suit it perfectly), while photos, video and screen grabs must be
resampled down to it, so fine detail physically cannot survive the trip.
NBTV Studio resamples in linear light with a Lanczos filter, and "Detail
boost" adds classic aperture correction so the few lines you do get carry
as much apparent detail as possible.  "Smooth" turns the chunky hard-pixel
preview into a proper Lanczos reconstruction - much closer to what a real
monitor's spot and phosphor show.  For still photos, try the X360/X480
photo-scan modes: ten times the lines down the same audio cable, just
slower."""


# ----------------------------------------------------------------------------
# Self-test: encode -> decode loopback for every colour system, headless.
# ----------------------------------------------------------------------------
def _selftest_one(mode, rate, cs, pattern, nframes, chunk=4096,
                  feed_offset=0):
    geom = Geometry(mode, rate)
    src = SourceManager()
    src.use_test(pattern)
    img = src.get_frame(0.0, geom.aspect)
    enc = Encoder(geom, cs, gain=0.9, fill=True)
    enc.set_frame(img)
    blocks = []
    for _ in range(nframes):
        audio, _prev = enc.encode_frame()
        blocks.append(audio)
    sig = np.concatenate(blocks)[feed_offset:]
    dec = Decoder(geom, cs)
    outs = []
    for i in range(0, len(sig), chunk):
        outs.extend(dec.feed(sig[i:i + chunk]))
    if not outs:
        return None, None, "no frames decoded"
    grid = source_to_grid(img, geom, True)
    if cs == "mono":
        exp = grid_to_display(rgb_to_y(grid), geom)
    else:
        exp = grid_to_display(grid, geom)
    got = outs[-1][0]
    if got.shape != exp.shape:
        return None, outs[-1][1], ("shape mismatch %s vs %s"
                                   % (got.shape, exp.shape))
    mae = float(np.mean(np.abs(got.astype(np.float64) - exp)))
    return mae, outs[-1][1], None


def selftest():
    print("NBTV Studio self-test")
    print("=" * 64)
    ok = True
    club = MODES[0]
    tests = [
        ("mono  / Colour bars / 48k", club, 48000, "mono",
         "Colour bars", 8, 0, 0.06),
        ("mono  / mid-stream start", club, 48000, "mono",
         "Grey staircase", 8, 1234, 0.06),
        ("fsc   / Colour bars / 48k", club, 48000, "fsc",
         "Colour bars", 10, 0, 0.12),
        ("lsc   / Colour bars / 48k", club, 48000, "lsc",
         "Colour bars", 10, 0, 0.12),
        ("yc    / Colour bars / 48k", club, 48000, "yc",
         "Colour bars", 8, 0, 0.12),
    ]
    x160 = next((m for m in MODES if m["name"].startswith("X160")), None)
    if x160:
        tests.append(("mono  / X160 wideband / 192k", x160, 192000, "mono",
                      "Crosshatch + circle test card", 4, 0, 0.06))
    for label, mode, rate, cs, pat, nf, off, thr in tests:
        try:
            mae, info, err = _selftest_one(mode, rate, cs, pat, nf,
                                           feed_offset=off)
        except Exception as e:
            import traceback
            traceback.print_exc()
            mae, info, err = None, None, "exception: %s" % e
        if err:
            print("FAIL  %-34s %s" % (label, err))
            ok = False
        else:
            good = mae < thr
            ok = ok and good
            print("%s  %-34s mae=%.4f (limit %.2f)  frames=%d locked=%s"
                  % ("PASS " if good else "FAIL ", label, mae, thr,
                     info.get("frame", 0), info.get("locked")))

    # WAV round trip (24-bit) ------------------------------------------------
    import tempfile
    try:
        mode = club
        geom = Geometry(mode, 48000)
        src = SourceManager()
        src.use_test("Colour bars")
        path = os.path.join(tempfile.gettempdir(), "nbtv_selftest.wav")
        render_wav_offline(geom, "mono", 0.9, None, src, True, path,
                           seconds=0.8, bits=24)
        rate, data = read_wav(path)
        dec = Decoder(geom, "mono")
        outs = []
        for i in range(0, len(data), 4096):
            outs.extend(dec.feed(data[i:i + 4096]))
        img = src.get_frame(0.0, geom.aspect)
        exp = grid_to_display(rgb_to_y(source_to_grid(img, geom, True)),
                              geom)
        if not outs:
            raise RuntimeError("no frames from WAV")
        mae = float(np.mean(np.abs(outs[-1][0] - exp)))
        good = (rate == 48000) and mae < 0.06
        ok = ok and good
        print("%s  %-34s mae=%.4f (limit 0.06, direct) rate=%d"
              % ("PASS " if good else "FAIL ", "WAV 24-bit round trip",
                 mae, rate))
        try:
            os.remove(path)
        except Exception:
            pass
    except Exception as e:
        import traceback
        traceback.print_exc()
        print("FAIL  WAV round trip: %s" % e)
        ok = False

    # File over NBTV (QR) round trips ----------------------------------------
    if get_cv2() is None:
        print("SKIP  file transfer tests (opencv-python not installed)")
    else:
        def _file_case(label, mode, rate, cutoff, nbytes, seed):
            geom = Geometry(mode, rate)
            rng = np.random.default_rng(seed)
            data = rng.integers(0, 256, nbytes, dtype=np.uint8).tobytes()
            snd = FileSender()
            snd.data, snd.name = data, "selftest.bin"
            snd.build(geom, cutoff, 1, 1)
            src = SourceManager()
            src.use_file_frames(snd.frames, snd.stats["period"], None)
            enc = Encoder(geom, "mono", 0.9, True)
            lpf = FIRLowpass(geom.rate, cutoff, channels=2)
            dec = Decoder(geom, "mono")
            rxr = FileReceiver(geom)
            t = 0.0
            for _ in range(len(snd.frames) + 3):
                enc.set_frame(src.get_frame(t, geom.aspect))
                t += 1.0 / geom.fps
                audio, _d = enc.encode_frame()
                for arr, _info in dec.feed(lpf.process(audio)):
                    rxr.offer(arr)
                if rxr.done:
                    break
            good = rxr.done and rxr.blob == data
            print("%s  %-34s %d B in %d QR frames%s"
                  % ("PASS " if good else "FAIL ", label, nbytes,
                     len(snd.frames),
                     "" if good else "  (incomplete or mismatch)"))
            return good

        try:
            if x160:
                ok = _file_case("file / X160 192k direct", x160, 192000,
                                None, 200, 42) and ok
            ok = _file_case("file / 32-line V 10kHz LPF", club, 48000,
                            10000.0, 24, 7) and ok
        except Exception as e:
            import traceback
            traceback.print_exc()
            print("FAIL  file transfer: %s" % e)
            ok = False

    print("=" * 64)
    print("RESULT:", "ALL TESTS PASSED" if ok else "SOME TESTS FAILED")
    return ok


def list_devices():
    sd = get_sd()
    if sd is None:
        print("sounddevice is not installed.  pip install sounddevice")
        return
    print(sd.query_devices())


def run_gui():
    if not HAVE_TK:
        print("tkinter is not available in this Python.  On Debian/Ubuntu: "
              "sudo apt install python3-tk")
        sys.exit(1)
    root = tk.Tk()
    app = App(root)
    root.mainloop()


def main(argv=None):
    ap = argparse.ArgumentParser(
        description="NBTV Studio - narrow-band television over audio, "
                    "TX and RX, classic and experimental wideband modes.")
    ap.add_argument("--selftest", action="store_true",
                    help="run headless encode->decode tests and exit")
    ap.add_argument("--list-devices", action="store_true",
                    help="list audio devices and exit")
    args = ap.parse_args(argv)
    if args.selftest:
        sys.exit(0 if selftest() else 1)
    if args.list_devices:
        list_devices()
        sys.exit(0)
    run_gui()


if __name__ == "__main__":
    main()
