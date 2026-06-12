# NBTV Studio By VA3JFL

**Narrow-band television over a sound card.** Transmit and receive mechanical-television-style video through audio — over a real cable, a WAV file, or a pure software loopback. Classic 32-line club NBTV, Baird 30-line, and experimental wideband modes up to 480 lines that turn a 24-bit/192 kHz sound card into a surprisingly capable video link. It can even send *files* over the video channel as a stream of QR frames.


[NBTV Studio main window](screenshot.jpg)
[Image Quality](qualitiy.png)


---

## Features

- **Full TX and RX in one window** — live side-by-side transmit and receive monitors, with a soft-loopback mode that needs no cable or audio hardware at all.
- **16 built-in modes** from Baird 30-line to 480-line photo-scan, plus a **custom mode editor**. Horizontal resolution follows the sample rate: `px/line ≈ 0.80 × sample_rate / (lines × fps)`.
- **Four colour systems**: monochrome, frame-sequential (R,G,B fields), line-sequential (R,G,B lines), and **Stereo Y/C** — luma on the left channel, alternating U/V chroma on the right, so a mono receiver still gets a clean black-and-white picture.
- **Any source**: built-in test cards, still images, animated GIFs, video files, webcam, or live screen capture of a region.
- **File over NBTV (QR)** — send any file as a cycling stream of QR-code frames; the receiver collects chunks in any order until the file is complete and CRC-verified.
- **Quality-first resampling**: sources are downscaled in linear light with a two-stage Lanczos filter, with optional **Detail boost** (classic TV aperture correction) so the few scan lines you get carry maximum apparent detail. Machine-generated frames at exact grid size (e.g. QR) bypass resampling entirely and arrive pixel-perfect.
- **Robust decoder**: edge-driven flywheel sync with coasting through missing pulses, per-line black clamp, frame integration (great for cleaning up noisy stills), brightness / contrast / saturation / tint, H/V flip, save-frame to PNG.
- **Authentic output filtering**: selectable low-pass filters (3.4–20 kHz) for real narrow-band conditions, or full-bandwidth "direct cable" mode.
- **WAV workflow**: record TX live (16/24-bit), render perfect WAVs offline faster than real time with no audio hardware, and decode WAVs back into the receiver at up to maximum speed.
- **Headless self-test**: encode→decode round trips for every colour system, a WAV round trip, and QR file-transfer round trips.

## Requirements

| Dependency | Needed for | Install |
|---|---|---|
| Python 3.8+ with Tkinter | the GUI | Debian/Ubuntu: `sudo apt install python3-tk` |
| `numpy`, `Pillow` | **required** (codec + UI) | `pip install numpy pillow` |
| `sounddevice` | live audio in/out | `pip install sounddevice` |
| `opencv-python` | video files, webcam, QR file transfer | `pip install opencv-python` |
| `mss` | faster screen capture (optional; falls back to PIL) | `pip install mss` |

Soft loopback, WAV render, and WAV decode all work **without** `sounddevice` — you only need it to drive a real cable. Wideband modes (96+ lines) need a 96/192 kHz-capable sound device when used live.

## Quick start

```bash
pip install numpy pillow sounddevice opencv-python
python nbtv_studio.py
```

First picture in 30 seconds, no hardware needed:

1. Leave **Mode** on a club mode and tick **Soft loopback**.
2. Press **START TX**, then **START RX**.
3. You should see the test card appear on the receive side, locked. Now switch **Source** to *Image file*, *Video file*, or *Webcam* and press **Load / configure…**.

To go over a real cable, plug line-out into line-in (or use two machines), pick devices under **Audio setup…**, untick soft loopback, and trim **Sync level** until the picture locks.

```bash
python nbtv_studio.py --selftest       # headless codec round-trip tests
python nbtv_studio.py --list-devices   # show audio devices
```

## Modes

| Mode | Lines | fps | Aspect | Scan | Notes |
|---|---|---|---|---|---|
| NBTV Club 32-line | 32 | 12.5 | 2:3 | vertical | the classic club standard |
| Baird 30-line | 30 | 12.5 | 3:7 | vertical | 1930s portrait format |
| Experimental 24-line | 24 | 12.5 | 2:3 | vertical | |
| Club 48-line | 48 | 12.5 | 4:3 | horizontal | |
| 1931-era 60-line | 60 | 20 | 4:3 | horizontal | |
| 90-line | 90 | 12.5 | 4:3 | horizontal | |
| Mid-30s 120-line | 120 | 12.5 | 4:3 | horizontal | |
| X96 wideband | 96 | 12.5 | 4:3 | horizontal | 96 kHz+ |
| X120 wideband | 120 | 25 | 4:3 | horizontal | 192 kHz |
| X160 wideband | 160 | 12.5 | 4:3 | horizontal | 192 kHz |
| X240 hi-def slow | 240 | 6.25 | 4:3 | horizontal | 192 kHz |
| X288 hi-def slow | 288 | 6.25 | 4:3 | horizontal | 192 kHz |
| X360 photo-scan | 360 | 3.125 | 4:3 | horizontal | 192 kHz — stills |
| X480 photo-scan | 480 | 2 | 4:3 | horizontal | 192 kHz — stills |
| UltraWide 32 | 32 | 12.5 | 2:3 | vertical | maximum horizontal detail |

New modes can be added live with **Custom mode…** — 480 lines over a patch cable at 192 kHz is a picture Baird would have given an arm for.

## Colour systems

- **Monochrome** — one composite signal, both channels identical.
- **Frame-sequential** — R, G, B sent as successive fields; the red field carries broad pulses on its first *two* lines so the receiver identifies the sequence. Needs a steady source; great for stills.
- **Line-sequential** — line colours rotate R, G, B and precess each frame.
- **Stereo Y/C** — left channel is a fully mono-compatible picture; the right channel carries alternating U/V chroma at 0.5 baseline.

## File over NBTV (QR)

Pick **Source → File over NBTV (QR)** and load any file. The sender splits it into base64 chunks, prepends a manifest frame (chunk count, length, CRC-32) and a filename frame, renders each as a QR code fitted to the current mode's scan grid, and cycles through them continuously. The receiver tries every QR version that fits the geometry, scores candidates by their finder patterns, and collects chunks in any order — start listening mid-stream and it still completes. Error-correction level and per-frame repeat (1–5) are adjustable; the dialog shows the effective data rate before you start. Wideband modes move files dramatically faster. Requires `opencv-python` on both ends.

## Picture quality

A 48-line mode carries exactly 48 picture rows — that's the medium. QR frames look razor sharp because they're generated natively on the scan grid; photos and video must be resampled down to it. NBTV Studio gets the most out of those lines:

- Downscaling runs in **linear light** with a two-stage Lanczos filter, so fine detail averages to the correct brightness instead of going muddy.
- **Detail boost** (TRANSMIT panel) applies mild aperture correction after the downscale — the same trick classic TV cameras used. On by default; toggles live; never touches exact-grid data frames.
- **Smooth** (RECEIVE panel) switches both monitors from honest hard pixels to a proper Lanczos reconstruction — much closer to what a real monitor's spot and phosphor show. *Save frame…* honours it too.
- Frame integration (2–16) on the receiver cleans up noisy stills beautifully.
- For photographs, try the X360/X480 photo-scan modes: ten times the lines down the same cable, just slower.

## Signal format

Levels and timing are NBTV-club inspired and fully self-consistent between this program's TX and RX:

```
video units 0..1 :  sync tip = 0.00, black = 0.30, peak white = 1.00
audio mapping    :  a = 2·v − 1   →  sync −1.0, black −0.4, white +1.0
line sync        :  12 % of the line period (per-mode adjustable)
frame sync       :  broad 50 % pulse on the first line of each frame
frame-sequential :  red field carries broad pulses on its first TWO lines
```

Interfacing with other NBTV hardware or software will likely mean trimming sync level, gain and polarity — that is half the fun.

## Troubleshooting

- **No audio devices / "sounddevice not installed"** — `pip install sounddevice`. Loopback and WAV workflows run fine without it.
- **Picture tears sideways** — nudge **Sync level**, and make sure nothing in the chain is filtering below the mode's bandwidth.
- **Wideband mode sounds wrong / won't lock live** — your device must genuinely run at 96/192 kHz; check **Audio setup…**.
- **Video/webcam/file-transfer greyed out or erroring** — install `opencv-python`.
- **Verify the codec itself** — `python nbtv_studio.py --selftest` should print `ALL TESTS PASSED`.

## License

Released under the [MIT License](LICENSE).
