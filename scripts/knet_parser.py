"""
Parser for NIED K-NET ASCII strong-motion waveform files (the standard
17-line-header + fixed-width-integer-count format served by
kyoshin.bosai.go.jp). Converts raw digitized counts to physical acceleration
(gal = cm/s^2) using each file's own Scale Factor line -- do not hardcode a
conversion factor, it differs per file/event.
"""
import re
from pathlib import Path

import numpy as np

HEADER_LINES = 17


HEADER_LABEL_WIDTH = 18  # K-NET ASCII headers are fixed-width: label padded to col 18, value after


def _parse_header(lines):
    h = {}
    for line in lines[:HEADER_LINES]:
        key = line[:HEADER_LABEL_WIDTH].strip()
        val = line[HEADER_LABEL_WIDTH:].strip()
        if key:
            h[key] = val
    return h


def read_knet_ascii(path):
    """Returns dict with header fields plus `accel_gal` (np.ndarray, cm/s^2)
    and `dt` (s, from Sampling Freq)."""
    path = Path(path)
    lines = path.read_text().splitlines()
    header = _parse_header(lines)

    scale = header["Scale Factor"]  # e.g. "7845(gal)/8223790"
    m = re.match(r"([\d.]+)\([a-zA-Z]+\)/([\d.]+)", scale)
    num, den = float(m.group(1)), float(m.group(2))
    scale_factor = num / den

    sf_hz = float(re.match(r"([\d.]+)", header["Sampling Freq(Hz)"]).group(1))
    dt = 1.0 / sf_hz

    data_lines = lines[HEADER_LINES:]
    counts = np.array(
        [int(v) for line in data_lines for v in line.split()], dtype=np.float64
    )
    # Raw counts carry a nonzero DC offset (sensor baseline); K-NET's own
    # "Max. Acc." header value is computed after removing it. Whole-record
    # mean subtraction reproduces the header's stated peak to 4 decimal
    # places on the sample file checked here (39.6585 vs 39.658 gal) --
    # simpler than isolating a pre-event window, and close enough because
    # these records are short (order 100s) with shaking a small fraction.
    counts = counts - counts.mean()
    accel_gal = counts * scale_factor

    return dict(
        header=header,
        event_id=path.stem,
        component=path.suffix.lstrip("."),  # EW / NS / UD
        origin_time=header["Origin Time"],
        magnitude=float(header["Mag."]),
        station_code=header["Station Code"],
        dt=dt,
        sampling_freq_hz=sf_hz,
        duration_s=float(re.match(r"([\d.]+)", header["Duration Time(s)"]).group(1)),
        accel_gal=accel_gal,
    )


def read_event(event_dir, event_id):
    """Reads all 3 components (EW, NS, UD) for one event. Returns dict of
    component -> read_knet_ascii() result. Missing components are omitted,
    not padded -- callers should check which directions are actually present."""
    event_dir = Path(event_dir)
    out = {}
    for comp in ("EW", "NS", "UD"):
        p = event_dir / f"{event_id}.{comp}"
        if p.exists():
            out[comp] = read_knet_ascii(p)
    return out


if __name__ == "__main__":
    import sys

    p = Path(sys.argv[1]) if len(sys.argv) > 1 else Path(
        "data/raw_ground_motion/selecteddata/knet/ascii/IBR0071809050511.EW"
    )
    rec = read_knet_ascii(p)
    print(f"event_id={rec['event_id']} component={rec['component']} "
          f"mag={rec['magnitude']} dt={rec['dt']}s n_samples={len(rec['accel_gal'])} "
          f"duration={rec['duration_s']}s")
    print(f"peak accel = {np.abs(rec['accel_gal']).max():.2f} gal "
          f"(header Max. Acc. = {rec['header']['Max. Acc. (gal)']})")
