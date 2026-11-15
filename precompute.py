import json
import cv2
import numpy as np

# ---------------------------------------------------------------------------
# CONFIGURATION (Matches original defaults)
# ---------------------------------------------------------------------------
VIDEO_IN = "input.mp4"
VIDEO_OUT = "motion.mp4"
DATA_OUT = "sonification_data.json"

PANEL_WIDTH = 640
SCALE_L = 24.0
N_BINS = 10

# Flow & Mask defaults (from original trackbars)
MIN_SPD = 0.3          # Min Spd x0.1 = 3 -> 0.3
COH_THRESH = 0.60      # Coherence % = 60
TIME_ALPHA = 0.30      # Time Alpha % = 30
DERIV_SMOOTH = 3.0     # Deriv Smooth = 3
BG_CUT = 12            # BG Cut = 12

# Gain & Curve defaults
LOW_FLOOR = 0.0
CURL_GAMMA = 5.0
CONC_GAMMA = 1.0
AMP_REF = 0.10
AMP_GAMMA = 1.0

# Vortex Tracker defaults
VORT_SMOOTH = 3.0
VORT_MIN = 0.50
VORT_MIN_AREA = 100
VORT_MAX_ASPECT = 2.5
VORT_MIN_FILL = 0.40
VORT_MEAN_MARGIN = 0.03
VORT_MIN_LIFE = 0.30
MAX_VORTICES = 20
MATCH_MIN_DIST = 36
TRACK_TIMEOUT = 0.5
VEL_MIN = 0.06
REPEAT_SLOW = 3.0
REPEAT_FAST = 0.5
STRIKE_JUMP = 0.12
RUNG_LO, RUNG_HI = 3, 17

C_ROOT = 130.8128
PENTA = np.array([1.0, 9 / 8, 5 / 4, 3 / 2, 5 / 3])


def ladder_freq(k):
    return float(C_ROOT * PENTA[k % 5] * (2.0 ** (k // 5)))


def vortex_gain(strength, floor):
    return floor + (1.0 - floor) * float(np.clip(strength, 0.0, 1.0)) ** CURL_GAMMA


def bin_gains(gamma, floor):
    return floor + (1.0 - floor) * (np.arange(N_BINS) / (N_BINS - 1.0)) ** gamma


def to_concentration(bgr_img):
    hsv = cv2.cvtColor(bgr_img, cv2.COLOR_BGR2HSV)
    return hsv[..., 1].astype(np.float32) / 255.0


def to_lcn_feature(bgr_img):
    v = np.max(bgr_img, axis=2).astype(np.float32)
    local_mean = cv2.GaussianBlur(v, (0, 0), sigmaX=12)
    diff = v - local_mean
    local_std = np.sqrt(cv2.GaussianBlur(diff * diff, (0, 0), sigmaX=12) + 25.0)
    lcn = np.clip((diff / local_std) * 45.0 + 128.0, 0, 255).astype(np.uint8)
    return v, lcn


def compute_curl_spectrum(u, v_cv, cos_lat, smooth_sigma=3.0):
    k = max(1.0, smooth_sigma)
    u_s = cv2.GaussianBlur(u, (0, 0), sigmaX=k)
    v_north_s = cv2.GaussianBlur(-v_cv, (0, 0), sigmaX=k)

    inv_cos = 1.0 / np.maximum(cos_lat, 0.25)
    du_dx = cv2.Sobel(u_s, cv2.CV_32F, 1, 0, ksize=3) * 0.125 * inv_cos
    du_dy = -cv2.Sobel(u_s, cv2.CV_32F, 0, 1, ksize=3) * 0.125
    dv_dx = cv2.Sobel(v_north_s, cv2.CV_32F, 1, 0, ksize=3) * 0.125 * inv_cos
    dv_dy = -cv2.Sobel(v_north_s, cv2.CV_32F, 0, 1, ksize=3) * 0.125

    zeta = dv_dx - du_dy
    s_n = du_dx - dv_dy
    s_s = dv_dx + du_dy

    zeta2 = zeta * zeta
    strain2 = s_n * s_n + s_s * s_s
    ow_purity = zeta2 / (zeta2 + strain2 + 1e-4)

    spd = np.hypot(u_s, v_north_s)
    spin_mag = np.abs(zeta) * SCALE_L
    spin_ratio = spin_mag / (spin_mag + spd + 1e-4)

    return np.clip(ow_purity * spin_ratio, 0.0, 1.0)


def region_stats(curl, conc, mask, spd, g_curl, g_conc):
    if not np.any(mask):
        return {
            "amp": 0.0,
            "mean_conc": 0.0,
            "mean_spd": 0.0,
            "conc_eff": [0.0] * N_BINS,
        }

    n_conc, _ = np.histogram(conc[mask], bins=N_BINS, range=(0.0, 1.0))
    conc_eff = (n_conc / n_conc.sum()) * (g_conc ** 2)
    amp = float(np.clip(np.count_nonzero(mask) / mask.size / AMP_REF, 0.0, 1.0) ** AMP_GAMMA)

    return {
        "amp": round(amp, 4),
        "mean_conc": round(float(np.mean(conc[mask])), 4),
        "mean_spd": round(float(np.mean(spd[mask])), 4),
        "conc_eff": [round(float(x), 5) for x in conc_eff],
    }


def rung_from_y(y, height):
    return RUNG_LO + (1.0 - float(np.clip(y / height, 0.0, 1.0))) * (RUNG_HI - RUNG_LO)


class VortexTracker:
    def __init__(self, width, height):
        self.W, self.H = width, height
        self.tracks = []
        self.next_id = 0
        self.rng = np.random.default_rng(5)

    def detect(self, curl_idx, moving_mask, vmin):
        s = cv2.GaussianBlur(curl_idx.astype(np.float32), (0, 0), VORT_SMOOTH)
        m = ((s > vmin) & moving_mask).astype(np.uint8)
        n, lab, stats, _ = cv2.connectedComponentsWithStats(m, connectivity=8)
        dets = []
        for k in range(1, n):
            x, y, w, h, area = (int(v) for v in stats[k])
            if area < VORT_MIN_AREA:
                continue
            if max(w, h) > VORT_MAX_ASPECT * max(min(w, h), 1) or area < VORT_MIN_FILL * w * h:
                continue
            comp = lab[y:y + h, x:x + w] == k
            sub = s[y:y + h, x:x + w]
            if float(sub[comp].mean()) < vmin + VORT_MEAN_MARGIN:
                continue
            wt = np.where(comp, sub ** 2, 0.0)
            tot = float(wt.sum()) + 1e-9
            yy, xx = np.mgrid[0:h, 0:w]
            dets.append({
                "x": x + float((wt * xx).sum()) / tot,
                "y": y + float((wt * yy).sum()) / tot,
                "strength": float(sub[comp].max()),
                "size": float(np.sqrt(area / np.pi)),
            })
        dets.sort(key=lambda d: -d["strength"])
        return dets[:MAX_VORTICES]

    def _strike(self, tr, now, vmin):
        r = rung_from_y(tr["y"], self.H)
        if tr["rung"] is None or abs(r - tr["rung"]) > 0.8:
            tr["rung"] = int(round(r))
        u = float(np.clip((tr["strength"] - vmin) / max(1.0 - vmin, 1e-3), 0.0, 1.0))
        period = (REPEAT_SLOW + (REPEAT_FAST - REPEAT_SLOW) * u) * (0.85 + 0.3 * self.rng.random())
        tr["last_strike"] = now
        tr["strike_strength"] = tr["strength"]
        tr["next_strike"] = now + period
        quad = (1 if tr["x"] >= self.W / 2 else 0) + (2 if tr["y"] >= self.H / 2 else 0)
        return {
            "freq": round(ladder_freq(tr["rung"]), 2),
            "vel": 0.0,
            "decay": round(0.5 + 1.6 * float(np.clip(tr["size"] / 40.0, 0.0, 1.0)), 3),
            "pan": round(float(-0.85 + 1.7 * np.clip(tr["x"] / self.W, 0.0, 1.0)), 3),
            "quad": quad,
            "strength": round(tr["strength"], 3),
            "rung": tr["rung"],
        }

    def update(self, curl_idx, moving_mask, now, vmin, floor):
        dets = self.detect(curl_idx, moving_mask, vmin)
        unmatched = list(self.tracks)
        touched = []
        for d in dets:
            best, best_d = None, max(MATCH_MIN_DIST, 1.5 * d["size"])
            for tr in unmatched:
                dist = np.hypot(tr["x"] - d["x"], tr["y"] - d["y"])
                if dist < best_d:
                    best, best_d = tr, dist
            if best is None:
                best = {
                    "id": self.next_id, "x": d["x"], "y": d["y"], "strength": d["strength"],
                    "size": d["size"], "rung": None, "last_strike": None,
                    "strike_strength": 0.0, "next_strike": 0.0, "last_seen": now, "born": now
                }
                self.next_id += 1
                self.tracks.append(best)
            else:
                unmatched.remove(best)
                for key in ("x", "y", "size"):
                    best[key] = 0.6 * d[key] + 0.4 * best[key]
                best["strength"] = 0.5 * d["strength"] + 0.5 * best["strength"]
            best["last_seen"] = now
            touched.append(best)

        self.tracks = [t for t in self.tracks if now - t["last_seen"] <= TRACK_TIMEOUT]

        events = []
        active_markers = []
        for tr in touched:
            g = vortex_gain(tr["strength"], floor)
            audible = (g >= VEL_MIN) and (now - tr["born"] >= VORT_MIN_LIFE)
            quad = (1 if tr["x"] >= self.W / 2 else 0) + (2 if tr["y"] >= self.H / 2 else 0)

            if audible:
                first = tr["last_strike"] is None
                due = now >= tr["next_strike"]
                jump = (tr["strength"] - tr["strike_strength"] > STRIKE_JUMP
                        and now - (tr["last_strike"] or 0.0) > 0.2)
                if first or due or jump:
                    ev = self._strike(tr, now, vmin)
                    ev["vel"] = round(float(g), 3)
                    events.append(ev)

            just_struck = (tr["last_strike"] is not None) and ((now - tr["last_strike"]) < 0.35)
            active_markers.append({
                "x": round(tr["x"] / self.W, 4),
                "y": round(tr["y"] / self.H, 4),
                "r": round(max(6.0, tr["size"]) / self.W, 4),
                "q": quad,
                "aud": 1 if audible else 0,
                "ring": round((now - tr["last_strike"]) / 0.35, 2) if just_struck else -1
            })

        return events, active_markers


def main():
    cap = cv2.VideoCapture(VIDEO_IN)
    if not cap.isOpened():
        raise FileNotFoundError(f"Cannot open {VIDEO_IN}")

    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    ret, frame = cap.read()
    if not ret:
        raise RuntimeError("Empty video")

    h_orig, w_orig = frame.shape[:2]
    panel_h = int(h_orig * (PANEL_WIDTH / w_orig))
    # Ensure even height for H.264 encoding
    if panel_h % 2 != 0:
        panel_h += 1

    mid_x, mid_y = PANEL_WIDTH // 2, panel_h // 2
    quad_boxes = [
        (0, 0, mid_x, mid_y),
        (mid_x, 0, PANEL_WIDTH, mid_y),
        (0, mid_y, mid_x, panel_h),
        (mid_x, mid_y, PANEL_WIDTH, panel_h),
    ]

    # Try H.264 ('avc1') first for universal web playback, fallback to 'mp4v'
    fourcc = cv2.VideoWriter_fourcc(*"avc1")
    writer = cv2.VideoWriter(VIDEO_OUT, fourcc, fps, (PANEL_WIDTH, panel_h))
    if not writer.isOpened():
        print("avc1 codec unavailable, falling back to mp4v (re-encode with ffmpeg for Safari/iOS if needed).")
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        writer = cv2.VideoWriter(VIDEO_OUT, fourcc, fps, (PANEL_WIDTH, panel_h))

    lat_rad = np.linspace(np.pi / 2, -np.pi / 2, panel_h, dtype=np.float32)[:, None]
    cos_lat = np.maximum(np.cos(lat_rad), 0.15)

    curr_panel = cv2.resize(frame, (PANEL_WIDTH, panel_h), interpolation=cv2.INTER_AREA)
    prev_v, prev_lcn = to_lcn_feature(curr_panel)
    ema_flow = np.zeros((panel_h, PANEL_WIDTH, 2), dtype=np.float32)

    g_curl = bin_gains(CURL_GAMMA, LOW_FLOOR)
    g_conc = bin_gains(CONC_GAMMA, LOW_FLOOR)
    tracker = VortexTracker(PANEL_WIDTH, panel_h)

    frames_data = []
    frame_idx = 0

    print(f"Processing {total_frames} frames at {fps:.2f} FPS...")

    while True:
        now = frame_idx / fps
        if frame_idx > 0:
            curr_v, curr_lcn = to_lcn_feature(curr_panel)
            flow = cv2.calcOpticalFlowFarneback(
                prev_lcn, curr_lcn, None,
                pyr_scale=0.5, levels=3, winsize=15,
                iterations=3, poly_n=5, poly_sigma=1.2, flags=0
            )
            ema_flow = TIME_ALPHA * flow + (1.0 - TIME_ALPHA) * ema_flow
            prev_v, prev_lcn = curr_v, curr_lcn

        u = ema_flow[..., 0] * cos_lat
        v_cv = ema_flow[..., 1]
        spd = np.hypot(u, v_cv)

        u_avg = cv2.GaussianBlur(u, (0, 0), sigmaX=7)
        v_avg = cv2.GaussianBlur(v_cv, (0, 0), sigmaX=7)
        spd_avg = cv2.GaussianBlur(spd, (0, 0), sigmaX=7)
        coherence = np.hypot(u_avg, v_avg) / (spd_avg + 1e-5)

        moving_mask = (prev_v > BG_CUT) & (coherence >= COH_THRESH) & (spd >= MIN_SPD)
        curl_idx = compute_curl_spectrum(u, v_cv, cos_lat, smooth_sigma=DERIV_SMOOTH)
        conc_map = to_concentration(curr_panel)

        # Render clean motion map video frame (overlays handled dynamically in HTML)
        curl_u8 = np.clip(curl_idx * 255.0, 0, 255).astype(np.uint8)
        regime_map = cv2.applyColorMap(curl_u8, cv2.COLORMAP_TURBO)
        regime_map[~moving_mask] = (0, 0, 0)
        writer.write(regime_map)

        # Quadrant statistics
        q_stats = []
        for (x0, y0, x1, y1) in quad_boxes:
            sl = (slice(y0, y1), slice(x0, x1))
            q_stats.append(region_stats(curl_idx[sl], conc_map[sl], moving_mask[sl], spd[sl], g_curl, g_conc))

        # Whole-frame summary stats for "All 4 Quadrants" mode
        all_stat = region_stats(curl_idx, conc_map, moving_mask, spd, g_curl, g_conc)
        all_stat["amp"] = round(float(np.mean([s["amp"] for s in q_stats])), 4)

        events, markers = tracker.update(curl_idx, moving_mask, now, VORT_MIN, LOW_FLOOR)

        frames_data.append({
            "t": round(now, 4),
            "q": q_stats,
            "all": all_stat,
            "ev": events,
            "mk": markers
        })

        frame_idx += 1
        if frame_idx % 60 == 0:
            print(f"  Processed {frame_idx}/{total_frames} frames...")

        ret, next_frame = cap.read()
        if not ret:
            break
        curr_panel = cv2.resize(next_frame, (PANEL_WIDTH, panel_h), interpolation=cv2.INTER_AREA)

    cap.release()
    writer.release()

    output_payload = {
        "fps": round(fps, 3),
        "width": PANEL_WIDTH,
        "height": panel_h,
        "frames": frames_data
    }

    with open(DATA_OUT, "w", encoding="utf-8") as f:
        json.dump(output_payload, f, separators=(",", ":"))

    print(f"Done! Created {VIDEO_OUT} and {DATA_OUT} ({len(frames_data)} frames).")


if __name__ == "__main__":
    main()