# streamlit_app.py
import io
from pathlib import Path
import numpy as np
import pandas as pd
import librosa
import altair as alt
import streamlit as st

# ---------------- core utilities ----------------

def moving_average(x: np.ndarray, win: int) -> np.ndarray:
    """NaN-aware moving average with reflect padding at the edges."""
    if win <= 1:
        return x
    x = np.asarray(x, dtype=float)
    n = x.shape[0]
    if n == 0 or n == 1:
        return x

    # handle NaNs with simple forward/backward fill before convolution
    mask = np.isfinite(x)
    if not np.all(mask):
        if not mask.any():
            # all NaN -> return as-is (all NaN)
            return x
        x = x.copy()
        # forward fill
        idx = np.where(mask, np.arange(n), 0)
        np.maximum.accumulate(idx, out=idx)
        x_ff = x[idx]
        # backward fill
        idxb = np.where(mask, np.arange(n), n - 1)
        idxb = np.minimum.accumulate(idxb[::-1])[::-1]
        x_bf = x[idxb]
        x[~mask] = (x_ff[~mask] + x_bf[~mask]) / 2.0

    k = int(win)
    if k < 1:
        return x

    pad = k // 2
    x_pad = np.pad(x, pad_width=pad, mode="reflect")
    kernel = np.ones(k, dtype=float) / k
    y = np.convolve(x_pad, kernel, mode="valid")
    return y


def norm_0_1(x: np.ndarray, robust: bool = True) -> np.ndarray:
    """Normalize array into [0, 1] with NaN-awareness and robust percentile option."""
    x = np.asarray(x, dtype=float)
    valid = np.isfinite(x)
    if not valid.any():
        # no valid data -> all NaN
        return np.full_like(x, np.nan)

    xv = x[valid]

    if robust:
        lo, hi = np.nanpercentile(xv, 5), np.nanpercentile(xv, 95)
        if not np.isfinite(lo) or not np.isfinite(hi) or hi <= lo:
            lo, hi = np.nanmin(xv), np.nanmax(xv)
    else:
        lo, hi = np.nanmin(xv), np.nanmax(xv)

    # Still degenerate -> return NaNs
    if not np.isfinite(lo) or not np.isfinite(hi) or hi <= lo:
        out = np.full_like(x, np.nan)
        return out

    y = (x - lo) / (hi - lo)
    y[~valid] = np.nan
    return np.clip(y, 0.0, 1.0)


# ---------------- features ----------------

def feature_pitch_violation(y: np.ndarray, sr: int, hop_length: int, frame_length: int,
                            fmin_hz=librosa.note_to_hz('C2'),
                            fmax_hz=librosa.note_to_hz('C7'),
                            smooth_win_frames: int = 9) -> np.ndarray:
    # Make YIN use an odd window to satisfy strict bound fmin > sr/(frame_length-1)
    yin_frame = int(frame_length) | 1  # ensure odd

    # Harmonic component for stable F0
    y_harm = librosa.effects.hpss(y)[0]

    # lower bound from librosa's constraint
    lb = sr / float(max(2, yin_frame - 1))
    yin_min = np.nextafter(lb, np.inf)
    fmin_safe = float(max(fmin_hz, yin_min))

    # keep upper bound sane
    fmax_safe = float(min(fmax_hz, sr / 2.0))
    if not (fmax_safe > fmin_safe):
        fmax_safe = fmin_safe * 2.0

    f0 = librosa.yin(
        y_harm,
        fmin=fmin_safe,
        fmax=fmax_safe,
        sr=sr,
        frame_length=yin_frame,
        hop_length=hop_length,
        trough_threshold=0.1,
    )

    f0[f0 <= 0] = np.nan
    f0_smooth = moving_average(f0, smooth_win_frames)
    return f0_smooth


def feature_spectral_centroid(y: np.ndarray, sr: int, hop_length: int,
                              frame_length: int,
                              smooth_win_frames: int = 9) -> np.ndarray:
    sc = librosa.feature.spectral_centroid(
        y=y,
        sr=sr,
        n_fft=frame_length,
        hop_length=hop_length,
        center=True,
        win_length=frame_length,
    )[0]
    sc_smooth = moving_average(sc, smooth_win_frames)
    return sc_smooth


def feature_inverse_ioi(y: np.ndarray,
                        sr: int,
                        hop_length: int,
                        window_s: float = 1.0,
                        center_onsets: bool = True,
                        smooth_win_frames: int = 9):
    o_env = librosa.onset.onset_strength(y=y, sr=sr)
    onsets_frames = librosa.onset.onset_detect(onset_envelope=o_env, sr=sr, units='frames')
    onset_times = librosa.frames_to_time(onsets_frames, sr=sr)
    onsets_samples = librosa.time_to_samples(onset_times, sr=sr)

    if len(onset_times) < 2:
        n_frames = int(np.ceil(len(y) / hop_length))
        return np.full(n_frames, np.nan), onset_times

    ioi = np.diff(onset_times)
    inv_ioi_values = 1.0 / ioi
    frame_times = librosa.frames_to_time(
        np.arange(int(np.ceil(len(y) / hop_length))), sr=sr, hop_length=hop_length
    )

    inv_ioi_series = np.full_like(frame_times, np.nan, dtype=float)
    half_w = max(1e-6, window_s / 2.0)

    for i, t_on in enumerate(onset_times[:-1]):
        v = inv_ioi_values[i]
        start = t_on - half_w if center_onsets else t_on
        end = t_on + half_w if center_onsets else t_on + window_s
        m = (frame_times >= start) & (frame_times < end)
        inv_ioi_series[m] = v

    inv_ioi_smooth = moving_average(inv_ioi_series, smooth_win_frames)
    return inv_ioi_smooth, onset_times


# ---------------- tension computation ----------------

def compute_tension(y: np.ndarray,
                    sr: int,
                    n_fft: int,
                    hop_length: int,
                    smooth_win_frames: int,
                    ioi_window_s: float,
                    w_pitch: float,
                    w_centroid: float,
                    w_ioi: float):
    # raw features
    f0_hz = feature_pitch_violation(
        y, sr, hop_length, n_fft,
        smooth_win_frames=smooth_win_frames
    )
    sc_hz = feature_spectral_centroid(
        y, sr, hop_length, n_fft,
        smooth_win_frames=smooth_win_frames
    )
    inv_ioi, onset_times = feature_inverse_ioi(
        y, sr, hop_length, window_s=ioi_window_s,
        center_onsets=True,
        smooth_win_frames=smooth_win_frames
    )

    n_frames = max(len(f0_hz), len(sc_hz), len(inv_ioi))
    times = librosa.frames_to_time(np.arange(n_frames), sr=sr, hop_length=hop_length)

    def pad(a):
        if len(a) == n_frames:
            return a
        out = np.full(n_frames, np.nan)
        out[:len(a)] = a
        return out

    f0_hz = pad(f0_hz)
    sc_hz = pad(sc_hz)
    inv_ioi = pad(inv_ioi)

    f0_norm = norm_0_1(f0_hz, robust=True)
    sc_norm = norm_0_1(sc_hz, robust=True)
    io_norm = norm_0_1(inv_ioi, robust=True)

    F = np.vstack([f0_norm, sc_norm, io_norm])
    W = np.asarray([w_pitch, w_centroid, w_ioi], dtype=float)
    W = np.clip(W, 0.0, None)
    if W.sum() <= 0:
        W = np.array([1.0, 1.0, 1.0])
    W = W / W.sum()

    M = np.isfinite(F).astype(float)
    den = np.sum(W[:, None] * M, axis=0)
    num = np.nansum(W[:, None] * F, axis=0)
    tension = np.full_like(den, np.nan, dtype=float)
    good = den > 0
    tension[good] = num[good] / den[good]

    tension = moving_average(tension, smooth_win_frames)

    return {
        "time_s": times,
        "tension": tension,
        "pitch_hz": f0_hz,
        "centroid_hz": sc_hz,
        "inv_ioi": inv_ioi,
        "pitch_norm": f0_norm,
        "centroid_norm": sc_norm,
        "inv_ioi_norm": io_norm,
        "onsets_s": onset_times,
    }


# ---------------- streamlit UI ----------------

st.set_page_config(
    page_title="EDM Build-up Tension Index",
    layout="wide",
)

st.title("EDM Build-up Tension Index Explorer")

st.markdown(
    """
Upload an EDM build-up (8–16 bars) and inspect:

- Pitch-violation proxy (rising F0)
- Spectral centroid (brightening)
- Inverse IOI (onset density)
- Combined tension curve
"""
)

with st.sidebar:
    st.header("Analysis settings")

    sr_opt = st.selectbox(
        "Target sample rate (Hz)",
        [22050, 32000, 44100, 48000],
        index=3,
    )
    n_fft = st.selectbox(
        "STFT / analysis window (samples)",
        [1024, 2048, 4096],
        index=1,
    )
    hop = st.selectbox(
        "Hop length (samples)",
        [128, 256, 512],
        index=1,
    )
    smooth = st.slider(
        "Smoothing window (frames)",
        min_value=1,
        max_value=31,
        value=19,
        step=2,
    )
    ioi_win = st.slider(
        "Onset IOI window (seconds)",
        min_value=0.25,
        max_value=4.0,
        value=2.0,
        step=0.25,
    )

    st.subheader("Feature weights")
    w_pitch = st.slider("Pitch proximity violation weight", 0.0, 2.0, 1.0, 0.1)
    w_centroid = st.slider("Spectral centroid / brightness weight", 0.0, 2.0, 1.0, 0.1)
    w_ioi = st.slider("Inverse IOI (density) weight", 0.0, 2.0, 1.0, 0.1)

uploaded = st.file_uploader(
    "Upload WAV/MP3/FLAC of a build-up section",
    type=["wav", "mp3", "flac", "ogg", "m4a"],
)


def load_audio(file_like, sr_target: int):
    """Load audio and also return raw bytes for UI playback."""
    if isinstance(file_like, (str, Path)):
        with open(file_like, "rb") as f:
            data = f.read()
    else:
        # Streamlit UploadedFile or any file-like
        if hasattr(file_like, "getvalue"):
            data = file_like.getvalue()
        else:
            data = file_like.read()

    bio = io.BytesIO(data)
    y, sr = librosa.load(bio, sr=sr_target, mono=True)
    return y, sr, data


if uploaded is not None:
    y, _sr, audio_bytes = load_audio(uploaded, sr_opt)
    st.audio(audio_bytes, format="audio/wav")
    st.write(f"Audio length: {len(y)/_sr:.2f} s · SR: {_sr} Hz")

    data = compute_tension(
        y, _sr,
        n_fft=int(n_fft),
        hop_length=int(hop),
        smooth_win_frames=int(smooth),
        ioi_window_s=float(ioi_win),
        w_pitch=float(w_pitch),
        w_centroid=float(w_centroid),
        w_ioi=float(w_ioi),
    )

    # --- Summary metrics ---
    t = data["tension"]
    valid = np.isfinite(t)
    t_valid = t[valid]
    c1, c2, c3 = st.columns(3)
    if t_valid.size > 0:
        avg = np.nanmean(t_valid)
        peak_idx = np.nanargmax(t_valid)
        peak_time = data["time_s"][valid][peak_idx]
        peak_val = t_valid[peak_idx]
        net_increase = t_valid[-1] - t_valid[0]
        c1.metric("Average tension", f"{avg:.2f}")
        c2.metric("Net increase", "nan" if np.isnan(net_increase) else f"{net_increase:.2f}")
        c3.metric("Frames (valid/total)", f"{valid.sum()}/{len(t)}")
    else:
        c1.metric("Average tension", "nan")
        c2.metric("Net increase", "nan")
        c3.metric("Frames (valid/total)", f"0/{len(t)}")

    # --- Interactive charts with Altair ---
    # Compute RMS envelope for a clean waveform overview
    rms = librosa.feature.rms(y=y, frame_length=int(n_fft), hop_length=int(hop))[0]
    # Pad/trim RMS to match frame timeline
    n_frames = len(data["tension"])
    if len(rms) != n_frames:
        rms_pad = np.full(n_frames, np.nan)
        rms_pad[:min(n_frames, len(rms))] = rms[:min(n_frames, len(rms))]
        rms = rms_pad

    df = pd.DataFrame({
        "time_s": data["time_s"],
        "tension": data["tension"],
        "pitch_hz": data["pitch_hz"],
        "centroid_hz": data["centroid_hz"],
        "inv_ioi": data["inv_ioi"],
        "pitch_norm": data["pitch_norm"],
        "centroid_norm": data["centroid_norm"],
        "inv_ioi_norm": data["inv_ioi_norm"],
    })
    df["rms"] = rms

    tab_overview, tab_norm, tab_raw, tab_data = st.tabs(["Overview", "Normalized Features", "Raw Features", "Data"])

    with tab_overview:
        st.caption("Zoom and hover for details. Onsets shown as dashed rules.")
        base_overview = alt.Chart(df).mark_line().encode(
            x=alt.X("time_s:Q", title="Time (s)"),
            y=alt.Y("rms:Q", title="Waveform RMS"),
            tooltip=["time_s:Q", alt.Tooltip("rms:Q", format=".3f")]
        )
        onsets_df = pd.DataFrame({"onset": data["onsets_s"]})
        onsets = alt.Chart(onsets_df).mark_rule(strokeDash=[4, 4], color="#888").encode(x="onset:Q")
        st.altair_chart((base_overview + onsets).interactive().properties(height=180), use_container_width=True)

        # Tension focus
        tension_chart = alt.Chart(df).mark_line(color="#d62728").encode(
            x=alt.X("time_s:Q", title="Time (s)"),
            y=alt.Y("tension:Q", title="Tension (0-1)"),
            tooltip=["time_s:Q", alt.Tooltip("tension:Q", format=".3f")]
        )
        st.altair_chart(tension_chart.properties(height=200), use_container_width=True)

    with tab_norm:
        st.caption("Normalized features (0–1) contributing to the tension index.")
        norm_cols = ["pitch_norm", "centroid_norm", "inv_ioi_norm"]
        show_norm = st.multiselect(
            "Normalized features",
            norm_cols,
            default=["pitch_norm", "centroid_norm", "inv_ioi_norm"],
            key=f"normcols_{uploaded.name}",
        )
        if show_norm:
            dfn = df[["time_s", *show_norm]].copy()
            dfn_long = dfn.melt(id_vars="time_s", var_name="feature", value_name="value")
            chart = alt.Chart(dfn_long).mark_line().encode(
                x=alt.X("time_s:Q", title="Time (s)"),
                y=alt.Y("value:Q", title="Normalized value (0–1)"),
                color=alt.Color("feature:N", title="Feature"),
                tooltip=["time_s:Q", "feature:N", alt.Tooltip("value:Q", format=".2f")]
            ).interactive()
            st.altair_chart(chart.properties(height=300), use_container_width=True)
        else:
            st.info("Pick at least one normalized feature to visualize.")

    with tab_raw:
        st.caption("Raw feature scales for reference.")
        raw_cols = ["pitch_hz", "centroid_hz", "inv_ioi"]
        show_raw = st.multiselect("Raw features", raw_cols, default=["pitch_hz", "centroid_hz"], key=f"rawcols_{uploaded.name}")
        log_scale = st.checkbox("Log frequency axis", value=True, key=f"logscale_{uploaded.name}")
        if show_raw:
            dfr = df[["time_s", *show_raw]].copy()
            dfr_long = dfr.melt(id_vars="time_s", var_name="feature", value_name="value")
            y_scale = alt.Scale(type="log") if log_scale else alt.Scale()
            raw_chart = alt.Chart(dfr_long).mark_line().encode(
                x=alt.X("time_s:Q", title="Time (s)"),
                y=alt.Y("value:Q", title="Value", scale=y_scale),
                color=alt.Color("feature:N", title="Feature"),
                tooltip=["time_s:Q", "feature:N", alt.Tooltip("value:Q", format=".2f")]
            ).interactive()
            st.altair_chart(raw_chart.properties(height=300), use_container_width=True)
        else:
            st.info("Pick at least one raw feature to visualize.")

    with tab_data:
        st.dataframe(df, use_container_width=True)
        # Correlation insight
        corr_cols = ["pitch_norm", "centroid_norm", "inv_ioi_norm", "tension"]
        corr = df[corr_cols].corr()
        corr_long = corr.reset_index().melt("index", var_name="feature", value_name="corr").rename(columns={"index": "target"})
        heat = alt.Chart(corr_long).mark_rect().encode(
            x=alt.X("feature:N", title="Feature"),
            y=alt.Y("target:N", title="Target"),
            color=alt.Color("corr:Q", scale=alt.Scale(scheme="redblue", domain=[-1, 1])),
            tooltip=["target:N", "feature:N", alt.Tooltip("corr:Q", format=".2f")]
        ).properties(height=200)
        st.altair_chart(heat, use_container_width=True)

    # basic insight
    if t_valid.size > 0:
        # simple slope as linear trend
        x = data["time_s"][valid]
        yv = t_valid
        # least squares slope
        A = np.vstack([x, np.ones_like(x)]).T
        slope, intercept = np.linalg.lstsq(A, yv, rcond=None)[0]
        peak_time = data["time_s"][np.nanargmax(data["tension"])]
        peak_val = np.nanmax(data["tension"])
        net_increase = t_valid[-1] - t_valid[0]

        if slope > 0.01 and net_increase > 0.2:
            summary = "Strong upward build-up"
        elif slope > 0.0 and net_increase > 0.05:
            summary = "Mild upward build-up"
        else:
            summary = "Flat or inconsistent build-up"

        st.write(
            f"{summary}. Peak tension {peak_val:.2f} at {peak_time:.2f}s · Trend slope {slope:.3f} per second"
        )
    else:
        st.write("No valid tension values to analyze.")

    # data export
    csv_bytes = df.to_csv(index=False).encode("utf-8")
    st.download_button(
        label="Download CSV",
        data=csv_bytes,
        file_name=f"{Path(uploaded.name).stem}_tension.csv",
        mime="text/csv",
    )
