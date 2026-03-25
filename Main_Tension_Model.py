import io
from pathlib import Path
import numpy as np
import pandas as pd
import librosa
import altair as alt
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import streamlit as st

#core utilities

def moving_average(x: np.ndarray, win: int) -> np.ndarray:
    # NaN-aware moving average with reflect padding at the edges.
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
            return x
        x = x.copy()
        # fwd fill
        idx = np.where(mask, np.arange(n), 0)
        np.maximum.accumulate(idx, out=idx)
        x_ff = x[idx]
        # bwd fill
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


# features

def feature_pitch_violation(y: np.ndarray, sr: int, hop_length: int, frame_length: int,
                            fmin_hz=librosa.note_to_hz('C2'),
                            fmax_hz=librosa.note_to_hz('C7'),
                            smooth_win_frames: int = 9) -> np.ndarray:
    # Make YIN use an odd window to satisfy strict bound fmin > sr/(frame_length-1)
    yin_frame = int(frame_length) | 1


    y_harm = librosa.effects.hpss(y)[0]

    lb = sr / float(max(2, yin_frame - 1))
    yin_min = np.nextafter(lb, np.inf)
    fmin_safe = float(max(fmin_hz, yin_min))

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


# tension computation

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


# publication figure

def generate_publication_figure(y: np.ndarray, sr: int, data: dict,
                                track_name: str = "") -> plt.Figure:
    # publication rcParams (local to this figure)

    with plt.rc_context({
        "font.family":       "serif",
        "font.size":         10,
        "axes.titlesize":    11,
        "axes.labelsize":    10,
        "legend.fontsize":   9,
        "xtick.labelsize":   9,
        "ytick.labelsize":   9,
        "axes.spines.top":   False,
        "axes.spines.right": False,
        "figure.dpi":        150,
    }):
        fig, axes = plt.subplots(
            5, 1,
            figsize=(7, 8),
            sharex=True,
            gridspec_kw={"hspace": 0.35},
        )
        fig.patch.set_facecolor("white")

        time_s = data["time_s"]
        t_audio = np.linspace(0, len(y) / sr, num=len(y), endpoint=False)

        # colour palette
        col_wave     = "#444444"
        col_pitch    = "#1f77b4"
        col_centroid = "#ff7f0e"
        col_ioi      = "#2ca02c"
        col_tension  = "#d62728"

        # Waveform
        ax = axes[0]
        ax.plot(t_audio, y, color=col_wave, linewidth=0.35)
        ax.set_ylabel("Amplitude")
        # ax.set_title(track_name if track_name else "Waveform", fontweight="bold")
        ax.set_xlim(0, t_audio[-1])

        # Pitch (normalised)
        ax = axes[1]
        ax.plot(time_s, data["pitch_norm"], color=col_pitch, linewidth=1.2)
        ax.set_ylabel("Pitch (norm.)")
        ax.set_ylim(0, 1.05)
        ax.yaxis.set_minor_locator(mticker.AutoMinorLocator())

        # Spectral Centroid (normalised)
        ax = axes[2]
        ax.plot(time_s, data["centroid_norm"], color=col_centroid, linewidth=1.2)
        ax.set_ylabel("Centroid (norm.)")
        ax.set_ylim(0, 1.05)
        ax.yaxis.set_minor_locator(mticker.AutoMinorLocator())

        # Inverse IOI (normalised)
        ax = axes[3]
        ax.plot(time_s, data["inv_ioi_norm"], color=col_ioi, linewidth=1.2)
        ax.set_ylabel("Inv. IOI (norm.)")
        ax.set_ylim(0, 1.05)
        ax.yaxis.set_minor_locator(mticker.AutoMinorLocator())

        # Tension index
        ax = axes[4]
        ax.plot(time_s, data["tension"], color=col_tension, linewidth=1.6)
        ax.set_ylabel("Tension Index")
        ax.set_ylim(0, 1.05)
        ax.set_xlabel("Time (s)")
        ax.yaxis.set_minor_locator(mticker.AutoMinorLocator())
        ax.xaxis.set_minor_locator(mticker.AutoMinorLocator())

        for a in axes:
            a.set_facecolor("white")

        fig.tight_layout()
    return fig


def generate_altair_unified_figure(df: pd.DataFrame, track_name: str = "") -> alt.VConcatChart:
    
    # Shared X-axis configuration for linked pan/zoom
    selection = alt.selection_interval(bind='scales', encodings=['x'])
    
    base = alt.Chart(df).encode(
        x=alt.X("time_s:Q", title="Time (s)"),
        tooltip=[alt.Tooltip("time_s:Q", format=".2f", title="Time (s)")]
    ).properties(width='container', height=120).add_params(selection)

    # Waveform (RMS)
    wave = base.mark_line(color="#444444").encode(
        y=alt.Y("rms:Q", title="RMS"),
        tooltip=[alt.Tooltip("time_s:Q", format=".2f"), alt.Tooltip("rms:Q", format=".3f")]
    )

    # Pitch (norm)
    pitch = base.mark_line(color="#1f77b4").encode(
        y=alt.Y("pitch_norm:Q", title="Pitch (norm)", scale=alt.Scale(domain=[0, 1])),
        tooltip=[alt.Tooltip("time_s:Q", format=".2f"), alt.Tooltip("pitch_norm:Q", format=".3f")]
    )

    # Centroid (norm)
    centroid = base.mark_line(color="#ff7f0e").encode(
        y=alt.Y("centroid_norm:Q", title="Centroid (norm)", scale=alt.Scale(domain=[0, 1])),
        tooltip=[alt.Tooltip("time_s:Q", format=".2f"), alt.Tooltip("centroid_norm:Q", format=".3f")]
    )

    # IOI (norm)
    ioi = base.mark_line(color="#2ca02c").encode(
        y=alt.Y("inv_ioi_norm:Q", title="Inv. IOI (norm)", scale=alt.Scale(domain=[0, 1])),
        tooltip=[alt.Tooltip("time_s:Q", format=".2f"), alt.Tooltip("inv_ioi_norm:Q", format=".3f")]
    )

    # Tension
    tension = base.mark_line(color="#d62728", strokeWidth=2).encode(
        y=alt.Y("tension:Q", title="Tension Index", scale=alt.Scale(domain=[0, 1])),
        tooltip=[alt.Tooltip("time_s:Q", format=".2f"), alt.Tooltip("tension:Q", format=".3f")]
    )

    return alt.vconcat(wave, pitch, centroid, ioi, tension).resolve_scale(x='shared')


def generate_matplotlib_overview(df: pd.DataFrame, data: dict, track_name: str = "") -> plt.Figure:
    with plt.rc_context({
        "font.family": "serif", "font.size": 10, "axes.titlesize": 11,
        "axes.labelsize": 10, "axes.spines.top": False, "axes.spines.right": False,
        "figure.dpi": 150,
    }):
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(7, 5), sharex=True)
        # RMS + Onsets
        ax1.plot(df["time_s"], df["rms"], color="#444444", lw=1)
        first_onset = True
        for o in data["onsets_s"]:
            ax1.axvline(o, color="#888888", ls="--", lw=0.8, alpha=0.5, 
                        label="Onsets" if first_onset else "")
            first_onset = False
        ax1.set_ylabel("RMS")
        ax1.set_title(f"Overview: {track_name}" if track_name else "Overview", fontweight="bold")
        # Tension
        ax2.plot(df["time_s"], df["tension"], color="#d62728", lw=1.5)
        ax2.set_ylabel("Tension Index")
        ax2.set_xlabel("Time (s)")
        ax2.set_ylim(0, 1.05)
        ax2.set_xlim(0, df["time_s"].iloc[-1])
        ax2.yaxis.set_minor_locator(mticker.AutoMinorLocator())
        plt.tight_layout()
    return fig


def generate_matplotlib_features(df: pd.DataFrame, cols: list, ylabel: str, title: str, 
                                 log_scale: bool = False) -> plt.Figure:
    with plt.rc_context({
        "font.family": "serif", "font.size": 10, "axes.titlesize": 11,
        "axes.labelsize": 10, "axes.spines.top": False, "axes.spines.right": False,
        "figure.dpi": 150,
    }):
        fig, ax = plt.subplots(figsize=(7, 4))
        for col in cols:
            label = col.replace("_norm", "").replace("_hz", "").replace("_", " ").title()
            ax.plot(df["time_s"], df[col], label=label, lw=1.2)
        ax.set_ylabel(ylabel)
        ax.set_xlabel("Time (s)")
        ax.set_title(title, fontweight="bold")
        if log_scale:
            ax.set_yscale("log")
        else:
            ax.set_ylim(bottom=0)
        ax.set_xlim(left=0)
        ax.legend(loc="best", frameon=True, framealpha=0.8)
        ax.yaxis.set_minor_locator(mticker.AutoMinorLocator())
        plt.tight_layout()
    return fig


# streamlit UI 

st.set_page_config(
    page_title="A Computational Model for Predicting Perceived Tension in Electronic Dance Music Build-ups",
    layout="wide",
)

st.title("A Computational Model for Predicting Perceived Tension in Electronic Dance Music Build-ups")

st.markdown("""
            Stefanos Biliousis | \
Aalborg University, Copenhagen, Denmark | \
    Sound and Music Perception and Cognition semester project
            """)

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
        [22050, 44100, 48000],
        index=2,
    )
    n_fft = st.selectbox(
        "STFT / analysis window (samples)",
        [1024, 2048, 4096],
        index=2,
    )
    hop = st.selectbox(
        "Hop length (samples)",
        [128, 256, 512],
        index=2,
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

    st.subheader("Display Options")
    display_mode = st.radio(
        "Analysis View Mode",
        ["Detailed Tabs", "Unified Interactive (Altair)", "Unified Publication (Matplotlib)"],
        index=0,
        help="Switch between granular tabs and all-in-one stacked views."
    )

uploaded = st.file_uploader(
    "Upload WAV/MP3/FLAC of a single build-up section",
    type=["wav", "mp3", "flac", "ogg", "m4a"],
    key="single_uploader",
)


def load_audio(file_like, sr_target: int):
    if isinstance(file_like, (str, Path)):
        with open(file_like, "rb") as f:
            data = f.read()
    else:
        if hasattr(file_like, "getvalue"):
            data = file_like.getvalue()
        else:
            data = file_like.read()

    bio = io.BytesIO(data)
    y, sr = librosa.load(bio, sr=sr_target, mono=True)
    return y, sr, data


# single-file detailed analysis
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

    # metrics
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

    # charts
    # RMS envelope
    rms = librosa.feature.rms(y=y, frame_length=int(n_fft), hop_length=int(hop))[0]
    # Pad RMS to match frame timeline
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

    if display_mode == "Detailed Tabs":
        tab_overview, tab_norm, tab_raw, tab_data = st.tabs(["Overview", "Normalized Features", "Raw Features", "Data"])

        with tab_overview:
            st.caption("Publication-style overview: RMS (with onsets) and Tension curve.")
            ov_fig = generate_matplotlib_overview(df, data, track_name=Path(uploaded.name).stem)
            st.pyplot(ov_fig)
            
            # download button
            buf = io.BytesIO()
            ov_fig.savefig(buf, format="png", dpi=300, bbox_inches="tight")
            st.download_button(
                label="Download Overview Figure (PNG)",
                data=buf.getvalue(),
                file_name=f"{Path(uploaded.name).stem}_overview.png",
                mime="image/png",
            )
            plt.close(ov_fig)

        with tab_norm:
            st.caption("Normalized features contributing to the tension index.")
            norm_cols = ["pitch_norm", "centroid_norm", "inv_ioi_norm"]
            show_norm = st.multiselect(
                "Features to include",
                norm_cols,
                default=norm_cols,
                key=f"normcols_{uploaded.name}",
            )
            if show_norm:
                # Dynamic title based on selection
                readable_names = [c.replace("_norm", "").replace("_", " ").title() for c in show_norm]
                if len(readable_names) > 1:
                    title_str = f"Normalized {', '.join(readable_names[:-1])} & {readable_names[-1]}"
                else:
                    title_str = f"Normalized {readable_names[0]}"
                
                norm_fig = generate_matplotlib_features(df, show_norm, "Normalized Value (0–1)", title_str)
                st.pyplot(norm_fig)
                
                # download button
                buf = io.BytesIO()
                norm_fig.savefig(buf, format="png", dpi=300, bbox_inches="tight")
                st.download_button(
                    label="Download Normalized Features (PNG)",
                    data=buf.getvalue(),
                    file_name=f"{Path(uploaded.name).stem}_norm_features.png",
                    mime="image/png",
                )
                plt.close(norm_fig)
            else:
                st.info("Pick at least one normalized feature to visualize.")

        with tab_raw:
            st.caption("Raw feature scales for reference.")
            raw_cols = ["pitch_hz", "centroid_hz", "inv_ioi"]
            show_raw = st.multiselect("Raw features", raw_cols, default=["pitch_hz", "centroid_hz"], key=f"rawcols_{uploaded.name}")
            log_scale = st.checkbox("Log frequency axis", value=True, key=f"logscale_{uploaded.name}")
            if show_raw:
                # Dynamic title based on selection
                readable_names = [c.replace("_hz", "").replace("_", " ").title() for c in show_raw]
                if len(readable_names) > 1:
                    title_str = f"Raw {', '.join(readable_names[:-1])} & {readable_names[-1]}"
                else:
                    title_str = f"Raw {readable_names[0]}"
                
                raw_fig = generate_matplotlib_features(df, show_raw, "Value", title_str, log_scale=log_scale)
                st.pyplot(raw_fig)
                
                # download button
                buf = io.BytesIO()
                raw_fig.savefig(buf, format="png", dpi=300, bbox_inches="tight")
                st.download_button(
                    label="Download Raw Features (PNG)",
                    data=buf.getvalue(),
                    file_name=f"{Path(uploaded.name).stem}_raw_features.png",
                    mime="image/png",
                )
                plt.close(raw_fig)
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


    elif display_mode == "Unified Interactive (Altair)":
        st.info("Interactive view: Pan/zoom with mouse. Export via the ⋯ menu in the top-right of the charts.")
        unified_alt = generate_altair_unified_figure(df, track_name=Path(uploaded.name).stem)
        st.altair_chart(unified_alt, use_container_width=True, theme="streamlit")

    elif display_mode == "Unified Publication (Matplotlib)":
        st.caption("Publication-ready static figure (Matplotlib).")
        pub_fig = generate_publication_figure(y, _sr, data, track_name=Path(uploaded.name).stem)
        st.pyplot(pub_fig)
        
        # offer PNG download for publication
        buf = io.BytesIO()
        pub_fig.savefig(buf, format="png", dpi=300, bbox_inches="tight")
        buf.seek(0)
        st.download_button(
            label="Download publication figure (PNG)",
            data=buf,
            file_name=f"{Path(uploaded.name).stem}_publication_figure.png",
            mime="image/png",
        )
        plt.close(pub_fig)

    # individual insight (always show)
    if t_valid.size > 0:
        # linear trend
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
        label="Download Analysis Data (CSV)",
        data=csv_bytes,
        file_name=f"{Path(uploaded.name).stem}_tension.csv",
        mime="text/csv",
    )




# overlay of multiple build-ups
st.header("Overlay of Tension Curves in batch")

corpus_files = st.file_uploader(
    "Upload up to 12 build-up sections for batch analysis",
    type=["wav", "mp3", "flac", "ogg", "m4a"],
    accept_multiple_files=True,
    key="corpus_uploader",
)

if corpus_files:
    max_tracks = 12
    corpus_rows = []
    used_files = corpus_files[:max_tracks]

    for f in used_files:
        y_i, sr_i, _ = load_audio(f, sr_opt)

        data_i = compute_tension(
            y_i, sr_i,
            n_fft=int(n_fft),
            hop_length=int(hop),
            smooth_win_frames=int(smooth),
            ioi_window_s=float(ioi_win),
            w_pitch=float(w_pitch),
            w_centroid=float(w_centroid),
            w_ioi=float(w_ioi),
        )

        tension_i = data_i["tension"]
        time_i = data_i["time_s"]

        valid_i = np.isfinite(tension_i)
        if not np.any(valid_i):
            continue

        time_v = time_i[valid_i]
        tens_v = tension_i[valid_i]

        if time_v[-1] <= 0:
            continue

        # normalize time to [0, 1]
        norm_time = time_v / time_v[-1]

        df_i = pd.DataFrame({
            "norm_time": norm_time,
            "tension": tens_v,
            "track": f.name,
        })
        corpus_rows.append(df_i)

    if corpus_rows:
        corpus_df = pd.concat(corpus_rows, ignore_index=True)

        st.caption(
            "Overlay of all final Tension Index curves, with time normalized to [0, 1] "
            "(0 = start of build-up, 1 = end)."
        )

        overlay_chart = alt.Chart(corpus_df).mark_line().encode(
            x=alt.X("norm_time:Q", title="Normalized build-up position (0 = start, 1 = end)"),
            y=alt.Y("tension:Q", title="Tension (0–1)"),
            color=alt.Color("track:N", title="Track"),
            tooltip=[
                "track:N",
                alt.Tooltip("norm_time:Q", format=".2f", title="Norm. pos"),
                alt.Tooltip("tension:Q", format=".3f"),
            ],
        ).interactive()

        # interpolate each curve on a common normalized grid
        grid = np.linspace(0.0, 1.0, 100)
        curves = []

        for track_name in corpus_df["track"].unique():
            df_t = corpus_df[corpus_df["track"] == track_name].sort_values("norm_time")
            x = df_t["norm_time"].values
            y = df_t["tension"].values

            if x.size < 2:
                continue

            # ensure coverage of full [0, 1] for interpolation
            if x[0] > 0.0:
                x = np.concatenate([[0.0], x])
                y = np.concatenate([[y[0]], y])
            if x[-1] < 1.0:
                x = np.concatenate([x, [1.0]])
                y = np.concatenate([y, [y[-1]]])

            y_interp = np.interp(grid, x, y)
            curves.append(y_interp)

        if curves:
            mean_curve = np.mean(np.vstack(curves), axis=0)
            mean_df = pd.DataFrame({
                "norm_time": grid,
                "tension": mean_curve,
            })

            # individual curves + mean (dashed black)
            mean_overlay = alt.Chart(mean_df).mark_line(
                strokeDash=[4, 4],
                strokeWidth=3,
                color="black",
            ).encode(
                x="norm_time:Q",
                y="tension:Q",
            )

            st.altair_chart(
                (overlay_chart + mean_overlay).properties(height=350),
                use_container_width=True,
            )

            # only the average curve
            st.subheader("Average tension profile across corpus")
            mean_only_chart = alt.Chart(mean_df).mark_line().encode(
                x=alt.X("norm_time:Q", title="Normalized build-up position (0 = start, 1 = end)"),
                y=alt.Y("tension:Q", title="Average tension (0–1)"),
                tooltip=[
                    alt.Tooltip("norm_time:Q", format=".2f", title="Norm. pos"),
                    alt.Tooltip("tension:Q", format=".3f"),
                ],
            ).interactive()

            st.altair_chart(
                mean_only_chart.properties(height=250),
                use_container_width=True,
            )
        else:
            # show overlay of raw curves
            st.altair_chart(
                overlay_chart.properties(height=350),
                use_container_width=True,
            )

        st.dataframe(corpus_df, use_container_width=True)
    else:
        st.info("No valid tension curves could be extracted from the uploaded files.")
