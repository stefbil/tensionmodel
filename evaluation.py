import streamlit as st
import pandas as pd
import numpy as np
import os
import time
import random
import uuid
from datetime import datetime

# -----------------------------------------
# PARTICIPANT ID GENERATION
# -----------------------------------------
def generate_participant_id():
    date_str = datetime.now().strftime("%Y%m%d")
    short_uid = uuid.uuid4().hex[:4].upper()
    return f"P-{date_str}-{short_uid}"

# -----------------------------------------
# SESSION STATE INITIALIZATION
# -----------------------------------------
default_states = {
    "phase": "intro",
    "participant_id": None,
    "track_order": [],
    "current_track_idx": 0,
    "start_time": None,
    "ratings": [],
    "last_slider_value": None,
    "slider_value": 0.5,
    "has_started_track": False,
    "audio_autoplay": False,
}

for key, val in default_states.items():
    if key not in st.session_state:
        st.session_state[key] = val

# -----------------------------------------
# CONSTANTS
# -----------------------------------------
TRACKS = [
    ("Track 1", "buildup01.mp3"),
    ("Track 2", "buildup02.mp3"),
    ("Track 3", "buildup03.mp3"),
    ("Track 4", "buildup04.mp3"),
    ("Track 5", "buildup05.mp3"),
    ("Track 6", "buildup06.mp3"),
    ("Track 7", "buildup07.mp3"),
    ("Track 8", "buildup08.mp3"),
    ("Track 9", "buildup09.mp3"),
    ("Track 10", "buildup10.mp3"),
    ("Track 11", "buildup11.mp3"),
    ("Track 12", "buildup12.mp3"),
]

AUDIO_DIR = "audio"
DATA_DIR = "data"
MASTER_CSV = os.path.join(DATA_DIR, "responses_master.csv")
os.makedirs(DATA_DIR, exist_ok=True)

# -----------------------------------------
# HELPERS
# -----------------------------------------
def reset_slider_state():
    st.session_state.slider_value = 0.5
    st.session_state.last_slider_value = None


def start_experiment():
    st.session_state.participant_id = generate_participant_id()
    st.session_state.track_order = TRACKS.copy()
    random.shuffle(st.session_state.track_order)
    st.session_state.current_track_idx = 0
    st.session_state.phase = "rating"
    st.session_state.start_time = None
    st.session_state.ratings = []
    st.session_state.has_started_track = False
    st.session_state.audio_autoplay = False
    reset_slider_state()


def save_current_track_ratings():
    if not st.session_state.ratings:
        return

    track_label, filename = st.session_state.track_order[st.session_state.current_track_idx]
    participant = st.session_state.participant_id

    df = pd.DataFrame(st.session_state.ratings)
    df["participant"] = participant
    df["track_label"] = track_label
    df["track_file"] = filename
    df = df[["participant", "track_label", "track_file", "time", "tension_rating"]]

    if os.path.exists(MASTER_CSV):
        df.to_csv(MASTER_CSV, mode="a", header=False, index=False)
    else:
        df.to_csv(MASTER_CSV, index=False)


def record_slider_if_changed():
    if st.session_state.start_time is None:
        return

    curr = st.session_state.slider_value
    last = st.session_state.last_slider_value

    if last is None or curr != last:
        elapsed = time.time() - st.session_state.start_time
        st.session_state.ratings.append({"time": elapsed, "tension_rating": curr})
        st.session_state.last_slider_value = curr


# -----------------------------------------
# INTRO SCREEN
# -----------------------------------------
def render_intro():
    st.title("EDM Tension Perception Test")

    st.markdown("""
    You will listen to **12 EDM build-ups** and continuously rate **musical tension**.

    ### NEW streamlined workflow:
    - The **track loads automatically**
    - The **slider appears immediately**
    - **Audio playback begins the moment you touch the slider**
    - You adjust the slider as tension rises/falls
    - Click **Finish Track** when the audio ends
    """)

    if st.button("Begin Experiment"):
        start_experiment()


# -----------------------------------------
# RATING SCREEN (NEW BEHAVIOR)
# -----------------------------------------
def render_rating():
    idx = st.session_state.current_track_idx
    total = len(st.session_state.track_order)

    # Finished all
    if idx >= total:
        st.session_state.phase = "done"
        return

    track_label, filename = st.session_state.track_order[idx]
    audio_path = os.path.join(AUDIO_DIR, filename)

    st.title("EDM Tension Perception Test")
    st.markdown(f"**Participant:** `{st.session_state.participant_id}`")
    st.markdown(f"**Track {idx+1}/{total}:** {track_label}")

    # ----------------------------------------------------
    # Handle autoplay AFTER slider first movement
    # ----------------------------------------------------
    if st.session_state.audio_autoplay:
        st.audio(audio_path, autoplay=True)
    else:
        st.audio(audio_path)

    st.success("Move the slider to begin playback and logging.")

    # ----------------------------------------------------
    # Slider always visible
    # ----------------------------------------------------
    new_slider_value = st.slider(
        "Tension (low → high)",
        0.0, 1.0,
        value=st.session_state.slider_value,
        step=0.01
    )
    st.session_state.slider_value = new_slider_value

    # ----------------------------------------------------
    # Detect first user interaction → start track + logging
    # ----------------------------------------------------
    if not st.session_state.has_started_track:
        # User moved slider for the first time →
        if st.session_state.slider_value != 0.5:
            st.session_state.start_time = time.time()
            st.session_state.has_started_track = True
            st.session_state.audio_autoplay = True
            st.session_state.ratings = []
            st.session_state.last_slider_value = None

    # ----------------------------------------------------
    # Sliding logs after start
    # ----------------------------------------------------
    if st.session_state.has_started_track:
        record_slider_if_changed()

    st.caption(f"Recorded samples so far: {len(st.session_state.ratings)}")

    # ----------------------------------------------------
    # Finish button
    # ----------------------------------------------------
    if st.button("Finish Track"):
        save_current_track_ratings()
        st.session_state.current_track_idx += 1
        st.session_state.start_time = None
        st.session_state.has_started_track = False
        st.session_state.audio_autoplay = False
        reset_slider_state()


# -----------------------------------------
# DONE SCREEN
# -----------------------------------------
def render_done():
    st.title("Experiment Complete 🎉")

    st.markdown(f"""
    Thank you!  
    Participant ID: **`{st.session_state.participant_id}`**
    """)

    st.subheader("Admin Panel")
    if os.path.exists(MASTER_CSV):
        df = pd.read_csv(MASTER_CSV)
        st.write(f"Total rows: {len(df)}")
        st.dataframe(df.head())
        st.download_button(
            "Download CSV",
            df.to_csv(index=False).encode("utf-8"),
            "responses_master.csv",
            "text/csv",
        )
    else:
        st.info("No data yet.")

    if st.button("Start Another Participant"):
        for key, val in default_states.items():
            st.session_state[key] = val


# -----------------------------------------
# ROUTER
# -----------------------------------------
if st.session_state.phase == "intro":
    render_intro()
elif st.session_state.phase == "rating":
    render_rating()
elif st.session_state.phase == "done":
    render_done()
