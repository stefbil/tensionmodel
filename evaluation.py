import streamlit as st
import pandas as pd
import numpy as np
import os
import time
import random
import uuid
from datetime import datetime

def generate_participant_id():
    # Example: P-20251118-7F3A
    date_str = datetime.now().strftime("%Y%m%d")
    short_uid = uuid.uuid4().hex[:4].upper()
    return f"P-{date_str}-{short_uid}"

if "phase" not in st.session_state:
    st.session_state.phase = "intro"  # "intro", "rating", "done"

if "participant_id" not in st.session_state:
    st.session_state.participant_id = None

if "track_order" not in st.session_state:
    st.session_state.track_order = []

if "current_track_idx" not in st.session_state:
    st.session_state.current_track_idx = 0

if "start_time" not in st.session_state:
    st.session_state.start_time = None

if "ratings" not in st.session_state:
    st.session_state.ratings = []

if "last_slider_value" not in st.session_state:
    st.session_state.last_slider_value = None

if "slider_value" not in st.session_state:
    st.session_state.slider_value = 0.5


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

def start_experiment():
    """Initialize session and randomized track order."""
    st.session_state.participant_id = generate_participant_id()
    st.session_state.track_order = TRACKS.copy()
    random.shuffle(st.session_state.track_order)
    st.session_state.current_track_idx = 0
    st.session_state.phase = "rating"
    st.session_state.start_time = None
    st.session_state.ratings = []
    st.session_state.last_slider_value = None
    st.session_state.slider_value = 0.5


def save_current_track_ratings():
    """Save the current track slider data to the master CSV."""
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

    st.session_state.ratings = []
    st.session_state.start_time = None
    st.session_state.last_slider_value = None
    st.session_state.slider_value = 0.5


def record_slider_if_changed():
    """Log a new sample only when slider value changes."""
    if st.session_state.start_time is None:
        return

    current_value = st.session_state.slider_value
    last_value = st.session_state.last_slider_value

    if (last_value is None) or (current_value != last_value):
        elapsed = time.time() - st.session_state.start_time
        st.session_state.ratings.append({
            "time": elapsed,
            "tension_rating": current_value
        })
        st.session_state.last_slider_value = current_value


# INTRO
def render_intro():
    st.title("EDM Tension Perception Test")

    st.markdown("""
    Welcome!  
    You will listen to **12 EDM build-ups**.  
    Your task is to continuously rate **how tense the music feels** using a slider.

    ### Instructions:
    1. Click **Begin Experiment**.
    2. A unique anonymous Participant ID will be assigned automatically.
    3. Each track will play in a random order.
    4. For each track:
        - Press **Play**.
        - Press **Start Track** when the audio begins.
        - Move the slider continuously according to your perceived tension.
        - Press **Finish Track** when the audio ends.

    Try to respond naturally — there are no right or wrong answers.
    """)

    if st.button("Begin Experiment"):
        start_experiment()



# RATING
def render_rating():
    total_tracks = len(st.session_state.track_order)
    idx = st.session_state.current_track_idx

    if idx >= total_tracks:
        st.session_state.phase = "done"
        return

    track_label, filename = st.session_state.track_order[idx]
    audio_path = os.path.join(AUDIO_DIR, filename)

    st.title("EDM Tension Perception Test")
    st.markdown(f"**Participant:** `{st.session_state.participant_id}`")
    st.markdown(f"**Track {idx + 1} of {total_tracks}** – {track_label}")

    st.audio(audio_path)

    st.markdown("""
    ### Instructions for this track:
    - Press **Play** on the player above.  
    - When the audio starts, click **Start Track**.  
    - Move the slider based on how the tension rises or falls.  
    - When the track ends, click **Finish Track**.
    """)

    if st.session_state.start_time is None:
        if st.button("Start Track"):
            st.session_state.start_time = time.time()
            st.session_state.ratings = []
            st.session_state.last_slider_value = None
            st.session_state.slider_value = 0.5
        return

    st.slider(
        "Tension (low → high)",
        min_value=0.0,
        max_value=1.0,
        step=0.01,
        key="slider_value",
    )

    record_slider_if_changed()

    st.caption(f"Recorded samples so far: {len(st.session_state.ratings)}")

    if st.button("Finish Track"):
        save_current_track_ratings()
        st.session_state.current_track_idx += 1

        if st.session_state.current_track_idx >= total_tracks:
            st.session_state.phase = "done"



# UI
def render_done():
    st.title("Experiment Complete 🎉")

    st.markdown(f"""
    Thank you for participating!  
    Your anonymous ID was **`{st.session_state.participant_id}`**  
    Your responses have been securely stored.
    """)

    st.markdown("---")
    st.subheader("Admin Panel (for experimenter)")

    if os.path.exists(MASTER_CSV):
        df_all = pd.read_csv(MASTER_CSV)
        st.write(f"Total rows collected: `{len(df_all)}`")
        st.dataframe(df_all.head())

        csv_bytes = df_all.to_csv(index=False).encode("utf-8")

        st.download_button(
            label="Download All Responses as CSV",
            data=csv_bytes,
            file_name="responses_master.csv",
            mime="text/csv",
        )
    else:
        st.info("No data collected yet.")

    if st.button("Start a New Participant"):
        st.session_state.phase = "intro"
        st.session_state.participant_id = None
        st.session_state.track_order = []
        st.session_state.current_track_idx = 0
        st.session_state.start_time = None
        st.session_state.ratings = []
        st.session_state.last_slider_value = None
        st.session_state.slider_value = 0.5


# MAIN ROUTER
if st.session_state.phase == "intro":
    render_intro()
elif st.session_state.phase == "rating":
    render_rating()
elif st.session_state.phase == "done":
    render_done()
