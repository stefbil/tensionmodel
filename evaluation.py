import streamlit as st
import pandas as pd
import numpy as np
import os
import time
import random
import uuid
from datetime import datetime

import gspread
from google.oauth2.service_account import Credentials


# GOOGLE SHEETS SETUP
SHEET_ID = "1ej1aYS1Ld5tvAMQpwXZA5wwwjzCi6roHaPP5NH51wTc"
SCOPE = ["https://www.googleapis.com/auth/spreadsheets"]

def get_gsheet():
    creds_dict = st.secrets["gcp_service_account"]
    client = gspread.service_account_from_dict(creds_dict)
    sheet = client.open_by_key(SHEET_ID).sheet1
    return sheet



# PARTICIPANT ID
def generate_participant_id():
    date_str = datetime.now().strftime("%Y%m%d")
    short_uid = uuid.uuid4().hex[:4].upper()
    return f"P-{date_str}-{short_uid}"


# SESSION STATE INITIALIZATION
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


# CONSTANTS
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


# HELPERS
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


def record_slider_if_changed():
    if st.session_state.start_time is None:
        return

    curr = st.session_state.slider_value
    last = st.session_state.last_slider_value

    if last is None or curr != last:
        elapsed = time.time() - st.session_state.start_time
        st.session_state.ratings.append({"time": elapsed, "tension": curr})
        st.session_state.last_slider_value = curr


def save_to_google_sheets():
    """Append current track's ratings to Google Sheets."""
    if not st.session_state.ratings:
        return

    sheet = get_gsheet()
    track_label, filename = st.session_state.track_order[st.session_state.current_track_idx]

    rows = []
    for r in st.session_state.ratings:
        rows.append([
            st.session_state.participant_id,
            track_label,
            filename,
            r["time"],
            r["tension"]
        ])

    sheet.append_rows(rows, value_input_option="RAW")


# INTRO SCREEN
def render_intro():
    st.title("EDM Tension Perception Test")

    st.markdown("""
    You will rate tension while listening to 12 EDM build-ups.

    ### How it works:
    - Slider appears immediately  
    - Audio starts automatically **when you touch the slider**
    - Slider movements are recorded in real time  
    - Press **Finish Track** when the audio ends  
    """)

    if st.button("Begin Experiment"):
        start_experiment()


# RATING SCREEN
def render_rating():
    idx = st.session_state.current_track_idx
    total = len(st.session_state.track_order)

    if idx >= total:
        st.session_state.phase = "done"
        return

    track_label, filename = st.session_state.track_order[idx]
    audio_path = os.path.join(AUDIO_DIR, filename)

    st.title("EDM Tension Perception Test")
    st.markdown(f"**Participant:** `{st.session_state.participant_id}`")
    st.markdown(f"**Track {idx+1}/{total}:** {track_label}")

    # Autoplay audio after slider touch
    if st.session_state.audio_autoplay:
        st.audio(audio_path, autoplay=True)
    else:
        st.audio(audio_path)

    st.success("Move the slider to begin playback and logging.")

    # Slider visible immediately
    new_slider = st.slider(
        "Tension (low → high)",
        0.0, 1.0,
        value=st.session_state.slider_value,
        step=0.01
    )
    st.session_state.slider_value = new_slider

    # First slider movement,start timer and autoplay
    if not st.session_state.has_started_track:
        if st.session_state.slider_value != 0.5:
            st.session_state.start_time = time.time()
            st.session_state.ratings = []
            st.session_state.has_started_track = True
            st.session_state.audio_autoplay = True

    if st.session_state.has_started_track:
        record_slider_if_changed()

    st.caption(f"Samples recorded: {len(st.session_state.ratings)}")

    # Finish Track
    if st.button("Finish Track"):
        save_to_google_sheets()

        st.session_state.current_track_idx += 1
        st.session_state.start_time = None
        st.session_state.has_started_track = False
        st.session_state.audio_autoplay = False
        reset_slider_state()



# DONE SCREEN
def render_done():
    st.title("Experiment Complete 🎉")
    st.markdown(f"Thank you! Your ID: **{st.session_state.participant_id}**")

    st.success("All data has been saved to Google Sheets.")

    if st.button("Start Another Participant"):
        for key, val in default_states.items():
            st.session_state[key] = val


# ROUTING
if st.session_state.phase == "intro":
    render_intro()
elif st.session_state.phase == "rating":
    render_rating()
elif st.session_state.phase == "done":
    render_done()

