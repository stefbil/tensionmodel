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

# GOOGLE SHEETS CONFIG
SHEET_ID = ${{ secrets.SHEET_ID }}

SCOPE = [
    "https://www.googleapis.com/auth/spreadsheets",
    "https://www.googleapis.com/auth/drive"
]

def get_gsheet():
    creds_dict = st.secrets["gcp_service_account"]
    credentials = Credentials.from_service_account_info(creds_dict, scopes=SCOPE)

    client = gspread.Client(auth=credentials)
    client.session = gspread.AuthorizedSession(credentials)

    sheet = client.open_by_key(SHEET_ID).sheet1
    return sheet


# PARTICIPANT ID
def generate_participant_id():
    date_str = datetime.now().strftime("%Y%m%d")
    short_uid = uuid.uuid4().hex[:4].upper()
    return f"P-{date_str}-{short_uid}"


# INITIAL SESSION STATE
default_states = {
    "phase": "intro",
    "participant_id": None,
    "track_order": [],
    "current_track_idx": 0,
    "start_time": None,
    "samples": [],              # continuous tension samples
    "slider_value": 0.5,
    "sampling": False,          # are we sampling yet?
    "last_sample_time": 0.0,    # last 100ms sample timestamp
    "audio_autoplay": False,    # autoplay audio after slider move
}

for key, value in default_states.items():
    if key not in st.session_state:
        st.session_state[key] = value


# TRACK LIST
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
def start_experiment():
    st.session_state.participant_id = generate_participant_id()
    st.session_state.track_order = TRACKS.copy()
    random.shuffle(st.session_state.track_order)
    st.session_state.current_track_idx = 0
    st.session_state.phase = "rating"

    st.session_state.samples = []
    st.session_state.sampling = False
    st.session_state.slider_value = 0.5
    st.session_state.audio_autoplay = False


def save_to_google_sheets():
    """Save all continuous samples (100ms) to Google Sheets."""
    if len(st.session_state.samples) == 0:
        return

    sheet = get_gsheet()
    track_label, filename = st.session_state.track_order[st.session_state.current_track_idx]

    rows = []
    for t, val in st.session_state.samples:
        rows.append([
            st.session_state.participant_id,
            track_label,
            filename,
            t,
            val
        ])

    sheet.append_rows(rows, value_input_option="RAW")


def perform_sampling():
    """Record slider values at 100ms intervals."""
    if not st.session_state.sampling:
        return

    now = time.time()
    elapsed = now - st.session_state.start_time

    if elapsed - st.session_state.last_sample_time >= 0.1:
        st.session_state.samples.append((elapsed, st.session_state.slider_value))
        st.session_state.last_sample_time = elapsed


# LAYOUTS
def render_intro():
    st.title("EDM Tension Perception Test")

    st.markdown("""
    ### How it works:
    - Slider appears immediately  
    - Audio auto-plays when the slider is touched  
    - Your tension rating is **sampled 10× per second**  
    - Press **Finish Track** when the audio ends  
    """)

    if st.button("Begin"):
        start_experiment()


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

    # Autoplay audio when sampling starts
    if st.session_state.audio_autoplay:
        st.audio(audio_path, autoplay=True)
    else:
        st.audio(audio_path)

    # Slider visible immediately
    new_slider = st.slider(
        "Tension (low → high)",
        0.0, 1.0,
        value=st.session_state.slider_value,
        step=0.01
    )
    st.session_state.slider_value = new_slider

    # First slider movement triggers sampling + audio autoplay
    if not st.session_state.sampling and new_slider != 0.5:
        st.session_state.sampling = True
        st.session_state.audio_autoplay = True
        st.session_state.start_time = time.time()
        st.session_state.samples = []
        st.session_state.last_sample_time = 0.0

    # Continuous sampling at 100ms
    perform_sampling()

    st.caption(f"Samples recorded: {len(st.session_state.samples)}")

    # Finish Track
    if st.button("Finish Track"):
        save_to_google_sheets()

        # reset track state
        st.session_state.current_track_idx += 1
        st.session_state.sampling = False
        st.session_state.samples = []
        st.session_state.slider_value = 0.5
        st.session_state.audio_autoplay = False


def render_done():
    st.title("Experiment Complete 🎉")
    st.write(f"Thanks! Your ID was: **{st.session_state.participant_id}**")
    st.success("Your responses have been saved.")

    if st.button("Start Another"):
        for k, v in default_states.items():
            st.session_state[k] = v



# ROUTER
if st.session_state.phase == "intro":
    render_intro()
elif st.session_state.phase == "rating":
    render_rating()
elif st.session_state.phase == "done":
    render_done()

