
import librosa
import streamlit as st
from helper import create_spectrogram, read_audio, record, save_record

st.header("1. Record your own voice")

if st.button(f"Click to Record"):
    record_state = st.text("Recording...")
    duration = 5  # seconds
    fs = 48000
    filename='input'
    myrecording = record(duration, fs)
    record_state.text(f"Saving sample as {filename}.mp3")

    path_myrecording = f"{filename}.mp3"

    save_record(path_myrecording, myrecording, fs)
    record_state.text(f"Done! Saved sample as {filename}.mp3")

    st.audio(read_audio(path_myrecording))

    fig = create_spectrogram(path_myrecording)
    st.pyplot(fig)