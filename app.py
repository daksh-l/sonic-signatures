import os
import pickle

import numpy as np
import streamlit as st
import matplotlib.pyplot as plt
import librosa
import pandas as pd

from fingerprint import compute_spectrogram, find_peaks, match_song, SAMPLE_RATE

DB_PATH = "database.pkl"

@st.cache_resource
def load_database():
    with open(DB_PATH, "rb") as f:
        return pickle.load(f)

database = load_database()
song_count = len({s for hits in database.values() for s, _ in hits})

def load_clip(f):
    # limiting duration to 30s for saving RAM and faster response
    clip, fs = librosa.load(f, sr=SAMPLE_RATE, mono=True, duration=30)
    return clip, fs

def identify(clip, fs):
    f, t, Sxx_db = compute_spectrogram(clip, fs=fs)
    freq_idx, time_idx = find_peaks(Sxx_db)
    matched, score, offset_counts = match_song(freq_idx, time_idx, database)
    return f, t, Sxx_db, freq_idx, time_idx, matched, score, offset_counts

st.title("Sonic Signatures")
st.write("Submission by: Daksh Leekha (250307) and Kanishk Parmar")
st.write(f"Database: {song_count} songs")

tab1, tab2 = st.tabs(["Single clip", "Batch"])

with tab1:
    uploaded = st.file_uploader("Upload a query clip (.mp3 or .wav)", type=["mp3", "wav"])

    if uploaded:
        clip, fs = load_clip(uploaded)

        with st.spinner("Identifying..."):
            f, t, Sxx_db, freq_idx, time_idx, matched, score, offset_counts = identify(clip, fs)

        if matched:
            st.success(f"Match: **{os.path.splitext(matched)[0]}** (score: {score})")
        else:
            st.warning("No match found in the database.")

        # spectrogram
        fig, ax = plt.subplots(figsize=(10, 3))
        ax.pcolormesh(t, f, Sxx_db, shading="gouraud", cmap="inferno")
        ax.set_ylim(0, 8000)
        ax.set_xlabel("Time (s)")
        ax.set_ylabel("Frequency (Hz)")
        ax.set_title("Spectrogram")
        fig.tight_layout()
        st.pyplot(fig)
        plt.close(fig)

        # constellation map
        fig, ax = plt.subplots(figsize=(10, 3))
        ax.pcolormesh(t, f, Sxx_db, shading="gouraud", cmap="inferno")
        ax.scatter(t[time_idx], f[freq_idx], color="cyan", s=6, marker="x")
        ax.set_ylim(0, 8000)
        ax.set_xlabel("Time (s)")
        ax.set_ylabel("Frequency (Hz)")
        ax.set_title("Constellation Map")
        fig.tight_layout()
        st.pyplot(fig)
        plt.close(fig)

        # offset histogram for the matched song
        fig, ax = plt.subplots(figsize=(10, 3))
        if matched and matched in offset_counts:
            counter = offset_counts[matched]
            offsets = list(counter.keys())
            counts = list(counter.values())
            ax.bar(offsets, counts, width=1.0)

            # zoom around the winning offset so the spike is visible
            best_offset = max(counter, key=counter.get)
            ax.set_xlim(best_offset - 100, best_offset + 100)
        ax.set_xlabel("Time offset (t_database - t_query)")
        ax.set_ylabel("Matching hash count")
        ax.set_title("Offset Histogram")
        fig.tight_layout()
        st.pyplot(fig)
        plt.close(fig)

with tab2:
    st.write("Upload multiple clips. Downloads a `results.csv` with `filename, prediction`.")

    files = st.file_uploader(
        "Upload query clips", type=["mp3", "wav"],
        accept_multiple_files=True, key="batch"
    )

    if files and st.button("Identify all"):
        rows = []
        bar = st.progress(0)

        for i, uf in enumerate(files):
            clip, fs = load_clip(uf)
            _, _, _, freq_idx, time_idx, matched, _, _ = identify(clip, fs)
            prediction = os.path.splitext(matched)[0] if matched else ""
            rows.append({"filename": uf.name, "prediction": prediction})
            bar.progress((i + 1) / len(files))

        df = pd.DataFrame(rows, columns=["filename", "prediction"])
        st.dataframe(df, use_container_width=True)
        st.download_button(
            "Download results.csv",
            df.to_csv(index=False).encode("utf-8"),
            "results.csv",
            "text/csv"
        )
