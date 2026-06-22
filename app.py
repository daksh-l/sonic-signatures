"""
EE200 Course Project — Q3B: Zapptain America

Streamlit app wrapping the audio fingerprinting identifier built in Q3A.

Two modes:
  - Single-clip mode: upload one query clip, see the spectrogram,
    constellation map, offset histogram, and the matched song.
  - Batch mode: upload several query clips at once, get back a
    results.csv with columns [filename, prediction].
"""

import os
import pickle

import numpy as np
import streamlit as st
import matplotlib.pyplot as plt
import librosa
import pandas as pd

from fingerprint import compute_spectrogram, find_peaks, match_song, SAMPLE_RATE


# ---------- setup ----------

st.set_page_config(page_title="Sonic Signatures", layout="wide")


@st.cache_resource
def load_database():
    with open("database.pkl", "rb") as f:
        return pickle.load(f)


database = load_database()
song_names = sorted({song_id for hits in database.values() for song_id, _ in hits})


def load_audio(uploaded_file):
    """Loads an uploaded audio file into a mono array at SAMPLE_RATE."""
    clip, fs = librosa.load(uploaded_file, sr=SAMPLE_RATE, mono=True)
    return clip, fs


def identify(clip, fs):
    """Runs the full pipeline on one clip and returns everything needed to plot + report."""
    f, t, Sxx_db = compute_spectrogram(clip, fs=fs)
    freq_idx, time_idx = find_peaks(Sxx_db)
    matched_song, score, offset_counts = match_song(freq_idx, time_idx, database)
    return {
        "f": f, "t": t, "Sxx_db": Sxx_db,
        "freq_idx": freq_idx, "time_idx": time_idx,
        "matched_song": matched_song, "score": score,
        "offset_counts": offset_counts,
    }


def plot_spectrogram(result):
    fig, ax = plt.subplots(figsize=(8, 3.5))
    ax.pcolormesh(result["t"], result["f"], result["Sxx_db"], shading="gouraud", cmap="inferno")
    ax.set_ylim(0, 8000)
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Frequency (Hz)")
    ax.set_title("Spectrogram")
    fig.tight_layout()
    return fig


def plot_constellation(result):
    fig, ax = plt.subplots(figsize=(8, 3.5))
    ax.pcolormesh(result["t"], result["f"], result["Sxx_db"], shading="gouraud", cmap="inferno")
    ax.scatter(result["t"][result["time_idx"]], result["f"][result["freq_idx"]],
               color="cyan", s=8, marker="x")
    ax.set_ylim(0, 8000)
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Frequency (Hz)")
    ax.set_title("Constellation Map")
    fig.tight_layout()
    return fig


def plot_offset_histogram(result):
    matched_song = result["matched_song"]
    offset_counts = result["offset_counts"]

    fig, ax = plt.subplots(figsize=(8, 3.5))

    if matched_song is not None and matched_song in offset_counts:
        counter = offset_counts[matched_song]
        offsets = list(counter.keys())
        counts = list(counter.values())
        ax.bar(offsets, counts, width=1.0, color="tab:blue")
        ax.set_title(f"Offset Histogram — {matched_song}")
    else:
        ax.set_title("Offset Histogram — no match found")

    ax.set_xlabel("Time offset (t_database - t_query)")
    ax.set_ylabel("Matching hash count")
    fig.tight_layout()
    return fig


# ---------- UI ----------

st.title("Sonic Signatures — Song Identifier")
st.caption("EE200 Course Project, Q3B")

tab1, tab2 = st.tabs(["Single-Clip Mode", "Batch Mode"])


# ----- Single-clip mode -----
with tab1:
    st.subheader("Identify one clip")
    st.write(f"Database loaded with {len(song_names)} songs.")

    uploaded = st.file_uploader("Upload a query clip", type=["mp3", "wav"], key="single")

    if uploaded is not None:
        clip, fs = load_audio(uploaded)

        with st.spinner("Fingerprinting and matching..."):
            result = identify(clip, fs)

        if result["matched_song"] is not None:
            song_label = os.path.splitext(result["matched_song"])[0]
            st.success(f"Match: **{song_label}**  (score: {result['score']})")
        else:
            st.error("No match found in the database.")

        col1, col2 = st.columns(2)
        with col1:
            st.pyplot(plot_spectrogram(result))
        with col2:
            st.pyplot(plot_constellation(result))

        st.pyplot(plot_offset_histogram(result))


# ----- Batch mode -----
with tab2:
    st.subheader("Identify multiple clips")
    st.write("Upload several query clips. The app outputs a `results.csv` "
             "with one row per clip: `filename, prediction`.")

    uploaded_files = st.file_uploader(
        "Upload query clips", type=["mp3", "wav"], accept_multiple_files=True, key="batch"
    )

    if uploaded_files:
        if st.button("Run batch identification"):
            rows = []
            progress = st.progress(0)

            for i, uf in enumerate(uploaded_files):
                clip, fs = load_audio(uf)
                result = identify(clip, fs)

                if result["matched_song"] is not None:
                    prediction = os.path.splitext(result["matched_song"])[0]
                else:
                    prediction = ""

                rows.append({"filename": uf.name, "prediction": prediction})
                progress.progress((i + 1) / len(uploaded_files))

            results_df = pd.DataFrame(rows, columns=["filename", "prediction"])
            st.dataframe(results_df, use_container_width=True)

            csv_bytes = results_df.to_csv(index=False).encode("utf-8")
            st.download_button(
                "Download results.csv",
                data=csv_bytes,
                file_name="results.csv",
                mime="text/csv",
            )
