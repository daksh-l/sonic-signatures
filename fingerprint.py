"""
Core audio fingerprinting functions for the song identifier.

This module contains the full pipeline:
spectrogram -> peak picking -> hashing -> database matching.

Logic is unchanged from the Colab notebook (EE200 Q3A); this file just
collects everything into one importable place for the Streamlit app.
"""

import numpy as np
from scipy.signal import spectrogram
from scipy.ndimage import maximum_filter
from collections import Counter

# ----- fixed parameters used throughout (same as in the notebook) -----
SAMPLE_RATE = 22050
NPERSEG = 2048
NOVERLAP = 1024
NEIGHBORHOOD_SIZE = 15
THRESHOLD = -40
FAN_VALUE = 5


def compute_spectrogram(clip, fs=SAMPLE_RATE):
    """Returns f, t, Sxx_db for an audio clip."""
    f, t, Sxx = spectrogram(clip, fs=fs, nperseg=NPERSEG, noverlap=NOVERLAP, window='hann')
    Sxx_db = 10 * np.log10(Sxx + 1e-10)
    return f, t, Sxx_db


def find_peaks(Sxx_db, neighborhood_size=NEIGHBORHOOD_SIZE, threshold=THRESHOLD):
    """Finds local maxima in the spectrogram above a loudness threshold."""
    local_max = maximum_filter(Sxx_db, size=neighborhood_size) == Sxx_db
    above_threshold = Sxx_db > threshold
    peak_mask = local_max & above_threshold

    freq_idx, time_idx = np.where(peak_mask)
    return freq_idx, time_idx


def generate_hashes(freq_idx, time_idx, fan_value=FAN_VALUE):
    """Pairs nearby peaks into (f1, f2, delta_t, t1) hashes."""
    peaks = sorted(zip(time_idx, freq_idx))  # sort by time

    hashes = []
    for i in range(len(peaks)):
        t1, f1 = peaks[i]
        for j in range(1, fan_value + 1):
            if i + j >= len(peaks):
                break
            t2, f2 = peaks[i + j]
            delta_t = t2 - t1
            hashes.append((f1, f2, delta_t, t1))

    return hashes


def add_song_to_database(database, song_id, freq_idx, time_idx, fan_value=FAN_VALUE):
    """Adds one song's hashes into the shared database dict."""
    hashes = generate_hashes(freq_idx, time_idx, fan_value=fan_value)

    for f1, f2, delta_t, t1 in hashes:
        hash_key = (f1, f2, delta_t)
        if hash_key not in database:
            database[hash_key] = []
        database[hash_key].append((song_id, t1))


def match_song(query_freq_idx, query_time_idx, database, fan_value=FAN_VALUE):
    """
    Matches a query clip's peaks against the database.

    Returns:
        best_song:    filename of the matched song (or None)
        best_score:   number of hashes agreeing on the winning offset
        offset_counts: dict song_id -> Counter(offset -> count), the full
                       histogram data, needed for displaying the offset
                       histogram in the app.
    """
    query_hashes = generate_hashes(query_freq_idx, query_time_idx, fan_value=fan_value)

    offset_counts = {}  # song_id -> Counter of offsets

    for f1, f2, delta_t, t_query in query_hashes:
        hash_key = (f1, f2, delta_t)

        if hash_key in database:
            for song_id, t_db in database[hash_key]:
                offset = t_db - t_query

                if song_id not in offset_counts:
                    offset_counts[song_id] = Counter()
                offset_counts[song_id][offset] += 1

    best_song = None
    best_score = 0

    for song_id, counter in offset_counts.items():
        top_offset, count = counter.most_common(1)[0]
        if count > best_score:
            best_score = count
            best_song = song_id

    return best_song, best_score, offset_counts


def fingerprint_clip(clip, fs=SAMPLE_RATE):
    """
    Full pipeline for one clip: spectrogram -> peaks.
    Returns f, t, Sxx_db, freq_idx, time_idx -- everything needed both
    for matching and for plotting the intermediate steps.
    """
    f, t, Sxx_db = compute_spectrogram(clip, fs=fs)
    freq_idx, time_idx = find_peaks(Sxx_db)
    return f, t, Sxx_db, freq_idx, time_idx
