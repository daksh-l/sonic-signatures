# Sonic Signatures - Song Identifier (EE200 Q3B)

A Shazam-style song identifier built from scratch using spectrogram peak
fingerprinting and hashing, wrapped in a Streamlit app.

Link: https://sonic-signatures-app.streamlit.app/

## Files

- `app.py` - the Streamlit app (single-clip mode + batch mode)
- `fingerprint.py` - core functions: spectrogram, peak picking, hashing, matching
- `database.pkl` - pre-built fingerprint database for the provided song library
- `requirements.txt` - dependencies


## Building database.pkl

`database.pkl` was built once in Colab from the provided song library, using
`fingerprint.py`'s `compute_spectrogram`, `find_peaks`, and
`add_song_to_database` functions on each song, then pickled. It must ship
inside this repo so the deployed app works immediately without rebuilding
the index at runtime.


## Modes

- **Single-clip mode**: upload one query clip (mp3/wav). Displays the
  spectrogram, constellation map, offset histogram, and the matched song.
- **Batch mode**: upload multiple query clips at once. Outputs
  `results.csv` with columns `filename, prediction`, where `prediction`
  is the matched song's filename without extension.
