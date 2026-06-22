# Sonic Signatures — Song Identifier (EE200 Q3B)

A Shazam-style song identifier built from scratch using spectrogram peak
fingerprinting and combinatorial hashing (Q3A), wrapped in a Streamlit app (Q3B).

## Files

- `app.py` — the Streamlit app (single-clip mode + batch mode)
- `fingerprint.py` — core pipeline: spectrogram, peak picking, hashing, matching
- `database.pkl` — pre-built fingerprint database for the provided song library
- `requirements.txt` — dependencies

## Running locally

```bash
pip install -r requirements.txt
streamlit run app.py
```

## Building database.pkl

`database.pkl` is built once in Colab from the provided song library, using
`fingerprint.py`'s `compute_spectrogram`, `find_peaks`, and
`add_song_to_database` functions on each song, then pickled. It must ship
inside this repo so the deployed app works immediately without rebuilding
the index at runtime.

## Deploying on Streamlit Community Cloud

1. Push this folder to a public GitHub repo.
2. Go to https://share.streamlit.io, sign in with GitHub.
3. Click "New app", select this repo, branch, and `app.py` as the entry point.
4. Deploy. Streamlit Cloud installs `requirements.txt` automatically.
5. Test both modes on the live link before submitting.

## Modes

- **Single-clip mode**: upload one query clip (mp3/wav). Displays the
  spectrogram, constellation map, offset histogram, and the matched song.
- **Batch mode**: upload multiple query clips at once. Outputs
  `results.csv` with columns `filename, prediction`, where `prediction`
  is the matched song's filename without extension.
