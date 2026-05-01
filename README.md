# ⟨ SOUND TRANSLATOR ⟩
### Inspired by *Project Hail Mary* (2026)

> *"Any structured pattern of sound carries information that can be decoded — if approached scientifically."*

A real-time acoustic recognition system built on the same scientific principles Ryland Grace uses to communicate with Rocky. Record any sound, assign it a meaning, and watch the system translate it back in real time using FFT analysis and cosine similarity pattern matching.

---

## What it does

```
You make a sound
       ↓
Microphone captures raw waveform
       ↓
Fast Fourier Transform decomposes it into frequencies
       ↓
64-bin spectral fingerprint is extracted
       ↓
Cosine similarity matches it against your dictionary
       ↓
Word appears on screen  ·  confidence score shown
```

The key difference from naive frequency matching: two sounds with the same dominant pitch but different harmonic structure (a whistle vs a hum at A4) will produce **different fingerprints** and will not be confused.

---

## Screenshots

```
┌─────────────────────────────────────────────────────┐
│  SPECTROGRAM  ·  frequency content over time  ────► │
│  ░░░▒▒▒████▓▓▓░░░▒▒████████▓▒░░░░░░▒▒▒████▓░░░░░  │
│  Scrolling road of colour — every sound leaves      │
│  a unique trail as it moves through time            │
├─────────────────────────────────────────────────────┤
│  FREQUENCY SPECTRUM  ·  FFT fingerprint             │
│  ·  ·  ·∙∘○●●●●○∘·  ·  ·  64 cyan dots =          │
│  exactly what gets stored and compared              │
└─────────────────────────────────────────────────────┘
```

---

## Installation

**Requirements:** Python 3.9+ · a working microphone

```bash
# Clone or download the project
git clone https://github.com/yourname/sound-translator
cd sound-translator

# Install dependencies
pip install -r requirements.txt

# Run
python3 sound_translator.py
```

**On some Linux systems you may also need:**
```bash
sudo apt install python3-tk portaudio19-dev
```

---

## How to use

### Building your dictionary

| Step | Action |
|---|---|
| 1 | Click **⏺ RECORD SOUND** |
| 2 | Make your sound (whistle, hum, tap, voice) |
| 3 | Click **⏹ STOP RECORDING** |
| 4 | Click **✎ LABEL LAST SOUND** |
| 5 | Type a word and press Enter |
| 6 | Repeat for each new sound |

### Translating in real time

Click **▶ START LISTENING** — known sounds now trigger their assigned word instantly, with a confidence percentage shown.

### Deleting entries

Select any entry in the dictionary list on the right, then click **🗑 DELETE SELECTED**.

---

## The science

### Fast Fourier Transform

The FFT decomposes a complex waveform into its constituent frequencies. For a discrete signal, this is:

```
X[k] = Σ x[n] · e^(−2πikn/N)    for k = 0, 1, ..., N−1
```

Each frequency bin `k` corresponds to a real frequency `f = k · Fs / N` where `Fs` is the sample rate.

### Spectral fingerprint

Rather than storing only the peak frequency, the system extracts a **64-bin L2-normalised spectral vector** across the full [80Hz–4000Hz] range. This captures the complete harmonic structure of a sound, not just its loudest component.

```python
pattern = np.interp(bin_freqs, measured_freqs, fft_magnitudes)
pattern = pattern / np.linalg.norm(pattern)   # L2 normalise
```

### Cosine similarity

Two fingerprints **a** and **b** are compared as vectors:

```
similarity = (a · b) / (‖a‖ · ‖b‖)
```

A score of `1.0` means identical spectral structure. The match threshold is set at `0.80` by default — adjustable in the config section at the top of `sound_translator.py`.

### Spectrogram

The rolling spectrogram plots frequency (Y axis) against time (X axis). Each new audio block generates one column of colour, which scrolls left as time advances — creating the continuous road of sound visualisation.

---

## Configuration

All tunable parameters are at the top of `sound_translator.py`:

```python
SAMPLE_RATE   = 44100   # Hz
BLOCK_SIZE    = 4096    # samples per audio block
MIN_FREQ      = 80      # Hz — ignore below (hum/rumble)
MAX_FREQ      = 4000    # Hz — ignore above
AMP_THRESHOLD = 0.008   # minimum amplitude to trigger analysis
N_BINS        = 64      # spectral fingerprint resolution
MIN_COSINE    = 0.80    # similarity threshold to accept a match
SMOOTH_FRAMES = 6       # frames averaged before matching
```

**Tips for better accuracy:**
- Use in a quiet room — background noise directly degrades FFT quality
- Produce consistent, sustained sounds rather than short clicks
- Whistling and sustained humming give the most reliable fingerprints
- Lower `MIN_COSINE` (e.g. `0.70`) to accept more matches; raise it for stricter recognition
- Lower `AMP_THRESHOLD` if the system misses quiet sounds

---

## File structure

```
sound-translator/
├── sound_translator.py      # main application
├── requirements.txt         # Python dependencies
├── README.md                # this file
└── sound_dictionary.json    # auto-created when you save first sound
```

The dictionary is saved automatically as `sound_dictionary.json` in the same folder. It persists between sessions — your vocabulary survives restarts.

---

## Limitations

This is a **Step 1–5 implementation** of Grace's pipeline. Step 6 — predictive generalisation — requires a trained neural network and is not included.

| Step | Status |
|---|---|
| Sound capture | ✅ Implemented |
| FFT analysis | ✅ Implemented |
| Pattern storage | ✅ Implemented |
| Meaning assignment | ✅ Human-assisted |
| Recognition & translation | ✅ Implemented |
| Predictive generalisation | ❌ Requires neural network |

The system will not generalise — a sound it has never seen exactly will not be predicted from similar known sounds. Every new sound must be manually labelled. This mirrors the most scientifically honest aspect of Grace's challenge: **meaning cannot be automated**.

---

## Academic context

This project was built alongside a reflective essay for **CCST9091 — Unveiling Science and Technology in Cinema**, The University of Hong Kong.

**Key references:**

- Sajedian, I., & Rho, J. (2019). Accurate and instant frequency estimation from noisy sinusoidal waves by deep learning. *Nanoscale and Microscale Thermophysical Engineering, 4*, 197.
- Oppenheim, J. N., & Magnasco, M. O. (2013). Human time-frequency acuity beats the Fourier uncertainty principle. *Physical Review Letters, 110*(4), 044301.
- Rubio-García, A., Muñiz-Rojas, J., & Mora-Torres, M. (2025). Animals as communication partners. *Animals, 16*(3), 375.
- Lee, W. S. et al. (2020). Fast frequency discrimination using a biomimetic membrane. *arXiv*.

---

## License

MIT — do whatever you want with it.

---

*Built with numpy · sounddevice · matplotlib · tkinter*
*Inspired by Andy Weir's Project Hail Mary*