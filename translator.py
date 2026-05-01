"""
╔══════════════════════════════════════════════════════════╗
║       SOUND TRANSLATOR v2  ·  Project Hail Mary          ║
║                                                          ║
║  NEW IN v2:                                              ║
║  • Scrolling spectrogram — see sounds as a road of       ║
║    colour moving through time                            ║
║  • Pattern matching — cosine similarity on 64-bin        ║
║    spectral fingerprint, not just peak frequency         ║
║  • Three meters: amplitude, confidence, raw similarity   ║
║                                                          ║
║  HOW TO USE:                                             ║
║  1. Watch the spectrogram — every sound leaves a trail   ║
║  2. Click RECORD SOUND, make your sound, STOP            ║
║  3. Click LABEL LAST SOUND — type a word                 ║
║  4. Click START LISTENING — known patterns translate     ║
╚══════════════════════════════════════════════════════════╝

Install:
    pip install sounddevice numpy scipy matplotlib --break-system-packages
Run:
    python3 sound_translator.py
"""

import tkinter as tk
from tkinter import ttk, simpledialog, messagebox
import numpy as np
import sounddevice as sd
import matplotlib
matplotlib.use("TkAgg")
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.colors import LinearSegmentedColormap
import json
import os
import threading
from collections import deque

# ─── Configuration ─────────────────────────────────────────────────────────────
SAMPLE_RATE   = 44100
BLOCK_SIZE    = 4096
MIN_FREQ      = 80       # Hz — ignore below (hum / rumble)
MAX_FREQ      = 4000     # Hz — ignore above
AMP_THRESHOLD = 0.008    # minimum amplitude to trigger analysis
DICT_FILE     = "sound_dictionary.json"
REFRESH_MS    = 55       # UI refresh interval (~18 fps)

# Pattern matching
N_BINS        = 64       # spectral fingerprint resolution
MIN_COSINE    = 0.80     # cosine similarity threshold to accept match
SMOOTH_FRAMES = 6        # average this many frames before matching

# Spectrogram dimensions
SPEC_W        = 220      # time columns (width)
SPEC_H        = 120      # frequency rows (height)

# ─── Colour palette ─────────────────────────────────────────────────────────────
BG     = "#08081a"
BG2    = "#0d0d25"
BG3    = "#060614"
GREEN  = "#00e676"
BLUE   = "#448aff"
ORANGE = "#ff6d00"
RED    = "#ff1744"
CYAN   = "#00e5ff"
DIM    = "#2a2a50"
TEXT   = "#c8c8f0"
WHITE  = "#ffffff"

# Spectrogram: silence -> glow -> bright signal
SPEC_CMAP = LinearSegmentedColormap.from_list(
    "hailmary",
    ["#08081a", "#0a0f35", "#0040a0", "#0099ff", "#00e676", "#ffee00"],
    N=256,
)


# ═══════════════════════════════════════════════════════════════════════════════
class SoundTranslator:

    def __init__(self, root: tk.Tk):
        self.root = root
        self.root.title("Sound Translator v2  ·  Project Hail Mary")
        self.root.configure(bg=BG)
        self.root.geometry("1340x810")

        # ── Audio state (written by audio thread, read by UI thread) ──────────
        self._lock         = threading.Lock()
        self.current_amp   = 0.0
        self.current_freq  = 0.0
        self.fft_smooth    = np.zeros(BLOCK_SIZE // 2 + 1)
        self.spec_data     = np.zeros((SPEC_H, SPEC_W))
        self.live_patterns = deque(maxlen=SMOOTH_FRAMES)

        # ── Recording state ───────────────────────────────────────────────────
        self.is_listening  = False
        self.is_recording  = False
        self._rec_buf      = []
        self.last_capture  = None   # (freq, pattern_ndarray)

        # ── Dictionary ────────────────────────────────────────────────────────
        # Each entry: {"word": str, "freq": float, "pattern": list[float]}
        self.dictionary: list = self._load_dict()

        # ── Build UI and start audio ──────────────────────────────────────────
        self._build_ui()
        self._start_stream()
        self._schedule_refresh()

    # ══════════════════════════════════════════════════════════════════════════
    #  DICTIONARY
    # ══════════════════════════════════════════════════════════════════════════

    def _load_dict(self) -> list:
        if not os.path.exists(DICT_FILE):
            return []
        try:
            with open(DICT_FILE) as f:
                raw = json.load(f)
            # Migrate v1 format {freq_str: word_str}
            if isinstance(raw, dict):
                out = []
                for k, v in raw.items():
                    if isinstance(v, dict):
                        out.append(v)
                    else:
                        out.append({"word": str(v),
                                    "freq": float(k),
                                    "pattern": None})
                return out
            return raw
        except Exception:
            return []

    def _save_dict(self):
        with open(DICT_FILE, "w") as f:
            json.dump(self.dictionary, f, indent=2)

    # ══════════════════════════════════════════════════════════════════════════
    #  PATTERN EXTRACTION & MATCHING
    # ══════════════════════════════════════════════════════════════════════════

    def _extract_pattern(self, audio: np.ndarray):
        """
        Return (dominant_freq_hz, unit_pattern_vector[N_BINS]).

        The fingerprint is a 64-bin L2-normalised spectral vector covering
        [MIN_FREQ, MAX_FREQ].  Cosine similarity of two such vectors measures
        how alike the harmonic structures are, not just whether they share a
        peak frequency — so 'A4 on a flute' and 'A4 on a whistle' will
        produce different patterns despite the same fundamental.
        """
        win   = audio * np.hanning(len(audio))
        fft_v = np.abs(np.fft.rfft(win))
        freqs = np.fft.rfftfreq(len(audio), 1.0 / SAMPLE_RATE)
        mask  = (freqs >= MIN_FREQ) & (freqs <= MAX_FREQ)

        dom_freq = 0.0
        if mask.any():
            idx      = int(np.argmax(fft_v[mask]))
            dom_freq = float(freqs[mask][idx])

        bin_freqs = np.linspace(MIN_FREQ, MAX_FREQ, N_BINS)
        if mask.any():
            pattern = np.interp(bin_freqs, freqs[mask], fft_v[mask])
        else:
            pattern = np.zeros(N_BINS)

        norm = np.linalg.norm(pattern)
        if norm > 0:
            pattern = pattern / norm

        return dom_freq, pattern

    @staticmethod
    def _cosine(a: np.ndarray, b: np.ndarray) -> float:
        na = np.linalg.norm(a)
        nb = np.linalg.norm(b)
        if na == 0 or nb == 0:
            return 0.0
        return float(np.dot(a / na, b / nb))

    def _match(self, live: np.ndarray):
        """
        Compare live fingerprint against every stored entry.
        Returns (word, confidence_pct) or (None, 0).
        confidence_pct is scaled: MIN_COSINE → 1.0 maps to 0 → 100 %.
        """
        if not self.dictionary:
            return None, 0.0
        best_word, best_sim = None, -1.0
        for entry in self.dictionary:
            stored = entry.get("pattern")
            if stored is None:
                continue
            sim = self._cosine(live, np.asarray(stored, dtype=np.float64))
            if sim > best_sim:
                best_sim, best_word = sim, entry["word"]
        if best_sim >= MIN_COSINE:
            conf = (best_sim - MIN_COSINE) / (1.0 - MIN_COSINE) * 100.0
            return best_word, min(100.0, conf)
        return None, 0.0

    def _best_raw_sim(self, live: np.ndarray) -> float:
        """Highest raw cosine similarity — used for the sim_bar display."""
        sims = [
            self._cosine(live, np.asarray(e["pattern"], dtype=np.float64))
            for e in self.dictionary
            if e.get("pattern") is not None
        ]
        return max(sims) if sims else 0.0

    # ══════════════════════════════════════════════════════════════════════════
    #  AUDIO CALLBACK  (runs in sounddevice thread)
    # ══════════════════════════════════════════════════════════════════════════

    def _audio_cb(self, indata, frames, time_info, status):
        audio = indata[:, 0].copy()
        amp   = float(np.max(np.abs(audio)))

        # FFT
        win   = audio * np.hanning(len(audio))
        fft_v = np.abs(np.fft.rfft(win))
        freqs = np.fft.rfftfreq(len(audio), 1.0 / SAMPLE_RATE)
        mask  = (freqs >= MIN_FREQ) & (freqs <= MAX_FREQ)

        # Exponential smoothing for the spectrum display
        self.fft_smooth = 0.65 * self.fft_smooth + 0.35 * fft_v

        # Dominant frequency
        dom_freq = 0.0
        if mask.any() and amp > AMP_THRESHOLD:
            idx      = int(np.argmax(fft_v[mask]))
            dom_freq = float(freqs[mask][idx])

        # Build spectrogram column: SPEC_H normalised bins
        spec_bins = np.linspace(MIN_FREQ, MAX_FREQ, SPEC_H)
        col = np.interp(spec_bins, freqs[mask], fft_v[mask]) \
              if mask.any() else np.zeros(SPEC_H)
        c_max = col.max()
        if c_max > 0:
            col = col / c_max

        # Accumulate live fingerprint for matching
        if amp > AMP_THRESHOLD:
            _, pat = self._extract_pattern(audio)
            self.live_patterns.append(pat)   # deque append is GIL-safe

        # Commit all shared state under lock
        with self._lock:
            self.current_amp  = amp
            self.current_freq = dom_freq
            # Roll spectrogram left; newest column on the right
            self.spec_data    = np.roll(self.spec_data, -1, axis=1)
            self.spec_data[:, -1] = col
            if self.is_recording:
                self._rec_buf.extend(audio.tolist())

    def _start_stream(self):
        try:
            self._stream = sd.InputStream(
                samplerate=SAMPLE_RATE,
                blocksize=BLOCK_SIZE,
                channels=1,
                callback=self._audio_cb,
            )
            self._stream.start()
            self._status("● MICROPHONE ACTIVE", GREEN)
        except Exception as e:
            self._status("● MIC ERROR — check device", RED)
            messagebox.showerror(
                "Audio Error",
                f"Cannot open microphone:\n{e}\n\n"
                "Install:\n"
                "pip install sounddevice --break-system-packages",
            )

    # ══════════════════════════════════════════════════════════════════════════
    #  UI CONSTRUCTION
    # ══════════════════════════════════════════════════════════════════════════

    def _build_ui(self):
        top = tk.Frame(self.root, bg=BG, pady=9)
        top.pack(fill="x", padx=24)
        tk.Label(top, text="⟨ SOUND TRANSLATOR  v2 ⟩",
                 font=("Courier New", 19, "bold"),
                 fg=GREEN, bg=BG).pack(side="left")
        tk.Label(top,
                 text="S(f,t) = |∫ x(τ)·w(τ−t)·e^(−2πifτ) dτ|²",
                 font=("Courier New", 10), fg=DIM, bg=BG).pack(side="right")
        tk.Frame(self.root, bg=DIM, height=1).pack(fill="x", padx=24)

        body = tk.Frame(self.root, bg=BG)
        body.pack(fill="both", expand=True, padx=24, pady=10)
        self._build_plots(body)
        self._build_sidebar(body)

    # ── Matplotlib panel ─────────────────────────────────────────────────────

    def _build_plots(self, parent):
        plt.style.use("dark_background")
        self.fig = plt.figure(figsize=(8.9, 5.9), facecolor=BG)
        gs = self.fig.add_gridspec(
            2, 1,
            height_ratios=[1.7, 1.0],
            hspace=0.50,
            left=0.07, right=0.97,
            top=0.93, bottom=0.08,
        )
        self.ax_spec = self.fig.add_subplot(gs[0])
        self.ax_fft  = self.fig.add_subplot(gs[1])
        self._init_spectrogram()
        self._init_fft()
        canvas = FigureCanvasTkAgg(self.fig, master=parent)
        w = canvas.get_tk_widget()
        w.configure(bg=BG, highlightthickness=0)
        w.pack(side="left", fill="both", expand=True)
        self.canvas = canvas

    def _init_spectrogram(self):
        ax = self.ax_spec
        ax.set_facecolor(BG3)
        ax.set_title(
            "SPECTROGRAM  ·  frequency content over time"
            "  ───────────────────────────────►",
            color=CYAN, fontsize=9, fontfamily="Courier New", pad=7,
        )
        ax.set_ylabel("Frequency (Hz)", color=DIM,
                      fontsize=7, fontfamily="Courier New")
        ax.set_xlabel("time  →", color=DIM,
                      fontsize=7, fontfamily="Courier New")
        ax.set_xticks([])
        ax.tick_params(colors=DIM, labelsize=6)
        for sp in ax.spines.values():
            sp.set_color(DIM)

        # Real Hz labels on Y axis
        n_ticks  = 7
        tick_idx = np.linspace(0, SPEC_H - 1, n_ticks).astype(int)
        tick_hz  = np.linspace(MIN_FREQ, MAX_FREQ, n_ticks).astype(int)
        ax.set_yticks(tick_idx)
        ax.set_yticklabels([str(h) for h in tick_hz])

        # Subtle guide lines at musical references
        for gf, name in {220: "A3", 440: "A4", 880: "A5", 1760: "A6"}.items():
            frac = (gf - MIN_FREQ) / (MAX_FREQ - MIN_FREQ)
            yi   = frac * (SPEC_H - 1)
            ax.axhline(yi, color="#181830", linewidth=0.8, linestyle=":")
            ax.text(3, yi + 1.2, name, color="#1e1e40",
                    fontsize=6, fontfamily="Courier New")

        self.spec_img = ax.imshow(
            self.spec_data,
            aspect="auto",
            origin="lower",
            cmap=SPEC_CMAP,
            interpolation="bilinear",
            vmin=0.0,
            vmax=1.0,
        )
        # Thin live-edge cursor
        ax.axvline(SPEC_W - 1, color=CYAN, linewidth=0.9, alpha=0.30)

    def _init_fft(self):
        ax = self.ax_fft
        ax.set_facecolor(BG3)
        ax.set_title(
            "FREQUENCY SPECTRUM  ·  FFT  "
            "(cyan dots = 64-bin fingerprint stored/matched)",
            color=GREEN, fontsize=9, fontfamily="Courier New", pad=7,
        )
        ax.set_xlim(MIN_FREQ, MAX_FREQ)
        ax.set_ylim(0, 1.08)
        ax.set_xlabel("Frequency (Hz)", color=DIM,
                      fontsize=7, fontfamily="Courier New")
        ax.set_ylabel("magnitude", color=DIM,
                      fontsize=7, fontfamily="Courier New")
        ax.tick_params(colors=DIM, labelsize=6)
        for sp in ax.spines.values():
            sp.set_color(DIM)
        ax.grid(True, color="#101028", linewidth=0.5)

        # Note reference lines
        for f, n in {131: "C3", 220: "A3", 262: "C4", 330: "E4",
                     440: "A4", 523: "C5", 880: "A5"}.items():
            if MIN_FREQ <= f <= MAX_FREQ:
                ax.axvline(f, color="#181830", linewidth=0.7, linestyle=":")
                ax.text(f + 5, 0.95, n, color="#202040",
                        fontsize=6, fontfamily="Courier New")

        # High-res smooth spectrum line
        px = np.linspace(MIN_FREQ, MAX_FREQ, 600)
        self.fft_line, = ax.plot(
            px, np.zeros(600), color=GREEN, linewidth=1.4, zorder=3)
        self.fft_fill = ax.fill_between(
            px, np.zeros(600), alpha=0.10, color=GREEN, zorder=2)

        # 64-bin fingerprint dots — exactly what gets stored and compared
        bx = np.linspace(MIN_FREQ, MAX_FREQ, N_BINS)
        self.bin_dots, = ax.plot(
            bx, np.zeros(N_BINS), "o",
            color=CYAN, markersize=2.8, alpha=0.65, zorder=4)

        # Peak frequency marker
        self.peak_vline = ax.axvline(
            440, color=ORANGE, linewidth=1.5, linestyle="--", alpha=0.0)
        self.peak_text  = ax.text(
            450, 0.86, "", color=ORANGE,
            fontsize=8, fontfamily="Courier New")

    # ── Sidebar ──────────────────────────────────────────────────────────────

    def _build_sidebar(self, parent):
        side = tk.Frame(parent, bg=BG2, width=268)
        side.pack(side="right", fill="y", padx=(12, 0))
        side.pack_propagate(False)
        p = {"padx": 13, "pady": 4}

        # Frequency readout
        fb = tk.Frame(side, bg=BG3)
        fb.pack(fill="x", **p)
        tk.Label(fb, text="DOMINANT FREQUENCY",
                 font=("Courier New", 7, "bold"), fg=DIM, bg=BG3
                 ).pack(pady=(7, 0))
        self.freq_lbl = tk.Label(
            fb, text=" ─── Hz",
            font=("Courier New", 28, "bold"), fg=GREEN, bg=BG3)
        self.freq_lbl.pack(pady=(0, 7))

        # Three progress bars
        mb = tk.Frame(side, bg=BG2)
        mb.pack(fill="x", **p)
        style = ttk.Style()
        style.theme_use("default")
        for name, color in [
            ("G.Horizontal.TProgressbar", GREEN),
            ("B.Horizontal.TProgressbar", BLUE),
            ("C.Horizontal.TProgressbar", CYAN),
        ]:
            style.configure(name, troughcolor=BG3,
                            background=color, thickness=9)

        for lbl, attr, sty in [
            ("SIGNAL LEVEL",       "amp_bar",  "G.Horizontal.TProgressbar"),
            ("MATCH CONFIDENCE",   "conf_bar", "B.Horizontal.TProgressbar"),
            ("PATTERN SIMILARITY", "sim_bar",  "C.Horizontal.TProgressbar"),
        ]:
            tk.Label(mb, text=lbl, font=("Courier New", 7, "bold"),
                     fg=DIM, bg=BG2).pack(anchor="w", pady=(4, 0))
            bar = ttk.Progressbar(mb, length=240, mode="determinate",
                                   style=sty)
            bar.pack(fill="x", pady=2)
            setattr(self, attr, bar)

        self.conf_lbl = tk.Label(mb, text="0 %",
                                  font=("Courier New", 9, "bold"),
                                  fg=BLUE, bg=BG2)
        self.conf_lbl.pack()

        # Translation box
        tb = tk.Frame(side, bg=BG3)
        tb.pack(fill="x", **p)
        tk.Label(tb, text="TRANSLATION",
                 font=("Courier New", 7, "bold"), fg=DIM, bg=BG3
                 ).pack(pady=(7, 0))
        self.trans_lbl = tk.Label(
            tb, text="···",
            font=("Courier New", 22, "bold"),
            fg=WHITE, bg=BG3, wraplength=230)
        self.trans_lbl.pack(pady=(0, 7))

        tk.Frame(side, bg=DIM, height=1).pack(fill="x", padx=13, pady=3)

        # Buttons
        bc = dict(font=("Courier New", 10, "bold"), relief="flat",
                  cursor="hand2", pady=9, width=24, bd=0)

        self.listen_btn = tk.Button(
            side, text="▶   START LISTENING",
            bg=GREEN, fg="#000000",
            command=self._toggle_listen, **bc)
        self.listen_btn.pack(fill="x", padx=13, pady=3)

        self.record_btn = tk.Button(
            side, text="⏺   RECORD SOUND",
            bg=BLUE, fg=WHITE,
            command=self._toggle_record, **bc)
        self.record_btn.pack(fill="x", padx=13, pady=3)

        tk.Button(side, text="✎   LABEL LAST SOUND",
                  bg="#1c1c40", fg=TEXT,
                  command=self._label_sound, **bc
                  ).pack(fill="x", padx=13, pady=3)

        tk.Frame(side, bg=DIM, height=1).pack(fill="x", padx=13, pady=5)

        # Dictionary list
        tk.Label(side, text="SOUND DICTIONARY",
                 font=("Courier New", 8, "bold"), fg=DIM, bg=BG2
                 ).pack(anchor="w", padx=13)

        df = tk.Frame(side, bg=BG3)
        df.pack(fill="both", expand=True, padx=13, pady=4)
        sb = tk.Scrollbar(df, bg=BG2, troughcolor=BG3,
                          highlightthickness=0)
        sb.pack(side="right", fill="y")
        self.dict_lb = tk.Listbox(
            df,
            font=("Courier New", 9),
            bg=BG3, fg=TEXT,
            selectbackground=GREEN, selectforeground="#000000",
            relief="flat", bd=0,
            yscrollcommand=sb.set,
            highlightthickness=0, activestyle="none",
        )
        self.dict_lb.pack(fill="both", expand=True)
        sb.config(command=self.dict_lb.yview)

        tk.Button(side, text="🗑   DELETE SELECTED",
                  bg="#1a0008", fg=RED,
                  command=self._delete_entry, **bc
                  ).pack(fill="x", padx=13, pady=(4, 5))

        self.status_lbl = tk.Label(
            side, text="● INITIALISING",
            font=("Courier New", 8), fg=DIM, bg=BG2)
        self.status_lbl.pack(pady=(0, 8))

        self._refresh_dict_display()

    # ══════════════════════════════════════════════════════════════════════════
    #  REFRESH LOOP  (UI thread, called every REFRESH_MS)
    # ══════════════════════════════════════════════════════════════════════════

    def _schedule_refresh(self):
        self._refresh()
        self.root.after(REFRESH_MS, self._schedule_refresh)

    def _refresh(self):
        # Snapshot shared audio state
        with self._lock:
            freq  = self.current_freq
            amp   = self.current_amp
            spec  = self.spec_data.copy()
            fft_s = self.fft_smooth.copy()

        # Frequency label
        self.freq_lbl.config(
            text=f"{freq:>7.1f} Hz"
            if (amp > AMP_THRESHOLD and freq > 0)
            else " ─── Hz"
        )

        # Signal level bar
        self.amp_bar["value"] = min(100.0, amp * 600.0)

        # ── Spectrogram ───────────────────────────────────────────────────────
        self.spec_img.set_data(spec)
        peak = spec.max()
        # Adaptive brightness: quiet sounds stay visible
        self.spec_img.set_clim(0.0, max(peak * 0.80, 0.02))

        # ── FFT spectrum ──────────────────────────────────────────────────────
        all_freqs = np.fft.rfftfreq(BLOCK_SIZE, 1.0 / SAMPLE_RATE)
        mask = (all_freqs >= MIN_FREQ) & (all_freqs <= MAX_FREQ)
        df   = all_freqs[mask]
        dv   = fft_s[mask]

        if dv.size > 0:
            peak2  = dv.max()
            norm   = (dv / peak2) if peak2 > 0 else dv

            # 600-point smooth line
            px = np.linspace(MIN_FREQ, MAX_FREQ, 600)
            py = np.interp(px, df, norm) if df.size > 1 else np.zeros(600)
            self.fft_line.set_data(px, py)

            # Rebuild fill
            self.fft_fill.remove()
            self.fft_fill = self.ax_fft.fill_between(
                px, py, alpha=0.10, color=GREEN, zorder=2)

            # 64-bin fingerprint dots
            bx = np.linspace(MIN_FREQ, MAX_FREQ, N_BINS)
            by = np.interp(bx, df, norm) if df.size > 1 else np.zeros(N_BINS)
            self.bin_dots.set_ydata(by)

            # Peak marker
            if amp > AMP_THRESHOLD and freq > 0:
                self.peak_vline.set_xdata([freq, freq])
                self.peak_vline.set_alpha(0.80)
                lx = min(freq + 25.0, MAX_FREQ - 260.0)
                self.peak_text.set_position((lx, 0.86))
                self.peak_text.set_text(f"← {freq:.0f} Hz")
            else:
                self.peak_vline.set_alpha(0.0)
                self.peak_text.set_text("")

        self.canvas.draw_idle()

        # ── Pattern matching (listen mode only) ───────────────────────────────
        if self.is_listening and amp > AMP_THRESHOLD and self.live_patterns:
            avg_pat = np.mean(list(self.live_patterns), axis=0)
            word, conf = self._match(avg_pat)
            raw_sim    = self._best_raw_sim(avg_pat)

            self.trans_lbl.config(
                text=word if word else "unknown",
                fg=GREEN if word else ORANGE,
            )
            self.conf_bar["value"] = conf
            self.conf_lbl.config(text=f"{conf:.0f} %")
            self.sim_bar["value"]  = raw_sim * 100.0

        elif not self.is_listening:
            self.trans_lbl.config(text="···", fg=WHITE)
            self.conf_bar["value"] = 0
            self.sim_bar["value"]  = 0
            self.conf_lbl.config(text="0 %")

    # ══════════════════════════════════════════════════════════════════════════
    #  BUTTON HANDLERS
    # ══════════════════════════════════════════════════════════════════════════

    def _toggle_listen(self):
        self.is_listening = not self.is_listening
        if self.is_listening:
            self.listen_btn.config(text="⏹   STOP LISTENING", bg=RED)
            self._status("● LISTENING  —  pattern matching active", GREEN)
        else:
            self.listen_btn.config(text="▶   START LISTENING", bg=GREEN)
            self.trans_lbl.config(text="···", fg=WHITE)
            self.conf_bar["value"] = 0
            self.sim_bar["value"]  = 0
            self.conf_lbl.config(text="0 %")
            self._status("● READY", GREEN)

    def _toggle_record(self):
        self.is_recording = not self.is_recording
        if self.is_recording:
            with self._lock:
                self._rec_buf = []
            self.record_btn.config(text="⏹   STOP RECORDING", bg=RED)
            self._status("● RECORDING  —  make your sound now…", RED)
        else:
            self.record_btn.config(text="⏺   RECORD SOUND", bg=BLUE)
            self._process_recording()

    def _process_recording(self):
        with self._lock:
            samples = list(self._rec_buf)
        if len(samples) < BLOCK_SIZE:
            self._status("● TOO SHORT  —  hold the sound longer", ORANGE)
            return
        audio            = np.array(samples, dtype=np.float32)
        freq, pat        = self._extract_pattern(audio)
        self.last_capture = (freq, pat)
        self._status(
            f"● CAPTURED  {freq:.1f} Hz  —  click LABEL LAST SOUND", BLUE)

    def _label_sound(self):
        if self.last_capture is None:
            messagebox.showwarning(
                "Nothing Recorded",
                "Record a sound first using RECORD SOUND,\n"
                "then click LABEL LAST SOUND.",
            )
            return
        freq, pat = self.last_capture
        word = simpledialog.askstring(
            "Assign Meaning",
            f"Dominant frequency :  {freq:.1f} Hz\n"
            f"Fingerprint bins   :  {N_BINS}  spectral values\n\n"
            f"What does this sound mean?",
            parent=self.root,
        )
        if not word or not word.strip():
            return
        word = word.strip().lower()
        self.dictionary.append({
            "word":    word,
            "freq":    round(freq, 1),
            "pattern": pat.tolist(),
        })
        self._save_dict()
        self._refresh_dict_display()
        self._status(
            f"● SAVED  '{word}'  ({freq:.1f} Hz)  ✓ pattern stored", GREEN)
        self.last_capture = None

    def _delete_entry(self):
        sel = self.dict_lb.curselection()
        if not sel:
            return
        idx = sel[0]
        if 0 <= idx < len(self.dictionary):
            removed = self.dictionary.pop(idx)
            self._save_dict()
            self._refresh_dict_display()
            self._status(f"● DELETED  '{removed['word']}'", ORANGE)

    # ══════════════════════════════════════════════════════════════════════════
    #  HELPERS
    # ══════════════════════════════════════════════════════════════════════════

    def _refresh_dict_display(self):
        self.dict_lb.delete(0, tk.END)
        for e in self.dictionary:
            tag = "✓ pattern" if e.get("pattern") else "─ no pattern"
            self.dict_lb.insert(
                tk.END,
                f"  {e['freq']:>7.1f} Hz  →  {e['word']}  [{tag}]",
            )

    def _status(self, text: str, color: str = TEXT):
        self.status_lbl.config(text=text, fg=color)

    def on_close(self):
        try:
            self._stream.stop()
            self._stream.close()
        except Exception:
            pass
        self.root.destroy()


# ══════════════════════════════════════════════════════════════════════════════
def main():
    root = tk.Tk()
    app  = SoundTranslator(root)
    root.protocol("WM_DELETE_WINDOW", app.on_close)
    root.update_idletasks()
    sw, sh = root.winfo_screenwidth(), root.winfo_screenheight()
    root.geometry(f"1340x810+{(sw - 1340) // 2}+{(sh - 810) // 2}")
    root.mainloop()


if __name__ == "__main__":
    main()