import os
import tkinter as tk
from tkinter import filedialog, messagebox
from tkinter import ttk

import numpy as np
import soundfile as sf
import librosa
import librosa.display
import matplotlib
matplotlib.use("TkAgg")
from matplotlib.figure import Figure
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2Tk

from scipy.signal import welch, get_window
from scipy.fft import rfft, rfftfreq


# --------- Helpers ---------
def to_mono(x):
    if x.ndim == 1:
        return x
    return np.mean(x, axis=1)

def safe_float(var, default):
    try:
        v = float(var.get())
        if np.isnan(v) or np.isinf(v):
            return default
        return v
    except Exception:
        return default

def safe_int(var, default):
    try:
        v = int(float(var.get()))
        return v
    except Exception:
        return default

def next_pow2(n):
    return 1 << (int(np.ceil(np.log2(max(1, n)))))

def db(x, floor_db=-120.0):
    x = np.maximum(x, 1e-20)
    out = 10.0 * np.log10(x)
    return np.maximum(out, floor_db)


# --------- Main App ---------
class SpectralGUI(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("Bioacoustics Spectral Explorer")
        self.geometry("1200x850")

        # State
        self.audio = None
        self.sr = None
        self.filename = None

        # Top controls
        ctrl = ttk.Frame(self, padding=8)
        ctrl.pack(side=tk.TOP, fill=tk.X)

        ttk.Button(ctrl, text="Open Audio", command=self.open_audio).pack(side=tk.LEFT, padx=4)
        ttk.Button(ctrl, text="Export Current PNG", command=self.export_png).pack(side=tk.LEFT, padx=4)

        ttk.Separator(ctrl, orient=tk.VERTICAL).pack(side=tk.LEFT, fill=tk.Y, padx=8)

        # Parameter controls
        self.fft_size_var = tk.StringVar(value="2048")
        self.window_ms_var = tk.StringVar(value="23.2")   # ~1024 at 44.1k
        self.hop_ms_var = tk.StringVar(value="10.0")
        self.max_freq_var = tk.StringVar(value="22050")
        self.db_range_var = tk.StringVar(value="80")      # spectrogram dyn range
        self.mels_var = tk.StringVar(value="128")
        self.window_type_var = tk.StringVar(value="hann")

        def make_labeled(entry_parent, text, var, width=7):
            f = ttk.Frame(entry_parent)
            ttk.Label(f, text=text).pack(side=tk.LEFT)
            e = ttk.Entry(f, textvariable=var, width=width)
            e.pack(side=tk.LEFT, padx=(4, 0))
            return f

        params = ttk.Frame(ctrl)
        params.pack(side=tk.LEFT, padx=8)

        make_labeled(params, "FFT:", self.fft_size_var).grid(row=0, column=0, padx=6, pady=2, sticky="w")
        make_labeled(params, "Win (ms):", self.window_ms_var).grid(row=0, column=1, padx=6, pady=2, sticky="w")
        make_labeled(params, "Hop (ms):", self.hop_ms_var).grid(row=0, column=2, padx=6, pady=2, sticky="w")
        make_labeled(params, "Max f (Hz):", self.max_freq_var).grid(row=0, column=3, padx=6, pady=2, sticky="w")
        make_labeled(params, "dB range:", self.db_range_var).grid(row=0, column=4, padx=6, pady=2, sticky="w")
        make_labeled(params, "Mel bins:", self.mels_var).grid(row=0, column=5, padx=6, pady=2, sticky="w")

        ttk.Label(params, text="Window:").grid(row=0, column=6, padx=(10, 2), sticky="e")
        ttk.Combobox(params, textvariable=self.window_type_var, values=[
            "hann", "hamming", "blackman", "nuttall", "flattop", "boxcar"
        ], width=10, state="readonly").grid(row=0, column=7, padx=2, sticky="w")

        ttk.Button(ctrl, text="Update Plots", command=self.update_all).pack(side=tk.LEFT, padx=10)

        # Tabs
        self.notebook = ttk.Notebook(self)
        self.notebook.pack(fill=tk.BOTH, expand=True)

        self.figures = {}
        self.canvases = {}

        self.add_tab("Waveform")
        self.add_tab("FFT Magnitude")
        self.add_tab("Welch PSD")
        self.add_tab("Spectrogram (STFT)")
        self.add_tab("Mel Spectrogram")
        self.add_tab("Autocorrelation")
        self.add_tab("Cepstrum")

        # Info bar
        self.status = tk.StringVar(value="Open an audio file to begin.")
        ttk.Label(self, textvariable=self.status, anchor="w").pack(fill=tk.X, padx=8, pady=4)

    def add_tab(self, name):
        frame = ttk.Frame(self.notebook)
        self.notebook.add(frame, text=name)

        fig = Figure(figsize=(10, 6), dpi=100)
        canvas = FigureCanvasTkAgg(fig, master=frame)
        toolbar = NavigationToolbar2Tk(canvas, frame, pack_toolbar=False)
        toolbar.update()
        toolbar.pack(side=tk.TOP, fill=tk.X)
        canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)

        self.figures[name] = fig
        self.canvases[name] = canvas

    # ---------- IO ----------
    def open_audio(self):
        path = filedialog.askopenfilename(
            title="Open audio",
            filetypes=[("Audio", "*.wav *.flac *.mp3 *.ogg *.m4a *.aac *.aiff *.aif *.wma"),
                       ("All files", "*.*")]
        )
        if not path:
            return
        try:
            # librosa handles most formats; preserves native sr by passing sr=None
            x, sr = librosa.load(path, sr=None, mono=False)
            if x.ndim > 1:
                x = to_mono(x.T)  # librosa returns shape (n_channels, n_samples) sometimes
            self.audio = x.astype(np.float32, copy=False)
            self.sr = int(sr)
            self.filename = os.path.basename(path)
            self.status.set(f"Loaded: {self.filename} | {len(self.audio)/self.sr:.2f} s @ {self.sr} Hz")
            # Set default max freq
            self.max_freq_var.set(str(self.sr / 2))
            # Snap FFT size to next power of two of default window
            self.update_all()
        except Exception as e:
            messagebox.showerror("Error", f"Could not open file:\n{e}")

    def export_png(self):
        if self.audio is None:
            messagebox.showinfo("Export", "Load an audio file first.")
            return
        tab = self.notebook.tab(self.notebook.select(), "text")
        fig = self.figures[tab]
        default_name = f"{os.path.splitext(self.filename or 'audio')[0]}_{tab.replace(' ', '_')}.png"
        out = filedialog.asksaveasfilename(
            title="Export PNG",
            defaultextension=".png",
            initialfile=default_name,
            filetypes=[("PNG", "*.png")]
        )
        if not out:
            return
        try:
            fig.savefig(out, bbox_inches="tight", dpi=200)
            self.status.set(f"Saved PNG: {out}")
        except Exception as e:
            messagebox.showerror("Error", f"Could not save:\n{e}")

    # ---------- Plotting ----------
    def update_all(self):
        if self.audio is None:
            self.status.set("No audio loaded. Open a file first.")
            return

        n_fft = max(256, safe_int(self.fft_size_var, 2048))
        win_ms = max(1.0, safe_float(self.window_ms_var, 23.2))
        hop_ms = max(0.5, safe_float(self.hop_ms_var, 10.0))
        mel_bins = max(16, safe_int(self.mels_var, 128))
        max_f = max(50.0, safe_float(self.max_freq_var, self.sr / 2))
        db_range = max(20.0, safe_float(self.db_range_var, 80.0))
        wtype = self.window_type_var.get()

        nperseg = int(self.sr * win_ms / 1000.0)
        nperseg = max(64, nperseg)
        nperseg = min(nperseg, n_fft)  # avoid larger window than FFT
        nperseg = next_pow2(nperseg)   # power of two window (fast)
        hop_length = int(self.sr * hop_ms / 1000.0)
        hop_length = max(1, hop_length)

        # Waveform
        self.plot_waveform()

        # FFT magnitude
        self.plot_fft(n_fft, wtype, max_f)

        # Welch PSD
        self.plot_welch_psd(nperseg, wtype, max_f)

        # Spectrogram (linear)
        self.plot_spectrogram(n_fft, nperseg, hop_length, wtype, db_range, max_f)

        # Mel spectrogram
        self.plot_mel(mel_bins, n_fft, hop_length, wtype, db_range, max_f)

        # Autocorrelation
        self.plot_autocorr(nperseg)

        # Cepstrum
        self.plot_cepstrum(n_fft, wtype, max_f)

        self.status.set(
            f"Updated: FFT={n_fft}, Win={nperseg} ({win_ms:.1f} ms), Hop={hop_ms:.1f} ms, "
            f"Mel={mel_bins}, Max f={max_f:.0f} Hz, dB range={db_range:.0f}"
        )

    def get_window(self, wtype, n):
        try:
            return get_window(wtype, n, fftbins=True)
        except Exception:
            return get_window("hann", n, fftbins=True)

    def plot_waveform(self):
        name = "Waveform"
        fig = self.figures[name]
        fig.clf()
        ax = fig.add_subplot(111)
        t = np.arange(len(self.audio)) / self.sr
        ax.plot(t, self.audio, linewidth=0.7)
        ax.set_title(f"Waveform: {self.filename or ''}")
        ax.set_xlabel("Time (s)")
        ax.set_ylabel("Amplitude")
        ax.grid(True, alpha=0.3)
        self.canvases[name].draw()

    def plot_fft(self, n_fft, wtype, max_f):
        name = "FFT Magnitude"
        fig = self.figures[name]
        fig.clf()
        ax = fig.add_subplot(111)

        x = self.audio
        n = min(len(x), n_fft)
        xseg = x[:n]
        w = self.get_window(wtype, n)
        X = rfft(xseg * w, n=n_fft)
        f = rfftfreq(n_fft, 1.0 / self.sr)
        mag = 20 * np.log10(np.maximum(np.abs(X), 1e-12))

        mask = f <= max_f
        ax.plot(f[mask], mag[mask], linewidth=0.8)
        ax.set_title("FFT Magnitude (first windowed segment)")
        ax.set_xlabel("Frequency (Hz)")
        ax.set_ylabel("Magnitude (dBFS)")
        ax.grid(True, alpha=0.3)
        self.canvases[name].draw()

    def plot_welch_psd(self, nperseg, wtype, max_f):
        name = "Welch PSD"
        fig = self.figures[name]
        fig.clf()
        ax = fig.add_subplot(111)

        f, Pxx = welch(self.audio, fs=self.sr, window=wtype, nperseg=nperseg, noverlap=nperseg//2, detrend=False)
        mask = f <= max_f
        ax.semilogy(f[mask], np.maximum(Pxx[mask], 1e-20))
        ax.set_title("Power Spectral Density (Welch)")
        ax.set_xlabel("Frequency (Hz)")
        ax.set_ylabel("Power / Hz")
        ax.grid(True, which="both", alpha=0.3)
        self.canvases[name].draw()

    def plot_spectrogram(self, n_fft, nperseg, hop_length, wtype, db_range, max_f):
        name = "Spectrogram (STFT)"
        fig = self.figures[name]
        fig.clf()
        ax = fig.add_subplot(111)

        S = librosa.stft(self.audio, n_fft=n_fft, hop_length=hop_length, win_length=nperseg,
                         window=wtype, center=True)
        S_mag = np.abs(S)**2
        S_db = librosa.power_to_db(S_mag, ref=np.max)
        vmax = 0
        vmin = vmax - db_range

        img = librosa.display.specshow(
            S_db, sr=self.sr, hop_length=hop_length, x_axis="time", y_axis="linear", ax=ax, cmap="magma",
            vmin=vmin, vmax=vmax
        )
        ax.set_ylim(0, max_f)
        ax.set_title("Linear-frequency Spectrogram (power, dB)")
        fig.colorbar(img, ax=ax, format="%+2.0f dB")
        self.canvases[name].draw()

    def plot_mel(self, n_mels, n_fft, hop_length, wtype, db_range, max_f):
        name = "Mel Spectrogram"
        fig = self.figures[name]
        fig.clf()
        ax = fig.add_subplot(111)

        S = librosa.feature.melspectrogram(
            y=self.audio, sr=self.sr, n_fft=n_fft, hop_length=hop_length, n_mels=n_mels,
            fmax=max_f, window=wtype, power=2.0
        )
        S_db = librosa.power_to_db(S, ref=np.max)
        vmax = 0
        vmin = vmax - db_range

        img = librosa.display.specshow(S_db, sr=self.sr, hop_length=hop_length, x_axis="time",
                                       y_axis="mel", fmax=max_f, ax=ax, cmap="magma",
                                       vmin=vmin, vmax=vmax)
        ax.set_title(f"Mel Spectrogram ({n_mels} mel bins, dB)")
        fig.colorbar(img, ax=ax, format="%+2.0f dB")
        self.canvases[name].draw()

    def plot_autocorr(self, nperseg):
        name = "Autocorrelation"
        fig = self.figures[name]
        fig.clf()
        ax = fig.add_subplot(111)

        # Use a windowed segment for clearer short-term periodicities
        n = min(len(self.audio), nperseg * 4)
        x = self.audio[:n]
        x = x - np.mean(x)
        ac = np.correlate(x, x, mode="full")
        ac = ac[ac.size // 2:]  # keep non-negative lags
        ac /= np.maximum(ac[0], 1e-20)
        lags = np.arange(ac.size) / self.sr

        ax.plot(lags, ac, linewidth=0.8)
        ax.set_xlim(0, min(0.5, len(x)/self.sr))  # show up to 0.5 s by default (good for many calls)
        ax.set_title("Autocorrelation (normalized)")
        ax.set_xlabel("Lag (s)")
        ax.set_ylabel("Correlation")
        ax.grid(True, alpha=0.3)
        self.canvases[name].draw()

    def plot_cepstrum(self, n_fft, wtype, max_f):
        name = "Cepstrum"
        fig = self.figures[name]
        fig.clf()
        ax = fig.add_subplot(111)

        n = min(len(self.audio), n_fft)
        xseg = self.audio[:n]
        w = self.get_window(wtype, n)
        X = rfft(xseg * w, n=n_fft)
        log_mag = np.log(np.maximum(np.abs(X), 1e-20))
        ceps = np.fft.irfft(log_mag)
        quefrency = np.arange(ceps.size) / self.sr

        # Often we look up to ~20 ms for pitch clues; adapt to sr
        qmax = min(0.02, quefrency[-1])
        imax = np.searchsorted(quefrency, qmax)

        ax.plot(quefrency[:imax], ceps[:imax], linewidth=0.8)
        ax.set_title("Real Cepstrum (low-quefrency region)")
        ax.set_xlabel("Quefrency (s)")
        ax.set_ylabel("Amplitude")
        ax.grid(True, alpha=0.3)
        self.canvases[name].draw()


if __name__ == "__main__":
    app = SpectralGUI()
    app.mainloop()
