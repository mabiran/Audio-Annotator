import os
import time
import tkinter as tk
from tkinter import filedialog, messagebox, simpledialog
from PIL import Image, ImageTk
import pandas as pd
import librosa
import matplotlib
matplotlib.use("Agg")
from matplotlib.figure import Figure
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import sounddevice as sd
import soundfile as sf
import threading
import numpy as np
import ttkbootstrap as ttk
from ttkbootstrap.constants import *
from pandas.errors import EmptyDataError

AUDIO_EXTS = ('.wav', '.mp3', '.flac', '.ogg', '.m4a')

class AudioAnnotator:
    def __init__(self, root):
        self.root = root
        self.root.title("AUT Animal Sound Annotator")
        self.root.geometry("1280x860")

        # --- state ---
        self.label_buttons = []
        self.audio_folder = ""
        self.output_folder = ""
        self.annotation_csv = ""
        self.csv_loaded = False
        self.audio_folder_set = False
        self.output_folder_set = False
        self.annotation_path_set = False
        self.file_list = []
        self.full_file_list = []
        self.annotated_files = set()
        self.current_index = 0
        self.df = None
        self.filename_column = 'File Name'

        self.audio_data = None
        self.full_audio_data = None
        self.sample_rate = None
        self.clicked_time = None
        self.current_filename = ""
        self.original_filename = ""
        self.current_metadata = None
        self.stop_event = threading.Event()
        self.play_thread = None

        self.label_options = ["Chitter", "Screech", "Grunt", "Hiss", "General Bird Sound"]
        self.segment_duration = 5
        self.output_sample_rate = 96000

        # Mel bounds (Hz)
        self.mel_fmin = 0.0
        self.mel_fmax = None

        # Display window/scroll
        self.view_window_s = 10.0      # seconds visible
        self.view_start_s = 0.0        # start time of visible window
        self.follow_playhead = tk.BooleanVar(value=True)

        # Contrast controls (in dB)
        self.db_floor = -80.0          # vmin
        self.db_range = 80.0           # vmax = floor + range

        # Cached mel
        self.S_dB = None               # full log-mel (np.ndarray [n_mels, n_frames])
        self.hop_length = 512
        self.n_fft = 2048
        self.n_mels = 256
        self.fmin_used = 0.0
        self.fmax_used = None

        # mpl refs
        self._fig = None
        self._canvas = None     # FigureCanvasTkAgg
        self._canvas_widget = None  # tk widget of the canvas
        self._ax = None
        self._img = None
        self._playline = None

        # container size tracking
        self._dpi = 100
        self._holder_w = 980
        self._holder_h = 560
        self._resize_after_id = None

        self.setup_gui()

    # ---------------- GUI ----------------
    def setup_gui(self):
        # Logo
        logo_frame = ttk.Frame(self.root); logo_frame.pack(pady=5)
        try:
            img = Image.open("aut_logo.png").resize((120, 60))
            self._logo = ImageTk.PhotoImage(img)
            ttk.Label(logo_frame, image=self._logo).pack()
        except Exception:
            ttk.Label(logo_frame, text="AUT").pack()

        # Top buttons (CSV optional)
        top = ttk.Frame(self.root); top.pack(pady=10)
        self.csv_btn   = ttk.Button(top, text="1. (Optional) Select CSV", bootstyle=SECONDARY, command=self.load_csv)
        self.audio_btn = ttk.Button(top, text="2. Select Audio Folder",  bootstyle=INFO,      state=tk.NORMAL,   command=self.set_audio_folder)
        self.output_btn= ttk.Button(top, text="3. Select Output Folder", bootstyle=SUCCESS,   state=tk.DISABLED, command=self.set_output_folder)
        self.annotation_btn = ttk.Button(top, text="4. Set Annotation CSV", bootstyle=DANGER, state=tk.DISABLED, command=self.set_annotation_csv)
        for b in (self.csv_btn, self.audio_btn, self.output_btn, self.annotation_btn):
            b.pack(side=tk.LEFT, padx=5)

        self.start_btn = ttk.Button(self.root, text="Start Annotation", bootstyle=WARNING, state=tk.DISABLED, command=self.start_annotation)
        self.start_btn.pack(pady=5)

        # Settings + Mel inputs
        settings = ttk.Frame(self.root); settings.pack(pady=6)
        ttk.Label(settings, text="Output Sample Rate (Hz):").pack(side=tk.LEFT)
        self.output_sr_entry = ttk.Entry(settings, width=8); self.output_sr_entry.insert(0, "44100")
        self.output_sr_entry.pack(side=tk.LEFT, padx=10)

        ttk.Label(settings, text="Segment Duration (s):").pack(side=tk.LEFT)
        self.segment_duration_entry = ttk.Entry(settings, width=5); self.segment_duration_entry.insert(0, "5")
        self.segment_duration_entry.pack(side=tk.LEFT, padx=(0, 15))

        ttk.Label(settings, text="Mel min (Hz):").pack(side=tk.LEFT)
        self.mel_min_entry = ttk.Entry(settings, width=7); self.mel_min_entry.insert(0, "0")
        self.mel_min_entry.pack(side=tk.LEFT, padx=(5, 10))

        ttk.Label(settings, text="Mel max (Hz):").pack(side=tk.LEFT)
        self.mel_max_entry = ttk.Entry(settings, width=7); self.mel_max_entry.insert(0, "")
        self.mel_max_entry.pack(side=tk.LEFT, padx=(5, 10))

        self.apply_mel_btn = ttk.Button(settings, text="Apply Mel Range", bootstyle=PRIMARY, command=self.apply_mel_bounds)
        self.apply_mel_btn.pack(side=tk.LEFT, padx=(0,10))

        # Contrast row
        contrast = ttk.Frame(self.root); contrast.pack(pady=4)
        ttk.Label(contrast, text="View Window (s)").pack(side=tk.LEFT)
        self.window_entry = ttk.Entry(contrast, width=6)
        self.window_entry.insert(0, str(int(self.view_window_s)))
        self.window_entry.pack(side=tk.LEFT, padx=6)
        ttk.Button(contrast, text="Apply", command=self.apply_window).pack(side=tk.LEFT, padx=(0,15))

        ttk.Label(contrast, text="dB Floor").pack(side=tk.LEFT)
        self.db_floor_scale = ttk.Scale(contrast, from_=-120, to=-10, value=self.db_floor, length=180, command=self._on_contrast_change)
        self.db_floor_scale.pack(side=tk.LEFT, padx=6)

        ttk.Label(contrast, text="Range").pack(side=tk.LEFT)
        self.db_range_scale = ttk.Scale(contrast, from_=20, to=120, value=self.db_range, length=180, command=self._on_contrast_change)
        self.db_range_scale.pack(side=tk.LEFT, padx=6)

        self.follow_chk = ttk.Checkbutton(contrast, text="Follow playhead", variable=self.follow_playhead, bootstyle=INFO)
        self.follow_chk.pack(side=tk.LEFT, padx=12)

        # Status labels
        self.status_label = ttk.Label(self.root, text=""); self.status_label.pack()
        self.time_label   = ttk.Label(self.root, text=""); self.time_label.pack()
        self.sr_label     = ttk.Label(self.root, text=""); self.sr_label.pack()

        # ===== MAIN AREA (grid) =====
        main = ttk.Frame(self.root); main.pack(fill=tk.BOTH, expand=True, padx=8, pady=6)
        main.grid_rowconfigure(0, weight=1)
        main.grid_columnconfigure(0, weight=1)
        main.grid_columnconfigure(1, weight=0, minsize=360)

        # Left: spectrogram container
        self.canvas_frame = ttk.Frame(main)
        self.canvas_frame.grid(row=0, column=0, sticky="nsew")

        # Plot holder
        self.plot_holder = ttk.Frame(self.canvas_frame, width=980, height=560)
        self.plot_holder.pack(fill=tk.BOTH, expand=True)
        self.plot_holder.pack_propagate(False)
        self.plot_holder.bind("<Configure>", self._on_holder_resize)

        # Time scrollbar
        sb_frame = ttk.Frame(self.canvas_frame)
        sb_frame.pack(fill=tk.X)
        ttk.Label(sb_frame, text="Scroll (s)").pack(side=tk.LEFT)
        self.time_scroll = ttk.Scale(sb_frame, from_=0.0, to=1.0, value=0.0, length=800, command=self._on_scroll)
        self.time_scroll.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=8)

        # Right: sidebar
        self.label_frame = ttk.Frame(main)
        self.label_frame.grid(row=0, column=1, sticky="ns", padx=(10, 0))
        self.label_frame.grid_propagate(False)
        self.label_frame.configure(width=360)

        # Label buttons
        self.labels_button_frame = ttk.Frame(self.label_frame)
        self.labels_button_frame.pack(pady=(10, 5), fill=tk.X)
        self.refresh_label_buttons()

        ttk.Label(self.label_frame, text="Manage Labels").pack(pady=(20, 5))
        self.add_label_btn = ttk.Button(self.label_frame, text="Add Label", bootstyle=SUCCESS, command=self.add_label, state=tk.DISABLED)
        self.remove_label_btn = ttk.Button(self.label_frame, text="Remove Label", bootstyle=DANGER, command=self.remove_label, state=tk.DISABLED)
        self.add_label_btn.pack(pady=2, fill=tk.X)
        self.remove_label_btn.pack(pady=2, fill=tk.X)

        ttk.Frame(self.label_frame).pack(fill=tk.BOTH, expand=True)

        # Controls
        self.controls_frame = ttk.Frame(self.label_frame)
        self.controls_frame.pack(side=tk.BOTTOM, fill=tk.X, pady=(6, 10))
        self.prev_btn = ttk.Button(self.controls_frame, text="Previous", bootstyle=SECONDARY, width=10,
                                   command=self.previous_audio, state=tk.DISABLED)
        self.play_btn = ttk.Button(self.controls_frame, text="Play",     bootstyle=PRIMARY,   width=9,
                                   command=self.play_audio,    state=tk.DISABLED)
        self.stop_btn = ttk.Button(self.controls_frame, text="Stop",     bootstyle=SECONDARY, width=9,
                                   command=self.stop_playback, state=tk.DISABLED)
        self.next_btn = ttk.Button(self.controls_frame, text="Next",     bootstyle=SECONDARY, width=10,
                                   command=self.next_audio,    state=tk.DISABLED)
        for b in (self.prev_btn, self.play_btn, self.stop_btn, self.next_btn):
            b.pack(side=tk.LEFT, padx=4)

    # ====== size tracking ======
    def _on_holder_resize(self, event):
        self._holder_w = max(400, event.width)
        self._holder_h = max(300, event.height)
        if self._resize_after_id:
            self.root.after_cancel(self._resize_after_id)
        self._resize_after_id = self.root.after(60, self._redraw_if_ready)

    def _redraw_if_ready(self):
        if self.audio_data is not None:
            self.draw_or_update_plot()

    # ---------- Labels list ----------
    def refresh_label_buttons(self):
        for w in self.labels_button_frame.winfo_children():
            w.destroy()
        self.label_buttons = []
        for label in self.label_options:
            btn = ttk.Button(self.labels_button_frame, text=label, width=22,
                             command=lambda l=label: self.label_segment(l),
                             bootstyle=SECONDARY, state=tk.NORMAL)
            btn.pack(pady=2, fill=tk.X)
            self.label_buttons.append(btn)

    def add_label(self):
        new_label = simpledialog.askstring("Add Label", "Enter new label name (default: NewLabel):",
                                           parent=self.root, initialvalue="NewLabel")
        if new_label and new_label not in self.label_options:
            self.label_options.append(new_label)
            self.refresh_label_buttons()

    def remove_label(self):
        remove_label = simpledialog.askstring("Remove Label", "Enter label name to remove:", parent=self.root)
        if remove_label and remove_label in self.label_options:
            self.label_options.remove(remove_label)
            self.refresh_label_buttons()

    # ---------- Playback ----------
    def stop_playback(self):
        self.stop_event.set()
        sd.stop()
        # leave playhead drawn at last position

    # ---------- CSV OPTIONAL ----------
    def load_csv(self):
        path = filedialog.askopenfilename(filetypes=[("CSV files", "*.csv")])
        if not path: return
        try:
            self.df = pd.read_csv(path, dtype=str)
        except EmptyDataError:
            self.df = pd.DataFrame()

        if self.df.empty:
            messagebox.showinfo("Info", "CSV is empty. All audio files in the selected folder will be used.")
            self.filename_column = None
            self.full_file_list = []
        else:
            default_col = self.df.columns[0]
            self.filename_column = simpledialog.askstring("CSV Column", "Enter the column name for filenames:",
                                                          initialvalue=default_col, parent=self.root)
            if not self.filename_column or self.filename_column not in self.df.columns:
                messagebox.showerror("Column Error", "Valid filename column not selected.")
                self.filename_column = None
                self.full_file_list = []
            else:
                self.full_file_list = self.df[self.filename_column].astype(str).tolist()

        self.csv_loaded = True
        self.status_label.config(text=f"CSV loaded. {len(self.full_file_list)} files listed (or will be auto-loaded).")

    def set_audio_folder(self):
        self.audio_folder = filedialog.askdirectory()
        if not self.audio_folder: return
        self.audio_folder_set = True

        if (self.df is None) or (self.filename_column is None) or (self.df is not None and self.df.empty):
            self.full_file_list = [f for f in os.listdir(self.audio_folder) if f.lower().endswith(AUDIO_EXTS)]
        else:
            names_in_csv = {os.path.basename(x).lower() for x in self.df[self.filename_column].astype(str).tolist()}
            self.full_file_list = [f for f in os.listdir(self.audio_folder)
                                   if f.lower().endswith(AUDIO_EXTS) and f.lower() in names_in_csv]

        self.output_btn.config(state=tk.NORMAL)
        self.status_label.config(text=f"Audio folder selected. {len(self.full_file_list)} files ready.")

    def set_output_folder(self):
        self.output_folder = filedialog.askdirectory()
        if self.output_folder:
            self.output_folder_set = True
            self.annotation_btn.config(state=tk.NORMAL)

    def set_annotation_csv(self):
        self.annotation_csv = filedialog.asksaveasfilename(defaultextension=".csv")
        if self.annotation_csv:
            self.annotation_path_set = True
            self.check_existing_annotations()
            self.start_btn.config(state=tk.NORMAL)
            self.status_label.config(text=f"Annotation CSV set. {len(self.file_list)} files remaining to annotate.")

    def check_existing_annotations(self):
        if os.path.exists(self.annotation_csv):
            existing_df = pd.read_csv(self.annotation_csv)
            if 'original_filename' in existing_df.columns:
                self.annotated_files = set(existing_df['original_filename'].astype(str))
        self.file_list = [f for f in self.full_file_list if f not in self.annotated_files]

    # ---------- Flow ----------
    def start_annotation(self):
        if not self.file_list:
            messagebox.showwarning("No Files", "No audio files available to annotate.")
            return

        self.output_sample_rate = int(self.output_sr_entry.get())
        self.segment_duration = int(self.segment_duration_entry.get())
        self.time_label.config(text="Click to select a point in the audio")
        self.current_index = 0
        self.load_audio_file()

        # enable controls
        self.play_btn.config(state=tk.NORMAL)
        self.stop_btn.config(state=tk.NORMAL)
        self.prev_btn.config(state=tk.DISABLED)
        self.next_btn.config(state=tk.NORMAL if self.current_index < len(self.file_list) - 1 else tk.DISABLED)

        self.add_label_btn.config(state=tk.NORMAL)
        self.remove_label_btn.config(state=tk.NORMAL)
        for btn in self.label_buttons:
            btn.config(state=tk.NORMAL)

    def _prepare_mel(self):
        """Compute and cache full log-mel for current file (mono)."""
        y = self.full_audio_data
        if y.ndim > 1:
            y = np.mean(y, axis=1)

        nyq = self.sample_rate / 2
        fmin = max(0.0, float(self.mel_fmin) if self.mel_fmin is not None else 0.0)
        fmax = float(self.mel_fmax) if self.mel_fmax is not None else nyq
        fmax = min(fmax, nyq)
        if fmax <= fmin + 200:
            fmax = min(fmin + 200.0, nyq)
        self.fmin_used = fmin
        self.fmax_used = fmax

        # choose FFT params for speed if very high sr
        if self.sample_rate >= 96000:
            self.n_fft = 4096
            self.hop_length = 512
        else:
            self.n_fft = 2048
            self.hop_length = 256

        S = librosa.feature.melspectrogram(
            y=y, sr=self.sample_rate,
            n_fft=self.n_fft, hop_length=self.hop_length,
            n_mels=self.n_mels, fmin=fmin, fmax=fmax, power=2.0
        )
        S_dB = librosa.power_to_db(S + 1e-12, ref=np.max)
        self.S_dB = S_dB.astype(np.float32)

        # Update scrollbar max based on total duration
        total_dur = len(self.full_audio_data) / float(self.sample_rate)
        max_start = max(0.0, total_dur - self.view_window_s)
        self.time_scroll.configure(from_=0.0, to=max_start)
        self.time_scroll.set(0.0)
        self.view_start_s = 0.0

    def load_audio_file(self):
        self.stop_playback()
        while self.current_index < len(self.file_list):
            self.original_filename = str(self.file_list[self.current_index])
            if self.df is not None and self.filename_column:
                rows = self.df[self.df[self.filename_column] == self.original_filename]
                self.current_metadata = rows.iloc[0] if not rows.empty else pd.Series({"original_filename": self.original_filename})
            else:
                self.current_metadata = pd.Series({"original_filename": self.original_filename})

            self.current_filename = os.path.join(self.audio_folder, self.original_filename)
            if not os.path.exists(self.current_filename):
                self.current_index += 1
                continue

            self.full_audio_data, self.sample_rate = sf.read(self.current_filename, dtype='float32')
            # limit display audio for playback snippets only (full used for mel)
            self.audio_data = self.full_audio_data
            self.sr_label.config(text=f"Sample rate: {self.sample_rate} Hz")

            if self.mel_max_entry.get().strip() == "":
                self.mel_max_entry.delete(0, tk.END)
                self.mel_max_entry.insert(0, str(int(self.sample_rate // 2)))
                self.mel_fmax = None

            # compute mel cache then draw
            self._prepare_mel()
            self.root.after_idle(self.draw_or_update_plot)
            return

        messagebox.showinfo("Done", "You’ve reached the end of the list.")

    # ---------- Mel bounds ----------
    def apply_mel_bounds(self):
        try:
            fmin_txt = self.mel_min_entry.get().strip()
            fmax_txt = self.mel_max_entry.get().strip()
            fmin = float(fmin_txt) if fmin_txt != "" else 0.0
            fmax = float(fmax_txt) if fmax_txt != "" else None
        except ValueError:
            messagebox.showerror("Invalid Input", "Please enter numeric values for Mel min/max (Hz).")
            return
        if fmin < 0:
            messagebox.showerror("Invalid Range", "Mel min must be ≥ 0."); return
        self.mel_fmin = fmin
        self.mel_fmax = fmax
        if self.audio_data is not None:
            self._prepare_mel()
            self.draw_or_update_plot()

    def apply_window(self):
        try:
            w = float(self.window_entry.get())
            if w <= 0: raise ValueError
        except Exception:
            messagebox.showerror("Invalid window", "Enter a positive number of seconds.")
            return
        self.view_window_s = w
        if self.audio_data is not None:
            total_dur = len(self.full_audio_data) / float(self.sample_rate)
            max_start = max(0.0, total_dur - self.view_window_s)
            self.time_scroll.configure(from_=0.0, to=max_start)
            if self.view_start_s > max_start:
                self.view_start_s = max_start
                self.time_scroll.set(self.view_start_s)
            self.draw_or_update_plot()

    def _on_contrast_change(self, _val=None):
        self.db_floor = float(self.db_floor_scale.get())
        self.db_range = float(self.db_range_scale.get())
        if self._img is not None:
            vmin = self.db_floor
            vmax = self.db_floor + self.db_range
            if vmax <= vmin + 1:
                vmax = vmin + 1
            self._img.set_clim(vmin=vmin, vmax=vmax)
            self._canvas.draw_idle()

    # ---------- Plotting with reuse ----------
    def _slice_current_window(self):
        if self.S_dB is None:
            return None, 0.0, 0.0
        start_col = int(self.view_start_s * self.sample_rate / self.hop_length)
        end_col = start_col + int(self.view_window_s * self.sample_rate / self.hop_length)
        start_col = max(0, start_col)
        end_col = min(self.S_dB.shape[1], end_col)
        if end_col <= start_col:
            end_col = min(self.S_dB.shape[1], start_col + 1)
        window = self.S_dB[:, start_col:end_col]
        x0 = self.view_start_s
        x1 = self.view_start_s + (end_col - start_col) * (self.hop_length / float(self.sample_rate))
        return window, x0, x1

    def draw_or_update_plot(self):
        if self.S_dB is None:
            return
        # slice window
        window, x0, x1 = self._slice_current_window()
        if window is None:
            return

        # Setup/reuse figure
        create_new = self._fig is None
        if create_new:
            w = max(600, self.plot_holder.winfo_width() or self._holder_w)
            h = max(360, self.plot_holder.winfo_height() or self._holder_h)
            self._fig = Figure(figsize=(w / self._dpi, h / self._dpi), dpi=self._dpi)
            self._ax = self._fig.add_subplot(111)
            self._img = self._ax.imshow(window, origin='lower', aspect='auto',
                                        extent=[x0, x1, self.fmin_used, self.fmax_used])
            self._ax.set_ylabel('Frequency (Hz)')
            self._ax.set_xlabel('Time (s)')
            self._ax.set_title(self.original_filename)
            # playhead line
            self._playline = self._ax.axvline(x=x0, linestyle='--', linewidth=1.5)

            self._canvas = FigureCanvasTkAgg(self._fig, master=self.plot_holder)
            self._canvas.draw()
            self._canvas_widget = self._canvas.get_tk_widget()
            self._canvas_widget.place(x=0, y=0, relwidth=1.0, relheight=1.0)
            self._canvas.mpl_connect("button_press_event", self.on_click)
        else:
            # update existing artists
            self._img.set_data(window)
            self._img.set_extent([x0, x1, self.fmin_used, self.fmax_used])
            # keep playline within new extent
            if self._playline is not None:
                x = self._playline.get_xdata()[0]
                if x < x0 or x > x1:
                    # if following, keep playhead visible at left
                    pass
            self._canvas.draw_idle()

        # apply current contrast
        self._on_contrast_change()

    # ---------- Interactions ----------
    def _on_scroll(self, _val):
        self.view_start_s = float(self.time_scroll.get())
        self.draw_or_update_plot()

    def on_click(self, event):
        self.stop_playback()
        if event.xdata is not None:
            self.clicked_time = max(0.0, float(event.xdata))
            self.time_label.config(text=f"Selected time: {round(self.clicked_time, 3)} s")
            self.center_on_time(self.clicked_time)
            self.play_audio()

    def center_on_time(self, t):
        half = self.view_window_s / 2.0
        new_start = max(0.0, t - half)
        total_dur = len(self.full_audio_data) / float(self.sample_rate)
        new_start = min(new_start, max(0.0, total_dur - self.view_window_s))
        self.view_start_s = new_start
        self.time_scroll.set(self.view_start_s)
        self.draw_or_update_plot()

    def _update_playhead(self, t):
        if self._playline is None:
            return
        try:
            self._playline.set_xdata([t, t])
            if self.follow_playhead.get():
                # auto-scroll if needed
                if t > self.view_start_s + self.view_window_s * 0.9:
                    self.view_start_s = t - self.view_window_s * 0.5
                    total_dur = len(self.full_audio_data) / float(self.sample_rate)
                    self.view_start_s = max(0.0, min(self.view_start_s, max(0.0, total_dur - self.view_window_s)))
                    self.time_scroll.set(self.view_start_s)
                    self.draw_or_update_plot()
            self._canvas.draw_idle()
        except Exception:
            pass

    def play_audio(self):
        def _play():
            try:
                self.stop_event.clear()
                start = int((self.clicked_time or 0.0) * self.sample_rate)
                end   = start + int(self.sample_rate * self.segment_duration)
                clip = self.full_audio_data[start:end]

                # start playhead
                t0 = time.time()
                self._update_playhead((self.clicked_time or 0.0))

                sd.play(clip, self.sample_rate)

                # UI playhead loop
                while not self.stop_event.is_set():
                    elapsed = time.time() - t0
                    t = (self.clicked_time or 0.0) + elapsed
                    self._update_playhead(t)
                    if elapsed >= self.segment_duration:
                        break
                    time.sleep(0.02)

                sd.wait()
            except Exception as e:
                messagebox.showerror("Playback Error", str(e))
        self.stop_playback()
        self.play_thread = threading.Thread(target=_play, daemon=True)
        self.play_thread.start()

    def label_segment(self, label):
        if not self.output_folder or not self.annotation_csv:
            messagebox.showerror("Error", "Output folder and CSV path must be set."); return

        base = os.path.splitext(self.original_filename)[0]
        start_sec = round(self.clicked_time or 0.0, 2)
        start = int(start_sec * self.sample_rate)
        end = start + int(self.sample_rate * self.segment_duration)
        clip = self.full_audio_data[start:end]

        # Cap resample at original SR
        try:
            desired_sr = int(self.output_sr_entry.get())
        except Exception:
            desired_sr = self.output_sample_rate
        # ensure we never upsample above original
        target_sr = min(desired_sr, int(self.sample_rate))
        if target_sr != int(self.sample_rate):
            # resample (preserve channels)
            if clip.ndim == 1:
                clip_rs = librosa.resample(clip, orig_sr=self.sample_rate, target_sr=target_sr)
            else:
                clip_rs = librosa.resample(clip.T, orig_sr=self.sample_rate, target_sr=target_sr).T
        else:
            clip_rs = clip

        seg_name = f"{base}_segment_{start_sec}_{label}.wav"
        sf.write(os.path.join(self.output_folder, seg_name), clip_rs, target_sr)

        row = self.current_metadata.to_dict() if isinstance(self.current_metadata, pd.Series) else {}
        row.update({"filename": base + ".wav", "original_filename": self.original_filename,
                    "Second": start_sec, "Label": label, "saved_clip": seg_name})
        new = pd.DataFrame([row])
        if not os.path.exists(self.annotation_csv): new.to_csv(self.annotation_csv, index=False)
        else: new.to_csv(self.annotation_csv, mode='a', index=False, header=False)

        self.clicked_time = None
        self.time_label.config(text="Click to select a point in the audio")

    # ---------- Nav ----------
    def previous_audio(self):
        self.stop_playback()
        if self.current_index > 0:
            self.current_index -= 1
            self.load_audio_file()
        self.next_btn.config(state=tk.NORMAL)
        if self.current_index == 0:
            self.prev_btn.config(state=tk.DISABLED)

    def next_audio(self):
        self.stop_playback()
        if self.current_index < len(self.file_list) - 1:
            self.current_index += 1
            self.load_audio_file()
        self.prev_btn.config(state=tk.NORMAL)
        if self.current_index >= len(self.file_list) - 1:
            self.next_btn.config(state=tk.DISABLED)


if __name__ == '__main__':
    app = ttk.Window(themename="flatly")
    AudioAnnotator(app)
    app.mainloop()
