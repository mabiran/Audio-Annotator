import os
import random
import tkinter as tk
from tkinter import filedialog, messagebox, simpledialog
from PIL import Image, ImageTk
import pandas as pd
import librosa
import librosa.display
import matplotlib
matplotlib.use("Agg")
from matplotlib.figure import Figure
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import sounddevice as sd
import soundfile as sf
import threading
import numpy as np
import gc
import ttkbootstrap as ttk
from ttkbootstrap.constants import *

class AudioAnnotator:
    def __init__(self, root):
        self.root = root
        self.root.title("AUT Animal Sound Annotator")
        self.root.geometry("1000x750")
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
        self.playback_thread = None
        self.stop_event = threading.Event()

        self.label_options = ["Chitter", "Screech", "Grunt", "Hiss", "General Bird Sound"]
        self.segment_duration = 5
        self.output_sample_rate = 44100

        self.setup_gui()

    def setup_gui(self):
        logo_frame = ttk.Frame(self.root)
        logo_frame.pack(pady=5)
        try:
            logo_image = Image.open("aut_logo.png").resize((120, 60))
            logo_photo = ImageTk.PhotoImage(logo_image)
            logo_label = ttk.Label(logo_frame, image=logo_photo)
            logo_label.image = logo_photo
            logo_label.pack()
        except Exception as e:
            print("AUT logo not found or failed to load.", e)

        self.btn_frame = ttk.Frame(self.root)
        self.btn_frame.pack(pady=10)

        self.csv_btn = ttk.Button(self.btn_frame, text="1. Select CSV", bootstyle=SECONDARY, command=self.load_csv)
        self.csv_btn.pack(side=tk.LEFT, padx=5)

        self.audio_btn = ttk.Button(self.btn_frame, text="2. Select Audio Folder", bootstyle=INFO, state=tk.DISABLED, command=self.set_audio_folder)
        self.audio_btn.pack(side=tk.LEFT, padx=5)

        self.output_btn = ttk.Button(self.btn_frame, text="3. Select Output Folder", bootstyle=SUCCESS, state=tk.DISABLED, command=self.set_output_folder)
        self.output_btn.pack(side=tk.LEFT, padx=5)

        self.annotation_btn = ttk.Button(self.btn_frame, text="4. Set Annotation CSV", bootstyle=DANGER, state=tk.DISABLED, command=self.set_annotation_csv)
        self.annotation_btn.pack(side=tk.LEFT, padx=5)

        self.start_btn = ttk.Button(self.root, text="Start Annotation", bootstyle=WARNING, state=tk.DISABLED, command=self.start_annotation)
        self.start_btn.pack(pady=5)

        self.settings_frame = ttk.Frame(self.root)
        self.settings_frame.pack(pady=5)

        ttk.Label(self.settings_frame, text="Output Sample Rate (Hz):").pack(side=tk.LEFT)
        self.output_sr_entry = ttk.Entry(self.settings_frame, width=6)
        self.output_sr_entry.insert(0, "44100")
        self.output_sr_entry.pack(side=tk.LEFT, padx=10)

        ttk.Label(self.settings_frame, text="Segment Duration (s):").pack(side=tk.LEFT)
        self.segment_duration_entry = ttk.Entry(self.settings_frame, width=4)
        self.segment_duration_entry.insert(0, "5")
        self.segment_duration_entry.pack(side=tk.LEFT)

        self.status_label = ttk.Label(self.root, text="")
        self.status_label.pack()

        self.time_label = ttk.Label(self.root, text="")
        self.time_label.pack()

        self.sr_label = ttk.Label(self.root, text="")
        self.sr_label.pack()

        self.main_frame = ttk.Frame(self.root)
        self.main_frame.pack(fill=tk.BOTH, expand=True)

        self.canvas_frame = ttk.Frame(self.main_frame)
        self.canvas_frame.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)

        self.label_frame = ttk.Frame(self.main_frame)
        self.label_frame.pack(side=tk.RIGHT, fill=tk.Y, padx=10)

        # This frame will contain just the label buttons and will be refreshed
        self.labels_button_frame = ttk.Frame(self.label_frame)
        self.labels_button_frame.pack(pady=(10, 5))

        # These are static (not refreshed)



        self.refresh_label_buttons()
        control_frame = ttk.Frame(self.root)
        control_frame.pack(pady=5)

        self.prev_btn = ttk.Button(control_frame, text="Previous", bootstyle=SECONDARY, command=self.previous_audio,
                                   state=tk.DISABLED)
        self.play_btn = ttk.Button(control_frame, text="Play", bootstyle=PRIMARY, command=self.play_audio,
                                   state=tk.DISABLED)
        self.stop_btn = ttk.Button(control_frame, text="Stop", bootstyle=SECONDARY, command=self.stop_playback,
                                   state=tk.DISABLED)
        self.next_btn = ttk.Button(control_frame, text="Next", bootstyle=SECONDARY, command=self.next_audio,
                                   state=tk.DISABLED)

        self.prev_btn.pack(side=tk.LEFT, padx=5)
        self.play_btn.pack(side=tk.LEFT, padx=5)
        self.stop_btn.pack(side=tk.LEFT, padx=5)
        self.next_btn.pack(side=tk.LEFT, padx=5)

        ttk.Label(self.label_frame, text="Manage Labels").pack(pady=(20, 5))
        self.add_label_btn = ttk.Button(self.label_frame, text="Add Label", bootstyle=SUCCESS, command=self.add_label,
                                        state=tk.DISABLED)
        self.remove_label_btn = ttk.Button(self.label_frame, text="Remove Label", bootstyle=DANGER,
                                           command=self.remove_label, state=tk.DISABLED)

        self.add_label_btn.pack(pady=2)
        self.remove_label_btn.pack(pady=2)

    def refresh_label_buttons(self):
        for widget in self.labels_button_frame.winfo_children():
            widget.destroy()

        for label in self.label_options:
            btn = ttk.Button(self.labels_button_frame, text=label, width=20,
                             command=lambda l=label: self.label_segment(l),
                             bootstyle=SECONDARY, state=tk.DISABLED)
            btn.pack(pady=2)
            self.label_buttons.append(btn)

    def add_label(self):
        new_label = simpledialog.askstring("Add Label", "Enter new label name (default: NewLabel):", parent=self.root, initialvalue="NewLabel")
        if new_label and new_label not in self.label_options:
            self.label_options.append(new_label)
            self.refresh_label_buttons()

    def remove_label(self):
        remove_label = simpledialog.askstring("Remove Label", "Enter label name to remove:", parent=self.root)
        if remove_label and remove_label in self.label_options:
            self.label_options.remove(remove_label)
            self.refresh_label_buttons()

    def stop_playback(self):
        self.stop_event.set()
        sd.stop()
        gc.collect()

    def load_csv(self):
        path = filedialog.askopenfilename(filetypes=[("CSV files", "*.csv")])
        if path:
            self.df = pd.read_csv(path, dtype=str)
            default_column = self.df.columns[0] if not self.df.empty else ""
            self.filename_column = simpledialog.askstring("CSV Column", "Enter the column name for filenames:", initialvalue=default_column, parent=self.root)
            if self.filename_column not in self.df.columns:
                messagebox.showerror("Column Error", f"Column '{self.filename_column}' not found in CSV.")
                return
            self.full_file_list = self.df[self.filename_column].tolist()
            self.csv_loaded = True
            self.audio_btn.config(state=tk.NORMAL)
            self.status_label.config(text=f"CSV loaded. {len(self.full_file_list)} files listed.")

    def set_audio_folder(self):
        self.audio_folder = filedialog.askdirectory()
        if self.audio_folder:
            self.audio_folder_set = True
            self.output_btn.config(state=tk.NORMAL)
            self.status_label.config(text=f"Audio folder selected.")

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
            self.annotated_files = set(existing_df['original_filename'].astype(str))
        self.file_list = [f for f in self.full_file_list if f not in self.annotated_files]

    def start_annotation(self):
        self.output_sample_rate = int(self.output_sr_entry.get())
        self.segment_duration = int(self.segment_duration_entry.get())
        self.time_label.config(text="Click to select a point in the audio")
        self.current_index = 0
        self.load_audio_file()
        self.play_btn.config(state=tk.NORMAL)
        self.stop_btn.config(state=tk.NORMAL)
        self.prev_btn.config(state=tk.NORMAL if self.current_index > 0 else tk.DISABLED)
        self.next_btn.config(state=tk.NORMAL if self.current_index < len(self.file_list) - 1 else tk.DISABLED)

        self.add_label_btn.config(state=tk.NORMAL)
        self.remove_label_btn.config(state=tk.NORMAL)

        for btn in self.label_buttons:
            btn.config(state=tk.NORMAL)

    def load_audio_file(self):
        self.stop_playback()
        while self.current_index < len(self.file_list):
            self.original_filename = str(self.file_list[self.current_index])
            self.current_metadata = self.df[self.df[self.filename_column] == self.original_filename].iloc[0]
            self.current_filename = os.path.join(self.audio_folder, self.original_filename)

            if not os.path.exists(self.current_filename):
                self.current_index += 1
                continue

            self.full_audio_data, self.sample_rate = sf.read(self.current_filename, dtype='float32')
            display_audio = self.full_audio_data[:min(len(self.full_audio_data), self.sample_rate * 60)]
            self.audio_data = display_audio
            self.sr_label.config(text=f"Sample rate: {self.sample_rate} Hz")
            self.draw_spectrogram()
            return

        messagebox.showinfo("Done", "You’ve reached the end of the list.")

    def draw_spectrogram(self):
        for widget in self.canvas_frame.winfo_children():
            widget.destroy()

        fig = Figure(figsize=(8, 4))
        ax = fig.add_subplot(111)
        y_data = self.audio_data[:, 0] if self.audio_data.ndim > 1 else self.audio_data
        fmax = self.sample_rate / 2
        n_mels = min(128, max(10, self.sample_rate // 400))
        S = librosa.feature.melspectrogram(y=y_data, sr=self.sample_rate, n_fft=2048, hop_length=512, n_mels=n_mels, fmax=fmax)
        S_dB = librosa.power_to_db(S, ref=np.max)
        img = librosa.display.specshow(S_dB, sr=self.sample_rate, ax=ax, x_axis='time', y_axis='mel', fmax=fmax)
        fig.colorbar(img, ax=ax, format='%+2.0f dB')
        ax.set_title(self.original_filename)

        canvas = FigureCanvasTkAgg(fig, master=self.canvas_frame)
        canvas.draw()
        canvas.get_tk_widget().pack()
        canvas.mpl_connect("button_press_event", self.on_click)
        gc.collect()

    def on_click(self, event):
        self.stop_playback()
        if event.xdata is not None:
            self.clicked_time = event.xdata
            self.time_label.config(text=f"Selected time: {round(self.clicked_time * 1000)} ms")
            self.play_audio()

    def play_audio(self):
        def _play():
            try:
                self.stop_event.clear()
                start_sample = int(self.clicked_time * self.sample_rate) if self.clicked_time else 0
                end_sample = start_sample + int(self.sample_rate * self.segment_duration)
                clip = self.full_audio_data[start_sample:end_sample]
                sd.play(clip, self.sample_rate)
                sd.wait()
                del clip
                gc.collect()
            except Exception as e:
                messagebox.showerror("Playback Error", str(e))

        self.stop_playback()
        threading.Thread(target=_play).start()

    def label_segment(self, label):
        if not self.output_folder or not self.annotation_csv:
            messagebox.showerror("Error", "Output folder and CSV path must be set.")
            return

        base_name = os.path.splitext(self.original_filename)[0]
        start_sec = round(self.clicked_time, 2) if self.clicked_time else 0
        start_sample = int(start_sec * self.sample_rate)
        end_sample = start_sample + int(self.sample_rate * self.segment_duration)
        clip = self.full_audio_data[start_sample:end_sample]

        if self.sample_rate != self.output_sample_rate:
            clip = librosa.resample(clip.T, orig_sr=self.sample_rate, target_sr=self.output_sample_rate).T

        segment_filename = f"{base_name}_segment_{start_sec}_{label}.wav"
        segment_path = os.path.join(self.output_folder, segment_filename)
        sf.write(segment_path, clip, self.output_sample_rate)
        del clip
        gc.collect()

        row_dict = self.current_metadata.to_dict()
        row_dict.update({"filename": base_name + ".wav", "original_filename": self.original_filename, "Second": start_sec, "Label": label})
        new_row_df = pd.DataFrame([row_dict])

        if not os.path.exists(self.annotation_csv):
            new_row_df.to_csv(self.annotation_csv, index=False)
        else:
            new_row_df.to_csv(self.annotation_csv, mode='a', index=False, header=False)

        self.clicked_time = None
        self.time_label.config(text="Click to select a point in the audio")

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