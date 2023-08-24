import tkinter as tk
from tkinter import filedialog, OptionMenu
import whisper
import time
import threading
from collections import deque
from transformers import pipeline

MAX_HISTORY_SIZE = 5
LANGUAGES = {
    "English": "en",
    "Chinese": "zh",
    "Spanish": "es",
    "Russian": "ru",
    "Korean": "ko",
    "French": "fr",
    "Japanese": "ja",
}

MODELS = {
    "Base": "base",
    "Small": "small",
    "Medium": "medium",
    "Large": "large-v1",
}

DEVICES = {
    "CPU": "cpu",
    "GPU": "cuda",
}

def transcribe_audio():
    # Disable the button during transcription
    summarize_button.config(state=tk.DISABLED)
    transcribe_button.config(state=tk.DISABLED)
    transcribe_button.config(text="Transcribing...")

    # Update the selected language and model
    selected_language_value = selected_language.get()
    selected_model_value = MODELS[selected_model.get()]
    selected_device_value = DEVICES[selected_device.get()]

    # Start a new thread for audio transcription
    transcription_thread = threading.Thread(target=perform_transcription, args=(LANGUAGES[selected_language_value], selected_model_value, selected_device_value))
    transcription_thread.start()

def perform_transcription(language, model, device):
    global text
    global transcribe_flag
    # Start the timer
    startTime = time.perf_counter()

    # Loading the model and audio
    whisperModel = whisper.load_model(model).to(device)
    whisperAudio = whisper.load_audio(selected_file.get()) # Use the selected file for transcription

    # This moves the audio to the same device as the model (cuda if enabled)
    whisper.log_mel_spectrogram(whisperAudio).to(device)

    # Transcribe the audio
    whisperResult = whisperModel.transcribe(whisperAudio, language=language)  # Specify language when possible (Boosts Accuracy of transcription)

    # Text processing
    text = whisperResult['text']
    segments = whisperResult['segments']
    transcribed_text = ""

    for segment in segments:
        segment_start = round(segment['start'], 2)
        segment_end = round(segment['end'], 2)
        segment_text = segment['text']
        transcribed_text += f"[{segment_start} - {segment_end}]: {segment_text}\n"

    completion_time = time.strftime("%H:%M:%S", time.gmtime(time.perf_counter() - startTime))
    transcribed_text += f"Completed in {completion_time}"
    transcribe_flag = 1

    # Update the text widget with the result
    root.after(0, update_text_widget, transcribed_text)

    # Enable the button after transcription is completed
    root.after(0, enable_transcribe_button)

def update_text_widget(text):
    result_text.config(state=tk.NORMAL)  # Enable the text widget
    result_text.delete(1.0, tk.END)
    result_text.insert(tk.END, text)
    result_text.config(state=tk.DISABLED)  # Disable the text widget

    # Adjust the height of the text widget based on its content
    num_lines = text.count('\n') + 1
    min_height = 10  # Set a minimum height for the text widget
    result_text.config(height=max(num_lines, min_height))

def enable_transcribe_button():
    transcribe_button.config(state=tk.NORMAL)
    transcribe_button.config(text="Transcribe Audio")
    summarize_button.config(state=tk.NORMAL)


def browse_file():
    global transcribe_flag
    file_path = filedialog.askopenfilename()
    transcribe_flag = 0
    if file_path:
        selected_file.set(file_path)
        update_transcribe_button_state()  # Update the state of the Transcribe Audio button

        # Add the selected file to history
        history.append(file_path)
        if len(history) > MAX_HISTORY_SIZE:
            history.popleft()
        update_history_menu()

def update_history_menu():
    history_menu['menu'].delete(0, tk.END)

    for item in history:
        history_menu['menu'].add_command(label=item, command=lambda file=item: selected_file.set(file))

def export_file():
    file_path = filedialog.asksaveasfilename(defaultextension=".txt", filetypes=[("Text Files", "*.txt")])
    if file_path:
        with open(file_path, "w") as file:
            file.write(result_text.get(1.0, tk.END))

def summarize_text():
    # Disable the button during transcription
    summarize_button.config(state=tk.DISABLED)
    transcribe_button.config(state=tk.DISABLED)
    summarize_button.config(text="Summarizing...")

    # Start a new thread for audio transcription
    transcription_thread = threading.Thread(target=perform_summary)
    transcription_thread.start()

def perform_summary():
    global transcribe_flag
    global text
    startTime = time.perf_counter()
    if transcribe_flag == 0:
        language = selected_language.get()
        model = MODELS[selected_model.get()]
        device = DEVICES[selected_device.get()]
        whisperModel = whisper.load_model(model).to(device)
        whisperAudio = whisper.load_audio(selected_file.get())
        whisperResult = whisperModel.transcribe(whisperAudio, language=language)

    # Get the text from the result_text widget
        text = whisperResult['text']

    min_length_value = int(min_length_entry.get())
    max_length_value = int(max_length_entry.get())

    # Perform text summarization
    summarizer = pipeline("summarization", model="facebook/bart-large-cnn")
    summary = summarizer(text, max_length=max_length_value, min_length=min_length_value, do_sample=False)
    summarized_text = summary[0]['summary_text']

    completion_time = time.strftime("%H:%M:%S", time.gmtime(time.perf_counter() - startTime))
    summarized_text += f"'\nCompleted in {completion_time}"

    # Update the text widget with the summarized text
    root.after(0, update_text_widget, summarized_text)

    # Enable the button after summarization is completed
    root.after(0, enable_summarize_button)

def enable_summarize_button():
    summarize_button.config(state=tk.NORMAL)
    summarize_button.config(text="Summarize Text")
    transcribe_button.config(state=tk.NORMAL)

def open_preferences():
    def save_preferences():
        # Perform any actions needed to save the preferences
        selected_language_value = selected_language.get()
        selected_model_value = selected_model.get()
        selected_device_value = selected_device.get()

        preferences_window.destroy()

    preferences_window = tk.Toplevel(root)
    preferences_window.title("Preferences")
    preferences_window.geometry("300x210")

    # Create a frame for the language dropdown
    language_frame = tk.Frame(preferences_window)
    language_frame.pack(pady=(10, 5), padx=10, anchor="w")

    # Create a label for the language dropdown
    language_label = tk.Label(language_frame, text="Language:")
    language_label.pack(side=tk.LEFT)

    # Truncate the language names for the dropdown
    truncated_languages = {key: value[:15] + "..." if len(value) > 15 else value for key, value in LANGUAGES.items()}

    # Create the language dropdown menu
    language_menu = OptionMenu(language_frame, selected_language, *truncated_languages.keys())
    language_menu.config(width=30)
    language_menu.pack(side=tk.LEFT)

    # Create a frame for the model dropdown
    model_frame = tk.Frame(preferences_window)
    model_frame.pack(pady=(0, 5), padx=10, anchor="w")

    # Create a label for the model dropdown with padding
    model_label = tk.Label(model_frame, text="Model:      ")
    model_label.pack(side=tk.LEFT)

    # Create the model dropdown menu
    model_menu = OptionMenu(model_frame, selected_model, *MODELS.keys())
    model_menu.config(width=30)
    model_menu.pack(side=tk.LEFT)

    # Create a frame for the device dropdown
    device_frame = tk.Frame(preferences_window)
    device_frame.pack(pady=(0, 5), padx=10, anchor="w")

    # Create a label for the device dropdown
    device_label = tk.Label(device_frame, text="Device:     ")
    device_label.pack(side=tk.LEFT)

    # Create the device dropdown menu
    device_menu = OptionMenu(device_frame, selected_device, *DEVICES.keys())
    device_menu.config(width=30)
    device_menu.pack(side=tk.LEFT)

    # Create a frame for the min length option
    min_length_frame = tk.Frame(preferences_window)
    min_length_frame.pack(pady=(0, 5), padx=10, anchor="w")

    # Create a label for the min length option
    min_length_label = tk.Label(min_length_frame, text="Min Length:")
    min_length_label.pack(side=tk.LEFT)

    # Create an entry box for the min length option
    min_length_entry = tk.Entry(min_length_frame)
    min_length_entry.pack(side=tk.LEFT)
    min_length_entry.insert(tk.END, "80")  # Set a default value

    # Create a frame for the max length option
    max_length_frame = tk.Frame(preferences_window)
    max_length_frame.pack(pady=(0, 5), padx=10, anchor="w")

    # Create a label for the max length option
    max_length_label = tk.Label(max_length_frame, text="Max Length:")
    max_length_label.pack(side=tk.LEFT)

    # Create an entry box for the max length option
    max_length_entry = tk.Entry(max_length_frame)
    max_length_entry.pack(side=tk.LEFT)
    max_length_entry.insert(tk.END, "200")  # Set a default value

    # Create a frame for the save button
    save_frame = tk.Frame(preferences_window)
    save_frame.pack(pady=10)

    # Create a button to save the preferences
    save_button = tk.Button(save_frame, text="Save", command=save_preferences)
    save_button.pack()

def update_selected_language(*args):
    # Perform any actions needed when the selected language is updated
    pass

def update_selected_model(*args):
    # Perform any actions needed when the selected model is updated
    pass

def update_selected_device(*args):
    # Perform any actions needed when the selected device is updated
    pass

def update_max_length_entry(*args):
    # Perform any actions needed when the selected device is updated
    pass

def update_min_length_entry(*args):
    # Perform any actions needed when the selected device is updated
    pass

def update_transcribe_button_state():
    if selected_file.get():
        transcribe_button.config(state=tk.NORMAL)
        summarize_button.config(state=tk.NORMAL)
    else:
        transcribe_button.config(state=tk.DISABLED)
        summarize_button.config(state=tk.DISABLED)


# Create the GUI
root = tk.Tk()
root.title("Audio Transcription")
root.geometry("800x450")

# Create a frame for the selected file and dropdown menu
button_frame = tk.Frame(root)
button_frame.pack(padx=10, pady=(0, 5), anchor="w")

# Create a button for preferences
preferences_button = tk.Button(button_frame, text="Preferences", command=open_preferences)
preferences_button.grid(row=0, column=0, padx=(0, 5))

# Create a button to browse for audio file
browse_button = tk.Button(button_frame, text="Browse", command=browse_file)
browse_button.grid(row=0, column=2, padx=(0,5))

# Create a button to start audio transcription
transcribe_button = tk.Button(button_frame, text="Transcribe Audio", command=transcribe_audio, state=tk.DISABLED)
transcribe_button.grid(row=0, column=3, padx=(0, 5))

# Add the "Summarize" button to the button_frame
summarize_button = tk.Button(button_frame, text="Summarize Text", command=summarize_text, state=tk.DISABLED)
summarize_button.grid(row=0, column=5, padx=(0, 5))

# Create a button to export the transcribed text
export_button = tk.Button(button_frame, text="Export File", command=export_file)
export_button.grid(row=0, column=6, padx=(0, 5))

# Create a StringVar to store the selected language
selected_language = tk.StringVar()
selected_language.trace("w", update_selected_language)

# Create a StringVar to store the selected model with a default value of "Base"
selected_model = tk.StringVar()
selected_model.trace("w", update_selected_model)

# Create a StringVar to store the selected device
selected_device = tk.StringVar()
selected_device.trace("w", update_selected_device)

max_length_entry = tk.StringVar()
max_length_entry.trace("w", update_max_length_entry)

min_length_entry = tk.StringVar()
min_length_entry.trace("w", update_min_length_entry)

# Create a frame for the selected file and dropdown menu
selected_frame = tk.Frame(root)
selected_frame.pack(padx=10, pady=(0, 5), anchor="w")

# Create a label to display the selected file
selected_file_label = tk.Label(selected_frame, text="Selected File:")
selected_file_label.pack(side=tk.LEFT)

# Create a dropdown menu to show the past files
selected_file = tk.StringVar()
history = deque(maxlen=MAX_HISTORY_SIZE)
history_menu = OptionMenu(selected_frame, selected_file, selected_file.get(), *history)
history_menu.config(width=70)
history_menu.pack(side=tk.LEFT, padx=(5, 0))

# Create a text widget to display the result
result_text = tk.Text(root, wrap=tk.WORD, state=tk.DISABLED)
result_text.pack(padx=10, pady=5, fill=tk.BOTH, expand=True)

# Create a vertical scrollbar for the text widget
scrollbar = tk.Scrollbar(result_text)
scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
result_text.config(yscrollcommand=scrollbar.set)
scrollbar.config(command=result_text.yview)

# Set the default language, model, and device
selected_language.set("English")
selected_model.set("Base")
selected_device.set("GPU")
max_length_entry.set("200")
min_length_entry.set("80")

# Run the GUI
root.mainloop()