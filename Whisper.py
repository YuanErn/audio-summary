import whisper
import json
import openai

# API Key from OpenAI
openai.api_key = "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"

whisperModel = whisper.load_model("medium")
whisperAudio = whisper.load_audio("C:/Users/Yuan Ern/Desktop/audio-summary/testing.mp3")# Path to file here

# This moves the audio to the same device as the model (cuda if enabled)
mel = whisper.log_mel_spectrogram(whisperAudio).to(whisperModel.device)

whisperResult = whisperModel.transcribe(whisperAudio, language = "Chinese")

# Settings for the model
response = openai.ChatCompletion.create(
    model="gpt-3.5-turbo", 
    messages=[{"role":"user", "content": whisperResult + "This is a meeting about a meeting sessions for the trial finance trial balance meeting. Here is the context. Summarize the meeting"}]
)

# Print the generated text
preTransform = str(response.choices[0])
postTransform = json.loads(preTransform)
answer = postTransform["message"]["content"]

print (answer)