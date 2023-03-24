import whisper
import json
import openai

# API Key from OpenAI
secret_file = open("API.txt", "r")
openai.api_key = secret_file.readline()
secret_file.close()

whisperModel = whisper.load_model("medium")
whisperAudio = whisper.load_audio("C:/Users/Yuan Ern/Desktop/audio-summary/testing.mp3")# Path to file here

# This moves the audio to the same device as the model (cuda if enabled)
mel = whisper.log_mel_spectrogram(whisperAudio).to(whisperModel.device)

whisperResult = str(whisperModel.transcribe(whisperAudio, language = "Chinese")) # Specify language when possible

# Context for the data above
meetingTopic = "a meeting session for trial balance dashboard review session" # What was the meeting about?
partiesInvolved = "developer and client" # Could be an internal meeting or one with clients
actionToDo = "summarize this meeting into meeting minutes" # What do u want to be done with this information
prompt = "This is " + meetingTopic + " between " + partiesInvolved + "." + actionToDo

# Settings for the model
response = openai.ChatCompletion.create(
    model="gpt-3.5-turbo", 
    messages=[{"role":"user", "content": prompt}]
)

# Print the generated text
preTransform = str(response.choices[0])
postTransform = json.loads(preTransform)
answer = postTransform["message"]["content"]

print (answer)