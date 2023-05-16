import whisper
import json
import openai
import time

# Timing the program
startTime = time.perf_counter()

# API Key from OpenAI
secret_file = open("API.txt", "r")
openai.api_key = secret_file.readline()
secret_file.close()

whisperModel = whisper.load_model("medium")
whisperAudio = whisper.load_audio("")# Path to file here

# This moves the audio to the same device as the model (cuda if enabled)
mel = whisper.log_mel_spectrogram(whisperAudio).to(whisperModel.device)

whisperResult = str(whisperModel.transcribe(whisperAudio, language = "Chinese"))# Specify language when possible (Boosts Accuracy of transcription)

# Context for the data. This will be passed on to ChatGPT
audioTopic = "" # What was the audio about?
partiesInvolved = "" # To whom did the voices in the audio belong to?
actionToDo = "" # What do u want to be done with this information
prompt = "This is " + audioTopic + " between " + partiesInvolved + "." + actionToDo

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
print ("Completed in", time.strftime("%H:%M:%S", time.gmtime(time.perf_counter() - startTime)))

