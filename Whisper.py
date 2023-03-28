import whisper
import json
import openai
import time

# Timing the program
startTime = time.time()

# API Key from OpenAI
secret_file = open("API.txt", "r")
openai.api_key = secret_file.readline()
secret_file.close()

whisperModel = whisper.load_model("medium")
whisperAudio = whisper.load_audio("C:/Users/Yuan Ern/Desktop/audio-summary/testing.mp3")# Path to file here

# This moves the audio to the same device as the model (cuda if enabled)
mel = whisper.log_mel_spectrogram(whisperAudio).to(whisperModel.device)

whisperResult = str(whisperModel.transcribe(whisperAudio, language = "Chinese"))# Specify language when possible

# Context for the data. This will be passed on to ChatGPT
audioTopic = "a meeting session for trial balance dashboard review session" # What was the audio about?
partiesInvolved = "developer and client" # To whom did the voices in the audio belong to?
actionToDo = "summarize this meeting into meeting minutes. Point form with more details into the topics" # What do u want to be done with this information
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
print ("Completed in", time.strftime("%H:%M:%S", time.gmtime(time.time() - startTime)))

