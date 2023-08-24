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
whisperAudio = whisper.load_audio("Trans1.mp4")# Path to file here

# This moves the audio to the same device as the model (cuda if enabled)
mel = whisper.log_mel_spectrogram(whisperAudio).to(whisperModel.device)

whisperResult = str(whisperModel.transcribe(whisperAudio, language = "English"))# Specify language when possible (Boosts Accuracy of transcription)

# Context for the data. This will be passed on to ChatGPT
audioTopic = "a meeting about creating Nprinting automated reporting for a current QlikSense dashboard" # What was the audio about?
partiesInvolved = "developer and IT from client side" # To whom did the voices in the audio belong to?
actionToDo = "Take note of the action items in detail. Include things that the client's user has requested" # What do u want to be done with this information
prompt = "This is " + audioTopic + " between " + partiesInvolved + "." + actionToDo + "The transcribed audio is in the triple backticks"+"'''{whisperResult}'''"
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

