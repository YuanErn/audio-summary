import openai
import json

# API Key from OpenAI
openai.api_key = "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"

# Settings for the model
response = openai.ChatCompletion.create(
    model="gpt-3.5-turbo", 
    messages=[{"role":"user", "content": "Say that this is the test going well."}]
)

# Print the generated text
preTransform = str(response.choices[0])
postTransform = json.loads(preTransform)
answer = postTransform["message"]["content"]

print (answer)