import os
from groq import Groq
os.environ["GROQ_API_KEY"] = "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"
client = Groq()
response = client.chat.completions.create(
    messages = [
        {"role": "user", "content": "Give me the names of the continents of the world"}
    ],
    model = "llama-3.1-8b-instant"
)
print(response)
