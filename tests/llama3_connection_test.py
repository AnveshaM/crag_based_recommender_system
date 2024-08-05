import os
from groq import Groq
os.environ["GROQ_API_KEY"] = "GROQ_API_KEY"
client = Groq()
response = client.chat.completions.create(
    messages = [
        {"role": "user", "content": "Give me the names of the continents of the world"}
    ],
    model = "llama-3.1-8b-instant"
)
print(response)
