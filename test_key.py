import os
from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()

client = OpenAI(
    base_url="https://openrouter.ai/api/v1",
    api_key=os.getenv("RAG_API_KEY"),
)

response = client.chat.completions.create(
    model="z-ai/glm-5.2:free",  # model
    messages=[{"role": "user", "content": "Say hello in one word"}],
)

print(response.choices[0].message.content)


# python -m test_key   >>>run this file standalone to test if the RAG_API_KEY is working