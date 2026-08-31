import sys
import os
from google import genai
from dotenv import load_dotenv

load_dotenv()
api_key = os.getenv("GEMINI_API_KEY")
client = genai.Client(api_key=api_key)

#print(sys.version)
#print(sys.executable)

response = client.models.generate_content(
    model="models/gemini-3.5-flash",
    contents="What is hello in german?"
)
print(response.text)