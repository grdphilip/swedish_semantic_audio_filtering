import asyncio

from elevenlabs.client import AsyncElevenLabs
from dotenv import load_dotenv
import os

load_dotenv()
api_key = os.getenv("PETTER_ELEVENLABS")

client = AsyncElevenLabs(
    api_key=api_key
)

eleven = AsyncElevenLabs(
  api_key=api_key # Defaults to ELEVENLABS_API_KEY
)

async def print_models() -> None:
    models = await eleven.voices.get_all()
    for voice in models.voices:
        print(voice.name, voice.labels, voice.voice_id)
        
        
voices = asyncio.run(print_models())
print(voices)
