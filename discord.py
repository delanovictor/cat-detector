import requests
import time
import os
from dotenv import load_dotenv

def send_message(text, image_path):
    TOKEN = os.getenv('DISCORD_TOKEN')
    CHANNEL_ID = os.getenv('DISCORD_CHANNEL_ID')

    BASE_URL = f"https://discord.com/api/v9"
    SEND_URL = BASE_URL + "/channels/{id}/messages"

    headers = {
        "Authorization": f"Bot {TOKEN}",
        "User-Agent": f"DiscordBot"
    }

    ts = time.time()
    
    files = {
        "file" : (f"cat-{ts}.jpg", open(image_path, 'rb'))
    }

    body = {
        "content": text,
    }

    r1 = requests.post(SEND_URL.format(id=CHANNEL_ID), headers=headers, json=body )

    r2 = requests.post(SEND_URL.format(id=CHANNEL_ID), headers=headers, json=body, files=files)

