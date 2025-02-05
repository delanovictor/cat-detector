import requests
import time
import os
from dotenv import load_dotenv

load_dotenv()
TOKEN = os.getenv('DISCORD_TOKEN')
CHANNEL_ID = os.getenv('DISCORD_CHANNEL_ID')
BASE_URL = f"https://discord.com/api/v9"
SEND_URL = BASE_URL + "/channels/{id}/messages"
headers = {
    "Authorization": f"Bot {TOKEN}",
    "User-Agent": f"DiscordBot"
}

def send_text_message(text):
    body = {
        "content": text
    }

    r1 = requests.post(SEND_URL.format(id=CHANNEL_ID), headers=headers, json=body )

    print("Mensagem Enviada!")

def send_cat_image(text, image_path):
    ts = time.time()

    files = {
        "file" : (f"cat-{ts}.jpg", open(image_path, 'rb'))
    }

    body = {
        "content": text
    }

    r1 = requests.post(SEND_URL.format(id=CHANNEL_ID), headers=headers, json=body )
    # print(r1.content)
    r2 = requests.post(SEND_URL.format(id=CHANNEL_ID), headers=headers, json=body, files=files)
    # print(r2.content)
    print('Mensagem Enviada!')
