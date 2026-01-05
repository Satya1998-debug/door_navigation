import base64
from os import path
import requests


def estimate_door_state_ollama_api(rgb_img, api_url):
    from ollama import chat

    response = chat(
        model='gemma3',
        messages=[
            {
            'role': 'user',
            'content': 'What is in this image? Be concise.',
            'images': [path],
            }
        ],
    )


if __name__ == "__main__":
    pass