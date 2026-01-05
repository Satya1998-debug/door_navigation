import base64
import requests



def estimate_door_state_ollama_api(rgb_img, api_url):
    OLLAMA_URL = "http://localhost:11434/v1/door-state"

    # Read and encode the image
    encoded_image = base64.b64encode(rgb_img).decode('utf-8')

    # Prepare the payload for the API request
    payload = {
        "image": encoded_image
    }

    # Send the request to the API
    response = requests.post(api_url, json=payload)

    # Check if the request was successful
    if response.status_code == 200:
        result = response.json()
        return result.get("door_state", "unknown")
    else:
        raise Exception(f"API request failed with status code {response.status_code}: {response.text}")