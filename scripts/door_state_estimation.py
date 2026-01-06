import base64
from os import path
from ollama import chat
import cv2

ollama_base_url = "http://localhost:11434"

def estimate_door_state_ollama_api(image_path: str) -> str:
    try:
        ok, buf = cv2.imencode('.jpg', cv2.imread(image_path))
        if not ok:
            raise RuntimeError(f"Failed to encode image at {image_path}.")
        rgb_img_bytes = buf.tobytes() 
        img_b64 = base64.b64encode(rgb_img_bytes).decode('utf-8')

        prompt = "Look at this image. Classify the door state as 'open', 'partly-open', or 'closed'. Also tell if any human is present. " \
        "Classify door_type (single/double), door_material (normal/glass)" \
        "Response structure: {'door_state': <state>, 'human_present': <yes/no>, 'door_type': <type>, 'door_material': <material>}. Only respond with the JSON structure."

        response = chat(
            model='qwen3-vl:4b',
            messages=[
                {
                    'role': 'user',
                    'content': prompt,
                    'images': [img_b64],
                }
            ],
            format="json"
        )

        
        # print(f"Ollama API response: {response}")

        res = response.message.content.strip().lower()
        if res:
            door_state = res
            print(f"Estimated door state: {door_state}")
            return door_state
        else:
            print("No valid response received from Ollama API.")
            return None
    
    except Exception as e:
        print(f"Error during estimate_door_state_ollama_api: {e}")
        return None


if __name__ == "__main__":
    path = "/home/RUS_CIP/st184744/codebase/door_navigation/scripts/data_new/latest_image_color_lab_35.jpg"
    estimate_door_state_ollama_api(path)