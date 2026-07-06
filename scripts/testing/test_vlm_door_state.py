"""
Simple test script for estimate_door_state_ollama_vlm function.
Usage: python test_vlm_door_state.py <image_path> [options]
"""

import argparse
import time
import cv2
import base64
from ollama import chat
import json


def test_vlm_door_state(image_path, model):
    # Load image
    print(f"Loading image: {image_path}")
    rgb_img = cv2.imread(image_path)
    
    # Call VLM estimation
    print("Calling estimate_door_state_ollama_vlm...")
    s_time = time.time()
    result = estimate_door_state_ollama_vlm(
        model,
        rgb_img,
        is_passable='False',
        door_open_percent='14',
        door_wall_angle="",
        left_right_door_angle='25',
        door_type='double'
    )
    e_time = time.time()
    print(f"Time taken: {e_time - s_time:.2f} seconds")

def estimate_door_state_ollama_vlm(model, rgb_img, is_passable="", door_open_percent="", door_wall_angle="", left_right_door_angle="", door_type=""):
    # directly use ollama api to estimate door state
    try:
        # Encode OpenCV image (BGR) as JPEG
        ok, buf = cv2.imencode('.jpg', rgb_img)
        if not ok:
            raise RuntimeError(f"Failed to encode image.")
        rgb_img_bytes = buf.tobytes() 
        img_b64 = base64.b64encode(rgb_img_bytes).decode('utf-8')

        prompt = f"""
            You are a robot perception assistant. Visually verify the door state in the image.

            Classify door state as: "open", "semi_open", "closed", or "unknown"
            Detect if a human is clearly visible near the door.

            Preliminary data (may be noisy):
            - is_passable: {is_passable}, open_percent: {door_open_percent}, wall_angle: {door_wall_angle}
            - lr_angle: {left_right_door_angle}, type: {door_type}

            Rules:
            - Fully open or unobstructed → "open"
            - Partially open → "semi_open"
            - Clearly closed → "closed"
            - Ambiguous/occluded → "unknown"
            - Human visible → "yes", otherwise "no"

            Generate a SHORT, polite spoken sentence ALWAYS:
            - If human present → request appropriate action (open door, open more, or keep open)
            - If no human → briefly describe the door scene (e.g., "The door appears closed" or "I see an open doorway"), then request if anyone can please open.
            
            Output ONLY valid JSON:
            {{
                "door_state": "<open|semi_open|closed|unknown>",
                "human_present": "<yes|no>",
                "conversation": "<single short sentence, always required>"
            }}
            """


        response = chat(
            model=model, # fast
            # model='qwen3-vl:4b-instruct',
            messages=[
                {
                    'role': 'user',
                    'content': prompt,
                    'images': [img_b64],
                }
            ],
            format="json"
        )

        """
        res = {
            'door_state': 'open',
            'human_present': 'no',
            'conversation': 'please open the door'
        }
        """
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
    model_list = [ 
                  #'qwen3-vl:4b', 
                  'qwen3-vl:4b-instruct',
                  #'qwen3-vl:2b-instruct-q4_K_M', 
                  #'qwen3-vl:2b-instruct-q8_0',
                  #'qwen2.5vl:7b'
                  ]
    for model in model_list:
        print(f"Testing model: {model}")
        img_id = 63 # Change this ID to test different images
        image_path = f"/home/ias/satya/catkin_ws/src/door_navigation/scripts/data_new/latest_image_color_lab_{img_id}.jpg"
        test_vlm_door_state(image_path, model)
