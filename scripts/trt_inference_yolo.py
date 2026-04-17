#!/usr/bin/env python3

import cv2
from ultralytics import YOLO
import numpy as np

from utils.config import MODEL_PATH, IMG_SIZE, CONFIDENCE_THRESHOLD

# TensorRT Python bindings in some versions still reference np.bool.
if "bool" not in np.__dict__:
    np.bool = np.bool_

def main():
    image_path = "/home/ias/satya/catkin_ws/src/door_navigation/scripts/data_new/latest_image_color_lab_19.jpg"
    save = "/home/ias/satya/catkin_ws/src/door_navigation/scripts/data_new/latest_image_color_lab_19_trt_yolo.png"  # e.g. "output.jpg"
    bgr_image = cv2.imread(image_path)
    if bgr_image is None:
        raise FileNotFoundError(f"Failed to read image: {image_path}")

    # Set task here (constructor) to avoid task auto-guess warning for .engine models.
    model = YOLO(MODEL_PATH, task='detect')

    # Force fixed inference size so TensorRT profile matches exported engine shape.
    results = model.predict(
        source=image_path,
        imgsz=IMG_SIZE,
        conf=CONFIDENCE_THRESHOLD,
        device="cuda:0",
        verbose=True,
    )
    
    # print results
    for result in results:
        print(f"got {len(result.boxes)} boxes in test image")
        for i, box in enumerate(result.boxes):
            print("for box-{}:".format(i))
            cls_id = int(box.cls[0])
            conf = float(box.conf[0])
            bbox = box.xyxy[0].cpu().numpy()  # get bbox coordinates
            print(f"Class ID: {cls_id}, Confidence: {conf:.2f}, BBox: {bbox}")
            
    # visualize results on the image
    annotated_image = results[0].plot()  # get the annotated image from the first result
    cv2.imwrite(save, annotated_image)
    print(f"Annotated image saved to: {save}")
    

if __name__ == "__main__":
    main()
