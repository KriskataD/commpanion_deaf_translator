import logging
import os
from pathlib import Path
from typing import Optional

import lmstudio as lms
import numpy as np
import torch
from ultralytics import YOLO


#-----------------------LLM Object Locator------------------------
class ObjectLocator:
    def __init__(self, model_name="mistral-7b-instruct-v0.3"):
        self.model = lms.llm(model_name)
        self.chat = lms.Chat()

    def locate(self, prompt: str, locations: str) -> str:
        instructions = (
            "You will be given a list of sentences and a question. "
            "Return every sentence from the list that answers the question. "
            "Do not add explanations. Do not add extra words. "
            "Just return the matching sentences from the list."
            "If there is no matching sentence say that the object is not present"
        )

        full_prompt = f"{instructions}\n\nSentences:\n{locations}\n\nQuestion: {prompt}\nAnswer:"
        self.chat.add_user_message(full_prompt)

        chunks = self.model.respond_stream(
            self.chat,
            on_message=self.chat.append,
        )

        return "".join(chunk.content for chunk in chunks).strip()


# Set performance settings
torch.set_num_threads(10)
torch.backends.mkldnn.enabled = True

# Logging
def setup_logging():
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
    return logging.getLogger(__name__)


# Load YOLOv8
def load_yolo_model(weights_path: Optional[str | os.PathLike[str]] = None):
    logger.info("Loading YOLOv8...")
    if weights_path is None:
        weights_path = Path(__file__).parent / "models" / "yolov8l-oiv7.pt"
    return YOLO(str(weights_path))  # change to 'yolov8s-oiv7.pt' or 'yolov8m-oiv7.pt' if desired (for less processing power)


def describe_all_objects(image_path_or_frame, yolo_model):
    """
    Describe objects in an image using YOLO detection.
    
    Args:
        image_path_or_frame: Either image path (str) or numpy array frame
        yolo_model: Loaded YOLO model
        
    Returns:
        str: Description of detected objects with their locations
    """
    results = yolo_model(image_path_or_frame)
    descriptions = []
    
    for r in results:
        width = r.orig_shape[1]
        height = r.orig_shape[0]
        for box in r.boxes:
            conf = box.conf[0].item()
            if conf < 0.4:
                continue

            label_idx = int(box.cls[0].item())
            label = yolo_model.names[label_idx]

            x_center = (box.xywh[0][0]).item()
            y_center = (box.xywh[0][1]).item()

            # Horizontal segmentation (9 regions)
            h_segment = width / 9
            if x_center < h_segment:
                horiz = "far left"
            elif x_center < 2 * h_segment:
                horiz = "left"
            elif x_center < 3 * h_segment:
                horiz = "close left"
            elif x_center < 4 * h_segment:
                horiz = "slightly left"
            elif x_center < 5 * h_segment:
                horiz = "middle"
            elif x_center < 6 * h_segment:
                horiz = "slightly right"
            elif x_center < 7 * h_segment:
                horiz = "close right"
            elif x_center < 8 * h_segment:
                horiz = "right"
            else:
                horiz = "far right"

            # Vertical segmentation (3 regions)
            v_segment = height / 3
            if y_center < v_segment:
                vert = "far away"
            elif y_center < 2 * v_segment:
                vert = "mid range"
            else:
                vert = "close up"

            descriptions.append(f"There is a {label} {horiz}, and {vert}.")

    if not descriptions:
        return "No objects found in the image."
    return "\n".join(descriptions)


logger = setup_logging()


def locate_objects_in_image(prompt: str, image_path_or_frame) -> str:
    """
    Locate objects in an image based on a text prompt.
    
    Args:
        prompt: The query to search for objects
        image_path_or_frame: Either image path (str) or numpy array frame
        
    Returns:
        str: Result of object location query
    """
    yolo_model = load_yolo_model()
    locator = ObjectLocator()

    logger.info("Detecting all objects in image using YOLO...")
    locations = describe_all_objects(image_path_or_frame, yolo_model)
    print("Detected objects:", locations)
    
    result = locator.locate(prompt, locations)
    print("Results:", result)
    
    return result


def locate_objects_in_frame(prompt: str, frame: np.ndarray) -> str:
    """
    Locate objects in a numpy frame based on a text prompt.
    
    Args:
        prompt: The query to search for objects
        frame: Captured image as numpy array
        
    Returns:
        str: Result of object location query
    """
    return locate_objects_in_image(prompt, frame)
