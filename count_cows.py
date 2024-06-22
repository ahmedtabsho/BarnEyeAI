import os
import cv2
from ultralytics import YOLO

def run_yolo():
    model_path = "/home/ahmedtab/Desktop/Final_Porject/Cow/best.pt"
    file_path = "/home/ahmedtab/Desktop/Final_Porject/Cow/cows2.jpeg"
    
    # Load the model
    model = YOLO(model_path)  # load a custom model

    unique_id = set()

    # Open the image file
    results = model.track(file_path, tracker="bytetrack.yaml", persist=True)
    img = results[0].plot()
    height, width, _ = img.shape

    boxes = results[0].boxes.xyxy.cpu().numpy().astype(int)
    ids = results[0].boxes.id.cpu().numpy().astype(int)
    for box, id in zip(boxes, ids):
        # Draw each bounding box with a smaller thickness
        cv2.rectangle(img, (box[0], box[1]), (box[2], box[3]), (0, 255, 0), 2)
        # Check if the id is unique
        int_id = int(id)
        if int_id not in unique_id:
            unique_id.add(int_id)

    counter = len(unique_id)

    # Ensure the static directory exists
    static_dir = 'static'
    if not os.path.exists(static_dir):
        os.makedirs(static_dir)

    img_path = "output_image.jpg"
    # Path to the output image
    output_path = os.path.join(static_dir, "output_image.jpg")

    # Delete the existing image if it exists
    if os.path.exists(output_path):
        os.remove(output_path)

    # Save the new image to the static directory
    cv2.imwrite(output_path, img)

    return img_path, counter
