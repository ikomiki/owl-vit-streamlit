import torch
from transformers import OwlViTProcessor, OwlViTForObjectDetection
from PIL import Image

def run_test():
    processor = OwlViTProcessor.from_pretrained('google/owlvit-base-patch32')
    model = OwlViTForObjectDetection.from_pretrained('google/owlvit-base-patch32')
    img = Image.open('dog.jpg')
    # Perform inference
    inputs = processor(text=["dog"], images=img, return_tensors="pt")
    with torch.no_grad():
        outputs = model(**inputs)
    # Use the newer post_process_grounded_object_detection method
    if hasattr(processor, "post_process_grounded_object_detection"):
        results = processor.post_process_grounded_object_detection(
            outputs, threshold=0.1, target_sizes=[img.size[::-1]]
        )
    else:
        results = processor.post_process_object_detection(
            outputs, threshold=0.1, target_sizes=[img.size[::-1]]
        )
    
    # Safely handle empty detections
    detections = results[0]
    boxes = detections.get("boxes")
    labels = detections.get("labels")
    if boxes is None or boxes.numel() == 0:
        print("No objects detected.")
    else:
        print("Detected boxes:", boxes.shape[0])
        print("First box:", boxes[0].tolist())
        if labels is not None:
            print("First label:", labels[0])

if __name__ == '__main__':
    run_test()
