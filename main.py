import streamlit as st
from PIL import Image
from transformers import OwlViTProcessor, OwlViTForObjectDetection
import torch
from utils import draw_boxes

@st.cache_resource
def load_model():
    """Load OwlViT model and processor once and cache the resources."""
    processor = OwlViTProcessor.from_pretrained("google/owlvit-base-patch32")
    model = OwlViTForObjectDetection.from_pretrained("google/owlvit-base-patch32")
    model = model.to("mps")
    model.eval()
    return processor, model

def main():
    uploaded_file = st.file_uploader("Choose an image...", type=["png", "jpg", "jpeg"])
    prompt = st.text_input("Enter a prompt (e.g., 'dog', 'cat', 'car')")
    run = st.button("Run Inference")

    if uploaded_file is not None:
        image = Image.open(uploaded_file).convert("RGB")
        st.image(image, caption="Uploaded Image", use_container_width=True)
    else:
        st.info("Please upload an image.")
        return

    if not prompt:
        st.info("Please enter a prompt.")
        return

    if run:
        processor, model = load_model()
        inputs = processor(text=[prompt], images=image, return_tensors="pt")
        inputs = {k: v.to("mps") if isinstance(v, torch.Tensor) else v for k, v in inputs.items()}
        
        with torch.no_grad():
            outputs = model(**inputs)
        # Post-process to get bounding boxes
        results = processor.post_process_object_detection(
            outputs, threshold=0.1, target_sizes=[image.size[::-1]]
        )
        print(f"results: {results}")
        if results and len(results[0]) > 0:
            boxes = []
            labels = []
            results_list = []
            
            # Move tensors to CPU for processing
            scores = results[0]["scores"].cpu()
            labels_indices = results[0]["labels"].cpu()
            boxes_tensor = results[0]["boxes"].cpu()

            for score, label_idx, box in zip(scores, labels_indices, boxes_tensor):
                if score > 0.3:
                    box_list = box.tolist()
                    boxes.append(box_list)
                    # Use the prompt as the label text since we only have one prompt
                    labels.append(prompt)
                    
                    results_list.append(
                        {
                            "Label": prompt,
                            "Score": f"{score:.2f}",
                            "Box": [round(b, 2) for b in box_list],
                        }
                    )
            
            if boxes:
                annotated_image = draw_boxes(image.copy(), boxes, labels)
                st.image(annotated_image, caption="Annotated Image", use_container_width=True)

                st.write("### Detection Results")
                st.dataframe(results_list)
            else:
                st.warning("No objects detected with the given prompt.")
        else:
            st.warning("No objects detected with the given prompt.")

if __name__ == "__main__":
    main()
