import streamlit as st
from PIL import Image
from transformers import OwlViTProcessor, OwlViTForObjectDetection
import torch
from utils import draw_boxes

def get_device():
    """利用可能なデバイス（CUDA, MPS, CPU）を判定して返します。"""
    if torch.cuda.is_available():
        return "cuda"
    elif torch.backends.mps.is_available():
        return "mps"
    else:
        return "cpu"

@st.cache_resource
def load_model():
    """OwlViTモデルとプロセッサを一度だけ読み込み、リソースをキャッシュします。"""
    device = get_device()
    processor = OwlViTProcessor.from_pretrained("google/owlvit-base-patch32")
    model = OwlViTForObjectDetection.from_pretrained("google/owlvit-base-patch32")
    model = model.to(device)
    model.eval()
    return processor, model

def main():
    st.title("OwlViT 物体検出アプリ")
    st.write("画像をアップロードし、テキストプロンプトを入力して物体を検出します。")
    
    device = get_device()
    st.sidebar.write(f"使用デバイス: {device.upper()}")
    
    # サイドバーの設定
    st.sidebar.header("設定")
    threshold = st.sidebar.slider("検出閾値 (Threshold)", 0.0, 1.0, 0.1, 0.01)
    line_width = st.sidebar.slider("線の太さ (Line Width)", 1, 20, 3)
    font_size = st.sidebar.slider("文字サイズ (Font Size)", 8, 128, 16)

    uploaded_file = st.file_uploader("画像を選択してください...", type=["png", "jpg", "jpeg"])
    prompt = st.text_input("プロンプトを入力してください (例: 'a dog, a cat, a photo of a male')")
    run = st.button("推論実行")

    if uploaded_file is not None:
        image = Image.open(uploaded_file).convert("RGB")
        st.image(image, caption="アップロードされた画像", use_container_width=True)
    else:
        st.info("画像をアップロードしてください。")
        return

    if not prompt:
        st.info("プロンプトを入力してください。")
        return

    # プロンプトを解析し、色を割り当てる
    prompts = [p.strip() for p in prompt.split(",") if p.strip()]
    if not prompts:
        st.info("有効なプロンプトを入力してください。")
        return

    colors = ["red", "green", "blue", "orange", "purple", "cyan", "magenta", "yellow"]
    prompt_colors = {p: colors[i % len(colors)] for i, p in enumerate(prompts)}

    if run:
        processor, model = load_model()
        inputs = processor(text=[prompts], images=image, return_tensors="pt")
        # Tensorの場合は適切なデバイスに移動
        inputs = {k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in inputs.items()}
        
        with torch.no_grad():
            outputs = model(**inputs)
        # バウンディングボックスを取得するための後処理
        results = processor.post_process_object_detection(
            outputs, threshold=threshold, target_sizes=[image.size[::-1]]
        )
        
        if results and len(results[0]) > 0:
            boxes = []
            labels = []
            box_colors = []
            results_list = []
            
            # 処理のためにTensorをCPUに移動
            scores = results[0]["scores"].cpu()
            labels_indices = results[0]["labels"].cpu()
            boxes_tensor = results[0]["boxes"].cpu()

            for score, label_idx, box in zip(scores, labels_indices, boxes_tensor):
                if score > threshold:
                    box_list = box.tolist()
                    boxes.append(box_list)
                    
                    # 対応するプロンプトと色を取得
                    # label_idx は prompts リストのインデックスに対応
                    idx = label_idx.item()
                    if idx < len(prompts):
                        label_text = prompts[idx]
                        color = prompt_colors[label_text]
                    else:
                        label_text = "Unknown"
                        color = "red"

                    labels.append(label_text)
                    box_colors.append(color)
                    
                    results_list.append(
                        {
                            "ラベル": label_text,
                            "スコア": f"{score:.2f}",
                            "バウンディングボックス": [round(b, 2) for b in box_list],
                            "色": color,
                        }
                    )
            
            if boxes:
                annotated_image = draw_boxes(image.copy(), boxes, labels, box_colors, line_width, font_size)
                st.image(annotated_image, caption="検出結果", use_container_width=True)

                st.write("### 検出結果一覧")
                st.dataframe(results_list)
            else:
                st.warning("指定されたプロンプトで物体は検出されませんでした。")
        else:
            st.warning("指定されたプロンプトで物体は検出されませんでした。")

if __name__ == "__main__":
    main()
