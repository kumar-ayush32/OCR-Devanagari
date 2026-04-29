import os
import torch
import cv2
import torchvision.transforms as transforms
from PIL import Image, ImageDraw, ImageFont
from model import CRNN
from vocab import VOCAB_SIZE, ctc_decode

class Config:
    DATASET_ROOT = r"E:\Project\OCR_Devnagri\ML project"
    CHECKPOINT   = os.path.join(DATASET_ROOT, "checkpoints", "best_model.pth")
    OUTPUT_IMAGE = os.path.join(DATASET_ROOT, "results", "result.png")
    OUTPUT_TXT   = os.path.join(DATASET_ROOT, "results", "result.txt")
    DEVICE       = "cuda" if torch.cuda.is_available() else "cpu"

def find_hindi_font():
    candidates = [
        "C:/Windows/Fonts/NirmalaUI.ttf",
        "C:/Windows/Fonts/NirmalaS.ttf",
        "C:/Windows/Fonts/Nirmala.ttf",
        "C:/Windows/Fonts/mangal.ttf",
        "C:/Windows/Fonts/Mangal.ttf",
    ]
    for path in candidates:
        if os.path.exists(path):
            print(f"Font found: {path}")
            return path
    print("No Hindi font found, using default")
    return None

def get_transform():
    return transforms.Compose([
        transforms.Grayscale(num_output_channels=1),
        transforms.Resize((32, 128)),
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))
    ])

def load_model(cfg):
    if not os.path.exists(cfg.CHECKPOINT):
        raise FileNotFoundError(f"Checkpoint not found: {cfg.CHECKPOINT}")

    size_mb = os.path.getsize(cfg.CHECKPOINT) / (1024 * 1024)
    if size_mb < 1.0:
        raise ValueError(f"Checkpoint looks corrupted ({size_mb:.2f} MB)")

    model = CRNN(vocab_size=VOCAB_SIZE).to(cfg.DEVICE)
    ckpt = torch.load(
        cfg.CHECKPOINT,
        map_location=cfg.DEVICE,
        weights_only=True
    )
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()

    epoch   = ckpt.get("epoch", "?")
    val_cer = ckpt.get("val_cer", None)
    cer_str = f"{val_cer*100:.2f}%" if val_cer is not None else "N/A"
    print(f"Model loaded | epoch: {epoch} | val CER: {cer_str} | device: {cfg.DEVICE}")

    return model

def decode_prediction(log_probs):
    indices = log_probs.argmax(dim=2).permute(1, 0)
    return ctc_decode(indices[0].tolist())

def save_result_image(image_path, predicted_text, output_path, font_path):
    orig = Image.open(image_path).convert("RGB")
    ow, oh = orig.size
    banner_h = 70
    try:
        font = ImageFont.truetype(font_path, size=32) if font_path else ImageFont.load_default()
    except Exception:
        font = ImageFont.load_default()

    label = f"OCR: {predicted_text}"
    dummy_img = Image.new("RGB", (10, 10))
    draw = ImageDraw.Draw(dummy_img)
    bbox = draw.textbbox((0, 0), label, font=font)
    tw, th = bbox[2] - bbox[0], bbox[3] - bbox[1]
    new_width = max(ow, tw + 40)
    banner = Image.new("RGB", (new_width, banner_h), color=(245, 245, 240))
    draw = ImageDraw.Draw(banner)
    tx = (new_width - tw) // 2
    ty = (banner_h - th) // 2
    draw.text((tx, ty), label, font=font, fill=(20, 20, 80))
    new_orig = Image.new("RGB", (new_width, oh), color=(245, 245, 240))
    new_orig.paste(orig, ((new_width - ow)//2, 0))
    combined = Image.new("RGB", (new_width, oh + banner_h), color=(245, 245, 240))
    combined.paste(new_orig, (0, 0))
    combined.paste(banner, (0, oh))
    combined.save(output_path)
    print(f"Result image saved: {output_path}")

def open_image(image_path, window_name="Image"):
    img = cv2.imread(image_path)
    if img is None:
        print("Image not found:", image_path)
        return
    cv2.imshow(window_name, img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

def ocr_image(image_path):
    if not os.path.exists(image_path):
        print(f"Image not found: {image_path}")
        return None
    cfg       = Config()
    font_path = find_hindi_font()
    model     = load_model(cfg)
    transform = get_transform()
    img        = Image.open(image_path).convert("RGB")
    img_tensor = transform(img).unsqueeze(0).to(cfg.DEVICE)

    with torch.no_grad():
        log_probs = model(img_tensor)

    prediction = decode_prediction(log_probs)
    with open(cfg.OUTPUT_TXT, "a", encoding="utf-8") as f:
        f.write(f"Image : {image_path}\n")
        f.write(f"Result: {prediction}\n")
        f.write("-" * 40 + "\n")
    print(f"Result text saved: {cfg.OUTPUT_TXT}")
    save_result_image(image_path, prediction, cfg.OUTPUT_IMAGE, font_path)
    open_image(cfg.OUTPUT_IMAGE, window_name="OCR Result")
    return prediction

def main():
    print("OCR Runner Started")
    print("Press 'q' and Enter anytime to quit\n")
    try:
        image_path = r"E:\Project\OCR_Devnagri\ML project\Custom_input\21.jpg"
        print("Processing...\n")
        ocr_image(image_path)
        print("\nDone.")
    except KeyboardInterrupt:
        print("\nInterrupted by user (Ctrl+C)")
    finally:
        print("Program finished")

if __name__ == "__main__":
    main()