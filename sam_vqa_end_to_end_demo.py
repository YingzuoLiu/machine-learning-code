"""
SAM + VQA End-to-End Demo
-------------------------
Pipeline:
1) Run SAM to get object masks (AutomaticMaskGenerator). Pick the largest or the most confident mask.
2) Crop/Mask the image to the selected region of interest (ROI).
3) Run a VQA model (BLIP VQA) on the ROI and on the full image for comparison.

Usage (needs Python 3.9+ and a GPU is recommended but not required):

    # 1) Install deps (≈ few minutes, requires internet):
    pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121  # pick cpu/your cuda
    pip install opencv-python pillow matplotlib numpy
    pip install git+https://github.com/facebookresearch/segment-anything.git
    pip install transformers accelerate timm

    # 2) Download a SAM checkpoint (choose one) from:
    #    https://github.com/facebookresearch/segment-anything#model-checkpoints
    # Example: sam_vit_h_4b8939.pth placed at ./checkpoints/sam_vit_h_4b8939.pth

    # 3) Run the script
    python sam_vqa_demo.py --image ./examples/cat_on_table.jpg \
                           --question "What color is the cat?" \
                           --sam-checkpoint ./checkpoints/sam_vit_h_4b8939.pth \
                           --sam-model-type vit_h \
                           --roi-mode largest

Notes:
- If you don't have a SAM checkpoint yet, you can still run with --roi-mode box to use a simple central box as a mock ROI.
- The VQA model used here is BLIP VQA (Salesforce/blip-vqa-base) via Hugging Face Transformers.
- This is a minimal educational demo (not production-optimized).  

"""
import argparse
import os
from typing import Optional, Tuple

import numpy as np
from PIL import Image
import cv2
import matplotlib.pyplot as plt

# Suppress TF/transformers warnings
os.environ["TOKENIZERS_PARALLELISM"] = "false"

# --- VQA (BLIP) ---
from transformers import BlipProcessor, BlipForQuestionAnswering

# --- SAM ---
try:
    from segment_anything import sam_model_registry, SamAutomaticMaskGenerator
    _HAS_SAM = True
except Exception:
    _HAS_SAM = False


def load_image(image_path: str) -> Image.Image:
    img = Image.open(image_path).convert("RGB")
    return img


def run_sam_automatic(image_pil: Image.Image,
                      sam_checkpoint: str,
                      model_type: str = "vit_h",
                      points_per_side: int = 32,
                      pred_iou_thresh: float = 0.88,
                      stability_score_thresh: float = 0.95) -> list:
    """Return list of masks (each a dict with 'segmentation' boolean ndarray)."""
    if not _HAS_SAM:
        raise RuntimeError("segment-anything is not installed. See comments at the top to install.")
    image = np.array(image_pil)
    sam = sam_model_registry[model_type](checkpoint=sam_checkpoint)
    mask_generator = SamAutomaticMaskGenerator(
        model=sam,
        points_per_side=points_per_side,
        pred_iou_thresh=pred_iou_thresh,
        stability_score_thresh=stability_score_thresh,
        crop_n_layers=1,
    )
    masks = mask_generator.generate(image)
    return masks


def pick_mask(masks: list, mode: str = "largest") -> Optional[np.ndarray]:
    """Pick one mask from SAM outputs. Returns boolean mask HxW or None."""
    if not masks:
        return None
    if mode == "largest":
        # pick the largest area
        idx = np.argmax([m.get("area", np.sum(m["segmentation"])) for m in masks])
        return masks[idx]["segmentation"].astype(bool)
    elif mode == "highest_iou":
        idx = np.argmax([m.get("predicted_iou", 0.0) for m in masks])
        return masks[idx]["segmentation"].astype(bool)
    else:
        # default fallback
        idx = np.argmax([m.get("area", np.sum(m["segmentation"])) for m in masks])
        return masks[idx]["segmentation"].astype(bool)


def mask_to_bbox(mask: np.ndarray) -> Optional[Tuple[int, int, int, int]]:
    """Convert a boolean mask to a tight bounding box (x1, y1, x2, y2)."""
    ys, xs = np.where(mask)
    if len(xs) == 0 or len(ys) == 0:
        return None
    x1, x2 = xs.min(), xs.max()
    y1, y2 = ys.min(), ys.max()
    return int(x1), int(y1), int(x2), int(y2)


def crop_with_mask(image_pil: Image.Image, mask: np.ndarray) -> Image.Image:
    """Apply mask and crop tight bbox."""
    image = np.array(image_pil)
    bbox = mask_to_bbox(mask)
    if bbox is None:
        return image_pil
    x1, y1, x2, y2 = bbox
    # Optional: black-out background to emphasize ROI
    masked = image.copy()
    masked[~mask] = 0
    roi = masked[y1:y2+1, x1:x2+1]
    return Image.fromarray(roi)


def simple_center_box(image_pil: Image.Image, scale: float = 0.5) -> Image.Image:
    """Fallback ROI crop: a central box crop covering `scale` of the image size."""
    w, h = image_pil.size
    bw, bh = int(w * scale), int(h * scale)
    x1 = (w - bw) // 2
    y1 = (h - bh) // 2
    x2 = x1 + bw
    y2 = y1 + bh
    return image_pil.crop((x1, y1, x2, y2))


def load_blip_vqa(device: str = "cpu"):
    processor = BlipProcessor.from_pretrained("Salesforce/blip-vqa-base")
    model = BlipForQuestionAnswering.from_pretrained("Salesforce/blip-vqa-base")
    model.to(device)
    model.eval()
    return processor, model


def answer_vqa(processor, model, image_pil: Image.Image, question: str, device: str = "cpu") -> str:
    inputs = processor(image=image_pil, text=question, return_tensors="pt").to(device)
    with torch.no_grad():
        out = model.generate(**inputs, max_new_tokens=20)
    return processor.decode(out[0], skip_special_tokens=True)


# torch import after setting up functions to avoid import cost when printing help
import torch  # noqa: E402


def visualize(full_img: Image.Image, roi_img: Image.Image, full_ans: str, roi_ans: str, question: str):
    plt.figure(figsize=(10, 6))
    plt.suptitle(f"Question: {question}")

    plt.subplot(1, 2, 1)
    plt.title(f"Full image\nAnswer: {full_ans}")
    plt.imshow(full_img)
    plt.axis('off')

    plt.subplot(1, 2, 2)
    plt.title(f"ROI (SAM/Box)\nAnswer: {roi_ans}")
    plt.imshow(roi_img)
    plt.axis('off')

    plt.tight_layout()
    plt.show()


def main():
    parser = argparse.ArgumentParser(description="SAM + VQA minimal demo")
    parser.add_argument("--image", type=str, required=True, help="Path to input image")
    parser.add_argument("--question", type=str, required=True, help="VQA question")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")

    # SAM options
    parser.add_argument("--sam-checkpoint", type=str, default=None, help="Path to SAM checkpoint .pth")
    parser.add_argument("--sam-model-type", type=str, default="vit_h", choices=["vit_h", "vit_l", "vit_b"])
    parser.add_argument("--roi-mode", type=str, default="largest", choices=["largest", "highest_iou", "box"],
                        help="How to pick ROI. Use 'box' to skip SAM and use a central crop.")

    args = parser.parse_args()

    # Load image
    img = load_image(args.image)

    # Prepare ROI via SAM (or fallback box)
    use_sam = (_HAS_SAM and args.sam_checkpoint and os.path.exists(args.sam_checkpoint) and args.roi-mode != "box")

    roi_img = None
    if args.roi_mode == "box" or not _HAS_SAM or not args.sam_checkpoint or not os.path.exists(args.sam_checkpoint):
        # Fallback simple ROI
        roi_img = simple_center_box(img, scale=0.6)
    else:
        masks = run_sam_automatic(img, args.sam_checkpoint, model_type=args.sam_model_type)
        if len(masks) == 0:
            roi_img = simple_center_box(img, scale=0.6)
        else:
            mask = pick_mask(masks, mode=args.roi_mode)
            roi_img = crop_with_mask(img, mask)

    # Load VQA (BLIP)
    processor, model = load_blip_vqa(device=args.device)

    # Answer on full image and on ROI
    full_ans = answer_vqa(processor, model, img, args.question, device=args.device)
    roi_ans = answer_vqa(processor, model, roi_img, args.question, device=args.device)

    print("Question:", args.question)
    print("Full image answer:", full_ans)
    print("ROI answer:", roi_ans)

    # Visualize
    visualize(img, roi_img, full_ans, roi_ans, args.question)


if __name__ == "__main__":
    main()
