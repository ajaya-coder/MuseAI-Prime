"""
img2img_style.py

Use a fine-tuned Stable Diffusion UNet to repaint a selfie/portrait
in the style of either Picasso or Rembrandt, while preserving the
overall face structure (identity) via controlled img2img strength.

Usage (from project root):

  python src/inference/img2img_style.py \
      --checkpoint outputs/sd_style_trained/checkpoints/sd_style_unet_epoch_010.pt \
      --input_image path/to/your_selfie.jpg \
      --artist picasso \
      --output_image outputs/picasso_selfie.png \
      --strength 0.55

  python src/inference/img2img_style.py \
      --checkpoint outputs/sd_style_trained/checkpoints/sd_style_unet_epoch_010.pt \
      --input_image path/to/your_selfie.jpg \
      --artist rembrandt \
      --output_image outputs/rembrandt_selfie.png \
      --strength 0.55
"""

import argparse
from pathlib import Path

import torch
from PIL import Image
from torchvision import transforms

from diffusers import StableDiffusionImg2ImgPipeline


# -----------------------------------------------------------------------------
# CONFIG
# -----------------------------------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parents[2]

BASE_MODEL_ID = "runwayml/stable-diffusion-v1-5"
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
DTYPE = torch.float16

IMAGE_SIZE = 512

# Same prompts as training – this anchors the style.
ARTIST_PROMPTS = {
    "picasso": (
        "a cubist oil painting portrait in the style of Pablo Picasso, "
        "bold color blocks, abstract facial planes, strong brush strokes"
    ),
    "rembrandt": (
        "a dramatic, high-contrast baroque oil painting portrait in the style of Rembrandt, "
        "rich chiaroscuro lighting, detailed skin texture, oil paint brushwork"
    ),
}


# -----------------------------------------------------------------------------
# UTILS
# -----------------------------------------------------------------------------

def load_image_for_sd(path: Path, image_size: int = IMAGE_SIZE) -> Image.Image:
    """
    Load, center-crop, and resize an input portrait to match the SD resolution.

    We keep it simple: center crop → resize to 512x512.
    For better face cropping, you can later reuse the MTCNN-based cropping
    from your preprocessing, but this is a good starting point.
    """
    img = Image.open(path).convert("RGB")

    # Simple center crop + resize
    transform = transforms.Compose([
        transforms.Resize(image_size, interpolation=transforms.InterpolationMode.BICUBIC),
        transforms.CenterCrop(image_size),
    ])
    img = transform(img)
    return img


def build_pipe_with_checkpoint(checkpoint_path: Path) -> StableDiffusionImg2ImgPipeline:
    """
    Load Stable Diffusion img2img pipeline and plug in the fine-tuned UNet.

    Args:
        checkpoint_path: path to a .pt file saved by train_style_sd.py
    """
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

    print(f"Loading base SD img2img pipeline: {BASE_MODEL_ID}")
    pipe = StableDiffusionImg2ImgPipeline.from_pretrained(
        BASE_MODEL_ID,
        torch_dtype=DTYPE,
        safety_checker=None,
    )

    pipe.to(DEVICE)

    # Load UNet weights
    print(f"Loading fine-tuned UNet weights from: {checkpoint_path}")
    ckpt = torch.load(checkpoint_path, map_location="cpu")
    unet_state = ckpt.get("unet_state_dict", None)
    if unet_state is None:
        raise KeyError("Checkpoint does not contain 'unet_state_dict'.")

    pipe.unet.load_state_dict(unet_state)
    print("✅ Fine-tuned UNet loaded into pipeline.")

    return pipe


# -----------------------------------------------------------------------------
# MAIN INFERENCE FUNCTION
# -----------------------------------------------------------------------------

def stylize_portrait(
    checkpoint_path: Path,
    input_image_path: Path,
    artist: str,
    output_image_path: Path,
    strength: float,
    guidance_scale: float = 7.5,
    num_inference_steps: int = 30,
):
    """
    Repaint an input portrait with a given artist's style using img2img.

    Args:
        checkpoint_path: path to fine-tuned UNet checkpoint .pt
        input_image_path: path to the user's selfie / portrait
        artist: "picasso" or "rembrandt"
        output_image_path: where to save the stylized image
        strength: how strong the edit is [0,1]; higher = more stylized / less identity
        guidance_scale: CFG scale (higher = follow text more, but risk weirdness)
        num_inference_steps: diffusion steps (30 is usually plenty)
    """
    artist = artist.lower()
    if artist not in ARTIST_PROMPTS:
        raise ValueError(f"Unsupported artist '{artist}'. Supported: {list(ARTIST_PROMPTS.keys())}")

    prompt = ARTIST_PROMPTS[artist]
    print(f"\nArtist: {artist}")
    print(f"Prompt: {prompt}")
    print(f"Strength: {strength}")

    # Build pipeline with fine-tuned UNet
    pipe = build_pipe_with_checkpoint(checkpoint_path)
    pipe.set_progress_bar_config(disable=False)

    # Load and preprocess input portrait
    init_image = load_image_for_sd(input_image_path, IMAGE_SIZE)
    print(f"Loaded input image: {input_image_path} → size {init_image.size}")

    # Run img2img
    with torch.autocast("cuda", enabled=(DEVICE.type == "cuda")):
        result = pipe(
            prompt=prompt,
            image=init_image,
            strength=strength,
            guidance_scale=guidance_scale,
            num_inference_steps=num_inference_steps,
        )

    stylized_image = result.images[0]

    output_image_path.parent.mkdir(parents=True, exist_ok=True)
    stylized_image.save(output_image_path)
    print(f"✅ Saved stylized portrait to: {output_image_path}")


# -----------------------------------------------------------------------------
# CLI
# -----------------------------------------------------------------------------

def parse_args():
    parser = argparse.ArgumentParser(description="MuseAI-Prime style img2img")

    parser.add_argument(
        "--checkpoint",
        type=str,
        required=True,
        help="Path to fine-tuned UNet checkpoint (.pt) from train_style_sd.py",
    )
    parser.add_argument(
        "--input_image",
        type=str,
        required=True,
        help="Path to input selfie/portrait image",
    )
    parser.add_argument(
        "--artist",
        type=str,
        choices=list(ARTIST_PROMPTS.keys()),
        required=True,
        help="Artist style to apply (picasso/rembrandt)",
    )
    parser.add_argument(
        "--output_image",
        type=str,
        required=True,
        help="Path to save the stylized output image",
    )
    parser.add_argument(
        "--strength",
        type=float,
        default=0.55,
        help="Img2img strength in [0,1]; higher => more style, less identity",
    )
    parser.add_argument(
        "--guidance_scale",
        type=float,
        default=7.5,
        help="Classifier-free guidance scale (e.g., 7.5 or 9.0)",
    )
    parser.add_argument(
        "--steps",
        type=int,
        default=30,
        help="Number of diffusion inference steps",
    )

    return parser.parse_args()


def main():
    args = parse_args()

    checkpoint = Path(args.checkpoint)
    input_image = Path(args.input_image)
    output_image = Path(args.output_image)

    stylize_portrait(
        checkpoint_path=checkpoint,
        input_image_path=input_image,
        artist=args.artist,
        output_image_path=output_image,
        strength=args.strength,
        guidance_scale=args.guidance_scale,
        num_inference_steps=args.steps,
    )


if __name__ == "__main__":
    main()