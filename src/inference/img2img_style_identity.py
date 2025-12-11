"""
img2img_style_identity.py

Advanced img2img script for MuseAI-Prime.

Goal: Take a selfie/portrait and repaint it in the style of
Picasso or Rembrandt using your fine-tuned Stable Diffusion UNet,
then choose the best result based on face identity similarity.

Usage example (from project root):

  python src/inference/img2img_style_identity.py \
      --checkpoint checkpoints/sd_style_unet_best.pt \
      --input_image data/content/faces/raw/004132.jpg \
      --artist picasso \
      --output_dir outputs/sd_results/picasso \
      --num_samples 6 \
      --strength 0.6
"""

import argparse
from pathlib import Path
from typing import List, Tuple

import torch
from torch import nn
from torchvision import transforms
from PIL import Image

from diffusers import StableDiffusionImg2ImgPipeline
from facenet_pytorch import InceptionResnetV1


# -----------------------------------------------------------------------------
# GLOBAL CONFIG
# -----------------------------------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parents[2]

BASE_MODEL_ID = "runwayml/stable-diffusion-v1-5"

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# IMPORTANT: use float32 at inference to avoid NaNs / black outputs
DTYPE = torch.float32

IMAGE_SIZE = 512

# Artist-specific prompts and negative prompts
ARTIST_PROMPTS = {
    "picasso": {
        "prompt": (
            "a cubist oil painting portrait in the style of Pablo Picasso, "
            "bold color blocks, abstract facial planes, strong brush strokes, "
            "canvas texture, expressive colors"
        ),
        "negative": (
            "photorealistic, 3d render, CGI, anime, cartoon, blurry, "
            "low detail, low resolution, distorted face, deformed"
        ),
    },
    "rembrandt": {
        "prompt": (
            "a baroque oil painting portrait in the style of Rembrandt, "
            "dramatic chiaroscuro lighting, rich shadows and highlights, "
            "detailed skin texture, painterly brush strokes, canvas texture"
        ),
        "negative": (
            "overexposed, flat lighting, harsh CGI, anime, cartoon, "
            "blurred, low detail, distorted face, deformed"
        ),
    },
}


# -----------------------------------------------------------------------------
# IMAGE UTILS
# -----------------------------------------------------------------------------

def load_image_center_crop(path: Path, image_size: int = IMAGE_SIZE) -> Image.Image:
    """
    Load an image, center-crop, and resize to target size.
    """
    img = Image.open(path).convert("RGB")

    transform = transforms.Compose([
        transforms.Resize(image_size, interpolation=transforms.InterpolationMode.BICUBIC),
        transforms.CenterCrop(image_size),
    ])
    img = transform(img)
    return img


def simple_face_crop_for_facenet(pil_img: Image.Image) -> Image.Image:
    """
    Simple central crop to approximate the face region for FaceNet.
    """
    w, h = pil_img.size
    crop_ratio = 0.7  # take central 70% region
    new_w = int(w * crop_ratio)
    new_h = int(h * crop_ratio)
    left = (w - new_w) // 2
    top = (h - new_h) // 2
    right = left + new_w
    bottom = top + new_h

    face_img = pil_img.crop((left, top, right, bottom))
    face_img = face_img.resize((160, 160), Image.BICUBIC)
    return face_img


# -----------------------------------------------------------------------------
# FACENET UTILS
# -----------------------------------------------------------------------------

class FaceEmbedder(nn.Module):
    """
    Wrapper around FaceNet (InceptionResnetV1) to compute normalized embeddings.
    """

    def __init__(self, device: torch.device):
        super().__init__()
        self.device = device
        # pretrained on VGGFace2 is strong for identity
        self.model = InceptionResnetV1(pretrained="vggface2").eval().to(device)

        # Standard FaceNet input normalization
        self.transform = transforms.Compose([
            transforms.ToTensor(),  # [0,1]
            transforms.Normalize(mean=[0.5, 0.5, 0.5],
                                 std=[0.5, 0.5, 0.5]),
        ])

    @torch.no_grad()
    def embed(self, pil_img: Image.Image) -> torch.Tensor:
        """
        Compute L2-normalized face embedding.
        """
        x = self.transform(pil_img).unsqueeze(0).to(self.device)  # (1,3,160,160)
        emb = self.model(x)  # (1,512)
        emb = emb / (emb.norm(dim=1, keepdim=True) + 1e-8)
        return emb.squeeze(0)  # (512,)


def cosine_similarity(a: torch.Tensor, b: torch.Tensor) -> float:
    """
    Cosine similarity between two normalized embeddings.
    """
    return float((a * b).sum().item())


# -----------------------------------------------------------------------------
# DIFFUSION PIPE UTILS
# -----------------------------------------------------------------------------

def build_pipe_with_checkpoint(checkpoint_path: Path) -> StableDiffusionImg2ImgPipeline:
    """
    Load SD img2img pipeline and plug in fine-tuned UNet.
    """
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

    print(f"Loading base SD img2img pipeline: {BASE_MODEL_ID}")
    pipe = StableDiffusionImg2ImgPipeline.from_pretrained(
        BASE_MODEL_ID,
        torch_dtype=DTYPE,        # use float32 at inference
        safety_checker=None,
    )
    pipe.to(DEVICE)

    # Load UNet weights
    print(f"Loading fine-tuned UNet from: {checkpoint_path}")
    ckpt = torch.load(checkpoint_path, map_location="cpu")
    unet_state = ckpt.get("unet_state_dict", None)
    if unet_state is None:
        raise KeyError("Checkpoint does not contain 'unet_state_dict'.")
    pipe.unet.load_state_dict(unet_state)
    print("✅ Fine-tuned UNet loaded.")

    return pipe


# -----------------------------------------------------------------------------
# MAIN LOGIC
# -----------------------------------------------------------------------------

def generate_stylized_candidates(
    pipe: StableDiffusionImg2ImgPipeline,
    init_image: Image.Image,
    artist: str,
    num_samples: int,
    strength: float,
    guidance_scale: float,
    steps: int,
    seeds: List[int],
) -> List[Image.Image]:
    """
    Generate multiple stylized candidates using img2img for a given artist.
    """
    artist_cfg = ARTIST_PROMPTS[artist]
    prompt = artist_cfg["prompt"]
    negative_prompt = artist_cfg["negative"]

    print(f"\nPrompt: {prompt}")
    print(f"Negative prompt: {negative_prompt}")
    print(f"Strength: {strength}, CFG: {guidance_scale}, Steps: {steps}")
    print(f"Generating {num_samples} candidates...")

    results = []

    for i in range(num_samples):
        seed = seeds[i]
        generator = torch.Generator(device=DEVICE).manual_seed(seed)
        print(f"  → Sample {i+1}/{num_samples}, seed={seed}")

        # NO autocast: keep everything in full float32 for stability
        out = pipe(
            prompt=prompt,
            image=init_image,
            strength=strength,
            guidance_scale=guidance_scale,
            num_inference_steps=steps,
            negative_prompt=negative_prompt,
            generator=generator,
        )

        img = out.images[0]
        results.append(img)

    return results


def select_best_by_identity(
    original_face_emb: torch.Tensor,
    candidates: List[Image.Image],
    face_embedder: FaceEmbedder,
) -> Tuple[int, float]:
    """
    Compute identity similarity for each candidate and pick the best one.

    Returns:
        best_index: index of best candidate
        best_score: cosine similarity score
    """
    best_idx = -1
    best_score = -1.0

    print("\nEvaluating identity similarity for candidates...")
    for idx, img in enumerate(candidates):
        face_img = simple_face_crop_for_facenet(img)
        emb = face_embedder.embed(face_img)
        score = cosine_similarity(original_face_emb, emb)
        print(f"  Candidate {idx}: cosine similarity = {score:.4f}")

        if score > best_score:
            best_score = score
            best_idx = idx

    print(f"\n✅ Best candidate: {best_idx} with similarity {best_score:.4f}")
    return best_idx, best_score


def stylize_with_identity_guidance(
    checkpoint_path: Path,
    input_image_path: Path,
    artist: str,
    output_dir: Path,
    num_samples: int,
    strength: float,
    guidance_scale: float,
    steps: int,
):
    """
    Main pipeline: stylize a selfie + pick the best identity-preserving candidate.
    """
    artist = artist.lower()
    if artist not in ARTIST_PROMPTS:
        raise ValueError(f"Unsupported artist '{artist}'. Supported: {list(ARTIST_PROMPTS.keys())}")

    output_dir.mkdir(parents=True, exist_ok=True)

    # Load SD pipe + UNet
    pipe = build_pipe_with_checkpoint(checkpoint_path)
    pipe.set_progress_bar_config(disable=False)

    # Load original portrait
    init_image = load_image_center_crop(input_image_path, IMAGE_SIZE)
    print(f"\nLoaded input image: {input_image_path} → size {init_image.size}")

    # Build FaceNet embedder
    face_embedder = FaceEmbedder(DEVICE)
    original_face = simple_face_crop_for_facenet(init_image)
    original_emb = face_embedder.embed(original_face)
    print("✅ Computed original face embedding.")

    # Seeds
    torch.manual_seed(42)
    seeds = torch.randint(0, 10_000_000, (num_samples,)).tolist()

    # Generate candidates
    candidates = generate_stylized_candidates(
        pipe=pipe,
        init_image=init_image,
        artist=artist,
        num_samples=num_samples,
        strength=strength,
        guidance_scale=guidance_scale,
        steps=steps,
        seeds=seeds,
    )

    # Save all candidates
    print("\nSaving all candidate images...")
    candidate_paths = []
    for idx, img in enumerate(candidates):
        candidate_path = output_dir / f"{artist}_candidate_{idx}.png"
        img.save(candidate_path)
        candidate_paths.append(candidate_path)
        print(f"  Saved candidate {idx}: {candidate_path}")

    # Pick best by identity
    best_idx, best_score = select_best_by_identity(
        original_face_emb=original_emb,
        candidates=candidates,
        face_embedder=face_embedder,
    )

    # Save best as the main result
    best_img = candidates[best_idx]
    best_path = output_dir / f"{artist}_best_identity.png"
    best_img.save(best_path)
    print(f"\n🎨 Final best {artist} portrait saved to: {best_path}")
    print(f"   Identity similarity score: {best_score:.4f}")


# -----------------------------------------------------------------------------
# CLI
# -----------------------------------------------------------------------------

def parse_args():
    parser = argparse.ArgumentParser(
        description="MuseAI-Prime: identity-guided img2img artist stylization"
    )

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
        help="Artist style to apply (picasso / rembrandt)",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        required=True,
        help="Directory to save all candidates + best result",
    )
    parser.add_argument(
        "--num_samples",
        type=int,
        default=6,
        help="How many candidates to generate per artist",
    )
    parser.add_argument(
        "--strength",
        type=float,
        default=0.6,
        help="Img2img strength [0,1]; higher => more style, less identity",
    )
    parser.add_argument(
        "--guidance_scale",
        type=float,
        default=8.0,
        help="Classifier-free guidance scale (e.g., 7.5-9.0)",
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

    checkpoint_path = Path(args.checkpoint)
    input_image_path = Path(args.input_image)
    output_dir = Path(args.output_dir)

    stylize_with_identity_guidance(
        checkpoint_path=checkpoint_path,
        input_image_path=input_image_path,
        artist=args.artist,
        output_dir=output_dir,
        num_samples=args.num_samples,
        strength=args.strength,
        guidance_scale=args.guidance_scale,
        steps=args.steps,
    )


if __name__ == "__main__":
    main()
