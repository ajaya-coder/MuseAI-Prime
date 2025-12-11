"""
train_style_sd.py

Strong style diffusion training for MuseAI-Prime using a pre-trained
Stable Diffusion v1.5 backbone.

- Fine-tunes ONLY the UNet on Picasso / Rembrandt style images.
- Uses light data augmentation.
- Trains in full float32 (no mixed precision) for numerical stability.
- Includes simple early stopping on validation loss.
"""

import os
import math
from pathlib import Path
from typing import Dict, List, Tuple

import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, random_split

from PIL import Image
from tqdm import tqdm
from torchvision import transforms
from torchvision.utils import save_image

from diffusers import StableDiffusionPipeline, DDPMScheduler

# -----------------------------------------------------------------------------
# CONFIG
# -----------------------------------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parents[2]

# Where your raw style images live
DATA_ROOT = PROJECT_ROOT / "data" / "style_raw"
ARTIST_FOLDERS: Dict[str, Path] = {
    "picasso": DATA_ROOT / "picasso",
    "rembrandt": DATA_ROOT / "rembrandt",
}

# Text prompts per artist — this is the conditioning signal
ARTIST_PROMPTS: Dict[str, str] = {
    "picasso": (
        "a cubist oil painting portrait in the style of Pablo Picasso, "
        "bold color blocks, abstract facial planes, strong brush strokes, "
        "canvas texture, expressive colors"
    ),
    "rembrandt": (
        "a dramatic, high-contrast baroque oil painting portrait in the style of Rembrandt, "
        "rich chiaroscuro lighting, detailed skin texture, oil paint brushwork, "
        "deep shadows and warm highlights"
    ),
}

# Base Stable Diffusion model to fine-tune
BASE_MODEL_ID = "runwayml/stable-diffusion-v1-5"

IMAGE_SIZE = 512

# Training hyperparams
BATCH_SIZE = 2
NUM_EPOCHS = 100
LEARNING_RATE = 5e-6          # lowered a bit for stability
WEIGHT_DECAY = 1e-2

VAL_SPLIT_RATIO = 0.2         # 20% val
EARLY_STOP_PATIENCE = 10      # epochs without improvement before stopping

NUM_WORKERS = 4

# Diffusion schedule for training
NUM_TRAIN_TIMESTEPS = 1000

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
DTYPE = torch.float32         # FULL FLOAT32 for stable training

# Output dirs
OUTPUT_ROOT = PROJECT_ROOT / "outputs" / "sd_style_trained"
CHECKPOINT_DIR = OUTPUT_ROOT / "checkpoints"
SAMPLES_DIR = OUTPUT_ROOT / "samples"
for d in [OUTPUT_ROOT, CHECKPOINT_DIR, SAMPLES_DIR]:
    d.mkdir(parents=True, exist_ok=True)

# -----------------------------------------------------------------------------
# DATASET
# -----------------------------------------------------------------------------

class StyleOnlyDataset(Dataset):
    """
    Loads style images from data/style_raw/<artist> and returns:
      - pixel_values: tensor in [-1, 1], shape [3, H, W]
      - prompt: text prompt encoding the desired style
      - artist_name: e.g. "picasso" or "rembrandt"
    """

    def __init__(
        self,
        artist_dirs: Dict[str, Path],
        artist_prompts: Dict[str, str],
        image_size: int = 512,
        augment: bool = True,
    ):
        super().__init__()

        self.image_paths: List[Path] = []
        self.artist_names: List[str] = []
        self.prompts: List[str] = []

        self.artists = sorted(artist_dirs.keys())
        self.artist_dirs = artist_dirs
        self.artist_prompts = artist_prompts

        valid_ext = {".jpg", ".jpeg", ".png", ".bmp", ".tiff"}

        for artist_name, artist_dir in artist_dirs.items():
            if not artist_dir.exists():
                print(f"⚠️  Warning: artist folder missing: {artist_dir}")
                continue

            files = [
                p for p in artist_dir.iterdir()
                if p.is_file() and p.suffix.lower() in valid_ext
            ]
            files.sort()

            prompt = self.artist_prompts.get(
                artist_name,
                f"a painting in the style of {artist_name}"
            )

            for p in files:
                self.image_paths.append(p)
                self.artist_names.append(artist_name)
                self.prompts.append(prompt)

        if len(self.image_paths) == 0:
            raise RuntimeError(f"No style images found under {artist_dirs}")

        print("\n[StyleOnlyDataset]")
        for a in self.artists:
            count = sum(1 for n in self.artist_names if n == a)
            print(f"  {a}: {count} images")

        base_transforms = [
            transforms.Resize(
                image_size,
                interpolation=transforms.InterpolationMode.BICUBIC,
            ),
            transforms.CenterCrop(image_size),
        ]

        if augment:
            # Gentle augmentations that won't break SD latents
            aug_transforms = [
                transforms.RandomHorizontalFlip(p=0.5),
                transforms.RandomApply(
                    [transforms.RandomAffine(degrees=10, translate=(0.05, 0.05))],
                    p=0.5,
                ),
                transforms.RandomApply(
                    [
                        transforms.ColorJitter(
                            brightness=0.15,
                            contrast=0.15,
                            saturation=0.15,
                        )
                    ],
                    p=0.5,
                ),
            ]
        else:
            aug_transforms = []

        self.transform = transforms.Compose(
            base_transforms
            + aug_transforms
            + [
                transforms.ToTensor(),  # [0,1]
                transforms.Normalize([0.5, 0.5, 0.5],
                                     [0.5, 0.5, 0.5]),  # -> [-1,1]
            ]
        )

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        path = self.image_paths[idx]
        artist_name = self.artist_names[idx]
        prompt = self.prompts[idx]

        image = Image.open(path).convert("RGB")
        pixel_values = self.transform(image)  # [3, H, W] in [-1,1]

        return {
            "pixel_values": pixel_values,
            "prompt": prompt,
            "artist_name": artist_name,
            "path": str(path),
        }


def make_dataloaders() -> Tuple[DataLoader, DataLoader]:
    """Create train and validation dataloaders with a fixed split."""
    full_dataset = StyleOnlyDataset(
        ARTIST_FOLDERS,
        ARTIST_PROMPTS,
        image_size=IMAGE_SIZE,
        augment=True,
    )

    total_len = len(full_dataset)
    val_len = int(total_len * VAL_SPLIT_RATIO)
    train_len = total_len - val_len

    train_dataset, val_dataset = random_split(
        full_dataset,
        [train_len, val_len],
        generator=torch.Generator().manual_seed(42),
    )

    print("\n[StyleOnlyDataset]")
    print(f"  Total style images: {total_len}")
    print(f"  Train: {train_len} images")
    print(f"  Val:   {val_len} images")

    train_loader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=NUM_WORKERS,
        pin_memory=True,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=NUM_WORKERS,
        pin_memory=True,
    )

    print(f"Train batches: {len(train_loader)}, Val batches: {len(val_loader)}")
    return train_loader, val_loader


# -----------------------------------------------------------------------------
# SAMPLING
# -----------------------------------------------------------------------------

def log_samples(
    pipe: StableDiffusionPipeline,
    epoch: int,
    num_per_artist: int = 2,
):
    """
    Generate and save some Picasso/Rembrandt samples with the fine-tuned pipe.
    """
    pipe.to(DEVICE)
    pipe.set_progress_bar_config(disable=True)

    images = []
    labels = []

    for artist_name, prompt in ARTIST_PROMPTS.items():
        for _ in range(num_per_artist):
            img = pipe(
                prompt,
                num_inference_steps=30,
                guidance_scale=7.5,
            ).images[0]
            images.append(img)
            labels.append(artist_name)

    # Convert to tensor grid
    tensors = []
    for img in images:
        t = transforms.ToTensor()(img)  # [0,1]
        tensors.append(t)
    grid = torch.stack(tensors, dim=0)

    out_path = SAMPLES_DIR / f"samples_epoch_{epoch:03d}.png"
    save_image(grid, out_path, nrow=num_per_artist)
    print(f"🖼  Saved sample grid to: {out_path}")


# -----------------------------------------------------------------------------
# TRAINING
# -----------------------------------------------------------------------------

def train():
    print("\n" + "=" * 70)
    print("MuseAI-Prime – Strong Style Diffusion Training (Stable Diffusion v1.5)")
    print("=" * 70)
    print(f"Device: {DEVICE}")
    print(f"Base model: {BASE_MODEL_ID}")
    print(f"Data root: {DATA_ROOT}")

    # 1. Dataloaders
    train_loader, val_loader = make_dataloaders()

    # 2. Load pre-trained Stable Diffusion
    print("\nLoading pre-trained Stable Diffusion pipeline...")
    pipe = StableDiffusionPipeline.from_pretrained(
        BASE_MODEL_ID,
        torch_dtype=DTYPE,
        safety_checker=None,
    )
    pipe.to(DEVICE)

    # DDPM scheduler for training
    noise_scheduler = DDPMScheduler.from_config(pipe.scheduler.config)
    noise_scheduler.num_train_timesteps = NUM_TRAIN_TIMESTEPS

    unet = pipe.unet
    vae = pipe.vae
    text_encoder = pipe.text_encoder
    tokenizer = pipe.tokenizer

    # Freeze VAE & text encoder; train only UNet for style
    vae.requires_grad_(False)
    text_encoder.requires_grad_(False)
    unet.requires_grad_(True)

    # Gradient checkpointing to save some memory
    unet.enable_gradient_checkpointing()

    optimizer = torch.optim.AdamW(
        unet.parameters(),
        lr=LEARNING_RATE,
        weight_decay=WEIGHT_DECAY,
    )

    best_val_loss = float("inf")
    epochs_without_improvement = 0

    global_step = 0

    # 3. Training loop
    for epoch in range(1, NUM_EPOCHS + 1):
        # ---------------------- TRAIN ----------------------
        unet.train()
        train_loss_sum = 0.0

        pbar = tqdm(train_loader, desc=f"Epoch {epoch}/{NUM_EPOCHS} [train]")

        for batch in pbar:
            pixel_values = batch["pixel_values"].to(DEVICE, dtype=DTYPE)  # [-1,1]
            prompts = batch["prompt"]

            # Encode images to latents with VAE
            with torch.no_grad():
                latents = vae.encode(pixel_values).latent_dist.sample()
                latents = latents * vae.config.scaling_factor

            # Noise & timesteps
            noise = torch.randn_like(latents)
            bsz = latents.shape[0]

            timesteps = torch.randint(
                0,
                noise_scheduler.config.num_train_timesteps,
                (bsz,),
                device=latents.device,
                dtype=torch.long,
            )

            noisy_latents = noise_scheduler.add_noise(latents, noise, timesteps)

            # Tokenize text
            text_inputs = tokenizer(
                list(prompts),
                padding="max_length",
                max_length=tokenizer.model_max_length,
                truncation=True,
                return_tensors="pt",
            )
            input_ids = text_inputs.input_ids.to(DEVICE)

            with torch.no_grad():
                encoder_hidden_states = text_encoder(input_ids)[0]

            # Predict noise
            model_pred = unet(
                noisy_latents,
                timesteps,
                encoder_hidden_states=encoder_hidden_states,
            ).sample

            loss = F.mse_loss(model_pred, noise, reduction="mean")

            optimizer.zero_grad(set_to_none=True)
            loss.backward()

            # Optional: gradient clipping for extra safety
            torch.nn.utils.clip_grad_norm_(unet.parameters(), max_norm=1.0)

            optimizer.step()

            global_step += 1
            train_loss_sum += float(loss.item())
            pbar.set_postfix({"loss": f"{loss.item():.4f}"})

        avg_train_loss = train_loss_sum / max(1, len(train_loader))

        # ---------------------- VALIDATE ----------------------
        unet.eval()
        val_loss_sum = 0.0

        with torch.no_grad():
            pbar_val = tqdm(val_loader, desc=f"Epoch {epoch}/{NUM_EPOCHS} [val]")
            for batch in pbar_val:
                pixel_values = batch["pixel_values"].to(DEVICE, dtype=DTYPE)
                prompts = batch["prompt"]

                latents = vae.encode(pixel_values).latent_dist.sample()
                latents = latents * vae.config.scaling_factor

                noise = torch.randn_like(latents)
                bsz = latents.shape[0]

                timesteps = torch.randint(
                    0,
                    noise_scheduler.config.num_train_timesteps,
                    (bsz,),
                    device=latents.device,
                    dtype=torch.long,
                )

                noisy_latents = noise_scheduler.add_noise(latents, noise, timesteps)

                text_inputs = tokenizer(
                    list(prompts),
                    padding="max_length",
                    max_length=tokenizer.model_max_length,
                    truncation=True,
                    return_tensors="pt",
                )
                input_ids = text_inputs.input_ids.to(DEVICE)

                encoder_hidden_states = text_encoder(input_ids)[0]

                model_pred = unet(
                    noisy_latents,
                    timesteps,
                    encoder_hidden_states=encoder_hidden_states,
                ).sample

                val_loss = F.mse_loss(model_pred, noise, reduction="mean")
                val_loss_sum += float(val_loss.item())
                pbar_val.set_postfix({"val_loss": f"{val_loss.item():.4f}"})

        avg_val_loss = val_loss_sum / max(1, len(val_loader))

        print(f"\nEpoch {epoch} – avg train loss: {avg_train_loss:.4f}")
        print(f"Epoch {epoch} – avg val loss:   {avg_val_loss:.4f}")

        # ---------------------- SAVE CHECKPOINT ----------------------
        ckpt_path = CHECKPOINT_DIR / f"sd_style_unet_epoch_{epoch:03d}.pt"
        torch.save(
            {
                "epoch": epoch,
                "unet_state_dict": unet.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "avg_train_loss": avg_train_loss,
                "avg_val_loss": avg_val_loss,
                "base_model_id": BASE_MODEL_ID,
                "artist_prompts": ARTIST_PROMPTS,
            },
            ckpt_path,
        )
        print(f"✅ Saved UNet checkpoint: {ckpt_path}")

        # ---------------------- SAMPLE IMAGES ----------------------
        if epoch % 2 == 0 or epoch == 1:
            try:
                print("Sampling with current fine-tuned UNet...")
                log_samples(pipe, epoch, num_per_artist=2)
            except Exception as e:
                print(f"⚠️ Sampling failed: {e}")

        # ---------------------- EARLY STOPPING ----------------------
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            epochs_without_improvement = 0

            best_ckpt_path = CHECKPOINT_DIR / "sd_style_unet_best.pt"
            torch.save(
                {
                    "epoch": epoch,
                    "unet_state_dict": unet.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "avg_train_loss": avg_train_loss,
                    "avg_val_loss": avg_val_loss,
                    "base_model_id": BASE_MODEL_ID,
                    "artist_prompts": ARTIST_PROMPTS,
                },
                best_ckpt_path,
            )
            print(f"🌟 New best model saved to: {best_ckpt_path}")
        else:
            epochs_without_improvement += 1
            print(f"No improvement for {epochs_without_improvement} epoch(s).")

        if epochs_without_improvement >= EARLY_STOP_PATIENCE:
            print("\n⏹ Early stopping: validation loss stopped improving.")
            break

    print("\nTraining complete!")
    print(f"Checkpoints saved under: {CHECKPOINT_DIR}")
    print(f"Samples saved under:    {SAMPLES_DIR}")


def main():
    train()


if __name__ == "__main__":
    main()
