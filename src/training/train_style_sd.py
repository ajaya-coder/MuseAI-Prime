"""
train_style_sd.py

Strong style diffusion training for MuseAI-Prime using a pre-trained
Stable Diffusion v1.5 backbone.

- Uses data augmentation on style images (Picasso / Rembrandt)
- Splits dataset into train/val
- Trains for up to 100 epochs with early stopping on val loss
"""

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
        "bold color blocks, abstract facial planes, strong brush strokes"
    ),
    "rembrandt": (
        "a dramatic, high-contrast baroque oil painting portrait in the style of Rembrandt, "
        "rich chiaroscuro lighting, detailed skin texture, oil paint brushwork"
    ),
}

# Base Stable Diffusion model to fine-tune
BASE_MODEL_ID = "runwayml/stable-diffusion-v1-5"

IMAGE_SIZE = 512          # SD v1.5 native resolution
BATCH_SIZE = 2            # 32GB V100 -> you can bump this if VRAM allows
NUM_EPOCHS = 100          # we now allow up to 100 epochs
LEARNING_RATE = 1e-5      # conservative LR for fine-tuning
NUM_WORKERS = 4

# Validation + early stopping
VAL_SPLIT = 0.1           # 10% of style images as validation
EARLY_STOP_PATIENCE = 8   # stop if no val improvement for 8 epochs
EARLY_STOP_MIN_DELTA = 1e-4

# Diffusion schedule for training (we use DDPM scheduler)
NUM_TRAIN_TIMESTEPS = 1000

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
DTYPE = torch.float16     # training still in fp16 for speed; we'll do fp32 at inference

# Output dirs
OUTPUT_DIR = PROJECT_ROOT / "outputs" / "sd_style_trained"
CHECKPOINT_DIR = PROJECT_ROOT / "checkpoints"
SAMPLES_DIR = OUTPUT_DIR / "samples"
for d in [OUTPUT_DIR, CHECKPOINT_DIR, SAMPLES_DIR]:
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
        augment: bool = False,
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

        # -------- TRANSFORMS --------
        if augment:
            # Aggressive but sane augmentations to enlarge style dataset
            self.transform = transforms.Compose([
                transforms.Resize(
                    int(image_size * 1.1),
                    interpolation=transforms.InterpolationMode.BICUBIC
                ),
                transforms.RandomResizedCrop(
                    image_size,
                    scale=(0.8, 1.0),
                    ratio=(0.9, 1.1),
                ),
                transforms.RandomHorizontalFlip(p=0.5),
                transforms.ColorJitter(
                    brightness=0.15,
                    contrast=0.15,
                    saturation=0.15,
                    hue=0.05,
                ),
                transforms.ToTensor(),                     # [0,1]
                transforms.Normalize([0.5, 0.5, 0.5],
                                     [0.5, 0.5, 0.5]),    # -> [-1,1]
            ])
        else:
            # Validation: deterministic, no augmentation
            self.transform = transforms.Compose([
                transforms.Resize(
                    image_size,
                    interpolation=transforms.InterpolationMode.BICUBIC
                ),
                transforms.CenterCrop(image_size),
                transforms.ToTensor(),
                transforms.Normalize([0.5, 0.5, 0.5],
                                     [0.5, 0.5, 0.5]),
            ])

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        path = self.image_paths[idx]
        artist_name = self.artist_names[idx]
        prompt = self.prompts[idx]

        image = Image.open(path).convert("RGB")
        pixel_values = self.transform(image)   # [3, H, W] in [-1,1]

        return {
            "pixel_values": pixel_values,
            "prompt": prompt,
            "artist_name": artist_name,
            "path": str(path),
        }


def make_dataloaders() -> Tuple[DataLoader, DataLoader, Dataset]:
    """
    Builds train & val dataloaders with a random split of the style images.
    """
    full_dataset = StyleOnlyDataset(
        ARTIST_FOLDERS,
        ARTIST_PROMPTS,
        image_size=IMAGE_SIZE,
        augment=True,          # training data gets augmentation
    )

    n_total = len(full_dataset)
    n_val = max(1, int(VAL_SPLIT * n_total))
    n_train = n_total - n_val

    generator = torch.Generator().manual_seed(42)
    train_dataset, val_dataset = random_split(
        full_dataset,
        [n_train, n_val],
        generator=generator,
    )

    # For validation we want deterministic transforms → wrap in a new dataset
    val_dataset = StyleOnlyDataset(
        ARTIST_FOLDERS,
        ARTIST_PROMPTS,
        image_size=IMAGE_SIZE,
        augment=False,
    )

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

    return train_loader, val_loader, full_dataset

# -----------------------------------------------------------------------------
# SAMPLING UTILITY
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

    for artist_name, prompt in ARTIST_PROMPTS.items():
        for _ in range(num_per_artist):
            img = pipe(
                prompt,
                num_inference_steps=30,
                guidance_scale=7.5,
            ).images[0]
            images.append(img)

    tensors = [transforms.ToTensor()(img) for img in images]
    grid = torch.stack(tensors, dim=0)

    out_path = SAMPLES_DIR / f"samples_epoch_{epoch:03d}.png"
    save_image(grid, out_path, nrow=num_per_artist)
    print(f"🖼  Saved sample grid to: {out_path}")

# -----------------------------------------------------------------------------
# TRAIN / VAL STEP HELPERS
# -----------------------------------------------------------------------------

def compute_unet_loss(
    unet,
    vae,
    text_encoder,
    tokenizer,
    noise_scheduler,
    batch,
) -> torch.Tensor:
    """
    Single forward loss for UNet, shared by train & val.
    """
    pixel_values = batch["pixel_values"].to(DEVICE, dtype=DTYPE)  # [-1,1]
    prompts = batch["prompt"]

    # Encode images to latents with VAE
    with torch.no_grad():
        latents = vae.encode(pixel_values).latent_dist.sample()
        latents = latents * vae.config.scaling_factor

    # Sample noise & timesteps
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

    # Get text embeddings
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

    loss = F.mse_loss(model_pred.float(), noise.float(), reduction="mean")
    return loss


def validate_unet(
    unet,
    vae,
    text_encoder,
    tokenizer,
    noise_scheduler,
    val_loader,
) -> float:
    """
    Evaluate UNet on the validation style set.
    """
    unet.eval()
    total_loss = 0.0

    with torch.no_grad():
        for batch in val_loader:
            loss = compute_unet_loss(
                unet,
                vae,
                text_encoder,
                tokenizer,
                noise_scheduler,
                batch,
            )
            total_loss += loss.item()

    avg_loss = total_loss / max(1, len(val_loader))
    return avg_loss

# -----------------------------------------------------------------------------
# MAIN TRAINING LOOP
# -----------------------------------------------------------------------------

def train():
    print("\n" + "=" * 70)
    print("MuseAI-Prime – Strong Style Diffusion Training (Stable Diffusion v1.5)")
    print("=" * 70)
    print(f"Device: {DEVICE}")
    print(f"Base model: {BASE_MODEL_ID}")
    print(f"Data root: {DATA_ROOT}")

    # 1. Load dataset + splits
    train_loader, val_loader, full_dataset = make_dataloaders()
    print(f"Total style images: {len(full_dataset)}")
    print(f"Train batches: {len(train_loader)}, Val batches: {len(val_loader)}")

    # 2. Load pre-trained Stable Diffusion
    print("\nLoading pre-trained Stable Diffusion pipeline...")
    pipe = StableDiffusionPipeline.from_pretrained(
        BASE_MODEL_ID,
        torch_dtype=DTYPE,
        safety_checker=None,
    )
    pipe.to(DEVICE)

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

    unet.enable_gradient_checkpointing()

    optimizer = torch.optim.AdamW(unet.parameters(), lr=LEARNING_RATE)

    global_step = 0
    best_val_loss = float("inf")
    epochs_without_improvement = 0

    # 3. Training loop
    for epoch in range(1, NUM_EPOCHS + 1):
        unet.train()
        epoch_loss = 0.0

        pbar = tqdm(train_loader, desc=f"Epoch {epoch}/{NUM_EPOCHS}")

        for batch in pbar:
            loss = compute_unet_loss(
                unet,
                vae,
                text_encoder,
                tokenizer,
                noise_scheduler,
                batch,
            )

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            global_step += 1
            epoch_loss += loss.item()
            pbar.set_postfix({"train_loss": f"{loss.item():.4f}"})

        avg_train_loss = epoch_loss / max(1, len(train_loader))

        # ----------------- VALIDATION -----------------
        val_loss = validate_unet(
            unet,
            vae,
            text_encoder,
            tokenizer,
            noise_scheduler,
            val_loader,
        )

        print(f"\nEpoch {epoch} – train loss: {avg_train_loss:.4f}, val loss: {val_loss:.4f}")

        # 4. Save per-epoch checkpoint
        ckpt_path = CHECKPOINT_DIR / f"sd_style_unet_epoch_{epoch:03d}.pt"
        torch.save(
            {
                "epoch": epoch,
                "unet_state_dict": unet.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "train_loss": avg_train_loss,
                "val_loss": val_loss,
                "base_model_id": BASE_MODEL_ID,
                "artist_prompts": ARTIST_PROMPTS,
            },
            ckpt_path,
        )
        print(f"✅ Saved UNet checkpoint: {ckpt_path}")

        # 5. Track best model + early stopping
        if val_loss < best_val_loss - EARLY_STOP_MIN_DELTA:
            best_val_loss = val_loss
            epochs_without_improvement = 0

            best_ckpt_path = CHECKPOINT_DIR / "sd_style_unet_best.pt"
            torch.save(
                {
                    "epoch": epoch,
                    "unet_state_dict": unet.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "train_loss": avg_train_loss,
                    "val_loss": val_loss,
                    "base_model_id": BASE_MODEL_ID,
                    "artist_prompts": ARTIST_PROMPTS,
                },
                best_ckpt_path,
            )
            print(f"🌟 New best model (val loss {best_val_loss:.4f}) saved to: {best_ckpt_path}")
        else:
            epochs_without_improvement += 1
            print(f"   No val improvement for {epochs_without_improvement} epoch(s).")

        # 6. Sample visuals
        if epoch % 5 == 0 or epoch == 1:
            try:
                print("Sampling with current fine-tuned UNet...")
                log_samples(pipe, epoch, num_per_artist=2)
            except Exception as e:
                print(f"⚠️ Sampling failed: {e}")

        # 7. Early stopping check
        if epochs_without_improvement >= EARLY_STOP_PATIENCE:
            print(f"\n⏹ Early stopping triggered after {epoch} epochs.")
            break

    print("\nTraining complete!")
    print(f"Checkpoints saved under: {CHECKPOINT_DIR}")
    print(f"Samples saved under:    {SAMPLES_DIR}")
    print(f"Best validation loss:   {best_val_loss:.4f}")


def main():
    train()


if __name__ == "__main__":
    main()
