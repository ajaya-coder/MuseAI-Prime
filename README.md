# 🎨 MuseAI-Prime  
### Identity-Preserving Artistic Portrait Stylization with Diffusion Models

MuseAI-Prime is a deep learning project that recreates human portrait images in the artistic vision of two famous painters, **Pablo Picasso** and **Rembrandt van Rijn**, all while preserving the subject’s facial identity. 

---

## ✨ Key Features

- 🎨 **Strong Style Transfer via Diffusion Models**
  - Fine-tunes the UNet of Stable Diffusion v1.5 for painter-specific aesthetics.
- 🧩 **Artist-Specific Conditioning**
  - Separate text prompts and learned representations for Picasso and Rembrandt.
- 🧑‍🦱 **Identity-Guided Img2Img Inference**
  - Uses FaceNet embeddings to select the most identity-preserving result.
- 🔁 **Data Augmentation**
  - Expands limited art datasets using classical image transformations.
- ⏹️ **Early Stopping**
  - Prevents overfitting and unnecessary compute during long diffusion training.
- 💻 **Reproducible CLI-based Workflow**
  - Entire pipeline is runnable from the terminal.

---

## 🧠 Technologies Used

- **PyTorch**
- **Stable Diffusion v1.5 (Diffusers)**
- **Hugging Face Transformers & Accelerate**
- **FaceNet (InceptionResnetV1)**
- **Torchvision**
- **CUDA (NVIDIA V100 GPUs)**
- **Python 3.10**
- **YAML-based configuration**

---

## 📁 Project Structure

```bash
MuseAI-Prime/
├── data/
│ ├── content/faces/raw/ # Input portraits / selfies
│ └── style_raw/
│ ├── picasso/ # Picasso training paintings
│ └── rembrandt/ # Rembrandt training paintings
│
├── src/
│ ├── preprocess/ # Data preprocessing & augmentation
│ ├── training/ # Diffusion fine-tuning
│ ├── inference/ # Identity-guided stylization
│ └── utils/
│
├── outputs/
│ └── sd_style_trained/
│ ├── checkpoints/ # Fine-tuned UNet weights
│ └── samples/ # Training sample grids
│
├── config.yaml
├── requirements.txt
└── README.md
```
---

## 🚀 Getting Started

MuseAI-Prime is designed to be run from the terminal using a dedicated Conda environment to ensure reproducibility and dependency stability.

### 🐍 Step 1: Create the Environment

First, create and activate a new Conda environment with Python 3.10:

```bash
conda create -n museai python=3.10
conda activate museai
pip install -r requirements.txt
```
---

### 🛠️ Step 2: Preprocessing

```bash
python src/preprocess/run_preprocessing.py
```
🔍 What this step does

- Loads raw portrait images and style paintings.

- Detects and crops faces (where applicable).

- Resizes images to Stable Diffusion’s native resolution (512×512).

- Applies classical data augmentation to expand the dataset:

  - random crops,
  - flips,
  - mild color jitter,
  - contrast and brightness shifts.

- Normalizes images and creates train/validation splits.

- Saves metadata used later during training.

---

### 🧪 Step 3: Training

```bash
python src/training/train_style_sd.py
```
⚙️ What this step does

- Loads a pre-trained Stable Diffusion v1.5 model.

- Freezes the VAE and text encoder.

- Fine-tunes only the UNet on Picasso and Rembrandt paintings.

- Conditions training using artist-specific text prompts.

- Uses mean squared error noise prediction loss.

- Monitors validation loss after every epoch.

- Applies Early Stopping when validation loss stops improving.

- Automatically saves:

  - per-epoch checkpoints,
  - the best model (sd_style_unet_best.pt),
  - visual sample grids for inspection.

---
  
### 🎨 Step 4: Stylization

Inference uses an img2img diffusion pipeline combined with identity scoring.

The system:

  - Generates multiple stylized candidates.
  - Computes FaceNet embeddings for each result.
  - Compares them to the original face.
  - Automatically selects the most identity-preserving output.

Picasso Stylization: 
```bash
python src/inference/img2img_style_identity.py \
  --checkpoint outputs/sd_style_trained/checkpoints/sd_style_unet_best.pt \
  --input_image data/content/faces/raw/"file-name".jpg \
  --artist picasso \
  --output_dir outputs/sd_results/picasso \
  --num_samples 6 \
  --strength 0.6 \
  --guidance_scale 8.0 \
  --steps 30
```

Rembrandt Stylization:
```bash
python src/inference/img2img_style_identity.py \
  --checkpoint outputs/sd_style_trained/checkpoints/sd_style_unet_best.pt \
  --input_image data/content/faces/raw/"file-name".jpg \
  --artist rembrandt \
  --output_dir outputs/sd_results/rembrandt \
  --num_samples 6 \
  --strength 0.55 \
  --guidance_scale 8.0 \
  --steps 30
```
