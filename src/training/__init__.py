"""
Training utilities for MuseAI-Prime.

This package currently provides:
- A basic conditional diffusion (DDPM-style) trainer over style images
  in `train_style_sd.py`.

Usage (from project root):

    python -m src.training.train_style_sd

or

    python src/training/train_style_sd.py

Make sure you have:
- data/style_raw/picasso/*.jpg (or png, jpeg, etc.)
- data/style_raw/rembrandt/*.jpg

These will be used to learn a diffusion prior over each artist's style.
"""
