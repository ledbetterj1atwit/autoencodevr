# Image Compression Using Auto-Encoder Convolutional Neural Network

This repo contains my attempt to make an autoencoder for compresing frames from a vr game(specifically ChilloutVR)

## Contents

- evidence_images: comparisons between algorithims and a script to generate them.
- logs: tensorboard compatible logs of some trainings
- model_checkpoints: saved weights for models
  - Model<letter>: The model.
    - <model-name>.data: The checkpoint.
    - <model-name>.index: Also part of the checkpoint.
    - model: The model's set-up.
- model_to_load: Put a checkpoint here to load it
- processed: (not here) Load dataset from here
- results: (not here) Generated result images
- autoencode.py: The main file, trains and validates the model.
- dataset.zip: A subsection of the dataset, smaller resolution.
- image_preprocess.py: Just halves the resolution of images, used to make dataset.
- notes.md: Some observations.

## Introduction

