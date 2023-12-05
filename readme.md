# Image Compression Using Auto-Encoder Convolutional Neural Network

This repo contains my attempt to make an autoencoder for compresing frames from a vr game(specifically ChilloutVR)

## Contents

- evidence_images: comparisons between algorithims and a script to generate them.
- logs: tensorboard compatible logs of some trainings
- model_checkpoints: saved weights for models
  - Model\<letter>: The model.
    - \<model-name>.data: The checkpoint.
    - \<model-name>.index: Also part of the checkpoint.
    - model: The model's set-up.
- model_to_load: Put a checkpoint here to load it
- processed: (not here) Load dataset from here
- results: (not here) Generated result images
- autoencode.py: The main file, trains and validates the model.
- dataset.zip: A subsection of the dataset, smaller resolution.
- image_preprocess.py: Just halves the resolution of images, used to make dataset.
- notes.md: Some observations.

## Dataset

it's [here](https://drive.google.com/file/d/1Y3DT43NK8L71z5UCDgO07CBggp0fGMuG/view?usp=drive_link)

## Introduction

Admitidly, most of the following is from my paper for my Principals of Machine Learning.

Virtual Reality (VR) is a well-known technology that is still actively being worked on.
In particular are constraints around processing, since VR applications need to render two high-resolution images from slightly different perspectives to create depth at a speed that does not feel “choppy,” causing motion sickness.
At the same time, physical constraints either require the user to tether their headset to their computer or to put a mobile computer on the headset itself.
Some users, however, find a compromise in wireless streaming applications like virtual desktop or Oculus AirLink.
These applications allow the user to have a tether-less experience while also having a full-fledged computer to run the VR application.
Often an issue is transferring the high resolution, that is, large file size,application frames over an at times unoptimal wireless network.
Compression is often used to reduce the file-size so less has to be transferred over the network, and so transfers per image are faster, assuming that compression and decompression times are not larger than the gained transfer speedup.
Image compression comes in two common forms, lossy and lossless.
Typically, lossy algorithms produce smaller files at the cost of some artifacting.

Machine learning (ML) approaches to image compression are uncommon and yet surprisingly effective.
The benefit ML approaches have is that they can learn how best to compress images in ways that may not be intuitive for a human to understand.
Additionally, they can be trained to work in specific environments with specific data, allowing them to work better than approaches that are vastly more general.
The model used in this paper is a Convolutional Neural Network (CNN).
This type of Neural Network works well on image data.

The goal of this research is to create an ML model that can compress images specifically for wireless VR streaming applications.
This requires that models are simultaneously, visually accurate, well compressed, and are compressed and decompressed quickly.
For this, I used a CNN set up as an auto-encoder.
Auto-encoders are Neural Networks(NN) that reduce the size of the data moving through them toward the middle and expand out again to the original size of the output to force the model to generalize.
This shaping is good for compression since it will cause the model to train both shrinking the file and also unshrinking it in the same model.
The model then only needs to focus on maintaining the original data at the output.

## Selection of Data

I just recorded 2 hours of chilloutvr play.
I used SteamVR's view of both eyes to propperly emulate the user's view.
Each frame was saved as a 1920x1080 pixel frame with 3 channels, saved to png format to avoid lossy compression.
The largest problem with this data was having way too much.
I recoreded at 5 frames per second and still ended up with 40Gb of data.
This was after editiing to remove frames that might have issues, namely those containing my desktop overlay.
I then halved the resolution to 960x540 to make it faster to train the model(faster mainly).

## Methods

I made several models, each model tuning to better balance between compression, time and quality.
The most important statistic is encode/decode time as latency is one of the biggest causes of motion sickness and general discomfort in wireless setups.
Quality is secondary as images that are recognisable enough are more important the recreating detail exactly.
The overall picture is more important than small details in most cases, though text specifically is an important detail since it is used to convey much application information to the user.
Last is size, since any reduction is useful and the user can be asked to select

Models were built off of a tutorial [from the keras developers](https://keras.io/examples/vision/autoencoder/).
The first model compressed images to a tiny size(4% of the original image) but was significantly worse at image accuracy and quality.
The second model was set up to not compress as small, but struggled with text as it had few filters to work with.
The third was given more filters and is considered the best model.
The fourth model used a different loss function but ultimately took longer to train with no discernible benefit.
All models took roughly the same time with encode/decode(10ms full trip).

Figure 1: Best Model Architecture ![Best Model Architecture](https://i.ibb.co/x5LPZKB/Model-Diag.png)

The best model was set up as seen in figure 1.
The encoder uses convolutional layers that have progressively fewer filters and progressively smaller kernels.
Each filter is a matrix that slides over each pixel in the input image, creating a new image with the same dimensions that has been transformed.
Between some layers are max pooling layers that take the max value of a sliding window
The decoder mirrors the encoder using convolutional transpose layers and upsampling to undo the pooling and convolutional layers.
However, it does not undo the encoder's final max pooling layer, and ends with a final convolutional transpose layer(3 filters\[one per channel], 5x5 kernel size, sigmoid activation).

Models were set up using tensorflow and keras.
Models were saved at regular intervals during training with tensorflow's checkpointing system.
As well, logs were kept using tensorflow's tensorboard callback, allowing me to look at training progress graphically.
Matplotlib allowed for the creation of plots to save and compare images from the dataset before and after encode/decode.

## Results

The following results are for the best model, which was described in Figure 1.
Compression ratio describes how much the image is compressed as a ratio of the original image's size.
A ratio of 0.25 means the compressed image is 1/4th the size of the original.
The best model compresses a 12441.6 kb image (540*960*3 tensor of 32-bit floats) to 2073.6 kb (135*240*8 tensor of 32-bit floats) for a compression ratio of 0.1666.
That's 1/6th the original size.

The model used MSE to calculate loss as it was the best loss function I was aware of at the time.
During training, the MSE went from 0.0026 initially to 0.0009 at the end of training.
This means the model is roughly 34% better than just scaling down and scaling up the image (which is what an untrained model is effectively doing).
The validation MSE was 0.0006.
While validation loss was lower than training loss, validation loss varied wildly per epoch.

The model takes on average 12.1 ms to inference through the whole model, making encode and decode roughly 6.05 ms each.
For comparison, [Oren Rippel and Lubomir Bourdev](https://arxiv.org/abs/1705.05823) found in their paper, Real-Time Adaptive Image Compression, that JPEG takes 18.6ms to encode and 13.0ms to decode and that WebP takes 67.0ms to encode and 83.7 ms to decode.
For an application running at 60 frames per second, the application needs to generate a frame every 16.6ms so jpeg and webp cause significant latency per frame.

## Discussion

This model is not very good, it needs to be changed, it should use a different loss fucntion(probably PSNR?), it could use a layer or regular neurons that flattens the images, something needs to be. 
The model does not handle text well at all.
While images overall look okay, it's much more artifacted than other algorithms.
The only benefit that the autoencoder adds is that it's encode and decode times are small.
A better model can likely be created, as other models exist that do better than mine. 
I am admittedly not familiar with how best to create a convolutional neural net.

## Summary

Autoencoder CNNs can provide great performance boost, but they need to be carefully tuned, something I'm not the best at.