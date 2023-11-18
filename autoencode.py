import glob
from math import ceil
import tensorflow as tf
import PIL
import PIL.Image
import keras
from keras import layers
import numpy as np
import matplotlib.pyplot as plt

# Declare some useful variables.
batch_size = 2  # No. images per batch
img_height = 540  # Height per image.
img_width = 960  # Width per image.
img_channels = 3  # Channels per image(RGB).
img_count = int(38458 / 8)  # No. of images in dataset
epochs = 20  # Times to run through train data and train.
save_freq = 2  # How often to save(epochs)


# Load dataset.
def decode_img(img_fp):
    """
    Takes an image filepath and loads the image data for the Dataset.

    Assumes png.
    """
    img = tf.io.read_file(img_fp)
    img = tf.io.decode_png(img, channels=img_channels)
    img = tf.image.resize(img, [img_height, img_width]) / 255  # Resizing transforms image data from uint(32b) to
    # float32, in range [0,255] but needs to be float32 range [0,1]. Thus, the /255.
    return img, img  # Dataset needs a tuple of input and label, for compression, the label is the input itself.


images = tf.data.Dataset.list_files("./processed/*").shuffle(38458, reshuffle_each_iteration=False)
val_size = int(img_count * 0.2)  # Grab 20% for validation and 80% for training.
vr_images_train = images.take(img_count - val_size)
vr_images_test = images.take(val_size)

vr_images_train = vr_images_train.map(decode_img, num_parallel_calls=tf.data.AUTOTUNE).batch(batch_size).prefetch(
    tf.data.AUTOTUNE)
vr_images_test = vr_images_test.map(decode_img, num_parallel_calls=tf.data.AUTOTUNE).batch(batch_size).prefetch(
    tf.data.AUTOTUNE)

# Setup model
input_shape = (img_height, img_width, img_channels)  # The input shape, (h,w,c)

encoder = keras.Sequential(  # All layers here must be "undone" in the decoder.
    [
        layers.InputLayer(input_shape=input_shape),  # in
        layers.Conv2D(16, (7, 7), activation='relu', padding='same'),
        layers.MaxPooling2D((2, 2), padding='same'),  # Pool, shrinks data by (2,2)
        layers.Conv2D(8, (5, 5), activation='relu', padding='same'),
        layers.MaxPooling2D((2, 2), padding='same'),  # Consider removing to compress less.
        layers.Conv2D(8, (3, 3), activation='relu', padding='same'),
        layers.MaxPooling2D((2, 2), padding='same')  # Middle, "Latent space"
    ]
)

decoder = keras.Sequential(
    [
        layers.Conv2DTranspose(8, (3, 3), activation='relu', padding='same'),  # Undo Convolution.
        layers.UpSampling2D((2, 2)),  # Upscale by same factor as pooling.
        layers.Conv2DTranspose(8, (5, 5), activation='relu', padding='same'),
        layers.UpSampling2D((2, 2)),
        layers.Conv2DTranspose(16, (7, 7), activation='relu', padding='same'),
        layers.UpSampling2D((2, 2)),
        layers.Conv2DTranspose(3, (5, 5), activation='sigmoid', padding='same'),
        # Convert convolutions to rbg bitplanes.
        layers.Cropping2D((2, 0))  # Crop to fit input dimensions. Don't know why I have to, but I do.
    ]
)

autoencoder = keras.Sequential(encoder.layers + decoder.layers)  # Combine encoder and decoder for training.
autoencoder.compile(optimizer='adam',
                    loss=keras.losses.MSE)  # Loss is MSE(bincrossentropy is usually described as worse???)

# Set up checkpointing
checkpoint_path = "model_checkpoints/autoencode"
chpt_cb = keras.callbacks.ModelCheckpoint(
    filepath=checkpoint_path+".ckpt",
    save_weights_only=True,
    verbose=1,
    save_freq=save_freq * ceil((img_count * .8) / batch_size)  # Every x (epochs)
)

if len(glob.glob("model_to_load/*")) != 0:  # Only load checkpoint in [./model_to_load]
    autoencoder.load_weights("model_to_load/" + checkpoint_path)

# Calculate current compression.
uncomp_byt = (input_shape[-3] * input_shape[-2] * input_shape[-1]) * 8  # (h,w,c) -> to bytes
comp_shape = encoder.layers[-1].output_shape
comp_byt = (comp_shape[-3] * comp_shape[-2] * comp_shape[-1]) * 8
print(f"Model compression: {uncomp_byt / 1000} kb -> {comp_byt / 1000} kb\nCompression ratio: {comp_byt / uncomp_byt}")

# Train
autoencoder.fit(
    vr_images_train,
    epochs=epochs,
    validation_data=vr_images_test,
    callbacks=[chpt_cb]  # Don't forget to save :)
)

autoencoder.save_weights(checkpoint_path+".done.ckpt")

# Show results.
n = 10  # number of images to show.
to_decode = vr_images_test.unbatch().take(n)
originals = [i[0] for i in to_decode]
for i in range(n):
    plt.figure()
    ax = plt.subplot(2, 2, 1)
    ax.get_xaxis().set_visible(False)  # Disable axes
    ax.get_yaxis().set_visible(False)
    ax.margins(tight=True)  # Tighen Margins
    ax.set_title('Original', fontstyle='oblique', fontfamily='serif', fontsize='medium')
    plt.imshow(originals[i].numpy().reshape(img_height, img_width, img_channels))

    ax = plt.subplot(2, 2, 2)
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)
    ax.margins(tight=True)
    ax.set_title('AI Compressed', fontstyle='oblique', fontfamily='serif', fontsize='medium')
    plt.imshow(
        autoencoder.predict(tf.reshape(originals[i], [1, img_height, img_width, img_channels])).reshape(img_height,
                                                                                                        img_width,
                                                                                                        img_channels))
    plt.savefig(f"./results/out{i}.png", dpi=500, bbox_inches='tight')
