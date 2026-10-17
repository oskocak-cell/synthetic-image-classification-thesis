#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# 1- Imports and Parameters

import os
import math
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")  # non-interactive backend for GPU server
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras import (
    layers, models, optimizers, callbacks, metrics, losses
)
from scipy.linalg import sqrtm

# Reproducibility
SEED = 42
tf.random.set_seed(SEED)
np.random.seed(SEED)

# Data
# PathMNIST: 28x28 RGB, 9 classes, imbalanced, pre-split by MedMNIST
IMAGE_SIZE = 28
CHANNELS   = 3
CLASSES    = 9

# Model
NOISE_EMBEDDING_SIZE = 32
CLASS_EMBEDDING_SIZE = 32  # same dimensionality as noise embedding to keep them symmetric when concatenated into U-Net
EMA = 0.999

# Training
BATCH_SIZE   = 64
WEIGHT_DECAY = 1e-4
EPOCHS       = 500
FID_FREQ     = 25   # compute FID every this many epochs
DIFFUSION_STEPS = 20  # steps used during FID generation and final generation

# Generation
GEN_BATCH_SIZE = 128

# Hyperparameter search grid
LR_OPTIONS       = [1e-4, 2e-4, 3e-4]
SCHEDULE_OPTIONS = ["linear", "offset_cosine"]

# Paths
PROJECT_DIR = "/path/to/thesis_project"
DATA_DIR    = PROJECT_DIR  # .npz files are directly here on GPU
SYNTH_DIR   = os.path.join(PROJECT_DIR, "data", "synthetic", "seed_42", "ddim", "path_mnist")
MODEL_DIR   = os.path.join(PROJECT_DIR, "models", "ddim", "path_mnist", "seed_42")

os.makedirs(SYNTH_DIR, exist_ok=True)
os.makedirs(MODEL_DIR, exist_ok=True)


# In[ ]:


# 2- Data Loading and Preprocessing

# Load preprocessed splits
# PathMNIST uses official MedMNIST pre-defined splits — not our stratified split.
# train: 89996, val: 10004, test: 7180 (test untouched, reserved for classification)
data    = np.load(os.path.join(DATA_DIR, "path_mnist.npz"))
X_train = data["X_train"]  # uint8, (89996, 28, 28, 3)
y_train = data["y_train"]  # int,   (89996,)
X_val   = data["X_val"]    # uint8, (10004, 28, 28, 3)
y_val   = data["y_val"]    # int,   (10004,)

print(f"X_train: {X_train.shape}, y_train: {y_train.shape}")
print(f"X_val:   {X_val.shape},   y_val:   {y_val.shape}")

# Normalize uint8 [0, 255] -> float32 [0.0, 1.0]
# DDIM operates in [0,1] unlike cGAN which uses [-1,1] with tanh output.
# Labels kept as integers — fed into a learned Embedding layer for class conditioning.
def preprocess(image, label):
    image = tf.cast(image, tf.float32) / 255.0
    return image, label


# In[ ]:


# 3- Diffusion Schedules
# Two schedules are evaluated in the hyperparameter search:
# 1- Linear schedule from Ho et al. (DDPM, 2020)
# 2- Offset cosine schedule from Nichol and Dhariwal (Improved DDPM, 2021)
# Offset cosine schedule prevents the schedule from spending too many steps near pure noise,
# where signal is already negligible. Controlled by offset s=0.008.

def linear_diffusion_schedule(diffusion_times):
    beta_1 = 1e-4
    beta_T = 0.02
    # Multiply by 1000 to match discrete DDPM scaling where t in [1,1000].
    # Without this, max noise at t=1 is only ~10% — too low for generation
    # which starts from pure noise.
    log_alpha_bar = -1000.0 * (
        beta_1 * diffusion_times +
        0.5 * (beta_T - beta_1) * diffusion_times ** 2
    )
    alpha_bars   = tf.exp(log_alpha_bar)
    signal_rates = tf.sqrt(tf.clip_by_value(alpha_bars, 1e-8, 1.0))
    noise_rates  = tf.sqrt(tf.clip_by_value(1.0 - alpha_bars, 1e-8, 1.0))
    return noise_rates, signal_rates


def offset_cosine_diffusion_schedule(diffusion_times):
    # Nichol and Dhariwal (2021), Eq. 17, offset s=0.008 prevents
    # alpha_bar from starting too close to 1 at t=0.
    s = 0.008
    alpha_bars = tf.cos((diffusion_times + s) / (1.0 + s) * math.pi / 2.0) ** 2
    # Normalise so alpha_bar_0 = 1. Ensures the cosine schedule starts at exactly 1.
    alpha_bars = alpha_bars / tf.cos(tf.cast(s / (1.0 + s) * math.pi / 2.0, tf.float32)) ** 2
    # clip_by_value on both rates prevents numerical issues at the boundaries t=0 and t=1.
    signal_rates = tf.sqrt(tf.clip_by_value(alpha_bars, 1e-8, 1.0))
    noise_rates  = tf.sqrt(tf.clip_by_value(1.0 - alpha_bars, 1e-8, 1.0))
    return noise_rates, signal_rates


# Map schedule name (string) to function -> used inside search loop
SCHEDULE_FN = {
    "linear":        linear_diffusion_schedule,
    "offset_cosine": offset_cosine_diffusion_schedule,
}


# In[ ]:


# 4- U-Net with Class Conditioning
# For the U-Net here we have a different structure:
# the network processes spatial feature maps at every layer,
# so class info should be injected in a way spatially compatible.
#
# PathMNIST-specific architecture note:
# IMAGE_SIZE=28. With 3 DownBlocks (pool_size=2): 28->14->7->3 — bottleneck too small.
# Reduced to 2 DownBlocks: 28->14->7, bottleneck at 7x7, 2 UpBlocks back to 28x28.

# 4a- Sinusoidal noise embedding
# Encodes the scalar noise variance as a high-dimensional vector so the
# network can distinguish noise levels. Kept exactly from the baseline.
def sinusoidal_embedding(x):
    frequencies = tf.exp(
        tf.linspace(
            tf.math.log(1.0),
            tf.math.log(1000.0),
            NOISE_EMBEDDING_SIZE // 2,
        )
    )
    angular_speeds = 2.0 * math.pi * frequencies
    embeddings = tf.concat(
        [tf.sin(angular_speeds * x), tf.cos(angular_speeds * x)], axis=3
    )
    return embeddings


# 4b- Residual/Down/Up blocks
# Kept from baseline. BatchNormalization(center=False, scale=False) ->
# acts as pure standardisation — learnable shift/scale removed to keep
# it lightweight and stable inside residual branches.
def ResidualBlock(width):
    def apply(x):
        input_width = x.shape[3]
        if input_width == width:
            residual = x
        else:
            residual = layers.Conv2D(width, kernel_size=1)(x)
        x = layers.BatchNormalization(center=False, scale=False)(x)
        x = layers.Conv2D(
            width, kernel_size=3, padding="same", activation="swish"
        )(x)
        x = layers.Conv2D(width, kernel_size=3, padding="same")(x)
        x = layers.Add()([x, residual])
        return x
    return apply


def DownBlock(width, block_depth):
    def apply(x):
        x, skips = x
        for _ in range(block_depth):
            x = ResidualBlock(width)(x)
            skips.append(x)
        x = layers.AveragePooling2D(pool_size=2)(x)
        return x
    return apply


def UpBlock(width, block_depth):
    def apply(x):
        x, skips = x
        x = layers.UpSampling2D(size=2, interpolation="bilinear")(x)
        for _ in range(block_depth):
            x = layers.Concatenate()([x, skips.pop()])
            x = ResidualBlock(width)(x)
        return x
    return apply


# 4c- U-Net builder with class conditioning
def build_unet():
    # Inputs
    noisy_images    = layers.Input(shape=(IMAGE_SIZE, IMAGE_SIZE, CHANNELS),
                                   name="noisy_images")
    noise_variances = layers.Input(shape=(1, 1, 1),
                                   name="noise_variances")
    class_labels    = layers.Input(shape=(),
                                   dtype=tf.int32, name="class_labels")

    # Noise embedding (sinusoidal)
    # noise_variances: (B,1,1,1) -> embedding: (B,1,1,NOISE_EMBEDDING_SIZE)
    # Upsampled to (B, IMAGE_SIZE, IMAGE_SIZE, NOISE_EMBEDDING_SIZE)
    noise_emb = layers.Lambda(sinusoidal_embedding,
                               name="sinusoidal_embedding")(noise_variances)
    noise_emb = layers.UpSampling2D(size=IMAGE_SIZE,
                                     interpolation="nearest")(noise_emb)

    # Class embedding (learned)
    # integer label -> (B, CLASS_EMBEDDING_SIZE) -> (B,1,1,CLASS_EMBEDDING_SIZE)
    # Upsampled to (B, IMAGE_SIZE, IMAGE_SIZE, CLASS_EMBEDDING_SIZE)
    # Symmetric to noise embedding: same broadcast mechanism, same spatial size.
    # The U-Net sees class information at every pixel from the very first layer.
    class_emb = layers.Embedding(CLASSES, CLASS_EMBEDDING_SIZE,
                                  name="class_embedding")(class_labels)
    class_emb = layers.Reshape((1, 1, CLASS_EMBEDDING_SIZE))(class_emb)
    class_emb = layers.UpSampling2D(size=IMAGE_SIZE,
                                     interpolation="nearest")(class_emb)

    # Initial projection + concatenate all conditioning
    # Project image to 32 channels, then concatenate noise + class embeddings.
    # CLASS_EMBEDDING_SIZE = NOISE_EMBEDDING_SIZE = 32:
    # symmetric dimensionality, no one dominates the other in the concatenation.
    x = layers.Conv2D(32, kernel_size=1, name="input_proj")(noisy_images)
    x = layers.Concatenate(name="conditioning_concat")(
        [x, noise_emb, class_emb]
    )

    # Encoder (downsampling path)
    # 2 DownBlocks for IMAGE_SIZE=28: 28->14->7 (bottleneck at 7x7)
    # 3 DownBlocks would give 28->14->7->3 — bottleneck too small
    skips = []
    x = DownBlock(32, block_depth=2)([x, skips])
    x = DownBlock(64, block_depth=2)([x, skips])

    # Bottleneck at 7x7
    x = ResidualBlock(96)(x)
    x = ResidualBlock(96)(x)

    # Decoder (upsampling path)
    x = UpBlock(64, block_depth=2)([x, skips])
    x = UpBlock(32, block_depth=2)([x, skips])

    # Output projection
    # kernel_initializer="zeros": predicts zero noise at init -> stable start
    outputs = layers.Conv2D(CHANNELS, kernel_size=1,
                             kernel_initializer="zeros",
                             name="output_proj")(x)

    return models.Model(
        [noisy_images, noise_variances, class_labels],
        outputs,
        name="unet"
    )


# In[ ]:


# 5- DiffusionModel class

# diffusion_schedule_fn lets the search loop instantiate a fresh model
# with a different schedule per run without any global variable changes.
# ema_network is built directly from build_unet() and set to trainable=False.

class DiffusionModel(models.Model):
    def __init__(self, diffusion_schedule_fn):
        super().__init__()
        self.normalizer          = layers.Normalization()
        self.network             = build_unet()
        self.ema_network         = build_unet()
        self.ema_network.trainable = False
        self.diffusion_schedule  = diffusion_schedule_fn
        # Initialise ema_network with same weights as network.
        # build_unet() returns a functional model so weights exist immediately.
        self.ema_network.set_weights(self.network.get_weights())

    def compile(self, **kwargs):
        super().compile(**kwargs)
        self.noise_loss_tracker = metrics.Mean(name="n_loss")

    @property
    def metrics(self):
        return [self.noise_loss_tracker]

    def denormalize(self, images):
        # Reverses normalizer.adapt() standardisation back to [0,1]
        images = self.normalizer.mean + images * self.normalizer.variance ** 0.5
        return tf.clip_by_value(images, 0.0, 1.0)

    # denoise() takes class_labels and passes it to the network
    # with noisy_images and noise_rates**2
    def denoise(self, noisy_images, noise_rates, signal_rates, class_labels, training):
        network = self.network if training else self.ema_network
        pred_noises = network(
            [noisy_images, noise_rates ** 2, class_labels], training=training
        )
        pred_images = (noisy_images - noise_rates * pred_noises) / signal_rates
        return pred_noises, pred_images

    def reverse_diffusion(self, initial_noise, diffusion_steps, class_labels):
        num_images     = tf.shape(initial_noise)[0]
        step_size      = 1.0 / diffusion_steps
        current_images = initial_noise
        for step in range(diffusion_steps):
            diffusion_times           = tf.ones((num_images, 1, 1, 1)) - step * step_size
            noise_rates, signal_rates = self.diffusion_schedule(diffusion_times)
            pred_noises, pred_images  = self.denoise(
                current_images, noise_rates, signal_rates, class_labels,
                training=False
            )
            next_diffusion_times                = diffusion_times - step_size
            next_noise_rates, next_signal_rates = self.diffusion_schedule(
                next_diffusion_times
            )
            current_images = (
                next_signal_rates * pred_images + next_noise_rates * pred_noises
            )
        return pred_images

    # generate() takes integer class_labels tensor.
    # Caller constructs it per class for controlled generation.
    def generate(self, num_images, diffusion_steps, class_labels):
        initial_noise    = tf.random.normal(
            shape=(num_images, IMAGE_SIZE, IMAGE_SIZE, CHANNELS)
        )
        generated_images = self.reverse_diffusion(
            initial_noise, diffusion_steps, class_labels
        )
        generated_images = self.denormalize(generated_images)
        return generated_images

    # tf.shape(images)[0] for dynamic batch size —
    # fixes the baseline hardcoded BATCH_SIZE bug.
    def train_step(self, data):
        images, class_labels = data
        images = self.normalizer(images, training=True)

        noises = tf.random.normal(
            shape=(tf.shape(images)[0], IMAGE_SIZE, IMAGE_SIZE, CHANNELS)
        )
        diffusion_times = tf.random.uniform(
            shape=(tf.shape(images)[0], 1, 1, 1), minval=0.0, maxval=1.0
        )
        noise_rates, signal_rates = self.diffusion_schedule(diffusion_times)
        noisy_images = signal_rates * images + noise_rates * noises

        with tf.GradientTape() as tape:
            pred_noises, pred_images = self.denoise(
                noisy_images, noise_rates, signal_rates, class_labels,
                training=True
            )
            noise_loss = self.loss(noises, pred_noises)

        gradients = tape.gradient(noise_loss, self.network.trainable_weights)
        self.optimizer.apply_gradients(
            zip(gradients, self.network.trainable_weights)
        )
        self.noise_loss_tracker.update_state(noise_loss)

        # EMA update — keeps a smoothed copy of network weights
        for weight, ema_weight in zip(
            self.network.weights, self.ema_network.weights
        ):
            ema_weight.assign(EMA * ema_weight + (1 - EMA) * weight)

        return {m.name: m.result() for m in self.metrics}

    def test_step(self, data):
        images, class_labels = data
        images = self.normalizer(images, training=False)

        # Dynamic batch size — fixes baseline bug where BATCH_SIZE was hardcoded
        batch_size = tf.shape(images)[0]
        noises = tf.random.normal(
            shape=(batch_size, IMAGE_SIZE, IMAGE_SIZE, CHANNELS)
        )
        diffusion_times = tf.random.uniform(
            shape=(batch_size, 1, 1, 1), minval=0.0, maxval=1.0
        )
        noise_rates, signal_rates = self.diffusion_schedule(diffusion_times)
        noisy_images = signal_rates * images + noise_rates * noises

        pred_noises, pred_images = self.denoise(
            noisy_images, noise_rates, signal_rates, class_labels,
            training=False
        )
        noise_loss = self.loss(noises, pred_noises)
        self.noise_loss_tracker.update_state(noise_loss)

        return {m.name: m.result() for m in self.metrics}


# In[ ]:


# 6- Callbacks

# 6a- FID helper
from scipy.linalg import sqrtm

def compute_fid(model, X_val, y_val, inception, real_val_features,
                diffusion_steps=DIFFUSION_STEPS, batch_size=64):
    # real_val_features: pre-computed InceptionV3 features of X_val (in cell 7)
    # Passed in to avoid recomputing every FID call.
    # real_val_features is a NumPy array and is unaffected by clear_session().

    def get_inception_features(images, batch_size=64):
        # images expected in [0, 255] float32
        # Resize in batches to avoid int32 overflow on large val sets
        all_features = []
        for i in range(0, len(images), batch_size):
            batch = images[i:i + batch_size]
            if batch.shape[-1] == 1:  # FashionMNIST grayscale -> RGB
                batch = np.repeat(batch, 3, axis=-1)
            batch = tf.image.resize(batch, (299, 299))
            batch = tf.keras.applications.inception_v3.preprocess_input(batch)
            feats = inception.predict(batch, verbose=0)
            all_features.append(feats)
        return np.concatenate(all_features, axis=0)

    # Generate synthetic images matching y_val class distribution
    synthetic_images = []
    for class_idx in range(CLASSES):
        class_mask = y_val == class_idx
        n_samples  = int(np.sum(class_mask))
        if n_samples == 0:
            continue
        class_generated = []
        n_batches = math.ceil(n_samples / GEN_BATCH_SIZE)
        for batch_idx in range(n_batches):
            batch_start  = batch_idx * GEN_BATCH_SIZE
            batch_end    = min(batch_start + GEN_BATCH_SIZE, n_samples)
            n_gen        = batch_end - batch_start
            class_labels = tf.fill((n_gen,), tf.cast(class_idx, tf.int32))
            generated    = model.generate(
                num_images=n_gen,
                diffusion_steps=diffusion_steps,
                class_labels=class_labels,
            )
            generated = (generated.numpy() * 255.0).clip(0, 255).astype("float32")
            class_generated.append(generated)
        synthetic_images.append(np.concatenate(class_generated, axis=0))
    synthetic_images   = np.concatenate(synthetic_images, axis=0)
    synthetic_features = get_inception_features(synthetic_images)

    # FID computation
    mu_real     = np.mean(real_val_features, axis=0)
    mu_synth    = np.mean(synthetic_features, axis=0)
    sigma_real  = np.cov(real_val_features, rowvar=False)
    sigma_synth = np.cov(synthetic_features, rowvar=False)
    diff        = mu_real - mu_synth

    eps     = 1e-6
    covmean = sqrtm(
        (sigma_real + eps * np.eye(sigma_real.shape[0])) @
        (sigma_synth + eps * np.eye(sigma_synth.shape[0]))
    )
    if np.iscomplexobj(covmean):
        covmean = covmean.real

    fid = np.dot(diff, diff) + np.trace(sigma_real + sigma_synth - 2 * covmean)
    return max(0.0, float(fid))


# 6b- CheckpointAndFID — used during hyperparameter search
class CheckpointAndFID(callbacks.Callback):
    def __init__(self, save_dir, inception, X_val, y_val,
                 real_val_features, fid_freq=FID_FREQ):
        self.save_dir          = save_dir
        self.inception         = inception
        self.X_val             = X_val
        self.y_val             = y_val
        self.real_val_features = real_val_features  # pre-computed, passed in
        self.fid_freq          = fid_freq
        self.fid_scores        = {}  # {epoch: fid_score}
        os.makedirs(save_dir, exist_ok=True)

    def on_epoch_end(self, epoch, logs=None):
        if (epoch + 1) % self.fid_freq == 0:
            # Save both network and ema_network weights
            checkpoint_dir = os.path.join(self.save_dir, f"epoch_{epoch+1:03d}")
            os.makedirs(checkpoint_dir, exist_ok=True)
            self.model.network.save_weights(
                os.path.join(checkpoint_dir, "network.weights.h5")
            )
            self.model.ema_network.save_weights(
                os.path.join(checkpoint_dir, "ema_network.weights.h5")
            )

            # Compute FID
            fid = compute_fid(
                model=self.model,
                X_val=self.X_val,
                y_val=self.y_val,
                inception=self.inception,
                real_val_features=self.real_val_features,
            )
            self.fid_scores[epoch + 1] = fid
            print(f"\n  -> Epoch {epoch+1}: FID = {fid:.4f}")


# 6c- SaveModels — used during final training only
class SaveModels(callbacks.Callback):
    """Saves weights + a generated image grid every save_freq epochs.
    Used for final training only — no FID computed here."""

    def __init__(self, save_dir, save_freq=50):
        self.save_dir  = save_dir
        self.save_freq = save_freq
        os.makedirs(save_dir, exist_ok=True)

    def on_epoch_end(self, epoch, logs=None):
        if (epoch + 1) % self.save_freq == 0:
            checkpoint_dir = os.path.join(self.save_dir, f"epoch_{epoch+1:03d}")
            os.makedirs(checkpoint_dir, exist_ok=True)

            # Save weights
            self.model.network.save_weights(
                os.path.join(checkpoint_dir, "network.weights.h5")
            )
            self.model.ema_network.save_weights(
                os.path.join(checkpoint_dir, "ema_network.weights.h5")
            )

            # Generate and save image grid — 5 images per class
            # PathMNIST is RGB — standard imshow, no squeeze or cmap needed
            fig, axes = plt.subplots(CLASSES, 5, figsize=(10, CLASSES * 2))
            for class_idx in range(CLASSES):
                class_labels = tf.fill((5,), tf.cast(class_idx, tf.int32))
                generated    = self.model.generate(
                    num_images=5,
                    diffusion_steps=DIFFUSION_STEPS,
                    class_labels=class_labels,
                )
                # Denormalize [0,1] -> [0,255] uint8 for display
                generated = (generated.numpy() * 255.0).clip(0, 255).astype("uint8")
                for img_idx in range(5):
                    axes[class_idx, img_idx].imshow(generated[img_idx])
                    axes[class_idx, img_idx].axis("off")
                    if img_idx == 0:
                        axes[class_idx, img_idx].set_ylabel(
                            f"Class {class_idx}", fontsize=8, rotation=90
                        )
            plt.suptitle(f"Epoch {epoch + 1}", fontsize=10)
            plt.tight_layout()
            plt.savefig(
                os.path.join(
                    checkpoint_dir, f"generated_epoch_{epoch+1:03d}.png"
                ),
                dpi=100, bbox_inches="tight"
            )
            plt.close()


# In[ ]:


# 7- Hyperparameter Search Loop

# 7a- Pre-compute real val InceptionV3 features once
# Deleted after use to free GPU memory. real_val_features is a plain
# numpy array and survives clear_session() unaffected.
# PathMNIST is RGB — no grayscale-to-RGB conversion needed here.
_inception_init = tf.keras.applications.InceptionV3(
    include_top=False, pooling="avg", input_shape=(299, 299, 3)
)

# Batch the resize to avoid memory spike from resizing all at once.
# PathMNIST val: 10004 x 28x28x3
_val_ds = tf.data.Dataset.from_tensor_slices(X_val.astype("float32"))
_val_ds = (
    _val_ds
    .batch(64)
    .map(lambda x: tf.keras.applications.inception_v3.preprocess_input(
        tf.image.resize(x, (299, 299))
    ), num_parallel_calls=tf.data.AUTOTUNE)
    .prefetch(tf.data.AUTOTUNE)
)
real_val_features = _inception_init.predict(_val_ds, verbose=0)

del _inception_init
del _val_ds
tf.keras.backend.clear_session()
import gc; gc.collect()
print(f"Real val features cached once: {real_val_features.shape}")  # (10004, 2048)

# 7b- Search loop
search_results = []

for lr in LR_OPTIONS:
    for schedule_name in SCHEDULE_OPTIONS:
        run_name = f"lr{lr}_sched{schedule_name}"
        print(f"\n{'='*60}")
        print(f"Starting run: {run_name}")
        print(f"{'='*60}")

        # Run-specific directory
        run_model_dir = os.path.join(MODEL_DIR, "search", run_name)
        os.makedirs(run_model_dir, exist_ok=True)

        # Clear session — ensures no weight leakage across runs
        tf.keras.backend.clear_session()

        # Reinstantiate inception fresh on clean graph
        inception = tf.keras.applications.InceptionV3(
            include_top=False, pooling="avg", input_shape=(299, 299, 3)
        )

        # Build train dataset
        train_dataset = tf.data.Dataset.from_tensor_slices((X_train, y_train))
        train_dataset = (
            train_dataset
            .shuffle(buffer_size=len(X_train), seed=SEED)
            .map(preprocess, num_parallel_calls=tf.data.AUTOTUNE)
            .batch(BATCH_SIZE, drop_remainder=True)
            .prefetch(tf.data.AUTOTUNE)
        )

        # Build val dataset — passed to fit() so Keras tracks both
        # n_loss (train) and val_n_loss (val) every epoch in history.
        # drop_remainder=False keeps the last smaller batch so no val samples are discarded.
        val_dataset = tf.data.Dataset.from_tensor_slices((X_val, y_val))
        val_dataset = (
            val_dataset
            .map(preprocess, num_parallel_calls=tf.data.AUTOTUNE)
            .batch(BATCH_SIZE, drop_remainder=False)
            .prefetch(tf.data.AUTOTUNE)
        )

        # Fresh model with the schedule for this run
        ddim = DiffusionModel(
            diffusion_schedule_fn=SCHEDULE_FN[schedule_name]
        )

        # Adapt normalizer on train split only.
        # Must be called after clear_session() and before compile().
        # Extracts images from (image, label) pairs via .map(lambda x, y: x).
        ddim.normalizer.adapt(train_dataset.map(lambda x, y: x))

        ddim.compile(
            optimizer=optimizers.AdamW(
                learning_rate=lr,
                weight_decay=WEIGHT_DECAY,
            ),
            loss=losses.MeanAbsoluteError(),
        )

        checkpoint_fid_cb = CheckpointAndFID(
            save_dir=run_model_dir,
            inception=inception,
            X_val=X_val,
            y_val=y_val,
            real_val_features=real_val_features,
            fid_freq=FID_FREQ,
        )

        # Train
        history = ddim.fit(
            train_dataset,
            epochs=EPOCHS,
            validation_data=val_dataset,
            callbacks=[checkpoint_fid_cb],
            verbose=1,
        )

        # Guard against empty FID scores
        if not checkpoint_fid_cb.fid_scores:
            print(f"WARNING: No FID scores recorded for run {run_name}. Skipping.")
            continue

        best_epoch = min(
            checkpoint_fid_cb.fid_scores,
            key=checkpoint_fid_cb.fid_scores.get
        )
        best_fid = checkpoint_fid_cb.fid_scores[best_epoch]

        # Save full training history (n_loss + val_n_loss every epoch)
        pd.DataFrame(history.history).to_csv(
            os.path.join(run_model_dir, "history.csv"), index_label="epoch"
        )

        # Save FID scores at checkpoint epochs
        pd.DataFrame(
            list(checkpoint_fid_cb.fid_scores.items()),
            columns=["epoch", "fid"]
        ).to_csv(
            os.path.join(run_model_dir, "fid_scores.csv"), index=False
        )

        print(f"\nRun {run_name} -> Best epoch: {best_epoch}, FID: {best_fid:.4f}")

        # Store results
        search_results.append({
            "run_name":   run_name,
            "lr":         lr,
            "schedule":   schedule_name,
            "best_epoch": best_epoch,
            "best_fid":   best_fid,
        })

        # Save search results after each run — safe against crashes
        pd.DataFrame(search_results).to_csv(
            os.path.join(MODEL_DIR, "search_results.csv"), index=False
        )

# 7c- Summary
print(f"\n{'='*60}")
print("Hyperparameter Search Results")
print(f"{'='*60}")
for r in sorted(search_results, key=lambda x: x["best_fid"]):
    print(
        f"{r['run_name']:<45}  Best Epoch: {r['best_epoch']:>3}  "
        f"FID: {r['best_fid']:>8.4f}"
    )

best_run = min(search_results, key=lambda x: x["best_fid"])
print(f"\nBest run:   {best_run['run_name']}")
print(f"Best epoch: {best_run['best_epoch']}")
print(f"Best FID:   {best_run['best_fid']:.4f}")
print(f"Final training with:")
print(f"  BATCH_SIZE = {BATCH_SIZE}")
print(f"  LR         = {best_run['lr']}")
print(f"  SCHEDULE   = {best_run['schedule']}")
print(f"  EPOCHS     = {best_run['best_epoch']}")

BEST_LR       = best_run["lr"]
BEST_SCHEDULE = best_run["schedule"]
BEST_EPOCH    = best_run["best_epoch"]


# In[ ]:


# 8- Final Training on Merged Train + Val

print(f"Final training with:")
print(f"  BATCH_SIZE = {BATCH_SIZE}")
print(f"  LR         = {BEST_LR}")
print(f"  SCHEDULE   = {BEST_SCHEDULE}")
print(f"  EPOCHS     = {BEST_EPOCH}")

# Merge train + val
# PathMNIST uses official MedMNIST pre-defined splits.
# We merge train+val for final training, consistent with EuroSAT and FashionMNIST strategy.
X_final = np.concatenate([X_train, X_val], axis=0)
y_final = np.concatenate([y_train, y_val], axis=0)
print(f"\nX_final: {X_final.shape}, y_final: {y_final.shape}")

# Build final dataset
final_dataset = tf.data.Dataset.from_tensor_slices((X_final, y_final))
final_dataset = (
    final_dataset
    .shuffle(buffer_size=len(X_final), seed=SEED)
    .map(preprocess, num_parallel_calls=tf.data.AUTOTUNE)
    .batch(BATCH_SIZE, drop_remainder=True)
    .prefetch(tf.data.AUTOTUNE)
)

# Final model directory
final_model_dir = os.path.join(MODEL_DIR, "final")
os.makedirs(final_model_dir, exist_ok=True)

# Clear session and build fresh model
tf.keras.backend.clear_session()

ddim_final = DiffusionModel(
    diffusion_schedule_fn=SCHEDULE_FN[BEST_SCHEDULE]
)

# Adapt normalizer on merged train+val.
# During search runs normalizer was fitted on train only — here it must
# see the full merged dataset since that is what the model trains on.
ddim_final.normalizer.adapt(final_dataset.map(lambda x, y: x))

ddim_final.compile(
    optimizer=optimizers.AdamW(
        learning_rate=BEST_LR,
        weight_decay=WEIGHT_DECAY,
    ),
    loss=losses.MeanAbsoluteError(),
)

# Train for exactly BEST_EPOCH epochs.
# No val data passed to fit() — train+val are merged, nothing held out.
history_final = ddim_final.fit(
    final_dataset,
    epochs=BEST_EPOCH,
    callbacks=[
        # Saves weights and image grid every 50 epochs for offline inspection.
        # No FID computed here.
        SaveModels(save_dir=final_model_dir, save_freq=50)
    ],
    verbose=1,
)

print(f"\nFinal training complete.")
# history_final contains only n_loss (no val_n_loss)
# since no validation data is passed
print(f"Final n_loss: {history_final.history['n_loss'][-1]:.4f}")

# Save training history
pd.DataFrame(history_final.history).to_csv(
    os.path.join(final_model_dir, "history.csv"), index_label="epoch"
)


# In[ ]:


# 9- Synthetic Data Generation and Saving
from sklearn.model_selection import train_test_split

# Match the class distribution of X_final (merged train+val)
# PathMNIST is imbalanced — class counts vary; val_ratio computed dynamically
unique_classes, class_counts = np.unique(y_final, return_counts=True)
print("Generating synthetic data matching y_final class distribution:")
for cls, cnt in zip(unique_classes, class_counts):
    print(f"  Class {cls}: {cnt} samples")

# Generate synthetic images per class in batches to avoid GPU OOM
synthetic_images = []
synthetic_labels = []

for class_idx, n_samples in zip(unique_classes, class_counts):
    print(f"  Generating {n_samples} images for class {class_idx}...")
    class_generated = []
    n_batches = math.ceil(n_samples / GEN_BATCH_SIZE)
    for batch_idx in range(n_batches):
        batch_start  = batch_idx * GEN_BATCH_SIZE
        batch_end    = min(batch_start + GEN_BATCH_SIZE, n_samples)
        n_gen        = batch_end - batch_start
        class_labels = tf.fill((n_gen,), tf.cast(class_idx, tf.int32))
        generated    = ddim_final.generate(
            num_images=n_gen,
            diffusion_steps=DIFFUSION_STEPS,
            class_labels=class_labels,
        )
        generated = (generated.numpy() * 255.0).clip(0, 255).astype("uint8")
        class_generated.append(generated)
    synthetic_images.append(np.concatenate(class_generated, axis=0))
    synthetic_labels.append(np.full(n_samples, class_idx, dtype=np.int64))

X_synthetic = np.concatenate(synthetic_images, axis=0)
y_synthetic = np.concatenate(synthetic_labels, axis=0)
print(f"\nX_synthetic: {X_synthetic.shape}, y_synthetic: {y_synthetic.shape}")

# Stratified split into synthetic train + val.
# Preserve the same train/val proportion as the original MedMNIST split.
# val_ratio computed dynamically from actual split sizes, not hardcoded.
original_total = len(X_train) + len(X_val)
val_ratio      = len(X_val) / original_total
print(f"\nApplying stratified split with val_ratio={val_ratio:.4f}")

# Note: train_test_split may cause +-1 rounding effect per class
X_synth_train, X_synth_val, y_synth_train, y_synth_val = train_test_split(
    X_synthetic,
    y_synthetic,
    test_size=val_ratio,
    stratify=y_synthetic,
    random_state=SEED,
)

print(f"Synthetic train: {X_synth_train.shape}, y: {y_synth_train.shape}")
print(f"Synthetic val:   {X_synth_val.shape},   y: {y_synth_val.shape}")

# Save
os.makedirs(SYNTH_DIR, exist_ok=True)
save_path = os.path.join(SYNTH_DIR, "path_mnist_synthetic.npz")
np.savez(
    save_path,
    X_train=X_synth_train,
    y_train=y_synth_train,
    X_val=X_synth_val,
    y_val=y_synth_val,
)
print(f"\nSaved synthetic dataset to: {save_path}")

