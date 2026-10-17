# cgan_path_mnist_runpod.py
# Full 6-run hyperparameter search for PathMNIST cGAN — RunPod version.
# PROJECT_DIR set to /root/thesis_project (RunPod default home).
# Directory names kept identical to cluster version (data_path, models_path).

# ── Cell 1 — Imports + paths ──────────────────────────────────────────────────
import numpy as np
import tensorflow as tf
from tensorflow.keras import (
    layers,
    models,
    callbacks,
    metrics,
    optimizers,
)
import os

SEED = 42
tf.random.set_seed(SEED)
np.random.seed(SEED)

PROJECT_DIR = "/root/thesis_project"
DATA_DIR    = PROJECT_DIR

SYNTH_DIR = os.path.join(PROJECT_DIR, "data_path", "synthetic", "seed_42", "cgan", "path_mnist")
MODEL_DIR = os.path.join(PROJECT_DIR, "models_path", "cgan", "path_mnist", "seed_42")
os.makedirs(SYNTH_DIR, exist_ok=True)
os.makedirs(MODEL_DIR, exist_ok=True)

# ── Cell 2 — Parameters ───────────────────────────────────────────────────────
IMAGE_SIZE = 28
CHANNELS   = 3   # RGB
CLASSES    = 9
GP_WEIGHT  = 10.0
EPOCHS     = 2000

Z_DIM        = 32
CRITIC_STEPS = 3

# betas=(0.0, 0.9) from WGAN-GP paper
BETAS = (0.0, 0.9)

LR_OPTIONS         = [1e-4, 5e-5]
BATCH_SIZE_OPTIONS = [32, 64, 128]
# Total: 2 LR x 3 batch sizes = 6 runs

# ── Cell 3 — Data loading ─────────────────────────────────────────────────────
data    = np.load(os.path.join(DATA_DIR, "path_mnist.npz"))
X_train = data["X_train"]  # (N, 28, 28, 3), uint8
y_train = data["y_train"]  # (N,), int
X_val   = data["X_val"]    # (M, 28, 28, 3), uint8
y_val   = data["y_val"]    # (M,), int

print(f"X_train: {X_train.shape}, y_train: {y_train.shape}")
print(f"X_val:   {X_val.shape},   y_val:   {y_val.shape}")
print(f"Pixel range: [{X_train.min()}, {X_train.max()}]")

def preprocess(img, label):
    img   = (tf.cast(img, "float32") - 127.5) / 127.5
    label = tf.one_hot(label, depth=CLASSES)
    return img, label

# ── Cell 4 — build_generator() ────────────────────────────────────────────────
# Adapted for PathMNIST: 28x28 RGB output, 9 classes.
# 7x7 start and 2 upsampling blocks. CHANNELS=3 for RGB output.
def build_generator(z_dim):
    generator_input = layers.Input(shape=(z_dim,))
    label_input     = layers.Input(shape=(CLASSES,))

    x = layers.Concatenate()([generator_input, label_input])

    # Dense projection -> 7x7 spatial feature map
    x = layers.Dense(7 * 7 * 128, use_bias=False)(x)
    x = layers.Reshape((7, 7, 128))(x)
    x = layers.BatchNormalization(momentum=0.9)(x)

    # Upsample: 7x7 -> 14x14
    x = layers.UpSampling2D()(x)
    x = layers.Conv2D(128, kernel_size=3, padding="same", use_bias=False)(x)
    x = layers.BatchNormalization(momentum=0.9)(x)
    x = layers.LeakyReLU(0.2)(x)

    # Upsample: 14x14 -> 28x28
    x = layers.UpSampling2D()(x)
    x = layers.Conv2D(64, kernel_size=3, padding="same", use_bias=False)(x)
    x = layers.BatchNormalization(momentum=0.9)(x)
    x = layers.LeakyReLU(0.2)(x)

    # Output: RGB, tanh -> [-1,1]
    generator_output = layers.Conv2D(
        CHANNELS, kernel_size=3, padding="same", activation="tanh"
    )(x)

    return models.Model(
        [generator_input, label_input], generator_output, name="generator"
    )

build_generator(Z_DIM).summary()

# ── Cell 5 — build_critic() ───────────────────────────────────────────────────
# Adapted for PathMNIST: 28x28 RGB input, 9 classes.
# Input channels: CHANNELS(3) + CLASSES(9) = 12.
# Downsampling: 28->14->7->4->scalar.
def build_critic():
    critic_input = layers.Input(shape=(IMAGE_SIZE, IMAGE_SIZE, CHANNELS))
    label_input  = layers.Input(shape=(IMAGE_SIZE, IMAGE_SIZE, CLASSES))

    # Concatenate: (28,28,3+9) -> (28,28,12)
    x = layers.Concatenate(axis=-1)([critic_input, label_input])

    # Downsample: 28->14
    x = layers.Conv2D(64, kernel_size=4, strides=2, padding="same", use_bias=False)(x)
    x = layers.LayerNormalization()(x)
    x = layers.LeakyReLU(0.2)(x)

    # Downsample: 14->7
    x = layers.Conv2D(128, kernel_size=4, strides=2, padding="same", use_bias=False)(x)
    x = layers.LayerNormalization()(x)
    x = layers.LeakyReLU(0.2)(x)
    x = layers.Dropout(0.3)(x)

    # Downsample: 7->4 (ceil(7/2)=4 with same padding)
    x = layers.Conv2D(128, kernel_size=4, strides=2, padding="same", use_bias=False)(x)
    x = layers.LayerNormalization()(x)
    x = layers.LeakyReLU(0.2)(x)
    x = layers.Dropout(0.3)(x)

    # 4->scalar (4-4+1=1 with valid padding)
    x = layers.Conv2D(1, kernel_size=4, strides=1, padding="valid")(x)
    critic_output = layers.Flatten()(x)

    return models.Model(
        [critic_input, label_input], critic_output, name="critic"
    )

build_critic().summary()

# ── Cell 6 — ConditionalWGAN class ───────────────────────────────────────────
class ConditionalWGAN(models.Model):
    def __init__(self, critic, generator, latent_dim, critic_steps, gp_weight):
        super(ConditionalWGAN, self).__init__()
        self.critic       = critic
        self.generator    = generator
        self.latent_dim   = latent_dim
        self.critic_steps = critic_steps
        self.gp_weight    = gp_weight

    def compile(self, c_optimizer, g_optimizer):
        super(ConditionalWGAN, self).compile()
        self.c_optimizer        = c_optimizer
        self.g_optimizer        = g_optimizer
        self.c_wass_loss_metric = metrics.Mean(name="c_wass_loss")
        self.c_gp_metric        = metrics.Mean(name="c_gp")
        self.c_loss_metric      = metrics.Mean(name="c_loss")
        self.g_loss_metric      = metrics.Mean(name="g_loss")

    @property
    def metrics(self):
        return [
            self.c_loss_metric,
            self.c_wass_loss_metric,
            self.c_gp_metric,
            self.g_loss_metric,
        ]

    def gradient_penalty(self, batch_size, real_images, fake_images, image_one_hot_labels):
        alpha        = tf.random.normal([batch_size, 1, 1, 1], 0.0, 1.0)
        diff         = fake_images - real_images
        interpolated = real_images + alpha * diff

        with tf.GradientTape() as gp_tape:
            gp_tape.watch(interpolated)
            pred = self.critic([interpolated, image_one_hot_labels], training=True)

        # Per-sample gradient penalty
        grads = gp_tape.gradient(pred, [interpolated])[0]
        norm  = tf.sqrt(tf.reduce_sum(tf.square(grads), axis=[1, 2, 3]))
        gp    = tf.reduce_mean((norm - 1.0) ** 2)
        return gp

    def train_step(self, data):
        real_images, one_hot_labels = data

        # Broadcast one-hot labels to spatial dimensions for the critic
        image_one_hot_labels = one_hot_labels[:, None, None, :]
        image_one_hot_labels = tf.repeat(image_one_hot_labels, repeats=IMAGE_SIZE, axis=1)
        image_one_hot_labels = tf.repeat(image_one_hot_labels, repeats=IMAGE_SIZE, axis=2)

        batch_size = tf.shape(real_images)[0]

        # Train critic
        for i in range(self.critic_steps):
            random_latent_vectors = tf.random.normal(shape=(batch_size, self.latent_dim))

            with tf.GradientTape() as tape:
                fake_images      = self.generator([random_latent_vectors, one_hot_labels], training=True)
                fake_predictions = self.critic([fake_images, image_one_hot_labels], training=True)
                real_predictions = self.critic([real_images, image_one_hot_labels], training=True)
                c_wass_loss      = tf.reduce_mean(fake_predictions) - tf.reduce_mean(real_predictions)
                c_gp             = self.gradient_penalty(batch_size, real_images, fake_images, image_one_hot_labels)
                c_loss           = c_wass_loss + c_gp * self.gp_weight

            c_gradient = tape.gradient(c_loss, self.critic.trainable_variables)
            self.c_optimizer.apply_gradients(zip(c_gradient, self.critic.trainable_variables))

        # Train generator
        random_latent_vectors = tf.random.normal(shape=(batch_size, self.latent_dim))

        with tf.GradientTape() as tape:
            fake_images      = self.generator([random_latent_vectors, one_hot_labels], training=True)
            fake_predictions = self.critic([fake_images, image_one_hot_labels], training=True)
            g_loss           = -tf.reduce_mean(fake_predictions)

        gen_gradient = tape.gradient(g_loss, self.generator.trainable_variables)
        self.g_optimizer.apply_gradients(zip(gen_gradient, self.generator.trainable_variables))

        # Update metrics
        self.c_loss_metric.update_state(c_loss)
        self.c_wass_loss_metric.update_state(c_wass_loss)
        self.c_gp_metric.update_state(c_gp)
        self.g_loss_metric.update_state(g_loss)

        return {m.name: m.result() for m in self.metrics}

# ── Cell 7 — Callbacks ────────────────────────────────────────────────────────
class CheckpointAndFID(callbacks.Callback):
    def __init__(self, save_dir, inception, X_val, y_val, latent_dim, save_freq=50, real_features=None):
        self.save_dir      = save_dir
        self.inception     = inception
        self.X_val         = X_val
        self.y_val         = y_val
        self.latent_dim    = latent_dim
        self.save_freq     = save_freq
        self.real_features = real_features  # pre-computed; survives clear_session()
        self.fid_scores    = {}             # {epoch: fid_score}
        os.makedirs(save_dir, exist_ok=True)

    def on_epoch_end(self, epoch, logs=None):
        if (epoch + 1) % self.save_freq == 0:
            checkpoint_dir = os.path.join(self.save_dir, f"epoch_{epoch+1:03d}")
            os.makedirs(checkpoint_dir, exist_ok=True)
            self.model.generator.save_weights(
                os.path.join(checkpoint_dir, "generator.weights.h5")
            )
            self.model.critic.save_weights(
                os.path.join(checkpoint_dir, "critic.weights.h5")
            )
            fid = compute_fid(
                generator=self.model.generator,
                latent_dim=self.latent_dim,
                X_val=self.X_val,
                y_val=self.y_val,
                inception=self.inception,
                real_features=self.real_features,
            )
            self.fid_scores[epoch + 1] = fid
            print(f"\n  \u2192 Epoch {epoch+1}: FID = {fid:.4f}")


class SaveModels(callbacks.Callback):
    """Periodic checkpoint saver with image generation — used for final training only."""
    def __init__(self, save_dir, latent_dim, save_freq=50):
        self.save_dir   = save_dir
        self.latent_dim = latent_dim
        self.save_freq  = save_freq
        os.makedirs(save_dir, exist_ok=True)

    def on_epoch_end(self, epoch, logs=None):
        if (epoch + 1) % self.save_freq == 0:
            checkpoint_dir = os.path.join(self.save_dir, f"epoch_{epoch+1:03d}")
            os.makedirs(checkpoint_dir, exist_ok=True)

            self.model.generator.save_weights(
                os.path.join(checkpoint_dir, "generator.weights.h5")
            )
            self.model.critic.save_weights(
                os.path.join(checkpoint_dir, "critic.weights.h5")
            )

            import matplotlib
            matplotlib.use("Agg")
            import matplotlib.pyplot as plt

            random_latent_vectors = tf.random.normal(shape=(5, self.latent_dim))
            fig, axes = plt.subplots(CLASSES, 5, figsize=(10, CLASSES * 2))

            for class_idx in range(CLASSES):
                one_hot = np.zeros((5, CLASSES))
                one_hot[:, class_idx] = 1.0
                one_hot   = tf.convert_to_tensor(one_hot, dtype=tf.float32)
                generated = self.model.generator([random_latent_vectors, one_hot], training=False)
                generated = (generated * 127.5 + 127.5).numpy().clip(0, 255).astype("uint8")

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
                os.path.join(checkpoint_dir, f"generated_epoch_{epoch+1:03d}.png"),
                dpi=100,
                bbox_inches="tight",
            )
            plt.close()

# ── Cell 8 — FID helper ───────────────────────────────────────────────────────
from scipy.linalg import sqrtm

def compute_fid(generator, latent_dim, X_val, y_val, inception, batch_size=64, real_features=None):
    def get_inception_features(images):
        all_features = []
        for i in range(0, len(images), batch_size):
            batch = images[i:i + batch_size].astype("float32")
            batch = tf.image.resize(batch, (299, 299))
            batch = tf.keras.applications.inception_v3.preprocess_input(batch)
            feats = inception.predict(batch, verbose=0)
            all_features.append(feats)
        return np.concatenate(all_features, axis=0)

    if real_features is None:
        real_features = get_inception_features(X_val.astype("float32"))

    synthetic_images = []
    for class_idx in range(CLASSES):
        class_mask = y_val == class_idx
        n_samples  = int(np.sum(class_mask))
        if n_samples == 0:
            continue
        z       = tf.random.normal(shape=(n_samples, latent_dim))
        one_hot = np.zeros((n_samples, CLASSES))
        one_hot[:, class_idx] = 1.0
        one_hot   = tf.convert_to_tensor(one_hot, dtype=tf.float32)
        generated = generator([z, one_hot], training=False)
        generated = (generated * 127.5 + 127.5).numpy().clip(0, 255).astype("float32")
        synthetic_images.append(generated)

    synthetic_images   = np.concatenate(synthetic_images, axis=0)
    synthetic_features = get_inception_features(synthetic_images)

    mu_real     = np.mean(real_features,      axis=0)
    mu_synth    = np.mean(synthetic_features,  axis=0)
    sigma_real  = np.cov(real_features,      rowvar=False)
    sigma_synth = np.cov(synthetic_features,  rowvar=False)

    diff = mu_real - mu_synth

    eps     = 1e-6
    covmean = sqrtm(
        (sigma_real  + eps * np.eye(sigma_real.shape[0])) @
        (sigma_synth + eps * np.eye(sigma_synth.shape[0]))
    )

    if np.iscomplexobj(covmean):
        covmean = covmean.real

    fid = np.dot(diff, diff) + np.trace(sigma_real + sigma_synth - 2 * covmean)
    return max(0.0, float(fid))

# ── Cell 9 — Hyperparameter search loop — all 6 runs ─────────────────────────
import pandas as pd

search_results = []

# Cache real val features once outside the loop
_inception_init = tf.keras.applications.InceptionV3(
    include_top=False, pooling="avg", input_shape=(299, 299, 3)
)
_real_ds = (
    tf.data.Dataset.from_tensor_slices(X_val.astype("float32"))
    .batch(64)
    .map(
        lambda x: tf.keras.applications.inception_v3.preprocess_input(
            tf.image.resize(x, (299, 299))
        ),
        num_parallel_calls=tf.data.AUTOTUNE,
    )
    .prefetch(tf.data.AUTOTUNE)
)
real_val_features = _inception_init.predict(_real_ds, verbose=0)
del _inception_init
del _real_ds
tf.keras.backend.clear_session()
import gc; gc.collect()
print(f"Real val features cached once: {real_val_features.shape}")

for lr in LR_OPTIONS:
    for batch_size in BATCH_SIZE_OPTIONS:
        run_name = f"lr{lr}_bs{batch_size}"
        print(f"\n{'='*60}")
        print(f"Starting run: {run_name}")
        print(f"{'='*60}")

        run_model_dir = os.path.join(MODEL_DIR, "search", run_name)
        os.makedirs(run_model_dir, exist_ok=True)

        train_dataset = tf.data.Dataset.from_tensor_slices((X_train, y_train))
        train_dataset = (
            train_dataset
            .shuffle(buffer_size=len(X_train), seed=SEED)
            .batch(batch_size, drop_remainder=True)
            .map(preprocess, num_parallel_calls=tf.data.AUTOTUNE)
            .prefetch(tf.data.AUTOTUNE)
        )

        tf.keras.backend.clear_session()
        inception = tf.keras.applications.InceptionV3(
            include_top=False, pooling="avg", input_shape=(299, 299, 3)
        )

        generator = build_generator(Z_DIM)
        critic    = build_critic()

        cgan = ConditionalWGAN(
            critic=critic,
            generator=generator,
            latent_dim=Z_DIM,
            critic_steps=CRITIC_STEPS,
            gp_weight=GP_WEIGHT,
        )

        cgan.compile(
            c_optimizer=optimizers.Adam(
                learning_rate=lr,
                beta_1=BETAS[0],
                beta_2=BETAS[1],
                weight_decay=1e-3,
            ),
            g_optimizer=optimizers.Adam(
                learning_rate=lr,
                beta_1=BETAS[0],
                beta_2=BETAS[1],
            ),
        )

        checkpoint_fid_cb = CheckpointAndFID(
            save_dir=run_model_dir,
            inception=inception,
            X_val=X_val,
            y_val=y_val,
            latent_dim=Z_DIM,
            save_freq=50,
            real_features=real_val_features,
        )

        history = cgan.fit(
            train_dataset,
            epochs=EPOCHS,
            callbacks=[checkpoint_fid_cb],
            verbose=1,
        )

        if checkpoint_fid_cb.fid_scores:
            best_epoch = min(checkpoint_fid_cb.fid_scores, key=checkpoint_fid_cb.fid_scores.get)
            best_fid   = checkpoint_fid_cb.fid_scores[best_epoch]
        else:
            print(f"WARNING: No FID scores recorded for run {run_name}. Skipping.")
            continue

        pd.DataFrame(history.history).to_csv(
            os.path.join(run_model_dir, "history.csv"), index_label="epoch"
        )
        pd.DataFrame(
            list(checkpoint_fid_cb.fid_scores.items()), columns=["epoch", "fid"]
        ).to_csv(os.path.join(run_model_dir, "fid_scores.csv"), index=False)

        print(f"\nRun {run_name} \u2192 Best epoch: {best_epoch}, FID: {best_fid:.4f}")

        search_results.append({
            "run_name":   run_name,
            "lr":         lr,
            "batch_size": batch_size,
            "best_epoch": best_epoch,
            "best_fid":   best_fid,
        })

        pd.DataFrame(search_results).to_csv(
            os.path.join(MODEL_DIR, "search_results.csv"), index=False
        )

# Summary
print(f"\n{'='*60}")
print("Hyperparameter Search Results \u2014 PathMNIST cGAN")
print(f"{'='*60}")
for r in sorted(search_results, key=lambda x: x["best_fid"]):
    print(
        f"{r['run_name']:<30} Batch: {r['batch_size']:>3}  "
        f"Best Epoch: {r['best_epoch']:>4}   FID: {r['best_fid']:>8.4f}"
    )

best_run = min(search_results, key=lambda x: x["best_fid"])
print(f"\nBest run:        {best_run['run_name']}")
print(f"Best epoch:      {best_run['best_epoch']}")
print(f"Best FID:        {best_run['best_fid']:.4f}")
print(f"Best LR:         {best_run['lr']}")
print(f"Best batch size: {best_run['batch_size']}")

BEST_LR         = best_run["lr"]
BEST_BATCH_SIZE = best_run["batch_size"]
BEST_EPOCH      = best_run["best_epoch"]

# ── Cell 10 — Final retraining on merged train+val ────────────────────────────
print(f"\nFinal training with:")
print(f"  BEST_LR={BEST_LR}")
print(f"  BEST_BATCH_SIZE={BEST_BATCH_SIZE}")
print(f"  BETAS={BETAS}")
print(f"  EPOCHS={BEST_EPOCH}")

X_final = np.concatenate([X_train, X_val], axis=0)
y_final = np.concatenate([y_train, y_val], axis=0)
print(f"X_final: {X_final.shape}, y_final: {y_final.shape}")

final_dataset = tf.data.Dataset.from_tensor_slices((X_final, y_final))
final_dataset = (
    final_dataset
    .shuffle(buffer_size=len(X_final), seed=SEED)
    .batch(BEST_BATCH_SIZE, drop_remainder=True)
    .map(preprocess, num_parallel_calls=tf.data.AUTOTUNE)
    .prefetch(tf.data.AUTOTUNE)
)

final_model_dir = os.path.join(MODEL_DIR, "final")
os.makedirs(final_model_dir, exist_ok=True)

tf.keras.backend.clear_session()
generator_final = build_generator(Z_DIM)
critic_final    = build_critic()

cgan_final = ConditionalWGAN(
    critic=critic_final,
    generator=generator_final,
    latent_dim=Z_DIM,
    critic_steps=CRITIC_STEPS,
    gp_weight=GP_WEIGHT,
)

cgan_final.compile(
    c_optimizer=optimizers.Adam(
        learning_rate=BEST_LR,
        beta_1=BETAS[0],
        beta_2=BETAS[1],
        weight_decay=1e-3,
    ),
    g_optimizer=optimizers.Adam(
        learning_rate=BEST_LR,
        beta_1=BETAS[0],
        beta_2=BETAS[1],
    ),
)

history_final = cgan_final.fit(
    final_dataset,
    epochs=BEST_EPOCH,
    callbacks=[SaveModels(save_dir=final_model_dir, latent_dim=Z_DIM, save_freq=50)],
    verbose=1,
)

print(f"Final training complete.")
print(f"Final g_loss:      {history_final.history['g_loss'][-1]:.4f}")
print(f"Final c_wass_loss: {history_final.history['c_wass_loss'][-1]:.4f}")

pd.DataFrame(history_final.history).to_csv(
    os.path.join(final_model_dir, "history.csv"), index_label="epoch"
)

# ── Cell 11 — Synthetic data generation + stratified split ───────────────────
from sklearn.model_selection import train_test_split

unique_classes, class_counts = np.unique(y_final, return_counts=True)
print("Generating PathMNIST synthetic data matching y_final class distribution:")
for cls, cnt in zip(unique_classes, class_counts):
    print(f"  Class {cls}: {cnt} samples")

synthetic_images = []
synthetic_labels = []

for class_idx, n_samples in zip(unique_classes, class_counts):
    z       = tf.random.normal(shape=(n_samples, Z_DIM))
    one_hot = np.zeros((n_samples, CLASSES))
    one_hot[:, class_idx] = 1.0
    one_hot   = tf.convert_to_tensor(one_hot, dtype=tf.float32)
    generated = cgan_final.generator([z, one_hot], training=False)
    generated = (generated * 127.5 + 127.5).numpy().clip(0, 255).astype("uint8")
    synthetic_images.append(generated)
    synthetic_labels.append(np.full(n_samples, class_idx, dtype=np.int64))

X_synthetic = np.concatenate(synthetic_images, axis=0)
y_synthetic = np.concatenate(synthetic_labels, axis=0)
print(f"\nX_synthetic: {X_synthetic.shape}, y_synthetic: {y_synthetic.shape}")

original_total = len(X_train) + len(X_val)
val_ratio      = len(X_val) / original_total

X_synth_train, X_synth_val, y_synth_train, y_synth_val = train_test_split(
    X_synthetic,
    y_synthetic,
    test_size=val_ratio,
    stratify=y_synthetic,
    random_state=SEED,
)

print(f"\nSynthetic train: {X_synth_train.shape}, y: {y_synth_train.shape}")
print(f"Synthetic val:   {X_synth_val.shape},   y: {y_synth_val.shape}")

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
