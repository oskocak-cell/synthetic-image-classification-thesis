# PathMNIST cGAN — Final training + synthetic generation (RunPod)

import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models, callbacks, metrics, optimizers
import matplotlib
matplotlib.use("Agg")   # set ONCE at top, before any plt import
import matplotlib.pyplot as plt
import os, gc
import pandas as pd
from scipy.linalg import sqrtm
from sklearn.model_selection import train_test_split

IMAGE_SIZE = 28
CHANNELS   = 3
CLASSES    = 9
GP_WEIGHT  = 10.0
Z_DIM      = 32
CRITIC_STEPS = 3
BETAS      = (0.0, 0.9)
SEED       = 42
tf.random.set_seed(SEED)
np.random.seed(SEED)

# ── Best hyperparameters from search ──
BEST_LR         = 1e-4
BEST_BATCH_SIZE = 32
BEST_EPOCH      = 150

PROJECT_DIR = "/workspace/thesis_project"
DATA_DIR    = PROJECT_DIR
SYNTH_DIR   = os.path.join(PROJECT_DIR, "data_path",   "synthetic", "seed_42", "cgan", "path_mnist")
MODEL_DIR   = os.path.join(PROJECT_DIR, "models_path", "cgan", "path_mnist", "seed_42")
os.makedirs(SYNTH_DIR, exist_ok=True)
os.makedirs(MODEL_DIR, exist_ok=True)

data    = np.load(os.path.join(DATA_DIR, "path_mnist.npz"))
X_train = data["X_train"]
y_train = data["y_train"]
X_val   = data["X_val"]
y_val   = data["y_val"]
print(f"X_train: {X_train.shape}  X_val: {X_val.shape}")

def preprocess(img, label):
    img   = (tf.cast(img, "float32") - 127.5) / 127.5
    label = tf.one_hot(label, depth=CLASSES)
    return img, label

def build_generator(z_dim):
    generator_input = layers.Input(shape=(z_dim,))
    label_input     = layers.Input(shape=(CLASSES,))
    x = layers.Concatenate()([generator_input, label_input])
    x = layers.Dense(7 * 7 * 128, use_bias=False)(x)
    x = layers.Reshape((7, 7, 128))(x)
    x = layers.BatchNormalization(momentum=0.9)(x)
    x = layers.UpSampling2D()(x)
    x = layers.Conv2D(128, kernel_size=3, padding="same", use_bias=False)(x)
    x = layers.BatchNormalization(momentum=0.9)(x)
    x = layers.LeakyReLU(0.2)(x)
    x = layers.UpSampling2D()(x)
    x = layers.Conv2D(64, kernel_size=3, padding="same", use_bias=False)(x)
    x = layers.BatchNormalization(momentum=0.9)(x)
    x = layers.LeakyReLU(0.2)(x)
    generator_output = layers.Conv2D(CHANNELS, kernel_size=3, padding="same", activation="tanh")(x)
    return models.Model([generator_input, label_input], generator_output, name="generator")

build_generator(Z_DIM).summary()

def build_critic():
    critic_input = layers.Input(shape=(IMAGE_SIZE, IMAGE_SIZE, CHANNELS))
    label_input  = layers.Input(shape=(IMAGE_SIZE, IMAGE_SIZE, CLASSES))
    x = layers.Concatenate(axis=-1)([critic_input, label_input])
    x = layers.Conv2D(64,  kernel_size=4, strides=2, padding="same", use_bias=False)(x)
    x = layers.LayerNormalization()(x)
    x = layers.LeakyReLU(0.2)(x)
    x = layers.Conv2D(128, kernel_size=4, strides=2, padding="same", use_bias=False)(x)
    x = layers.LayerNormalization()(x)
    x = layers.LeakyReLU(0.2)(x)
    x = layers.Dropout(0.3)(x)
    x = layers.Conv2D(128, kernel_size=4, strides=2, padding="same", use_bias=False)(x)
    x = layers.LayerNormalization()(x)
    x = layers.LeakyReLU(0.2)(x)
    x = layers.Dropout(0.3)(x)
    x = layers.Conv2D(1, kernel_size=4, strides=1, padding="valid")(x)
    critic_output = layers.Flatten()(x)
    return models.Model([critic_input, label_input], critic_output, name="critic")

build_critic().summary()

class ConditionalWGAN(models.Model):
    def __init__(self, critic, generator, latent_dim, critic_steps, gp_weight):
        super().__init__()
        self.critic       = critic
        self.generator    = generator
        self.latent_dim   = latent_dim
        self.critic_steps = critic_steps
        self.gp_weight    = gp_weight

    def compile(self, c_optimizer, g_optimizer):
        super().compile()
        self.c_optimizer        = c_optimizer
        self.g_optimizer        = g_optimizer
        self.c_wass_loss_metric = metrics.Mean(name="c_wass_loss")
        self.c_gp_metric        = metrics.Mean(name="c_gp")
        self.c_loss_metric      = metrics.Mean(name="c_loss")
        self.g_loss_metric      = metrics.Mean(name="g_loss")

    @property
    def metrics(self):
        return [self.c_loss_metric, self.c_wass_loss_metric, self.c_gp_metric, self.g_loss_metric]

    def gradient_penalty(self, batch_size, real_images, fake_images, image_one_hot_labels):
        alpha        = tf.random.normal([batch_size, 1, 1, 1], 0.0, 1.0)
        interpolated = real_images + alpha * (fake_images - real_images)
        with tf.GradientTape() as gp_tape:
            gp_tape.watch(interpolated)
            pred = self.critic([interpolated, image_one_hot_labels], training=True)
        grads = gp_tape.gradient(pred, [interpolated])[0]
        norm  = tf.sqrt(tf.reduce_sum(tf.square(grads), axis=[1, 2, 3]))
        return tf.reduce_mean((norm - 1.0) ** 2)

    def train_step(self, data):
        real_images, one_hot_labels = data
        image_one_hot_labels = tf.repeat(tf.repeat(one_hot_labels[:, None, None, :], IMAGE_SIZE, axis=1), IMAGE_SIZE, axis=2)
        batch_size = tf.shape(real_images)[0]

        for _ in range(self.critic_steps):
            z = tf.random.normal(shape=(batch_size, self.latent_dim))
            with tf.GradientTape() as tape:
                fake_images = self.generator([z, one_hot_labels], training=True)
                c_wass_loss = tf.reduce_mean(self.critic([fake_images, image_one_hot_labels], training=True)) -                               tf.reduce_mean(self.critic([real_images, image_one_hot_labels], training=True))
                c_gp   = self.gradient_penalty(batch_size, real_images, fake_images, image_one_hot_labels)
                c_loss = c_wass_loss + c_gp * self.gp_weight
            self.c_optimizer.apply_gradients(zip(tape.gradient(c_loss, self.critic.trainable_variables), self.critic.trainable_variables))

        z = tf.random.normal(shape=(batch_size, self.latent_dim))
        with tf.GradientTape() as tape:
            fake_images = self.generator([z, one_hot_labels], training=True)
            g_loss = -tf.reduce_mean(self.critic([fake_images, image_one_hot_labels], training=True))
        self.g_optimizer.apply_gradients(zip(tape.gradient(g_loss, self.generator.trainable_variables), self.generator.trainable_variables))

        self.c_loss_metric.update_state(c_loss)
        self.c_wass_loss_metric.update_state(c_wass_loss)
        self.c_gp_metric.update_state(c_gp)
        self.g_loss_metric.update_state(g_loss)
        return {m.name: m.result() for m in self.metrics}

class SaveModels(callbacks.Callback):
    """Saves weights + a generated image grid every save_freq epochs.
    matplotlib.use('Agg') is set at module level (top of script) — NOT here."""
    def __init__(self, save_dir, latent_dim, save_freq=50):
        self.save_dir   = save_dir
        self.latent_dim = latent_dim
        self.save_freq  = save_freq
        os.makedirs(save_dir, exist_ok=True)

    def on_epoch_end(self, epoch, logs=None):
        if (epoch + 1) % self.save_freq == 0:
            checkpoint_dir = os.path.join(self.save_dir, f"epoch_{epoch+1:03d}")
            os.makedirs(checkpoint_dir, exist_ok=True)

            self.model.generator.save_weights(os.path.join(checkpoint_dir, "generator.weights.h5"))
            self.model.critic.save_weights(   os.path.join(checkpoint_dir, "critic.weights.h5"))

            # Generate sample grid — matplotlib already imported at module level
            z = tf.random.normal(shape=(5, self.latent_dim))
            fig, axes = plt.subplots(CLASSES, 5, figsize=(10, CLASSES * 2))
            for class_idx in range(CLASSES):
                one_hot   = tf.convert_to_tensor(np.eye(CLASSES)[class_idx:class_idx+1].repeat(5, axis=0), dtype=tf.float32)
                generated = (self.model.generator([z, one_hot], training=False) * 127.5 + 127.5).numpy().clip(0, 255).astype("uint8")
                for img_idx in range(5):
                    axes[class_idx, img_idx].imshow(generated[img_idx])
                    axes[class_idx, img_idx].axis("off")
                    if img_idx == 0:
                        axes[class_idx, img_idx].set_ylabel(f"Class {class_idx}", fontsize=8)
            plt.suptitle(f"Epoch {epoch + 1}", fontsize=10)
            plt.tight_layout()
            plt.savefig(os.path.join(checkpoint_dir, f"generated_epoch_{epoch+1:03d}.png"), dpi=100, bbox_inches="tight")
            plt.close()
            print(f"  → Checkpoint + image saved at epoch {epoch+1}")

# ── Cell 10: Final retraining on merged train+val ────────────────────────────
print(f"Final training — LR={BEST_LR}, batch={BEST_BATCH_SIZE}, epochs={BEST_EPOCH}")

X_final = np.concatenate([X_train, X_val], axis=0)
y_final = np.concatenate([y_train, y_val], axis=0)
print(f"X_final: {X_final.shape}, y_final: {y_final.shape}")

final_dataset = (
    tf.data.Dataset.from_tensor_slices((X_final, y_final))
    .shuffle(buffer_size=len(X_final), seed=SEED)
    .batch(BEST_BATCH_SIZE, drop_remainder=True)
    .map(preprocess, num_parallel_calls=tf.data.AUTOTUNE)
    .prefetch(tf.data.AUTOTUNE)
)

final_model_dir = os.path.join(MODEL_DIR, "final")
os.makedirs(final_model_dir, exist_ok=True)

tf.keras.backend.clear_session()
gc.collect()
generator_final = build_generator(Z_DIM)
critic_final    = build_critic()

cgan_final = ConditionalWGAN(
    critic=critic_final, generator=generator_final,
    latent_dim=Z_DIM, critic_steps=CRITIC_STEPS, gp_weight=GP_WEIGHT,
)
cgan_final.compile(
    c_optimizer=optimizers.Adam(learning_rate=BEST_LR, beta_1=BETAS[0], beta_2=BETAS[1], weight_decay=1e-3),
    g_optimizer=optimizers.Adam(learning_rate=BEST_LR, beta_1=BETAS[0], beta_2=BETAS[1]),
)

history_final = cgan_final.fit(
    final_dataset,
    epochs=BEST_EPOCH,
    callbacks=[SaveModels(save_dir=final_model_dir, latent_dim=Z_DIM, save_freq=50)],
    verbose=1,
)

pd.DataFrame(history_final.history).to_csv(os.path.join(final_model_dir, "history.csv"), index_label="epoch")
print(f"Final training complete. g_loss={history_final.history['g_loss'][-1]:.4f}")

# ── Cell 11: Synthetic data generation + stratified split ────────────────────
unique_classes, class_counts = np.unique(y_final, return_counts=True)
print("Generating synthetic data matching y_final class distribution:")
for cls, cnt in zip(unique_classes, class_counts):
    print(f"  Class {cls}: {cnt} samples")

synthetic_images, synthetic_labels = [], []
for class_idx, n_samples in zip(unique_classes, class_counts):
    z       = tf.random.normal(shape=(n_samples, Z_DIM))
    one_hot = tf.convert_to_tensor(np.eye(CLASSES)[class_idx:class_idx+1].repeat(n_samples, axis=0), dtype=tf.float32)
    generated = cgan_final.generator([z, one_hot], training=False)
    generated = (generated * 127.5 + 127.5).numpy().clip(0, 255).astype("uint8")
    synthetic_images.append(generated)
    synthetic_labels.append(np.full(n_samples, class_idx, dtype=np.int64))

X_synthetic = np.concatenate(synthetic_images, axis=0)
y_synthetic = np.concatenate(synthetic_labels, axis=0)
print(f"X_synthetic: {X_synthetic.shape}, y_synthetic: {y_synthetic.shape}")

val_ratio = len(X_val) / (len(X_train) + len(X_val))
X_synth_train, X_synth_val, y_synth_train, y_synth_val = train_test_split(
    X_synthetic, y_synthetic, test_size=val_ratio, stratify=y_synthetic, random_state=SEED,
)
print(f"Synthetic train: {X_synth_train.shape}  Synthetic val: {X_synth_val.shape}")

os.makedirs(SYNTH_DIR, exist_ok=True)
save_path = os.path.join(SYNTH_DIR, "path_mnist_synthetic.npz")
np.savez(save_path, X_train=X_synth_train, y_train=y_synth_train, X_val=X_synth_val, y_val=y_synth_val)
print(f"Saved synthetic dataset to: {save_path}")