import tensorflow_datasets as tfds

dataset, info = tfds.load("gtzan", with_info=True, as_supervised=True)
train_ds = dataset["train"]
test_ds = dataset["test"]

NUM_CLASSES = info.features["label"].num_classes
AUTOTUNE = tf.data.AUTOTUNE


def waveform_to_melspectrogram(audio, label):
    
    target_len = 660000
    audio = audio[:target_len]
    zero_padding = tf.maximum(target_len - tf.shape(audio)[0], 0)
    audio = tf.pad(audio, [[0, zero_padding]])

    spectrogram = tf.signal.stft(audio, frame_length=1024, frame_step=512)
    spectrogram = tf.abs(spectrogram)

    num_mel_bins = 128
    linear_to_mel_weight_matrix = tf.signal.linear_to_mel_weight_matrix(
        num_mel_bins, spectrogram.shape[-1], 22050, 80.0, 8000.0
    )
    mel_spectrogram = tf.tensordot(spectrogram, linear_to_mel_weight_matrix, 1)
    mel_spectrogram.set_shape(spectrogram.shape[:-1].concatenate([num_mel_bins]))

    mel_spectrogram = tf.expand_dims(mel_spectrogram, -1)
    return mel_spectrogram, label


BATCH_SIZE = 16

train_ds = train_ds.map(waveform_to_melspectrogram, num_parallel_calls=AUTOTUNE)
train_ds = train_ds.shuffle(1000).batch(BATCH_SIZE).prefetch(AUTOTUNE)

test_ds = test_ds.map(waveform_to_melspectrogram, num_parallel_calls=AUTOTUNE)
test_ds = test_ds.batch(BATCH_SIZE).prefetch(AUTOTUNE)


model = tf.keras.Sequential([
    tf.keras.layers.Conv2D(32, (3,3), activation="relu", input_shape=(None, None, 1)),
    tf.keras.layers.MaxPooling2D((2,2)),
    
    tf.keras.layers.Conv2D(64, (3,3), activation="relu"),
    tf.keras.layers.MaxPooling2D((2,2)),
    
    tf.keras.layers.Conv2D(128, (3,3), activation="relu"),
    tf.keras.layers.MaxPooling2D((2,2)),
    
    tf.keras.layers.GlobalAveragePooling2D(),
    tf.keras.layers.Dense(64, activation="relu"),
    tf.keras.layers.Dense(NUM_CLASSES, activation="softmax")
])

model.compile(
    optimizer="adam",
    loss="sparse_categorical_crossentropy",
    metrics=["accuracy"]
)

model.summary()

EPOCHS = 10
history = model.fit(train_ds, validation_data=test_ds, epochs=EPOCHS)

test_loss, test_acc = model.evaluate(test_ds)
print(f"Test accuracy: {test_acc:.2f}")