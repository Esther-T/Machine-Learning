I built a convolutional neural network using TensorFlow to classify music genres from raw audio using the GTZAN dataset

Key features:
- Loads GTZAN dataset using TensorFlow Datasets with supervised labels
- Converts raw audio waveforms into mel-spectrograms using STFT and mel filter banks
- Standardizes input length by trimming or zero-padding audio samples
- Uses an efficient tf.data pipeline with mapping, shuffling, batching, and prefetching
- Implements a CNN with multiple Conv2D and MaxPooling layers for feature extraction
- Supports variable input dimensions with global average pooling
- Trains using Adam optimizer and sparse categorical crossentropy loss
- Evaluates model performance on a held-out test dataset
- End-to-end pipeline from raw audio input to genre classification predictions
