"""
train_model.py
---------------
Trains and evaluates a neural network classifier
to detect attention states (Focused / Unfocused / Drowsy)
from preprocessed EEG data.
"""

import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.utils.class_weight import compute_class_weight
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv1D, MaxPooling1D, Flatten, Dense, Dropout, BatchNormalization
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau

from preprocessing import preprocess_data


def train_eeg_model():
    """Loads preprocessed data, builds, trains, and evaluates the model."""
    data = preprocess_data()
    X, y = data["features"], data["labels"]

    print(f"Features: {X.shape}, Labels: {y.shape}")

    # Normalize EEG features per channel
    X = (X - np.mean(X, axis=0)) / (np.std(X, axis=0) + 1e-8)

    X = X.reshape(X.shape[0], X.shape[1], X.shape[2])  # (samples, 512, 14)
    y = to_categorical(y, num_classes=3)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=np.argmax(y, axis=1)
    )

    # Compute class weights for imbalance handling
    y_integers = np.argmax(y_train, axis=1)
    class_weights = compute_class_weight('balanced', classes=np.unique(y_integers), y=y_integers)
    class_weights = dict(enumerate(class_weights))
    print("Class Weights:", class_weights)

    # Build improved CNN
    model = Sequential([
        Conv1D(64, 5, activation='relu', input_shape=(X_train.shape[1], X_train.shape[2])),
        BatchNormalization(),
        MaxPooling1D(2),
        Dropout(0.25),

        Conv1D(128, 3, activation='relu'),
        BatchNormalization(),
        MaxPooling1D(2),
        Dropout(0.25),

        Conv1D(256, 3, activation='relu'),
        BatchNormalization(),
        MaxPooling1D(2),
        Dropout(0.25),

        Flatten(),
        Dense(128, activation='relu'),
        BatchNormalization(),
        Dropout(0.3),

        Dense(3, activation='softmax')
    ])

    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

    checkpoint_path = "model/best_model.keras"

    callbacks = [
        EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True),
        ModelCheckpoint(filepath=checkpoint_path, monitor='val_accuracy', save_best_only=True),
        ReduceLROnPlateau(monitor='val_accuracy', factor=0.5, patience=3, min_lr=1e-6)
    ]

    history = model.fit(
        X_train, y_train,
        epochs=30,
        batch_size=32,
        validation_split=0.2,
        callbacks=callbacks,
        class_weight=class_weights,
        verbose=1
    )

    loss, acc = model.evaluate(X_test, y_test, verbose=0)
    print(f"\nTest Accuracy: {acc * 100:.2f}%")

    y_pred = np.argmax(model.predict(X_test), axis=1)
    y_true = np.argmax(y_test, axis=1)

    print("\nClassification Report:")
    print(classification_report(y_true, y_pred, target_names=["Focused", "Unfocused", "Drowsy"]))
    print("\nConfusion Matrix:")
    print(confusion_matrix(y_true, y_pred))

    model.save("model/eeg_attention_model.keras")
    print("\nModel saved to 'model/eeg_attention_model.keras'")

    return model, X_test, y_test


if __name__ == "_main_":
    train_eeg_model()