import os
import numpy as np
import pandas as pd
import yaml
import tensorflow as tf


def main():
    os.makedirs("models", exist_ok=True)

    # Load parameters
    with open("params.yaml", "r") as f:
        params = yaml.safe_load(f)

    train_params = params["train"]

    # Load processed data
    x_train = np.load("data/processed/x_train.npy")
    y_train = np.load("data/processed/y_train.npy")
    x_val = np.load("data/processed/x_val.npy")
    y_val = np.load("data/processed/y_val.npy")

    # Build model
    model = tf.keras.Sequential([
        tf.keras.layers.Flatten(input_shape=(28, 28)),
        tf.keras.layers.Dense(
            train_params["hidden_units"],
            activation="relu"
        ),
        tf.keras.layers.Dropout(
            train_params["dropout"]
        ),
        tf.keras.layers.Dense(10, activation="softmax")
    ])

    # Compile model
    optimizer = tf.keras.optimizers.Adam(
        learning_rate=train_params["learning_rate"]
    )

    model.compile(
        optimizer=optimizer,
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"]
    )

    # Train model
    history = model.fit(
        x_train,
        y_train,
        validation_data=(x_val, y_val),
        epochs=train_params["epochs"],
        batch_size=train_params["batch_size"]
    )

    # Save model
    model.save("models/model.h5")

    # Save training history
    history_df = pd.DataFrame(history.history)
    history_df.to_csv("models/history.csv", index=False)

    print("Training completed.")
    print("Model saved to models/model.h5")
    print("History saved to models/history.csv")


if __name__ == "__main__":
    main()