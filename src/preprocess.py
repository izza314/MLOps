import os
import numpy as np
import yaml
from sklearn.model_selection import train_test_split


def main():
    os.makedirs("data/processed", exist_ok=True)

    with open("params.yaml", "r") as f:
        params = yaml.safe_load(f)

    val_size = params["preprocess"]["validation_split"]
    random_state = params["preprocess"]["random_state"]

    x_train = np.load("data/raw/x_train.npy")
    y_train = np.load("data/raw/y_train.npy")
    x_test = np.load("data/raw/x_test.npy")
    y_test = np.load("data/raw/y_test.npy")

    x_train = x_train.astype("float32") / 255.0
    x_test = x_test.astype("float32") / 255.0

    mean = np.mean(x_train)
    std = np.std(x_train)

    x_train = (x_train - mean) / std
    x_test = (x_test - mean) / std

    x_train, x_val, y_train, y_val = train_test_split(
        x_train,
        y_train,
        test_size=val_size,
        random_state=random_state
    )

    np.save("data/processed/x_train.npy", x_train)
    np.save("data/processed/y_train.npy", y_train)
    np.save("data/processed/x_val.npy", x_val)
    np.save("data/processed/y_val.npy", y_val)
    np.save("data/processed/x_test.npy", x_test)
    np.save("data/processed/y_test.npy", y_test)

    print("Preprocessing completed.")
    print("Train:", x_train.shape, y_train.shape)
    print("Validation:", x_val.shape, y_val.shape)
    print("Test:", x_test.shape, y_test.shape)
    print("Pixel range:", x_train.min(), "to", x_train.max())


if __name__ == "__main__":
    main()