"""Small TensorFlow classifier example for the classical side of hybrid QML."""


def train_tiny_classifier(epochs: int = 20) -> float:
    try:
        import numpy as np
        import tensorflow as tf
    except ImportError as exc:
        raise SystemExit("Install the unified stack: pip install -r requirements.txt") from exc

    tf.random.set_seed(7)
    x = np.array(
        [
            [-1.0, -1.0],
            [-1.0, 1.0],
            [1.0, -1.0],
            [1.0, 1.0],
        ],
        dtype="float32",
    )
    y = np.array([[0.0], [1.0], [1.0], [0.0]], dtype="float32")

    model = tf.keras.Sequential(
        [
            tf.keras.layers.Input(shape=(2,)),
            tf.keras.layers.Dense(4, activation="tanh"),
            tf.keras.layers.Dense(1, activation="sigmoid"),
        ]
    )
    model.compile(optimizer=tf.keras.optimizers.Adam(0.05), loss="binary_crossentropy")
    model.fit(x, y, epochs=epochs, verbose=0)
    loss = model.evaluate(x, y, verbose=0)
    return float(loss)


if __name__ == "__main__":
    print(f"final_loss={train_tiny_classifier():.6f}")

