import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow import keras
from matplotlib import pyplot as plt
from sklearn.model_selection import train_test_split

df = pd.read_csv("/content/drive/MyDrive/Colab Notebooks/digit_recognizer/train.csv")
df_test = pd.read_csv("/content/drive/MyDrive/Colab Notebooks/digit_recognizer/test.csv")
df_test_labels = pd.read_csv("/content/drive/MyDrive/Colab Notebooks/digit_recognizer/sample_submission.csv")

model = tf.keras.Sequential([
    tf.keras.layers.Dense(28 * 28, activation="relu"),
    tf.keras.layers.Dense(128, activation="relu"),
    tf.keras.layers.Dense(32, activation="tanh"),
    tf.keras.layers.Dense(10)                                                    
])

model.compile(optimizer=keras.optimizers.Adam(learning_rate=.002),
              loss = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
              metrics=['accuracy'])

x_train, y_train = df.iloc[:, 1:].to_numpy(), df["label"].to_numpy()
x_test, y_test = df_test.to_numpy(), df_test_labels.to_numpy()

model.fit(x_train, y_train, epochs=20)

plt.figure(figsize=(10, 10))
for i in range(36):
    pred = model.predict(x_test[i].reshape(-1, 784))[0].tolist()
    pred = pred.index(max(pred))
    plt.subplot(6, 6, i+1)
    plt.imshow(x_test[i].reshape(28, 28))
    plt.xlabel("Prediction: {}".format(pred))
    plt.xticks([])
    plt.yticks([])
plt.show()