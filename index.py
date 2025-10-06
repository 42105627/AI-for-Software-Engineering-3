# ---- IRIS DECISION TREE ----
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, classification_report
import pandas as pd

# 1. Load and view dataset
iris = load_iris(as_frame=True)
X, y = iris.data, iris.target

# 2. Train–test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 3. Train model
clf = DecisionTreeClassifier(random_state=42)
clf.fit(X_train, y_train)

# 4. Predict and evaluate
y_pred = clf.predict(X_test)
print("Accuracy:", accuracy_score(y_test, y_pred))
print("Precision:", precision_score(y_test, y_pred, average='macro'))
print("Recall:", recall_score(y_test, y_pred, average='macro'))
print("\nClassification Report:\n", classification_report(y_test, y_pred))



# ---- MNIST CNN ----
import tensorflow as tf
from tensorflow.keras import layers, models
import matplotlib.pyplot as plt

# 1. Load data
(x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()
x_train, x_test = x_train/255.0, x_test/255.0

# 2. Build CNN
model = models.Sequential([
    layers.Reshape((28,28,1), input_shape=(28,28)),
    layers.Conv2D(32, (3,3), activation='relu'),
    layers.MaxPooling2D(2,2),
    layers.Conv2D(64, (3,3), activation='relu'),
    layers.MaxPooling2D(2,2),
    layers.Flatten(),
    layers.Dense(128, activation='relu'),
    layers.Dense(10, activation='softmax')
])

# 3. Compile & train
model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])
history = model.fit(x_train, y_train, epochs=5,
                    validation_data=(x_test, y_test))

# 4. Evaluate
test_loss, test_acc = model.evaluate(x_test, y_test)
print("Test Accuracy:", test_acc)

# 5. Plot accuracy curve
plt.plot(history.history['accuracy'], label='train')
plt.plot(history.history['val_accuracy'], label='val')
plt.legend(), plt.title("Training vs Validation Accuracy")
plt.show()

# 6. Show 5 predictions
import numpy as np
preds = model.predict(x_test[:5])
for i in range(5):
    plt.imshow(x_test[i], cmap='gray')
    plt.title(f"Predicted: {np.argmax(preds[i])}")
    plt.show()





# ---- NER + SIMPLE SENTIMENT ----
import spacy
from textblob import TextBlob

nlp = spacy.load("en_core_web_sm")

reviews = [
    "I love my new Samsung Galaxy S21, the camera is fantastic!",
    "The battery life of this Apple Watch is terrible.",
    "Sony headphones deliver amazing sound quality."
]

for review in reviews:
    doc = nlp(review)
    print(f"\nReview: {review}")
    print("Entities:")
    for ent in doc.ents:
        print(f"  {ent.text} -> {ent.label_}")
    sentiment = TextBlob(review).sentiment.polarity
    label = "Positive" if sentiment > 0 else "Negative" if sentiment < 0 else "Neutral"
    print("Sentiment:", label)




# WRONG
model.compile(loss='categorical_crossentropy',)

# CORRECT for integer labels
model.compile(loss='sparse_categorical_crossentropy',)




# ---- streamlit_app.py ----
import streamlit as st
import numpy as np
import tensorflow as tf
from PIL import Image

model = tf.keras.models.load_model('mnist_model.h5')
st.title("🖋️ MNIST Digit Classifier")

uploaded = st.file_uploader("Upload a 28x28 grayscale digit image")
if uploaded:
    img = Image.open(uploaded).convert('L').resize((28,28))
    arr = np.array(img)/255.0
    pred = np.argmax(model.predict(arr.reshape(1,28,28,1)))
    st.image(img, caption=f"Predicted Digit: {pred}", width=150)

