import joblib
import numpy as np
import matplotlib.pyplot as plt
from skimage import io, color, transform

svm = joblib.load("../model.pkl")


# -------------------------
# PREDICT NEW IMAGES + EXPLAIN
# -------------------------
def predict_image(path):
    img = preprocess_image(path)
    gray = color.rgb2gray(img)

    hog_features, hog_image = extract_hog_features(gray)
    color_features = extract_color_features(img)

    features = np.concatenate([hog_features, color_features])
    features = scaler.transform([features])

    prediction = svm.predict(features)[0]
    explanation = explain_prediction(img)

    # plt.figure()
    # plt.imshow(img)
    # plt.axis('off')
    # plt.title(f"Prediction: {prediction}")
    # plt.show()

    # Show original + HOG
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))

    axes[0].imshow(img)
    axes[0].set_title("Original")
    axes[0].axis("off")

    axes[1].imshow(hog_image, cmap='gray')
    axes[1].set_title("HOG (edges)")
    axes[1].axis("off")

    plt.show()

    print("\nImage:", path)
    print("Prediction:", prediction)
    print("Reason:")
    for e in explanation:
        print("-", e)


# -------------------------
# TEST ON NEW IMAGES
# -------------------------
test_folder = "../TestImages"

if os.path.exists(test_folder):
    for file in os.listdir(test_folder):

        if not file.lower().endswith(('.png', '.jpg', '.jpeg')):
            continue

        predict_image(os.path.join(test_folder, file))