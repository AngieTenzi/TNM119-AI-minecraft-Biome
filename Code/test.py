import os
import joblib
import numpy as np
import matplotlib.pyplot as plt

from skimage.feature import hog
from skimage import io, color, transform

from sklearn.model_selection import train_test_split
from sklearn.svm import LinearSVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.preprocessing import StandardScaler


# -------------------------
# CONFIG
# -------------------------
dataset_path = "../BiomeData"
test_folder = "../TestImages"
MODEL_PATH = "model_bundle.pkl"

biome_map = {
    "biome_1": "Plains",
    "biome_4": "Forest",
    "biome_3": "Mountains",
    "biome_5": "Taiga",
    "biome_2": "Desert",
    "biome_12": "Snow-Tundra",
    "biome_35": "Savanna",
    "biome_16": "Beach"
}


# -------------------------
# PREPROCESSING
# -------------------------
def preprocess_image(path):
    img = io.imread(path)
    img = transform.resize(img, (64, 64))

    # Handle grayscale images
    if len(img.shape) == 2:
        img = np.stack([img] * 3, axis=-1)

    # Handle RGBA images
    elif img.shape[2] == 4:
        img = img[:, :, :3]

    return img


# -------------------------
# HOG FEATURES
# -------------------------
def extract_hog_features(gray_img):
    return hog(
        gray_img,
        orientations=8,
        pixels_per_cell=(16, 16),
        cells_per_block=(2, 2),
        block_norm='L2-Hys'
    )


# -------------------------
# COLOR FEATURES
# -------------------------
def extract_color_features(img):
    hist_r = np.histogram(img[:, :, 0], bins=16, range=(0, 1))[0]
    hist_g = np.histogram(img[:, :, 1], bins=16, range=(0, 1))[0]
    hist_b = np.histogram(img[:, :, 2], bins=16, range=(0, 1))[0]
    return np.concatenate([hist_r, hist_g, hist_b])


# -------------------------
# EXPLANATION FUNCTION
# -------------------------
def explain_prediction(img):
    avg_color = np.mean(img, axis=(0, 1))
    brightness = np.mean(color.rgb2gray(img))

    explanation = []

    if avg_color[0] > 0.6 and avg_color[1] > 0.6:
        explanation.append("Bright yellowish colors -> likely desert or beach")

    if avg_color[2] > avg_color[0]:
        explanation.append("Bluish tones -> could be snow or water nearby")

    if brightness < 0.4:
        explanation.append("Dark image -> dense biome like forest or taiga")

    if brightness > 0.7:
        explanation.append("Very bright -> open biome like plains or desert")

    if not explanation:
        explanation.append("Biome determined mainly from texture patterns")

    return explanation


# -------------------------
# LOAD DATASET
# -------------------------
def load_dataset():
    X = []
    y = []
    class_counts = {name: 0 for name in biome_map.values()}
    count = 0

    print("Loading dataset...")

    for folder_name in os.listdir(dataset_path):
        if folder_name not in biome_map:
            continue

        label = biome_map[folder_name]
        folder = os.path.join(dataset_path, folder_name)

        if not os.path.isdir(folder):
            continue

        for file in os.listdir(folder):
            if not file.lower().endswith(('.png', '.jpg', '.jpeg')):
                continue

            path = os.path.join(folder, file)

            try:
                img = preprocess_image(path)
                gray = color.rgb2gray(img)

                hog_features = extract_hog_features(gray)
                color_features = extract_color_features(img)
                features = np.concatenate([hog_features, color_features])

                X.append(features)
                y.append(label)
                class_counts[label] += 1

                count += 1
                if count % 500 == 0:
                    print(f"Processed {count} images...")

            except Exception as e:
                print(f"Error processing {path}: {e}")

    X = np.array(X)
    y = np.array(y)

    print("\nDataset loaded successfully.")
    print(f"Total images: {len(X)}")
    print("\nImages per class:")
    for biome, num in class_counts.items():
        print(f"{biome}: {num}")

    return X, y


# -------------------------
# TRAIN AND SAVE MODEL
# -------------------------
def train_and_save_model():
    X, y = load_dataset()

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    print("\nTrain/Test split:")
    print(f"Training samples: {len(X_train)}")
    print(f"Testing samples: {len(X_test)}")

    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    print("\nTraining SVM...")
    svm = LinearSVC(C=1.0, max_iter=5000)
    svm.fit(X_train, y_train)

    print("Training Random Forest...")
    rf = RandomForestClassifier(n_estimators=150, max_depth=20, random_state=42)
    rf.fit(X_train, y_train)

    svm_preds = svm.predict(X_test)
    rf_preds = rf.predict(X_test)

    print("\n====================")
    print("MODEL EVALUATION")
    print("====================")

    print("\nSVM accuracy:", accuracy_score(y_test, svm_preds))
    print("\nSVM classification report:")
    print(classification_report(y_test, svm_preds))
    print("SVM confusion matrix:")
    print(confusion_matrix(y_test, svm_preds))

    print("\nRandom Forest accuracy:", accuracy_score(y_test, rf_preds))
    print("\nRandom Forest classification report:")
    print(classification_report(y_test, rf_preds))
    print("Random Forest confusion matrix:")
    print(confusion_matrix(y_test, rf_preds))

    joblib.dump({
        "svm": svm,
        "rf": rf,
        "scaler": scaler
    }, MODEL_PATH)

    print(f"\nModel saved successfully to {MODEL_PATH}")

    return svm, rf, scaler


# -------------------------
# LOAD OR TRAIN MODEL
# -------------------------
def load_or_train_model():
    if os.path.exists(MODEL_PATH):
        print(f"Found trained model: {MODEL_PATH}")
        print("Loading saved model...")

        data = joblib.load(MODEL_PATH)
        svm = data["svm"]
        rf = data["rf"]
        scaler = data["scaler"]

        print("Model loaded successfully.")
        return svm, rf, scaler

    else:
        print("No trained model found.")
        print("Training new model...")
        return train_and_save_model()


# -------------------------
# PREDICT NEW IMAGES + SHOW
# -------------------------
def predict_image(path, svm, rf, scaler):
    img = preprocess_image(path)
    gray = color.rgb2gray(img)

    hog_features = extract_hog_features(gray)
    color_features = extract_color_features(img)

    features = np.concatenate([hog_features, color_features])
    features = scaler.transform([features])

    svm_prediction = svm.predict(features)[0]
    rf_prediction = rf.predict(features)[0]
    explanation = explain_prediction(img)

    plt.figure()
    plt.imshow(img)
    plt.axis("off")
    plt.title(f"SVM: {svm_prediction} | RF: {rf_prediction}")
    plt.show()

    print("\n-------------------------")
    print("Image:", path)
    print("SVM Prediction:", svm_prediction)
    print("RF Prediction:", rf_prediction)
    print("Reason:")
    for e in explanation:
        print("-", e)


# -------------------------
# MAIN
# -------------------------
svm, rf, scaler = load_or_train_model()

print("\n====================")
print("NEW IMAGE PREDICTIONS")
print("====================")

if os.path.exists(test_folder):
    found_images = False

    for file in os.listdir(test_folder):
        if not file.lower().endswith(('.png', '.jpg', '.jpeg')):
            continue

        found_images = True
        predict_image(os.path.join(test_folder, file), svm, rf, scaler)

    if not found_images:
        print("No image files found in ../TestImages")
else:
    print("Test folder does not exist:", test_folder)