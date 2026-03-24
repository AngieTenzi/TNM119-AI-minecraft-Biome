import os
import numpy as np

from skimage.feature import hog
from skimage import io, color, transform

from sklearn.model_selection import train_test_split
from sklearn.svm import LinearSVC
from sklearn.ensemble import RandomForestClassifier

from sklearn.metrics import accuracy_score
from sklearn.preprocessing import StandardScaler


# -------------------------
# PREPROCESSING
# -------------------------
def preprocess_image(path):
    img = io.imread(path)
    img = transform.resize(img, (64, 64))  # smaller = MUCH faster
    return img


# -------------------------
# HOG FEATURES (texture)
# -------------------------
def extract_hog_features(gray_img):
    features = hog(
        gray_img,
        orientations=8,
        pixels_per_cell=(16, 16),
        cells_per_block=(2, 2),
        block_norm='L2-Hys'
    )
    return features


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
        explanation.append("Bright yellowish colors → likely desert or beach")

    if avg_color[2] > avg_color[0]:
        explanation.append("Bluish tones → could be snow or water nearby")

    if brightness < 0.4:
        explanation.append("Dark image → dense biome like forest or taiga")

    if brightness > 0.7:
        explanation.append("Very bright → open biome like plains or desert")

    if not explanation:
        explanation.append("Biome determined mainly from texture patterns")

    return explanation


# -------------------------
# LOAD DATASET
# -------------------------
X = []
y = []

dataset_path = "../BiomeData"

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

count = 0

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

        img = preprocess_image(path)
        gray = color.rgb2gray(img)

        hog_features = extract_hog_features(gray)
        color_features = extract_color_features(img)

        features = np.concatenate([hog_features, color_features])

        X.append(features)
        y.append(label)

        count += 1
        if count % 500 == 0:
            print(f"Processed {count} images...")

X = np.array(X)
y = np.array(y)


# -------------------------
# TRAIN / TEST SPLIT
# -------------------------
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)


# -------------------------
# SCALE FEATURES
# -------------------------
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)


# -------------------------
# TRAIN MODELS
# -------------------------
svm = LinearSVC(C=1.0, max_iter=5000)
svm.fit(X_train, y_train)

rf = RandomForestClassifier(n_estimators=150, max_depth=20)
rf.fit(X_train, y_train)


# -------------------------
# EVALUATE
# -------------------------
svm_preds = svm.predict(X_test)
rf_preds = rf.predict(X_test)

print("\nSVM accuracy:", accuracy_score(y_test, svm_preds))
print("RF accuracy:", accuracy_score(y_test, rf_preds))


# -------------------------
# PREDICT NEW IMAGES + EXPLAIN
# -------------------------
def predict_image(path):
    img = preprocess_image(path)
    gray = color.rgb2gray(img)

    hog_features = extract_hog_features(gray)
    color_features = extract_color_features(img)

    features = np.concatenate([hog_features, color_features])
    features = scaler.transform([features])

    prediction = svm.predict(features)[0]
    explanation = explain_prediction(img)

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