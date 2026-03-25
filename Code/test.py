import os
import joblib
import numpy as np
import matplotlib.pyplot as plt
from skimage.color import rgb2hsv

from skimage.feature import hog
from skimage import io, color, transform

from sklearn.model_selection import train_test_split
from sklearn.svm import LinearSVC
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.preprocessing import StandardScaler

from lime import lime_image
from skimage.segmentation import mark_boundaries
from skimage.segmentation import slic

def segmenter(img):
    return slic(img, n_segments=70, compactness=10)


# Preset paths
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


# Preprocessing
def preprocess_image(path):
    img = io.imread(path)
    img = transform.resize(img, (160, 160))

    if len(img.shape) == 2:
        img = np.stack([img] * 3, axis=-1)
    elif img.shape[2] == 4:
        img = img[:, :, :3]

    return img


# Hog Features
def extract_hog_features(gray_img):
    h = gray_img.shape[0]
    top = gray_img[:h//2, :]
    bot = gray_img[h//2:, :]

    hog_top = hog(
        top,
        orientations=8,
        pixels_per_cell=(8, 8),
        cells_per_block=(2, 2),
        block_norm='L2-Hys',
    )
    hog_bot = hog(
        bot,
        orientations=8,
        pixels_per_cell=(8, 8),
        cells_per_block=(2, 2),
        block_norm='L2-Hys',
    )
    hog_full, hog_img = hog(
        gray_img,
        orientations=8,
        pixels_per_cell=(8, 8),
        cells_per_block=(2, 2),
        block_norm='L2-Hys',
        visualize=True
    )
    # Normalization for visuals
    hog_img = (hog_img - hog_img.min()) / (hog_img.max() - hog_img.min())

    hog_feat = np.concatenate([hog_top*0, hog_bot])

    return hog_feat, hog_img


# Color Features
def extract_color_features(img):
    hist_r = np.histogram(img[:, :, 0], bins=16, range=(0, 1))[0]
    hist_g = np.histogram(img[:, :, 1], bins=16, range=(0, 1))[0]
    hist_b = np.histogram(img[:, :, 2], bins=16, range=(0, 1))[0]
    return np.concatenate([hist_r, hist_g, hist_b])

# HSV Features
def extract_hsv_features(img):
    h = img.shape[0]

    img = rgb2hsv(img)
    img_top = img[:h//2, :, :]
    img_bot = img[h//2:, :, :]

    # For the top part of the image we try to mask out the sky parts
    h_top, s_top, v_top = img_top[:, :, 0], img_top[:, :, 1], img_top[:, :, 2]

    thresh_h = 0.58
    sky_mask = (h_top > thresh_h)
    ground_mask = ~sky_mask

    hist_h_top = np.histogram(h_top[ground_mask], bins=16, range=(0, 1))[0]
    hist_s_top = np.histogram(s_top[ground_mask], bins=16, range=(0, 1))[0]
    hist_v_top = np.histogram(v_top[ground_mask], bins=16, range=(0, 1))[0]
    hist_top = np.concatenate([hist_h_top, hist_s_top, hist_v_top])


    hist_h_bot = np.histogram(img_bot[:, :, 0], bins=16, range=(0, 1))[0]
    hist_s_bot = np.histogram(img_bot[:, :, 1], bins=16, range=(0, 1))[0]
    hist_v_bot = np.histogram(img_bot[:, :, 2], bins=16, range=(0, 1))[0]
    hist_bot = np.concatenate([hist_h_bot, hist_s_bot, hist_v_bot])

    # plt.imshow(h_top*ground_mask)
    # plt.show()

    return np.concatenate([hist_top, hist_bot])


# Lime Predict
def lime_predict(images):
    features_list = []

    for img in images:
        # Resize // preprocessing
        img = transform.resize(img, (160, 160))

        if len(img.shape) == 2:
            img = np.stack([img]*3, axis=-1)
        elif img.shape[2] == 4:
            img = img[:, :, :3]

        gray = color.rgb2gray(img)

        hog_features, _ = extract_hog_features(gray)
        hsv_features = extract_hsv_features(img)

        features = np.concatenate([hog_features, hsv_features])
        features_list.append(features)

    features_array = scaler.transform(features_list)

    return svm.decision_function(features_array)


def explain_with_lime(path):
    img = preprocess_image(path)

    explainer = lime_image.LimeImageExplainer()

    explanation = explainer.explain_instance(
        img,
        lambda x: lime_predict(x),
        top_labels=1,
        hide_color=0,
        num_samples=500,
        segmentation_fn = segmenter
    )

    temp, mask = explanation.get_image_and_mask(
        explanation.top_labels[0],
        positive_only=True,
        num_features=5,
        hide_rest=True
    )

    return temp, mask


# Training model here --------------------------------------
# Loading dataset
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
        count = 0

        if not os.path.isdir(folder):
            continue

        for file in os.listdir(folder):
            if not file.lower().endswith(('.png', '.jpg', '.jpeg')):
                continue

            path = os.path.join(folder, file)

            try:
                img = preprocess_image(path)
                gray = color.rgb2gray(img)

                hog_features, _  = extract_hog_features(gray)
                hsv_features = extract_hsv_features(img)

                features = np.concatenate([hog_features, hsv_features])
                X.append(features)
                y.append(label)
                class_counts[label] += 1

                count += 1
                if count % 500 == 0:
                    print(f"Processed {count} images...")
                    break

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


# Train and save model
def train_and_save_model():
    X, y = load_dataset()

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, stratify=y
    )

    print("\nTrain/Test split:")
    print(f"Training samples: {len(X_train)}")
    print(f"Testing samples: {len(X_test)}")

    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    print("\nTraining SVM...")
    svm = LinearSVC(C=1.0, max_iter=5000)
    #svm = SVC(kernel="rbf")
    svm.fit(X_train, y_train)


    svm_preds = svm.predict(X_test)

    print("MODEL EVALUATION")

    print("\nSVM accuracy:", accuracy_score(y_test, svm_preds))
    print("\nSVM classification report:")
    print(classification_report(y_test, svm_preds))
    print("SVM confusion matrix:")
    print(confusion_matrix(y_test, svm_preds))


    joblib.dump({
        "svm": svm,
        "scaler": scaler,
        "acc": accuracy_score(y_test, svm_preds),
    }, MODEL_PATH)

    print(f"\nModel saved successfully to {MODEL_PATH}")

    return svm, scaler


# load model (or call the train and save model function if no model found)
def load_or_train_model():
    if os.path.exists(MODEL_PATH):
        print(f"Found trained model: {MODEL_PATH}")
        print("Loading saved model...")

        data = joblib.load(MODEL_PATH)
        svm = data["svm"]
        scaler = data["scaler"]
        acc = data["acc"]

        print("Model loaded successfully.")
        print(f"Accuracy: {acc}")

        return svm, scaler

    else:
        print("No trained model found.")
        print("Training new model...")
        return train_and_save_model()


# Predict new images
def predict_image(path, svm, scaler):
    img = preprocess_image(path)
    gray = color.rgb2gray(img)

    hog_features, hog_image = extract_hog_features(gray)
    hsv_features = extract_hsv_features(img)

    features = np.concatenate([hog_features, hsv_features])
    features = scaler.transform([features])

    svm_prediction = svm.predict(features)[0]

    print("\n-------------------------")
    print("Image:", path)
    print("SVM Prediction:", svm_prediction)

    temp, mask = explain_with_lime(path)
    lime_img = mark_boundaries(temp, mask)

    fig, axes = plt.subplots(2, 3, figsize=(7, 7))

    axes[0, 0].imshow(img)
    axes[0, 0].set_title(f"OG img | SVM Prediction: {svm_prediction}")
    axes[0, 0].axis("off")

    #hog_mask = mark_boundaries(hog_image, mask)
    axes[0, 1].imshow(hog_image, cmap="gray")
    axes[0, 1].set_title("HOG Image")
    axes[0, 1].axis("off")

    axes[0, 2].imshow(lime_img)
    axes[0, 2].set_title("LIME explanation")
    axes[0, 2].axis("off")

    hsv_img = color.rgb2hsv(img)

    axes[1, 0].imshow(hsv_img[:, :, 0], cmap="hsv")
    axes[1, 0].set_title("Hue Image")
    axes[1, 0].axis("off")

    axes[1, 1].imshow(hsv_img[:, :, 1], cmap="hsv")
    axes[1, 1].set_title("Saturation Image")
    axes[1, 1].axis("off")

    axes[1, 2].imshow(hsv_img[:, :, 2], cmap="hsv")
    axes[1, 2].set_title("Value Image")
    axes[1, 2].axis("off")

    plt.tight_layout()
    # plt.show()

    # Create output folder if it doesn't exist
    os.makedirs("Output_2", exist_ok=True)

    # Create filename based on image name
    base_name = os.path.basename(path)
    name_without_ext = os.path.splitext(base_name)[0]

    save_path = os.path.join("Output_2", f"{name_without_ext}_result.png")

    plt.savefig(save_path, dpi=300)
    plt.close()  # VERY important when looping

    print(f"Saved figure to: {save_path}")


# Loading new images and predicts
svm, scaler = load_or_train_model()

print("NEW IMAGE PREDICTIONS")
# Go through TestImages folder and predict images

if os.path.exists(test_folder):
    found_images = False

    for file in os.listdir(test_folder):
        if not file.lower().endswith(('.png', '.jpg', '.jpeg')):
            continue

        found_images = True
        predict_image(os.path.join(test_folder, file), svm, scaler)

    if not found_images:
        print("No image files found in ../TestImages")
else:
    print("Test folder does not exist:", test_folder)