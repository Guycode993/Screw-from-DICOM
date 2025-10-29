# Screw-from-DICOM
Detection of cervical screw brands from anonymized DICOM images
# just with raw PNG files with noise reduction and contrast enhancement functions.
import os
import cv2
import numpy as np
from sklearn.cluster import MiniBatchKMeans
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from tensorflow.keras.applications import ResNet50
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.resnet50 import preprocess_input
from tqdm import tqdm

# -------------------------------
# CONFIG
# -------------------------------
dataset_root = r"C:\Users\Ramesh\anaconda3\envs\imagecv2\ML health projects\png_dataset_5"
num_clusters = 200  # visual words
target_size = (224, 224)  # for ResNet
use_sobel = True
cache_dir = "feature_cache"

# Ensure cache folder exists
os.makedirs(cache_dir, exist_ok=True)

X_cache = os.path.join(cache_dir, "X.npy")
y_cache = os.path.join(cache_dir, "y.npy")
kmeans_cache = os.path.join(cache_dir, "kmeans_model.npy")

# -------------------------------
# STEP 1 — Noise Reduction and Contrast Enhancement
# -------------------------------
def noise_reduction(img):
    # Bilateral filter preserves edges while reducing noise
    return cv2.bilateralFilter(img, d=9, sigmaColor=75, sigmaSpace=75)

def contrast_enhancement(img):
    # CLAHE (Contrast Limited Adaptive Histogram Equalization)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
    return clahe.apply(img)

def preprocess_image(img_path):
    img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        return None
    img = noise_reduction(img)
    img = contrast_enhancement(img)
    return img

# -------------------------------
# STEP 2 — Sobel Edge Image (with noise reduction + contrast enhancement)
# -------------------------------
def sobel_edge_image(img_path):
    img = preprocess_image(img_path)
    if img is None:
        return None

    img = img.astype(np.float32)
    sobelx = cv2.Sobel(img, cv2.CV_32F, 1, 0, ksize=3)
    sobely = cv2.Sobel(img, cv2.CV_32F, 0, 1, ksize=3)
    sobel = np.sqrt(sobelx**2 + sobely**2)
    if np.max(sobel) > 0:
        sobel = (sobel / np.max(sobel)) * 255.0
    sobel = sobel.astype(np.uint8)
    return sobel

# -------------------------------
# STEP 3 — Build or Load BoVW Dictionary
# -------------------------------
print("📦 Checking for cached visual vocabulary...")

if os.path.exists(kmeans_cache):
    print("✅ Found cached KMeans model.")
    kmeans_data = np.load(kmeans_cache, allow_pickle=True).item()
    kmeans = kmeans_data["kmeans"]
    kaze = cv2.KAZE_create()
else:
    print("⚙️ Building new KMeans visual vocabulary...")
    descriptor_list = []
    kaze = cv2.KAZE_create()

    for cls in os.listdir(dataset_root):
        class_path = os.path.join(dataset_root, cls)
        if not os.path.isdir(class_path):
            continue
        for fname in os.listdir(class_path):
            if not fname.lower().endswith(('.jpg', '.jpeg', '.png')):
                continue
            img_path = os.path.join(class_path, fname)
            img = preprocess_image(img_path)
            if img is None:
                continue
            if use_sobel:
                img_sobel = sobel_edge_image(img_path)
                if img_sobel is not None:
                    kps, des = kaze.detectAndCompute(img_sobel, None)
                    if des is not None and len(des) > 0:
                        descriptor_list.extend(des.astype(np.float32))
            else:
                kps, des = kaze.detectAndCompute(img, None)
                if des is not None and len(des) > 0:
                    descriptor_list.extend(des.astype(np.float32))

    if not descriptor_list:
        raise RuntimeError("No KAZE descriptors found. Check dataset or file types.")

    descriptor_array = np.vstack(descriptor_list).astype(np.float32)
    kmeans = MiniBatchKMeans(n_clusters=num_clusters, random_state=42, batch_size=1000)
    kmeans.fit(descriptor_array)
    np.save(kmeans_cache, {"kmeans": kmeans})
    print("✅ KMeans visual vocabulary saved to cache.")

# -------------------------------
# STEP 4 — Helper Functions for BoVW with preprocessing options
# -------------------------------
def compute_bovw(img_path, kaze, kmeans, use_sobel=False, use_preprocessing=False):
    if use_sobel:
        img = sobel_edge_image(img_path)
        if img is None:
            return np.zeros(kmeans.n_clusters, dtype=np.float32)
    elif use_preprocessing:
        img = preprocess_image(img_path)
        if img is None:
            return np.zeros(kmeans.n_clusters, dtype=np.float32)
    else:
        img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
        if img is None:
            return np.zeros(kmeans.n_clusters, dtype=np.float32)

    kps, des = kaze.detectAndCompute(img, None)
    if des is None or len(des) == 0:
        return np.zeros(kmeans.n_clusters, dtype=np.float32)
    preds = kmeans.predict(des.astype(np.float32))
    hist, _ = np.histogram(preds, bins=np.arange(kmeans.n_clusters + 1))
    hist = hist.astype(np.float32)
    hist /= (hist.sum() + 1e-6)
    return hist

# -------------------------------
# STEP 5 — CNN Feature Extraction
# -------------------------------
print("🔍 Loading ResNet50 for feature extraction...")
resnet_model = ResNet50(weights='imagenet', include_top=False, pooling='avg')

def extract_resnet_features(img_path, model, use_sobel=False, target_size=(224,224)):
    if use_sobel:
        sobel = sobel_edge_image(img_path)
        if sobel is None:
            return np.zeros(model.output_shape[1], dtype=np.float32)
        sobel = cv2.cvtColor(sobel, cv2.COLOR_GRAY2RGB)
        sobel = cv2.resize(sobel, target_size)
        x = np.expand_dims(sobel.astype(np.float32), axis=0)
    else:
        img = image.load_img(img_path, target_size=target_size)
        x = np.expand_dims(image.img_to_array(img).astype(np.float32), axis=0)
    x = preprocess_input(x)
    features = model.predict(x, verbose=0)
    return features.flatten().astype(np.float32)

# -------------------------------
# STEP 6 — Load Cached Features or Extract
# -------------------------------
if os.path.exists(X_cache) and os.path.exists(y_cache):
    print("✅ Found cached features. Loading from disk...")
    X = np.load(X_cache)
    y = np.load(y_cache)
else:
    print("🚀 Extracting features (BoVW + CNN + Sobel + Preprocessing)...")
    X, y = [], []
    for cls in os.listdir(dataset_root):
        class_path = os.path.join(dataset_root, cls)
        if not os.path.isdir(class_path):
            continue
        for fname in tqdm(os.listdir(class_path), desc=f"Class {cls}"):
            if not fname.lower().endswith(('.jpg', '.jpeg', '.png')):
                continue
            img_path = os.path.join(class_path, fname)

            # Compute BoVW histograms with preprocessing
            bovw_orig = compute_bovw(img_path, kaze, kmeans, use_sobel=False, use_preprocessing=True)
            bovw_sobel = compute_bovw(img_path, kaze, kmeans, use_sobel=True)

            # Extract CNN features
            cnn_orig = extract_resnet_features(img_path, resnet_model, use_sobel=False)
            cnn_sobel = extract_resnet_features(img_path, resnet_model, use_sobel=True)

            # Concatenate all features
            hybrid = np.concatenate([bovw_orig, bovw_sobel, cnn_orig, cnn_sobel]).astype(np.float32)
            X.append(hybrid)
            y.append(cls)

    X = np.array(X, dtype=np.float32)
    y = np.array(y)
    np.save(X_cache, X)
    np.save(y_cache, y)
    print("💾 Features saved to cache for future runs.")

print(f"✅ Feature matrix shape: {X.shape}")

# -------------------------------
# STEP 7 — Normalize + Train SVM
# -------------------------------
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y, test_size=0.2, random_state=42, stratify=y
)

clf = SVC(kernel='rbf', C=10, gamma='scale')
clf.fit(X_train, y_train)
y_pred = clf.predict(X_test)

print("\n📊 Classification Report:")
print(classification_report(y_test, y_pred))
