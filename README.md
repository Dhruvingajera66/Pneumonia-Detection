# 🩺 Pneumonia Detection from Chest X-Rays

Deep learning models that classify chest X-ray images as **NORMAL** or **PNEUMONIA**, plus a **Streamlit** web app for uploading an X-ray and getting a prediction with a confidence score.

> ⚠️ **Not a medical device.** This is a learning and portfolio project. It must not be used for clinical diagnosis or treatment decisions.

---

## Models

| Version | File | Architecture | Input |
|---|---|---|---|
| **V1** | `final_model/Version_1.h5` | Custom CNN: 4 × (Conv2D + MaxPool), Dense 128, softmax | 256×256 RGB, pixels scaled to `[0, 1]` |
| **V2** | `final_model/Version_2.h5` | **MobileNetV2** (ImageNet, frozen) + GlobalAvgPool, Dense 128, Dropout 0.5, softmax | 256×256 RGB, raw `0–255` pixels (the model rescales to `[-1, 1]` itself) |

Both models are trained with:
- **Train / validation split by file** (80 / 20, fixed seed), so no image appears in both sets
- **Data augmentation** on every epoch: flip, rotation, zoom, translation, contrast
- **Class weights** to handle the imbalance (about 3× more pneumonia than normal images)
- **Adam** (lr 1e-4) and sparse categorical cross-entropy

### Results

Test-set results will be added after the models are retrained with the fixed training pipeline (see [changelog](#changelog)). The `.h5` files currently in the repo were trained before those fixes.

---

## Dataset

[Chest X-Ray Images (Pneumonia)](https://www.kaggle.com/datasets/paultimothymooney/chest-xray-pneumonia) on Kaggle (pediatric chest X-rays).

| Split | NORMAL | PNEUMONIA | Total |
|---|---|---|---|
| `data/train` | 1,349 | 3,883 | 5,232 |
| `data/test` | 234 | 390 | 624 |

The training script takes 20% of `data/train` as the validation set. The dataset is not stored in this repo; download it and arrange it like this:

```
data/
├── train/
│   ├── NORMAL/
│   └── PNEUMONIA/
└── test/
    ├── NORMAL/
    └── PNEUMONIA/
```

---

## Quick start

### 1. Set up Python

```bash
python3 -m venv .venv
source .venv/bin/activate
```

### 2. Train a model

```bash
pip install -r training/requirements.txt
cd training
python model_1.py   # custom CNN   → saves ../final_model/Version_1.h5
python model_2.py   # MobileNetV2  → saves ../final_model/Version_2.h5
```

Each script prints the test accuracy when training finishes.

### 3. Run the web app

```bash
pip install -r frontend/requirements.txt
cd frontend
streamlit run app.py
```

Upload a `.jpg` / `.png` chest X-ray and click **Predict**. The app shows the image with the predicted class and confidence. It uses `Version_1.h5` by default.

---

## Project structure

```
Pneumonia-Detection/
├── final_model/
│   ├── Version_1.h5        # Custom CNN
│   └── Version_2.h5        # MobileNetV2 transfer learning
├── training/
│   ├── model_1.py          # Train the custom CNN
│   ├── model_2.py          # Train the MobileNetV2 model
│   └── requirements.txt
├── frontend/
│   ├── app.py              # Streamlit web app
│   └── requirements.txt
└── README.md
```

---

## Changelog

**Training pipeline fixes**
- **Validation leak fixed.** The train/val split used `take()`/`skip()` on a dataset that reshuffles every epoch, so validation images leaked into training and validation accuracy was inflated. It now uses `validation_split` + `subset` with a fixed seed (checked: 4,186 train / 1,046 val, 0 overlap).
- **Augmentation fixed.** `.cache()` ran after the random augmentation, freezing the first epoch's augmented images for every later epoch. Caching now happens before augmentation.
- **MobileNetV2 input scale fixed.** It was fed `[0, 1]` pixels instead of the `[-1, 1]` its ImageNet weights expect. A `Rescaling` layer inside the model now handles this.
- Removed an unused `cv2` import that crashed `model_1.py` on a fresh install, and renamed `fontend/` to `frontend/`.

---

## Roadmap

- [ ] Retrain both models with the fixed pipeline and publish test metrics (accuracy, precision, recall, confusion matrix)
- [ ] Move the `.h5` model files to GitHub Releases to keep the repo small
- [ ] Let the app switch between V1 and V2
- [ ] Serve the model as a FastAPI endpoint
