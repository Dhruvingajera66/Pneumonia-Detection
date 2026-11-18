# 🩺 Pneumonia Detection from Chest X-Rays

![Python](https://img.shields.io/badge/Python-3.10%2B-3776AB?logo=python&logoColor=white)
![TensorFlow](https://img.shields.io/badge/TensorFlow-Keras-FF6F00?logo=tensorflow&logoColor=white)
![Streamlit](https://img.shields.io/badge/Streamlit-app-FF4B4B?logo=streamlit&logoColor=white)
![License](https://img.shields.io/badge/License-MIT-green)

Deep learning models that classify chest X-ray images as **NORMAL** or **PNEUMONIA**, plus a **Streamlit** web app for uploading an X-ray and getting a prediction with a confidence score.

> ⚠️ **Not a medical device.** This is a learning and portfolio project. It must not be used for clinical diagnosis or treatment decisions.

**How it works**

```
Chest X-ray (.jpg/.png) → resize to 256×256 → CNN / MobileNetV2 → NORMAL or PNEUMONIA + confidence %
```

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

Test-set results will be added after the models are retrained with the fixed training pipeline (see [CHANGELOG.md](CHANGELOG.md)). The `.h5` files currently in the repo were trained before those fixes.

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
├── CHANGELOG.md
├── LICENSE
└── README.md
```

---

## Recent changes

Fixed a validation leak, frozen augmentation and the MobileNetV2 input scale in the training pipeline. Details in [CHANGELOG.md](CHANGELOG.md).

---

## Roadmap

- [ ] Retrain both models with the fixed pipeline and publish test metrics (accuracy, precision, recall, confusion matrix)
- [ ] Move the `.h5` model files to GitHub Releases to keep the repo small
- [ ] Let the app switch between V1 and V2
- [ ] Serve the model as a FastAPI endpoint

---

## License

[MIT](LICENSE) © Dhruvin Gajera
