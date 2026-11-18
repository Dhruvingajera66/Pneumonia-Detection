# Changelog

## 2026-11-18: Training pipeline fixes

- **Validation leak fixed.** The train/val split used `take()`/`skip()` on a dataset that reshuffles every epoch, so validation images leaked into training and validation accuracy was inflated. It now uses `validation_split` + `subset` with a fixed seed (checked: 4,186 train / 1,046 val, 0 overlap).
- **Augmentation fixed.** `.cache()` ran after the random augmentation, freezing the first epoch's augmented images for every later epoch. Caching now happens before augmentation.
- **MobileNetV2 input scale fixed.** It was fed `[0, 1]` pixels instead of the `[-1, 1]` its ImageNet weights expect. A `Rescaling` layer inside the model now handles this, so `Version_2.h5` takes raw `0–255` pixels.
- Class weights are computed from the training split only, and the test set is no longer shuffled.
- Removed an unused `cv2` import that crashed `model_1.py` on a fresh install.
- Renamed `fontend/` to `frontend/` and added `frontend/requirements.txt`.

The `.h5` files in `final_model/` were trained before these fixes and have not been retrained yet.

## 2025-02-16: Initial release

- Custom CNN (`model_1.py`) and MobileNetV2 transfer-learning model (`model_2.py`)
- Streamlit app for uploading an X-ray and getting a prediction
