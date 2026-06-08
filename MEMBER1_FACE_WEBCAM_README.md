# Member 1: Webcam Face Detection and Crop

Deliverable flow:

```text
Webcam
  ↓
Face Detection
  ↓
Face Crop
  ↓
Emotion Model Hook
```

## Setup

```powershell
pip install -r requirements_member1_webcam.txt
```

## Run

```powershell
python member1_webcam_face_crop.py
```

If your webcam is not camera `0`, try:

```powershell
python member1_webcam_face_crop.py --camera 1
```

## Save Face Crops

Press `s` while the webcam window is open, or auto-save crops every second:

```powershell
python member1_webcam_face_crop.py --save-crops
```

Saved crops go to `face_crops/`.

## Connect an Emotion Model

If the next member has a Keras/TensorFlow model, run:

```powershell
python member1_webcam_face_crop.py --emotion-model path/to/model.h5
```

The code sends each cropped face image into `EmotionModel.predict()` in
`member1_webcam_face_crop.py`.

