from fastapi import FastAPI, UploadFile, File
from fastapi.responses import JSONResponse
import tensorflow as tf
import cv2
import os
import numpy as np
import h5py
import json
import shutil
import tempfile
from PIL import Image, ExifTags
import io

print("TF:", tf.__version__)

app = FastAPI()

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, "age_gender_model.h5")
MAX_IMAGE_DIM = 640  # now actually used


def load_model_compat(path):
    def fix_config(cfg):
        if isinstance(cfg, dict):
            inner = cfg.get("config", {})
            if isinstance(inner, dict):
                if isinstance(inner.get("dtype"), dict):
                    inner["dtype"] = inner["dtype"].get("config", {}).get("name", "float32")
                inner.pop("quantization_config", None)
                inner.pop("optional", None)
                if cfg.get("class_name") == "InputLayer":
                    if "batch_shape" in inner:
                        inner["batch_input_shape"] = inner.pop("batch_shape")
                for v in inner.values():
                    fix_config(v)
            cfg.pop("module", None)
            cfg.pop("registered_name", None)
            for key in ["build_config", "call_spec", "keras_version"]:
                cfg.pop(key, None)
            for v in list(cfg.values()):
                fix_config(v)
        elif isinstance(cfg, list):
            for item in cfg:
                fix_config(item)

    tmp = tempfile.NamedTemporaryFile(suffix=".h5", delete=False)
    tmp.close()
    shutil.copy2(path, tmp.name)
    try:
        with h5py.File(tmp.name, "r+") as f:
            raw = f.attrs.get("model_config")
            if raw is not None:
                model_config = json.loads(raw)
                fix_config(model_config)
                f.attrs["model_config"] = json.dumps(model_config)
        model = tf.keras.models.load_model(tmp.name, compile=False)
    finally:
        os.unlink(tmp.name)
    return model


model = load_model_compat(MODEL_PATH)
print("Model loaded successfully")

print("Model outputs:")
for i, out in enumerate(model.outputs):
    print(f"  [{i}] name={out.name}  shape={out.shape}")

gender_idx, age_idx = 0, 1
for i, out in enumerate(model.outputs):
    name = out.name.lower()
    if "gender" in name:
        gender_idx = i
    elif "age" in name:
        age_idx = i

print(f"Using output indices → gender={gender_idx}, age={age_idx}")

gender_dict = {0: "Male", 1: "Female"}

FACE_PROTO = os.path.join(BASE_DIR, "opencv_face_detector.pbtxt")
FACE_MODEL = os.path.join(BASE_DIR, "opencv_face_detector_uint8.pb")
faceNet = cv2.dnn.readNet(FACE_MODEL, FACE_PROTO)


def fix_exif_rotation(pil_image):
    """Rotate image to correct orientation based on EXIF data."""
    try:
        exif_data = pil_image._getexif()
        if exif_data is None:
            return pil_image
        orientation_key = next(
            (k for k, v in ExifTags.TAGS.items() if v == "Orientation"), None
        )
        if orientation_key is None or orientation_key not in exif_data:
            return pil_image
        orientation = exif_data[orientation_key]
        rotation_map = {3: 180, 6: 270, 8: 90}
        if orientation in rotation_map:
            pil_image = pil_image.rotate(rotation_map[orientation], expand=True)
    except Exception:
        pass  # No EXIF or unreadable — leave image as-is
    return pil_image


def resize_if_large(image, max_dim=MAX_IMAGE_DIM):
    """Downscale image if either dimension exceeds max_dim."""
    h, w = image.shape[:2]
    if max(h, w) <= max_dim:
        return image
    scale = max_dim / max(h, w)
    new_w, new_h = int(w * scale), int(h * scale)
    return cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_AREA)


def detect_faces(net, frame, conf_threshold=0.7):
    h, w = frame.shape[:2]
    blob = cv2.dnn.blobFromImage(
        frame, 1.0, (300, 300), [104, 117, 123], False, False
    )
    net.setInput(blob)
    detections = net.forward()
    boxes = []
    for i in range(detections.shape[2]):
        confidence = detections[0, 0, i, 2]
        if confidence > conf_threshold:
            x1 = int(detections[0, 0, i, 3] * w)
            y1 = int(detections[0, 0, i, 4] * h)
            x2 = int(detections[0, 0, i, 5] * w)
            y2 = int(detections[0, 0, i, 6] * h)
            boxes.append([x1, y1, x2, y2])
    return boxes


def decode_image_from_bytes(contents):
    """Decode image bytes, fixing EXIF rotation if needed."""
    # Fix EXIF rotation via PIL first
    try:
        pil_img = Image.open(io.BytesIO(contents)).convert("RGB")
        pil_img = fix_exif_rotation(pil_img)
        # Convert PIL → OpenCV (BGR)
        image = cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR)
    except Exception:
        # Fallback to raw OpenCV decode if PIL fails
        nparr = np.frombuffer(contents, np.uint8)
        image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    return image


@app.get("/")
def home():
    return {"message": "Server Running"}


@app.get("/health")
def health():
    return {"status": "ok"}


@app.get("/model-status")
def model_status():
    outputs = [{"index": i, "name": out.name, "shape": str(out.shape)}
               for i, out in enumerate(model.outputs)]
    return {
        "loaded": True,
        "outputs": outputs,
        "gender_idx": gender_idx,
        "age_idx": age_idx,
    }


@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    try:
        contents = await file.read()
        image = decode_image_from_bytes(contents)

        if image is None:
            return JSONResponse(status_code=400, content={"error": "Invalid image"})

        # FIX: actually use MAX_IMAGE_DIM to resize large images
        image = resize_if_large(image)

        faces = detect_faces(faceNet, image)
        if len(faces) == 0:
            return JSONResponse(status_code=400, content={"error": "No face detected"})

        padding = 10
        x1, y1, x2, y2 = faces[0]
        face = image[
            max(0, y1 - padding): min(y2 + padding, image.shape[0]),
            max(0, x1 - padding): min(x2 + padding, image.shape[1]),
        ]

        # FIX: guard against empty crop
        if face.size == 0:
            return JSONResponse(status_code=400, content={"error": "Face crop failed"})

        face_gray = cv2.cvtColor(face, cv2.COLOR_BGR2GRAY)
        face_resized = cv2.resize(face_gray, (128, 128))
        face_norm = face_resized / 255.0
        face_input = face_norm.reshape(1, 128, 128, 1)

        pred = model.predict(face_input, verbose=0)

        gender_raw = float(pred[gender_idx].flatten()[0])
        age_raw = float(pred[age_idx].flatten()[0])

        # FIX: clamp gender to [0,1] before rounding to avoid KeyError
        gender_raw = max(0.0, min(1.0, gender_raw))
        gender = gender_dict[int(round(gender_raw))]
        age = max(0, int(round(age_raw)))

        return {"gender": gender, "age": age}

    except Exception as e:
        return JSONResponse(status_code=500, content={"error": str(e)})


@app.post("/predict-debug")
async def predict_debug(file: UploadFile = File(...)):
    try:
        contents = await file.read()
        image = decode_image_from_bytes(contents)

        if image is None:
            return JSONResponse(status_code=400, content={"error": "Invalid image"})

        image = resize_if_large(image)

        faces = detect_faces(faceNet, image)
        if len(faces) == 0:
            return JSONResponse(status_code=400, content={"error": "No face detected"})

        padding = 10
        x1, y1, x2, y2 = faces[0]
        face = image[
            max(0, y1 - padding): min(y2 + padding, image.shape[0]),
            max(0, x1 - padding): min(x2 + padding, image.shape[1]),
        ]

        if face.size == 0:
            return JSONResponse(status_code=400, content={"error": "Face crop failed"})

        face_gray = cv2.cvtColor(face, cv2.COLOR_BGR2GRAY)
        face_resized = cv2.resize(face_gray, (128, 128))
        face_input = (face_resized / 255.0).reshape(1, 128, 128, 1)

        pred = model.predict(face_input, verbose=0)

        raw_outputs = {f"output_{i}_name": model.outputs[i].name for i in range(len(pred))}
        raw_values = {f"output_{i}_value": float(pred[i].flatten()[0]) for i in range(len(pred))}

        gender_raw = max(0.0, min(1.0, float(pred[gender_idx].flatten()[0])))
        age_raw = float(pred[age_idx].flatten()[0])

        return {
            "output_names": raw_outputs,
            "raw_values": raw_values,
            "gender_idx_used": gender_idx,
            "age_idx_used": age_idx,
            "predicted_gender": gender_dict[int(round(gender_raw))],
            "predicted_age": max(0, int(round(age_raw))),
            "face_box": faces[0],
            "image_shape": list(image.shape),
        }

    except Exception as e:
        return JSONResponse(status_code=500, content={"error": str(e)})
