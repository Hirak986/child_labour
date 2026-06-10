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

print("TF:", tf.__version__)

app = FastAPI()

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, "age_gender_model.h5")
MAX_IMAGE_DIM = 640  #

def load_model_compat(path):
    """
    Load model with compatibility fixes applied to a TEMP COPY only.
    Never mutates the original .h5 file on disk.
    """
    def fix_config(cfg):
        if isinstance(cfg, dict):
            inner = cfg.get("config", {})
            if isinstance(inner, dict):
                # Fix DTypePolicy dict → plain string
                if isinstance(inner.get("dtype"), dict):
                    inner["dtype"] = inner["dtype"].get("config", {}).get("name", "float32")

                # Remove unsupported keys from ANY layer
                inner.pop("quantization_config", None)
                inner.pop("optional", None)

                # Fix InputLayer specifically
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
print(" Model loaded successfully")


print("Model outputs:")
for i, out in enumerate(model.outputs):
    print(f"  [{i}] name={out.name}  shape={out.shape}")

# -----------------------------------------------------------------------
# Resolve which output index is gender and which is age by output name.
# Falls back to the original assumption (0=gender, 1=age) if names are
# ambiguous — but the startup log above will tell you if that's wrong.
# -----------------------------------------------------------------------
gender_idx, age_idx = 0, 1  # safe defaults matching your local code

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

# Reads EXIF orientation tag and rotates the image correctly
def fix_exif_rotation(pil_image): ...
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
        nparr = np.frombuffer(contents, np.uint8)
        image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

        if image is None:
            return JSONResponse(status_code=400, content={"error": "Invalid image"})

        faces = detect_faces(faceNet, image)
        if len(faces) == 0:
            return JSONResponse(status_code=400, content={"error": "No face detected"})

        padding = 10
        x1, y1, x2, y2 = faces[0]
        face = image[
            max(0, y1 - padding): min(y2 + padding, image.shape[0]),
            max(0, x1 - padding): min(x2 + padding, image.shape[1]),
        ]

        # ✅ Exact same preprocessing as local code
        face_gray = cv2.cvtColor(face, cv2.COLOR_BGR2GRAY)
        face_resized = cv2.resize(face_gray, (128, 128))
        face_norm = face_resized / 255.0
        face_input = face_norm.reshape(1, 128, 128, 1)

        pred = model.predict(face_input, verbose=0)

        # ✅ Use resolved indices, not hardcoded 0/1
        gender_raw = float(pred[gender_idx].flatten()[0])
        age_raw = float(pred[age_idx].flatten()[0])

        gender = gender_dict[int(round(gender_raw))]
        age = max(0, int(round(age_raw)))

        return {"gender": gender, "age": age}

    except Exception as e:
        return JSONResponse(status_code=500, content={"error": str(e)})


# ---------------------------------------------------------------------------
# Debug endpoint — call this once after deploying to verify everything is
# wired correctly. Returns raw model outputs so you can spot any swap.
# ---------------------------------------------------------------------------
@app.post("/predict-debug")
async def predict_debug(file: UploadFile = File(...)):
    try:
        contents = await file.read()
        nparr = np.frombuffer(contents, np.uint8)
        image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

        if image is None:
            return JSONResponse(status_code=400, content={"error": "Invalid image"})

        faces = detect_faces(faceNet, image)
        if len(faces) == 0:
            return JSONResponse(status_code=400, content={"error": "No face detected"})

        padding = 10
        x1, y1, x2, y2 = faces[0]
        face = image[
            max(0, y1 - padding): min(y2 + padding, image.shape[0]),
            max(0, x1 - padding): min(x2 + padding, image.shape[1]),
        ]

        face_gray = cv2.cvtColor(face, cv2.COLOR_BGR2GRAY)
        face_resized = cv2.resize(face_gray, (128, 128))
        face_input = (face_resized / 255.0).reshape(1, 128, 128, 1)

        pred = model.predict(face_input, verbose=0)

        raw_outputs = {
            f"output_{i}_name": model.outputs[i].name
            for i in range(len(pred))
        }
        raw_values = {
            f"output_{i}_value": float(pred[i].flatten()[0])
            for i in range(len(pred))
        }

        gender_raw = float(pred[gender_idx].flatten()[0])
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
