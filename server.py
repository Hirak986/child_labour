from fastapi import FastAPI, UploadFile, File
from fastapi.responses import JSONResponse
import cv2
import os
import numpy as np
import tensorflow as tf

app = FastAPI()

# ================= BASE DIRECTORY =================

BASE_DIR = os.path.dirname(
    os.path.abspath(__file__)
)

# ================= LOAD MODEL =================

MODEL_PATH = os.path.join(
    BASE_DIR,
    "age_gender_model.h5"
)

try:

    model = tf.keras.models.load_model(
        MODEL_PATH,
        compile=False,
        safe_mode=False
    )

    print("Model loaded successfully")

except Exception as e:

    print(f"Model loading error: {e}")

    model = None

# ================= GENDER LABELS =================

gender_dict = {
    0: "Male",
    1: "Female"
}

# ================= FACE DETECTOR =================

FACE_PROTO = os.path.join(
    BASE_DIR,
    "opencv_face_detector.pbtxt"
)

FACE_MODEL = os.path.join(
    BASE_DIR,
    "opencv_face_detector_uint8.pb"
)

faceNet = cv2.dnn.readNet(
    FACE_MODEL,
    FACE_PROTO
)

# ================= DETECT FACES =================

def detect_faces(
    net,
    frame,
    conf_threshold=0.7
):

    h, w = frame.shape[:2]

    blob = cv2.dnn.blobFromImage(
        frame,
        1.0,
        (300, 300),
        [104, 117, 123],
        swapRB=False,
        crop=False
    )

    net.setInput(blob)

    detections = net.forward()

    boxes = []

    for i in range(detections.shape[2]):

        confidence = detections[0, 0, i, 2]

        if confidence > conf_threshold:

            x1 = int(
                detections[0, 0, i, 3] * w
            )

            y1 = int(
                detections[0, 0, i, 4] * h
            )

            x2 = int(
                detections[0, 0, i, 5] * w
            )

            y2 = int(
                detections[0, 0, i, 6] * h
            )

            boxes.append(
                [x1, y1, x2, y2]
            )

    return boxes

# ================= HOME =================

@app.get("/")
def home():

    return {
        "message": "Server Running"
    }

# ================= HEALTH CHECK =================

@app.get("/health")
def health():

    return {
        "status": "ok"
    }

# ================= PREDICT =================

@app.post("/predict")
async def predict(
    file: UploadFile = File(...)
):

    try:

        # ---------- MODEL CHECK ----------

        if model is None:

            return JSONResponse(
                status_code=500,
                content={
                    "error": "Model not loaded"
                }
            )

        # ---------- READ IMAGE ----------

        contents = await file.read()

        nparr = np.frombuffer(
            contents,
            np.uint8
        )

        image = cv2.imdecode(
            nparr,
            cv2.IMREAD_COLOR
        )

        if image is None:

            return JSONResponse(
                status_code=400,
                content={
                    "error": "Invalid image"
                }
            )

        # ---------- DETECT FACE ----------

        faces = detect_faces(
            faceNet,
            image
        )

        if len(faces) == 0:

            return JSONResponse(
                status_code=400,
                content={
                    "error": "No face detected"
                }
            )

        # ---------- TAKE FIRST FACE ----------

        padding = 10

        x1, y1, x2, y2 = faces[0]

        face = image[
            max(0, y1 - padding):
            min(y2 + padding, image.shape[0]),

            max(0, x1 - padding):
            min(x2 + padding, image.shape[1])
        ]

        # ---------- PREPROCESS ----------

        face_gray = cv2.cvtColor(
            face,
            cv2.COLOR_BGR2GRAY
        )

        face_resized = cv2.resize(
            face_gray,
            (128, 128)
        )

        face_norm = face_resized.astype(
            "float32"
        ) / 255.0

        face_input = face_norm.reshape(
            1,
            128,
            128,
            1
        )

        # ---------- PREDICTION ----------

        pred = model.predict(
            face_input,
            verbose=0
        )

        gender = gender_dict[
            int(
                round(
                    float(pred[0][0][0])
                )
            )
        ]

        age = int(
            round(
                float(pred[1][0][0])
            )
        )

        # ---------- RESPONSE ----------

        return {
            "gender": gender,
            "age": age
        }

    except Exception as e:

        return JSONResponse(
            status_code=500,
            content={
                "error": str(e)
            }
        )
@app.get("/model-status")
def model_status():

    if model is None:
        return {"loaded": False}

    return {"loaded": True}
