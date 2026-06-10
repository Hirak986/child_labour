import cv2
import cvlib as cv
import numpy as np
from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.responses import JSONResponse

# Initialize FastAPI application
app = FastAPI(
    title="Gender Detection API",
    description="Upload an image to detect faces and predict gender (Male/Female).",
    version="1.0.0"
)

@app.get("/")
def read_root():
    """Health check endpoint."""
    return {"status": "online", "message": "Gender Detection API is running"}

@app.post("/predict-gender/")
async def predict_gender(file: UploadFile = File(...)):
    """
    Upload an image file to detect the gender of faces present.
    Supported formats: JPEG, PNG
    """
    # 1. Validate file extension
    if not file.filename.lower().endswith(('.png', '.jpg', '.jpeg')):
        raise HTTPException(status_code=400, detail="Invalid image format. Please upload JPG or PNG.")

    try:
        # 2. Read image bytes into numpy array
        contents = await file.read()
        nparr = np.frombuffer(contents, np.uint8)
        image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

        if image is None:
            raise HTTPException(status_code=400, detail="Could not decode the uploaded image.")

        # 3. Detect faces in the image
        faces, confidences = cv.detect_face(image)

        if not faces:
            return JSONResponse(content={
                "face_count": 0,
                "predictions": [],
                "message": "No faces detected in the image."
            })

        results = []

        # 4. Loop through detected faces and predict gender
        for idx, face in enumerate(faces):
            start_x, start_y, end_x, end_y = face

            # Ensure coordinates are within image boundaries
            start_x, start_y = max(0, start_x), max(0, start_y)
            end_x, end_y = min(image.shape[1], end_x), min(image.shape[0], end_y)

            # Crop the detected face
            face_crop = np.copy(image[start_y:end_y, start_x:end_x])

            if face_crop.shape[0] < 10 or face_crop.shape[1] < 10:
                continue # Skip tiny, unresolvable artifacts

            # Predict gender on the cropped face
            labels, confs = cv.detect_gender(face_crop)

            # Get highest confidence prediction
            best_idx = np.argmax(confs)
            gender_label = labels[best_idx]
            gender_confidence = float(confs[best_idx])

            results.append({
                "face_index": idx + 1,
                "gender": gender_label,
                "confidence": round(gender_confidence, 4),
                "box": {
                    "xmin": start_x,
                    "ymin": start_y,
                    "xmax": end_x,
                    "ymax": end_y
                }
            })

        return JSONResponse(content={
            "face_count": len(results),
            "predictions": results
        })

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Internal Server Error: {str(e)}")
