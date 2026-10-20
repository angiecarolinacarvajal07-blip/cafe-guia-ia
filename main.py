from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from ultralytics import YOLO
from PIL import Image, UnidentifiedImageError
import io

app = FastAPI(title="Café Guía - Detector orientativo", version="1.1.0")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["GET", "POST", "OPTIONS"],
    allow_headers=["*"],
)

# Modelo público DECAFIA. Solo detecta roya, daño por gorgojos defoliadores (etiqueta coco)
# y minador. No puede identificar todas las plagas del café.
try:
    model = YOLO("estebanr25/decafia")
except Exception as e:
    print(f"No se pudo cargar el modelo DECAFIA: {e}")
    model = None

@app.get("/")
def health():
    return {
        "ok": True,
        "service": "Cafe Guia detector",
        "classes": ["roya", "coco", "minador"],
        "notice": "Resultado orientativo; no es diagnóstico definitivo."
    }

@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    if not file.content_type or not file.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="Sube un archivo de imagen válido.")
    raw = await file.read()
    if not raw:
        raise HTTPException(status_code=400, detail="La imagen está vacía.")
    try:
        image = Image.open(io.BytesIO(raw)).convert("RGB")
    if model is None:
        raise HTTPException(
            status_code=503,
            detail="El modelo de IA no está disponible."
        )

    try:
        results = model.predict(image, conf=0.50, verbose=False)
    except Exception as exc:
        raise HTTPException(
            status_code=500,
            detail="No se pudo analizar la imagen."
        ) from exc

    detections = []
    for box in results[0].boxes:
        class_id = int(box.cls.item())
        confidence = float(box.conf.item())
        detections.append({
            "class_name": str(model.names[class_id]).lower(),
            "confidence": confidence
        })

    detections.sort(
        key=lambda item: item["confidence"],
        reverse=True
    )
    return {"detections": detections}
        confidence = float(box.conf.item())
        detections.append({"class_name": str(model.names[class_id]).lower(), "confidence": confidence})
    detections.sort(key=lambda item: item["confidence"], reverse=True)
    return {"detections": detections}
