import io, os, logging
from contextlib import asynccontextmanager
from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from PIL import Image, UnidentifiedImageError
from huggingface_hub import hf_hub_download
from ultralytics import YOLO

logging.basicConfig(level=logging.INFO)
log=logging.getLogger("cafe-guia")
MODEL_REPO=os.getenv("MODEL_REPO", "estebanr25/decafia")
MODEL_FILE=os.getenv("MODEL_FILE", "decafia_clean_best.pt")
CONFIDENCE=float(os.getenv("CONFIDENCE", "0.50"))
model=None; model_error=None; model_path=None

@asynccontextmanager
async def lifespan(app: FastAPI):
    global model, model_error, model_path
    try:
        log.info("Descargando pesos: %s/%s", MODEL_REPO, MODEL_FILE)
        model_path=hf_hub_download(repo_id=MODEL_REPO, filename=MODEL_FILE)
        log.info("Pesos descargados; cargando modelo YOLO")
        loaded=YOLO(model_path, task="detect")
        model=loaded; model_error=None
        log.info("MODELO LISTO. Clases: %s", loaded.names)
    except Exception as exc:
        model=None; model_error=f"{type(exc).__name__}: {exc}"
        log.exception("ERROR AL CARGAR MODELO")
    yield

app=FastAPI(title="Cafe Guia - Detector orientativo", version="1.3.0", lifespan=lifespan)
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_credentials=False, allow_methods=["GET","POST","OPTIONS"], allow_headers=["*"])

def status_payload():
    return {"ok":True,"service":"Cafe Guia detector","model_available":model is not None,"model_repo":MODEL_REPO,"model_file":MODEL_FILE,"model_path":model_path,"classes":list(model.names.values()) if model is not None else [],"model_error":model_error,"notice":"Este modelo solo reconoce roya, daño por gorgojos defoliadores (coco) y minador. No detecta las 16 categorías."}

@app.get("/")
def root(): return status_payload()

@app.get("/health")
def health(): return {"status":"alive","model_available":model is not None}

@app.post("/predict")
async def predict(file: UploadFile=File(...)):
    if model is None:
        raise HTTPException(status_code=503, detail={"message":"El servidor está activo, pero el modelo de IA no se pudo cargar.","model_error":model_error,"check":"Abre la página principal del servicio para ver model_error."})
    if not file.content_type or not file.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="Sube un archivo de imagen válido.")
    raw=await file.read()
    if not raw: raise HTTPException(status_code=400, detail="La imagen está vacía.")
    if len(raw)>12*1024*1024: raise HTTPException(status_code=413, detail="La imagen supera 12 MB. Usa una foto más pequeña.")
    try: image=Image.open(io.BytesIO(raw)).convert("RGB")
    except (UnidentifiedImageError,OSError,ValueError): raise HTTPException(status_code=400, detail="No se pudo leer la imagen.")
    try: results=model.predict(image, conf=CONFIDENCE, imgsz=640, device="cpu", verbose=False)
    except Exception as exc:
        log.exception("Falló la predicción")
        raise HTTPException(status_code=500, detail=f"El modelo cargó, pero falló al analizar la imagen ({type(exc).__name__}).") from exc
    detections=[]; result=results[0]
    if result.boxes is not None:
        for box in result.boxes:
            cid=int(box.cls.item()); conf=float(box.conf.item()); names=model.names
            cname=str(names.get(cid,cid) if isinstance(names,dict) else names[cid]).lower()
            detections.append({"class_name":cname,"confidence":round(conf,4),"box":[round(float(v),2) for v in box.xyxy[0].tolist()]})
    detections.sort(key=lambda x:x["confidence"], reverse=True)
    return {"detections":detections,"model":f"{MODEL_REPO}/{MODEL_FILE}","notice":"Resultado orientativo: este modelo reconoce únicamente roya, daño por gorgojos defoliadores (coco) y minador. La ausencia de detecciones no confirma que la planta esté sana."}
