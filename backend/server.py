from fastapi import FastAPI, Form, UploadFile, File, HTTPException
from fastapi.responses import JSONResponse
from fastapi.staticfiles import StaticFiles
import asyncio
from contextlib import asynccontextmanager
from fastapi.middleware.cors import CORSMiddleware
from PIL import Image
import io
import torch
from pathlib import Path
import dotenv
import tempfile
import os
from .paths import OUTPUT_DIR, ROOT_DIR, resolve_project_path
from .predictor import predict_country, predict_region
from .torch_main import load_model_checkpoint, crop_resize, stretch_resize

dotenv_file = ROOT_DIR / '.env'
dotenv.load_dotenv(dotenv_file, override=True)
env = os.environ.copy()

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

print("MODULE IMPORTED")

job_queue = asyncio.Queue()

model_country = None
model_region = None

async def inference_worker():
    print("worker started")
    while True:
        image, pretty, act, future, temp_path = await job_queue.get()
        try:
            img = image.convert("RGB")
            img_crop = crop_resize(img)
            img_stretch = stretch_resize(img)
            samples = [(img_crop, temp_path), (img_stretch, temp_path)]
            
            result = {"status": "ok", "original_image": f"/output/{os.path.basename(temp_path)}"}
            
            if model_country:
                country_preds = predict_country(model_country, samples, show_pictures=True, IS_PRETTY=pretty, device=DEVICE)
                result["country_predictions"] = country_preds
            
            if model_region:
                region_preds = predict_region(model_region, samples, show_pictures=True, IS_PRETTY=pretty, device=DEVICE)
                result["region_predictions"] = region_preds
            
            print("setting future")
            future.set_result(result)
        except Exception as e:
            future.set_exception(e)
        finally:
            job_queue.task_done()

@asynccontextmanager
async def lifespan(app: FastAPI):
    global model_country, model_region
    print("LIFESPAN ENTER")
    try:
        ckpt_path = os.getenv("CKPT")
        if ckpt_path:
            ckpt_path = resolve_project_path(ckpt_path)
        model_country, _ = load_model_checkpoint(path=ckpt_path, device=DEVICE, num_classes=56, is_reg=False)
        print("Country model loaded")
    except Exception as e:
        print(f"Failed to load country model: {e}")
        model_country = None
    
    try:
        ckpt_reg_path = os.getenv("CKPT_REG")
        if ckpt_reg_path:
            ckpt_reg_path = resolve_project_path(ckpt_reg_path)
        model_region, _ = load_model_checkpoint(path=ckpt_reg_path, device=DEVICE, num_classes=7, is_reg=True)
        print("Region model loaded")
    except Exception as e:
        print(f"Failed to load region model: {e}")
        model_region = None
    
    asyncio.create_task(inference_worker())
    yield
    print("LIFESPAN EXIT")


app = FastAPI(lifespan=lifespan)
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
app.mount("/output", StaticFiles(directory=OUTPUT_DIR), name="output")

allowed_origins = [
    origin.strip()
    for origin in os.getenv("ALLOWED_ORIGINS", "*").split(",")
    if origin.strip()
]
app.add_middleware(
    CORSMiddleware,
    allow_origins=allowed_origins,
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/health")
async def health():
    return {
        "status": "ok",
        "country_model": model_country is not None,
        "region_model": model_region is not None,
    }


@app.post("/inference")
@app.post("/api/inference")
async def inference(
    image: UploadFile = File(...),
    pretty: str = Form(...),
    act: str = Form(...)
):
    print(f"Received: image={image.filename}, pretty={pretty}, act={act}")
    if act != "run":
        return {"error": "invalid action"}

    pretty_bool = pretty.lower() in ("true", "1", "yes")

    if job_queue.qsize() >= 8:
        raise HTTPException(status_code=503, detail="Server is busy, try again shortly")

    loop = asyncio.get_running_loop()
    future = loop.create_future()

    contents = await image.read()
    if len(contents) > 15 * 1024 * 1024:
        raise HTTPException(status_code=413, detail="Image is too large (max 15 MB)")
    try:
        img = Image.open(io.BytesIO(contents)).convert("RGB")
    except Exception as exc:  # noqa: BLE001
        raise HTTPException(status_code=400, detail=f"Invalid image: {exc}") from exc

    with tempfile.NamedTemporaryFile(suffix='.jpg', delete=False) as tmp:
        img.save(tmp.name)
        temp_path = tmp.name

    try:
        await job_queue.put((img, pretty_bool, act, future, temp_path))

        result = await future
        print("returning")
        print(result)
        return JSONResponse(content=result)
    finally:
        os.unlink(temp_path)