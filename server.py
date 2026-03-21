from fastapi import FastAPI, Form, UploadFile, File
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
from predictor import predict_country, predict_region
from torch_main import load_model_checkpoint, crop_resize, stretch_resize

dotenv_file = Path(__file__).parent / '.env'
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
        model_country, _ = load_model_checkpoint(path=ckpt_path, device=DEVICE, num_classes=56, is_reg=False)
        print("Country model loaded")
    except Exception as e:
        print(f"Failed to load country model: {e}")
        model_country = None
    
    try:
        ckpt_reg_path = os.getenv("CKPT_REG")
        model_region, _ = load_model_checkpoint(path=ckpt_reg_path, device=DEVICE, num_classes=7, is_reg=True)
        print("Region model loaded")
    except Exception as e:
        print(f"Failed to load region model: {e}")
        model_region = None
    
    asyncio.create_task(inference_worker())
    yield
    print("LIFESPAN EXIT")


app = FastAPI(lifespan=lifespan)
app.mount("/output", StaticFiles(directory="output"), name="output")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173"],  # or "*"
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.post("/inference")
async def inference(
    image: UploadFile = File(...),
    pretty: str = Form(...),
    act: str = Form(...)
):
    print(f"Received: image={image.filename}, pretty={pretty}, act={act}")
    if act != "run":
        return {"error": "invalid action"}

    pretty_bool = pretty.lower() in ("true", "1", "yes")

    loop = asyncio.get_running_loop()
    future = loop.create_future()

    contents = await image.read()
    img = Image.open(io.BytesIO(contents)).convert("RGB")

    # Save to temp file for gradcam
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

