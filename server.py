from fastapi import FastAPI, Form, UploadFile, File
from fastapi.responses import JSONResponse
import asyncio
from contextlib import asynccontextmanager
from fastapi.middleware.cors import CORSMiddleware
from PIL import Image
import io

print("MODULE IMPORTED")

job_queue = asyncio.Queue()

async def inference_worker():
    print("worker started")
    while True:
        image, pretty, act, future = await job_queue.get()
        try:
            result = {
                "image": image,
                "pretty": pretty,
                "act": act,
                "status": "ok"
            }
            print("setting future")
            future.set_result(result)
        except Exception as e:
            future.set_exception(e)
        finally:
            job_queue.task_done()

@asynccontextmanager
async def lifespan(app: FastAPI):
    print("LIFESPAN ENTER")
    asyncio.create_task(inference_worker())
    yield
    print("LIFESPAN EXIT")


app = FastAPI(lifespan=lifespan)

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
    if act != "run":
        return {"error": "invalid action"}

    pretty_bool = pretty.lower() in ("true", "1", "yes")

    loop = asyncio.get_running_loop()
    future = loop.create_future()

    contents = await image.read()
    img = Image.open(io.BytesIO(contents))

    await job_queue.put((img, pretty_bool, act, future))

    result = await future
    print("returning")
    print(result)
    return JSONResponse(content=result)

