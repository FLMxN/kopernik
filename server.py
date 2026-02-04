from fastapi import FastAPI, Form
import asyncio
from torch_main import main

job_queue = asyncio.Queue()

async def inference_worker():
    while True:
        image, pretty, act = await job_queue.get()
        print(f"running inference for {image} (pretty={pretty})")
        await asyncio.sleep(2)
        print(f"finished {image}")
        job_queue.task_done()

async def lifespan(app: FastAPI):
    asyncio.create_task(inference_worker())
    yield

app = FastAPI(lifespan=lifespan)

@app.post("/inference")
async def inference(
    image: str = Form(...),
    pretty: str = Form(...),
    act: str = Form(...)
):
    if act != "run":
        return {"error": "invalid action"}
    
    pretty_bool = pretty.lower() in ("true", "1", "yes")
    
    await job_queue.put((image, pretty_bool, act))
    return {"status": "job enqueued", "image": image}
