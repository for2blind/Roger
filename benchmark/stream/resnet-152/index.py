#!/usr/bin/env python
from quart import Quart, jsonify, request
import os,time
import torch
from torch.cuda import current_device
import subprocess
import re
from hypercorn.config import Config
from hypercorn.asyncio import serve
import asyncio
from multiprocessing import Process

os.chdir(os.path.dirname(os.path.abspath(__file__)))

def get_cuda_device_mapping():
    try:
        output = subprocess.check_output(["nvidia-smi", "-L"], universal_newlines=True)
    except subprocess.CalledProcessError:
        print(
            "nvidia-smi could not be executed. Are you sure the system has an NVIDIA GPU and nvidia-smi is installed?"
        )
        return {}
    except FileNotFoundError:
        print("nvidia-smi was not found. Are you sure it's installed?")
        return {}
    devices = re.findall(r"GPU (\d+): (.* \(UUID: GPU-(.*)\))", output)
    uuid_to_id = {uuid: int(id) for id, _, uuid in devices}
    return uuid_to_id


def get_device_id_by_uuid(target_uuid):
    uuid_to_id = get_cuda_device_mapping()
    return uuid_to_id.get(target_uuid, None)



if os.environ.get('infer_device') is None:
    infer_device = ('cuda' if torch.cuda.is_available() else 'cpu')
    os.environ['infer_device'] = infer_device
elif os.environ.get('infer_device') == 'cuda' and not torch.cuda.is_available():
    os.environ['infer_device'] = 'cpu'
infer_device = os.environ.get('infer_device')
if infer_device != "cpu":
    cuda_id = get_device_id_by_uuid(
        os.environ.get("NVIDIA_VISIBLE_DEVICES", "0").replace("GPU-", "")
    )
    if cuda_id is None:
        print("CUDA device not found")
        exit(1)
    infer_device = f"cuda:{cuda_id}"
function_time_out = float(os.environ.get("read_timeout", "10s").replace("s", ""))

app = Quart(__name__)
model_name = 'resnet-152'

def load_model():
    with torch.inference_mode():
        global loaded, model, x
        x = torch.rand(1, 3, 224, 224).to(infer_device)
        model= torch.load(f'/data/model/openfaas/gss/resnet-152.pt').to(infer_device)
        output = model(x)
    return True


async def inference():
    async with inference_semaphore:
        if infer_device != "cpu":
            s = torch.cuda.Stream(infer_device)
            with torch.cuda.stream(s):
                with torch.inference_mode():
                    output = model(x)
                    s.synchronize()
        else:
            with torch.inference_mode():
                output = model(x)
        return output


global loaded, model, x

loaded = False
inference_semaphore = asyncio.Semaphore(2)
loaded = load_model()


async def serve_loaded_endpoint():
    app_loaded = Quart("LoadedApp")
    global loaded
    @app_loaded.route("/loaded", methods=["GET"])
    async def check_loaded():
        try:
            if loaded:
                return jsonify({"result": True}), 200
            else:
                return jsonify({"result": False}), 500
        except:
            return jsonify({"result": False}), 500

    config = Config()
    config.bind = ["0.0.0.0:5001"]
    await serve(app_loaded, config)


@app.route(
    "/", defaults={"path": ""}, methods=["GET", "PUT", "POST", "PATCH", "DELETE"]
)
@app.route("/<path:path>", methods=["GET", "PUT", "POST", "PATCH", "DELETE"])
async def call_handler(path):
    try:
        start_time = time.time()        
        await inference()
        end_time = time.time()
        res = {"start_time": start_time,"end_time":end_time,"infer_device": infer_device, "exec_time": end_time - start_time}
        return jsonify(res),200
    except Exception as e:
        return f"infernece except:{e}", 500


if __name__ == "__main__":
    p = Process(target=asyncio.run, args=(serve_loaded_endpoint(),))
    p.daemon = True
    p.start()
    config = Config()
    # config.worker_class = "asyncio"
    # config.workers = 2
    config.bind = ["0.0.0.0:5000"]
    asyncio.run(serve(app, config))
