#!/usr/bin/env python
from quart import Quart, jsonify, request
import os
import random
import torch
from torch.cuda import current_device
import aiohttp
import subprocess
import re
from hypercorn.config import Config
from hypercorn.asyncio import serve
import asyncio
from multiprocessing import Process
from graph_structure import SimpleWrapperGMInput, SimpleWrapperGMList

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


infer_device = (
    "cpu" if not torch.cuda.is_available() else os.environ.get("infer_device", "cpu")
)
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


async def call_next_function():
    global final_submod
    timeout = aiohttp.ClientTimeout(total=function_time_out)
    uid = random.randint(0, 10e6)
    try:
        async with aiohttp.ClientSession() as session:
            async with session.get(
                f"http://172.169.8.253:31112/function/opt-66b-submod-{final_submod}-latency-64.cdgp#{str(uid)}",
                timeout=timeout,
            ) as response:
                return await response.text(), response.status
    except:
        return "call subgraph exception", 500


def check_cuda_memory(final_submod, new_final_submod):
    # Calculate the required memory in bytes
    total_memory_required = 0
    for i in range(final_submod, new_final_submod):
        model_path = f"/data/model/openfaas/cdgp/subgraphs/opt-66b/LATENCY_OAP/10.0-64/opt-66b-submod-{i}.pt"
        input_path = f"/data/model/openfaas/cdgp/subgraphs/opt-66b/LATENCY_OAP/10.0-64/opt-66b-inputs-{i}.pt"
        total_memory_required += os.path.getsize(model_path) + os.path.getsize(
            input_path
        )
    # Convert free memory to bytes
    free_memory = torch.cuda.mem_get_info(current_device())[0]
    # torch.cuda.get_device_properties(current_device()).total_memory- torch.cuda.memory_allocated()
    return free_memory >= total_memory_required * 1.2


def load_model(start_submod, end_submod):
    with torch.inference_mode():
        global loaded, mm, ii, model, x
        for i in range(start_submod, end_submod):
            mm.append(
                torch.load(
                    f"/data/model/openfaas/cdgp/subgraphs/opt-66b/LATENCY_OAP/10.0-64/opt-66b-submod-{i}.pt",
                    map_location=infer_device,
                )
            )
            ii.append(
                torch.load(
                    f"/data/model/openfaas/cdgp/subgraphs/opt-66b/LATENCY_OAP/10.0-64/opt-66b-inputs-{i}.pt",
                    map_location=infer_device,
                )
            )
    return True


def do_merge_model():
    with torch.inference_mode():
        global loaded, mm, ii, model, x
        model = SimpleWrapperGMList(mm)
        x = SimpleWrapperGMInput(ii, device=infer_device)
        loaded = True
    return loaded


async def reduce_model(new_model_num):
    global loaded, mm, ii, model, x
    mm = mm[:new_model_num]
    ii = ii[:new_model_num]
    return True


async def inference():
    async with inference_semaphore:
        s = torch.cuda.Stream(infer_device)
        with torch.cuda.stream(s):
            with torch.inference_mode():
                output = model(x)
                s.synchronize()
                return output


global model_nums, final_submod, loaded, model, x
model_nums = int(os.environ.get("model_nums", 1))
function_model_name = "opt-66b-submod-0"
current_model = int(function_model_name.split("-")[-1])
model_name = "opt-66b"
final_submod = current_model + model_nums
loaded = False
global mm
global ii
mm = []
ii = []
inference_semaphore = asyncio.Semaphore(2)
loaded = load_model(current_model, final_submod)


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


@app.route("/merge/<int:new_model_num>", methods=["GET"])
async def merge_model(new_model_num):
    try:
        global final_submod
        global model_nums
        if new_model_num == 0:
            return {"merged": final_submod - current_model}, 200
        new_final_submod = current_model + new_model_num
        if new_final_submod < final_submod:
            asyncio.create_task(reduce_model(new_model_num))
            return f"Successfully reduce merged subgraphs to {model_nums}", 200
            # return f"Can not reduce subgraphs from {model_nums} to {new_model_num}", 500
        if new_final_submod == final_submod:
            return f"Aleady merged {model_nums} models", 200
        if not check_cuda_memory(final_submod, new_final_submod):
            return (f"Merge error:CUDA {current_device()} out of memory!", 500)
        global loaded
        load_model(final_submod, new_final_submod)
        return f"Merge {new_model_num} Success! Loading", 200
    except Exception as e:
        return f"Merge {new_model_num} Error: {e}!", 500




@app.route("/change/<int:new_model_num>", methods=["GET"])
async def change_model(new_model_num):
    try:
        global final_submod
        global model_nums
        if new_model_num == 0:
            return {"changed": final_submod - current_model}, 200
        new_final_submod = current_model + new_model_num
        if new_final_submod < final_submod:
            do_merge_model()
            final_submod = new_final_submod
            model_nums = new_model_num
            return f"Successfully changed subgraphs to {model_nums}", 200
            # return f"Can not reduce subgraphs from {model_nums} to {new_model_num}", 500
        if new_final_submod == final_submod:
            return f"The model has aleady merged {model_nums} models", 200
        if not check_cuda_memory(final_submod, new_final_submod):
            return (f"Change error:CUDA {current_device()} out of memory!", 500)
        global loaded
        do_merge_model()
        final_submod = new_final_submod
        model_nums = new_model_num
        return f"Change {new_model_num} Success!", 200
    except Exception as e:
        return f"Change {new_model_num} Error: {e}!", 500


@app.route(
    "/", defaults={"path": ""}, methods=["GET", "PUT", "POST", "PATCH", "DELETE"]
)
@app.route("/<path:path>", methods=["GET", "PUT", "POST", "PATCH", "DELETE"])
async def call_handler(path):
    try:
        await inference()
        if final_submod >= 64:
            return {"result": "last"}, 200
        else:
            response, status_code = await call_next_function()
            return response, status_code
    except:
        return "infernece except", 500


if __name__ == "__main__":
    p = Process(target=asyncio.run, args=(serve_loaded_endpoint(),))
    p.daemon = True
    p.start()
    config = Config()
    # config.worker_class = "asyncio"
    # config.workers = 2
    config.bind = ["0.0.0.0:5000"]
    asyncio.run(serve(app, config))
