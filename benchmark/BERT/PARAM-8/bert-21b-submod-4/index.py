#!/usr/bin/env python
from quart import Quart, request, jsonify
import os
import time
import random
import json
import torch
import aiohttp
import subprocess
import re
from hypercorn.config import Config
from hypercorn.asyncio import serve
import asyncio
from multiprocessing import Process

def get_cuda_device_mapping():
    try:
        # Run nvidia-smi to get the device information
        output = subprocess.check_output(["nvidia-smi", "-L"], universal_newlines=True)
    except subprocess.CalledProcessError:
        print("nvidia-smi could not be executed. Are you sure the system has an NVIDIA GPU and nvidia-smi is installed?")
        return {}
    except FileNotFoundError:
        print("nvidia-smi was not found. Are you sure it's installed?")
        return {}

    # Parse the output using regex
    devices = re.findall(r"GPU (\d+): (.* \(UUID: GPU-(.*)\))", output)

    # Build a mapping of UUID to device ID
    uuid_to_id = {uuid: int(id) for id, _, uuid in devices}
    
    return uuid_to_id

def get_device_id_by_uuid(target_uuid):
    uuid_to_id = get_cuda_device_mapping()
    
    return uuid_to_id.get(target_uuid, None)

model_name='bert-21b-submod-4'
stage = 4
pipeline_length = 8
if os.environ.get('retry_check') is None:
    os.environ['retry_check'] = 'false'
    retry_check = False
elif os.environ['retry_check'] == 'true':
    retry_check = True
else:
    retry_check = False
# retry_check = True
loaded = False
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
function_time_out = float(os.environ.get('read_timeout', "10s").replace("s",""))

with torch.inference_mode():
    model= torch.load('/data/model/openfaas/cdgp/subgraphs/bert-21b/PARAM_OAP/8/bert-21b-submod-4.pt', map_location=infer_device)
    # warm up 
    if hasattr(model, "gm_list"):
        for gm in model.gm_list:
            if hasattr(gm, "_buffers"):
                for k, v in gm._buffers.items():
                    if isinstance(v, torch.Tensor):
                        setattr(gm, k, v.to(infer_device))
    
    x = torch.load("/data/model/openfaas/cdgp/subgraphs/bert-21b/PARAM_OAP/8/bert-21b-inputs-4.pt", map_location=infer_device)
    output = model(*x)
            
    loaded = True


app = Quart(__name__)

async def serve_loaded_endpoint():
    app_loaded = Quart("LoadedApp")

    @app_loaded.route('/loaded', methods=['GET'])
    async def check_loaded():
        try:
            if loaded:
                return jsonify({'result': True}), 200
            else:
                return jsonify({'result': False}), 500
        except:
            return jsonify({'result': False}), 500

    config = Config()
    config.bind = ["0.0.0.0:5001"]
    await serve(app_loaded, config)

async def call_next_function(start_time,uid):
    timeout = aiohttp.ClientTimeout(total=function_time_out)
    retry_count = 0
    while True:
        # try:
        async with aiohttp.ClientSession() as session:
            async with session.get("http://172.169.8.253:31112/function/bert-21b-submod-5-param-8.cdgp?uid="+str(uid), timeout=timeout) as response:
    
                result = await response.text()
                if response.status == 200:
                    json_response = json.loads(result)
                    json_response['t_exec'] =  json_response['end_time'] - start_time
                    return json.dumps(json_response), 200
                elif retry_check:                        
                    retry_count += 1
                    if await ask_for_retry(retry_count,start_time,uid):
                        # print(f"Retrying...{retry_count}")
                        continue  
                return result,response.status                      
                    # if response.status == 500:
                    #     return result,response.status
                    #     try:
                    #         json_response = json.loads(result)
                    #         json_response['t_exec'] =  json_response['end_time'] - start_time
                    #         return json_response, response.status
                    #     except:
                    #         # print(f"denied retry.{retry_count}")
                    #         return {'result':f'Retry denied:{result}','end_time':time.time(),'t_exec':time.time()-start_time,'stage':stage}, 500   
                    # else:
                    #     return {'result':f'Retry denied:{result}','end_time':time.time(),'t_exec':time.time()-start_time,'stage':stage}, 500  
                               
                    # else:
                    #     # print(f"do not retry.{result}")
                    #     return {'result':f'{result} & retry_check not enabled','end_time':time.time(),'t_exec':time.time()-start_time,'stage':stage}, 500
        # except Exception as e:
        #     print(f"ERROR:{e}")
        #     return {'result':f'call subgraph exception: {e}','end_time':time.time(),'t_exec':time.time()-start_time,'stage':stage}, 500


async def ask_for_retry(retry_count,start_time,uid):
    async with aiohttp.ClientSession() as session:
        retry_request_data = {
            "model_name": f"'bert-21b-submod-4'",
            "stage": f"4",
            "uid":uid,
            "t_exec": f"{time.time()-start_time}",
            "retry_count": retry_count,
            'pipeline_length': pipeline_length,
        }
        async with session.post("http://roger-retry-service.roger/retry_check", json=retry_request_data) as response:
            if response.status == 200:
                json_response = json.loads(await response.text())
                # print(f"retry reply:{json_response}")
                return json_response.get('retry', False)
            else:
                return False 
            
inference_semaphore = asyncio.Semaphore(2)
if infer_device != "cpu":
    s = torch.cuda.Stream(infer_device)
async def inference():
    async with inference_semaphore:
        if infer_device != "cpu":
            with torch.cuda.stream(s):
                with torch.inference_mode():
                    output = model(*x)
                    s.synchronize()
        else:
            with torch.inference_mode():
                output = model(*x)
        return output 

@app.route('/', defaults={'path': ''}, methods=['GET', 'PUT', 'POST', 'PATCH', 'DELETE'])
@app.route('/<path:path>', methods=['GET', 'PUT', 'POST', 'PATCH', 'DELETE'])
async def call_handler(path):
    try:
        start_time = time.time()
        await inference()
        
        uid = request.args.get('uid') or str(random.randint(0, 10e6))
        # print("uid",uid)
        response, status_code = await call_next_function(start_time,uid)
        # print()
        return response, status_code
    except Exception as e:
        return str({'result':f'infernece except: {e}','end_time':time.time(),'t_exec':time.time()-start_time,'stage':4}), 500
    



if __name__ == '__main__':
    p = Process(target=asyncio.run, args=(serve_loaded_endpoint(),))
    p.daemon = True
    p.start()
    config = Config()
    # config.worker_class = "asyncio"
    # config.workers = 2
    config.bind = ["0.0.0.0:5000"]
    asyncio.run(serve(app, config))
