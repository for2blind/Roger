import os,datetime,time
import pandas as pd
current_path = os.path.dirname(os.path.abspath(__file__))
os.chdir(current_path)
import kubernetes
import pytz
import uuid
import subprocess, asyncio, requests
from concurrent.futures import ProcessPoolExecutor
from collections import defaultdict

kubernetes.config.load_kube_config()
v1 = kubernetes.client.CoreV1Api()
namespace = 'cdgp'
scaling = 'none'
scheduler = 'dasheng'
slo = 5
exp="E2_3"
EVALUATE_WORKLOAD = ['roger']
benchmark_version = '1.0.3.roger'

columns = ['uuid','wrkname','scaling','scheduler','start_time','end_time','slo','collected','csv']
evaluation_record_path=f"../../evaluation/metrics/evaluation_record/evaluation_record_{exp}.csv"
if not os.path.exists(evaluation_record_path):
    evaluation_record = pd.DataFrame(columns=columns)
    evaluation_record.to_csv(evaluation_record_path, index=False)
evaluation_record  = pd.read_csv(evaluation_record_path,names=columns)


url_list = {
    'OPT-66B': 'http://172.169.8.253:31112/function/opt-66b-submod-0-param-8.cdgp',
    'BERT-21B': 'http://172.169.8.253:31112/function/bert-21b-submod-0-param-8.cdgp',
    'LLAMA-7B': 'http://172.169.8.253:31112/function/llama-7b-submod-0-param-7.cdgp',
    'BERT-QA': 'http://172.169.8.253:31112/function/bert-qa.cdgp',    
    'Resnet-50': 'http://172.169.8.253:31112/function/resnet-50.cdgp',
    'Resnet-152': 'http://172.169.8.253:31112/function/resnet-152.cdgp',
}

models = list(url_list.keys())
def build_url_list(wrkname):
    return url_list

def deploy_model(wrkname):
    cmd = f'bash bash_deploy_{wrkname}.sh'
    os.system(cmd)  
    return True

async def release_model(wrkname):
    # cmd = f'bash bash_delete_all.sh'
    cmd = f'bash bash_delete.sh'
    os.system(cmd)    
    return True

def check_if_service_ready(model, wrkname):
    url_list = build_url_list(wrkname)    
    url = url_list[model]
    try:
        r = requests.get(url, timeout=5)
        if r.status_code == 200:
            print(f'service of {model} is ready')
            return True
        else:
            print(f'service of {model} is not ready, return code: {r.status_code}')
            return False
    except Exception as e:
        print(f'service of {model} is error', e)
        return False
    

# using request test if the wrk is ready
async def check_wrk_service_ready(wrkname):
    models_to_deploy = models.copy()
    while True:
        for model in models_to_deploy:
            status = check_if_service_ready(model, wrkname)
            if status == True:
                models_to_deploy.remove(model)
        if len(models_to_deploy) == 0:
            print(f'All service in wrk {wrkname} is ready')
            break

async def remove_dead_pod(pod_name):
    try:
        pod = v1.read_namespaced_pod(pod_name, namespace)
        if not pod.spec.containers[0].resources.limits.get('nvidia.com/gpu'):
            return
        if not pod.status.container_statuses or not pod.status.container_statuses[0].state.running:
            os.system(f'kubectl delete pod {pod.metadata.name} -n {namespace}')
    except:
        pass

async def check_wrk_pod_ready(wrkname, namespace):
    while True:
        pod_status = {}
        pods = v1.list_namespaced_pod(namespace).items
        for pod in pods:
            if pod.status.container_statuses:
                pod_status[pod.metadata.name] = pod.status.container_statuses[0].ready
            else:
                pod_status[pod.metadata.name] = False

        pod_ready = all(status for status in pod_status.values())
        # delete the pod of status "CrashLoopBackOff"
        if not pod_ready:
            print(f'{datetime.datetime.now()}, pod of {wrkname} is not yet ready')
            for pod in pod_status:
                if not pod_status[pod]:
                    await remove_dead_pod(pod)
        else:
            print(f'{wrkname} is ready')
            return pod_ready
        print(f'wait for 30 seconds to check again')
        await asyncio.sleep(30)


async def start_request_to_wrks(wrkname, scaling, scheduler):
    evaluation_record = pd.read_csv(evaluation_record_path,names=columns)
    if len(evaluation_record[(evaluation_record['wrkname']== wrkname)   & (evaluation_record['scaling'] == scaling) & (evaluation_record['scheduler'] == scheduler)]) > 0:
        print(f'{wrkname} {scaling} {scheduler} has been evaluated')
        # return
    duration = 620 * 5
    start_datetime = datetime.datetime.now(pytz.timezone('Asia/Shanghai'))
    estimated_end_time = start_datetime + datetime.timedelta(seconds=duration)    
    print(f"Workload started at: {start_datetime.strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Estimated end time: {estimated_end_time.strftime('%Y-%m-%d %H:%M:%S')}")

    start_time = time.time().__int__()    
    # print("Usage: python3 round_robbin_wrk.py  <wrkname> <scaling> <scheduler> <slo> <start_time>")
    # cmd = f"python3 cv_wula_RR_exp_{exp.split('_')[0]}.py {wrkname} {scaling} {scheduler} {slo} {start_time} {exp}"
    cmd = f"python3 cv_wula_exp_{exp.split('_')[0]}.py {wrkname} {scaling} {scheduler} {slo} {start_time} {exp}"
    # when workload is finished, record the time
    print("run ",cmd)
    process = subprocess.Popen(cmd, shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    process.wait()
    stdout, stderr = process.communicate()
    print(stdout.decode(),stderr.decode())
    end_time = time.time().__int__()
    evaluation_uuid = uuid.uuid4()
    resp_path = f'../metrics/wula_RR/{exp}/{wrkname}/{scheduler}/{start_time}/'
    tmp_record = pd.DataFrame([[evaluation_uuid,wrkname,scaling,scheduler,start_time,end_time,slo,0,resp_path]],columns=columns)
    evaluation_record = pd.concat([evaluation_record,tmp_record])
    evaluation_record.to_csv(evaluation_record_path,index=False,header=False)

async def replace_retry_service(wrkname):
    cmd_copy = f'kubectl cp ../../retry_check_service/retry_check_service_{wrkname}.py roger-retry-7c786bd9dd-2hfmg:/app/retry_check_service_{wrkname}.py -n roger'
    os.system(cmd_copy) 
    cmd_exec = f'kubectl exec roger-retry-7c786bd9dd-2hfmg -n roger -- /bin/bash -c "cd /app/ ; pkill -9 python ; sleep 5 ; python /app/retry_check_service_{wrkname}.py" &'
    print(cmd_exec)
    os.system(cmd_exec) 
    return True

async def run_wrk_benchmark(wrkname):    
    if len(evaluation_record[(evaluation_record['wrkname']== wrkname) & (evaluation_record['scaling'] == scaling) & (evaluation_record['scheduler'] == scheduler)]) > 0:
        print(f'{wrkname} has been evaluated')
    # await release_model(wrkname)
    # await asyncio.sleep(10)
    # deploy_model(wrkname)
    print(f'{wrkname} is deployed, sleeping to function ready!')
    await asyncio.sleep(10)
    await replace_retry_service(wrkname)
    await check_wrk_pod_ready(wrkname, namespace)
    await asyncio.sleep(10)
    await check_wrk_service_ready(wrkname)
    print(f'all service of {wrkname} is ready, start benchmark!')
    await start_request_to_wrks(wrkname, scaling, scheduler)
    print(f'{wrkname} done')
    # release wrk
    # await asyncio.sleep(30)
    # await release_model(wrkname)
    # await asyncio.sleep(30)




if __name__ == '__main__':
    for wrk in EVALUATE_WORKLOAD:
        wrkname = wrk
        asyncio.run(run_wrk_benchmark(wrkname))
    # asyncio.run(run_wrk_benchmark())
    # asyncio.run(release_model('LATENCY'))
    