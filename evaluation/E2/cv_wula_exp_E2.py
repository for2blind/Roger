import pandas as pd
from datetime import datetime
import os, sys,subprocess, math
import numpy as np
import logging, time
from cachetools import TTLCache,cached
import asyncio
from collections import defaultdict
import multiprocessing
current_path = os.path.dirname(os.path.abspath(__file__))
os.chdir(current_path)

logging.basicConfig(filename='cv_wula.log',level=logging.DEBUG)
GATEWAY = 'http://172.169.8.253:31112'
namespace = 'cdgp'

exp="E2_1"

mu_theory = {
    'OPT-66B': 0.08,
    'BERT-21B': 0.075,
    'LLAMA-7B': 0.075,
    'BERT-QA': 0.025,    
    'Resnet-50':0.025,
    'Resnet-152': 0.025,
}
url_list = {
    'OPT-66B': 'http://172.169.8.253:31112/function/opt-66b-submod-0-param-8.cdgp',
    'BERT-21B': 'http://172.169.8.253:31112/function/bert-21b-submod-0-param-8.cdgp',
    'LLAMA-7B': 'http://172.169.8.253:31112/function/llama-7b-submod-0-param-7.cdgp',
    'BERT-QA': 'http://172.169.8.253:31112/function/bert-qa.cdgp',    
    'Resnet-50': 'http://172.169.8.253:31112/function/resnet-50.cdgp',
    'Resnet-152': 'http://172.169.8.253:31112/function/resnet-152.cdgp'
}

def build_url_list(wrkname):
    return url_list
models = list(url_list.keys())


def generate_gamma_distribution(t: float, mu: float, sigma: float):
    beta = sigma ** 2 / mu
    alpha = mu / beta
    n = int(math.ceil(t / mu))
    s = t * 2
    while s > t * 1.5:
        dist = np.random.gamma(alpha, beta, n)
        for i in range(1, n):
            dist[i] += dist[i-1]
        s = dist[-1]
    return dist
    
def to_file(distFile: str, dist: list[float]):
    os.makedirs(os.path.dirname(distFile), exist_ok=True)
    with open(distFile, "w+") as f:
        for d in dist:
            f.write(f"{d}\n")


def build_request_distribution(wrkname:str, benchmark:str, scheduler:str, slo:str, start_time:str, mu:float, cv:float,duration:int):
    sigma = mu * cv
    output_path = f'../workloads/cvd/{wrkname}/{scheduler}/{start_time}/{cv}_{mu}_{duration}/'
    if not os.path.exists(output_path):
        os.makedirs(output_path)
    dist = generate_gamma_distribution(duration, mu, sigma)
    to_file(os.path.join(output_path, f"{benchmark}-dist.txt"), dist)

async def run_http_request_with_wula(wrkname:str, benchmark:str, scheduler:str, slo:str, start_time:str, mu:float, cv:float,url:str, synctime:str,duration:int):
    resp_path = f'../metrics/wula/{exp}/{wrkname}/{scheduler}/{start_time}/{cv}_{mu}_{duration}/'
    cvd_path = f'../workloads/cvd/{wrkname}/{scheduler}/{start_time}/{cv}_{mu}_{duration}/'
    os.makedirs(os.path.dirname(resp_path), exist_ok=True)
    request_cmd = f'../workloads/wula -name {benchmark} -dist {benchmark} -dir {cvd_path} -dest {resp_path}/{benchmark}.csv -url {url} -SLO {slo} -synctime {synctime}'
    print(request_cmd)
    logging.debug(request_cmd)
    process = await asyncio.create_subprocess_shell(request_cmd)
    await process.wait()

def get_mu(input_substring):
    # return 0.016
    try:
        mu = float(mu_theory[input_substring])
        print(mu)
        logging.debug(mu)
        return mu
    except Exception as e:
        logging.debug(e)
        print(e)
        return 0.05
    # df = pd.read_csv("/home/pengshijie/dybranch/OCD/model_config.csv")
    # filtered_df = df[df['model'].str.lower().str.contains(input_substring.lower())]
    # if not filtered_df.empty:
    #     max_inference_time = filtered_df.loc[filtered_df['stages'].idxmax(), 'inference_time']
    #     max_inference_time_float = float(max_inference_time) / 2
    #     return max_inference_time_float
    # else:
    #     return 0.05

async def main():

    CVs = [0.1, 1, 2, 4, 8] #for different workload.
    duration = 600
    # build request distribution
    benchmark_entry = build_url_list(wrkname)
    for cv in CVs:
        try:
            tasks = []
            synctime =  int((time.time()+10)*1000)
            for benchmark, url in benchmark_entry.items():
                print(benchmark)
                if not any(model.lower() in benchmark.lower() for model in models):
                    continue
                # start_time = expr timestamp=`date +%s%3N
                mu = get_mu(benchmark)
                build_request_distribution(wrkname, benchmark, scheduler, slo, start_time, mu, cv,duration)
                print(f"{benchmark} {cv} {mu} {url}")
                logging.debug(f"{benchmark} {cv} {mu} {url}")
                tasks.append(run_http_request_with_wula(wrkname, benchmark, scheduler, slo, start_time, mu, cv,url, synctime,duration))
            await asyncio.gather(*tasks)
            await asyncio.sleep(10)          
        except Exception as e:
            print(f"error {e}")
            return e


if __name__ == '__main__':
    start_time = time.time().__int__()
    default_values = ['cv_wula.py', 'ME','none', 'dasheng', '10', start_time,exp,8]
    args = sys.argv
    if len(args) > 1:
        args = sys.argv
    else:
        args = default_values
    if len(args) == 0:
        print("Usage: python3 cv_wula.py  <wrkname> <scaling> <scheduler> <slo> <start_time> <exp>")
        # exit(1)
    wrkname = args[1]
    scaling = args[2]
    scheduler=args[3]
    slo = args[4]
    start_time = args[5]
    exp=args[6]
    print(args)
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    loop.run_until_complete(main())  # <-- Use this instead
    loop.close()  # <-- Stop the event loop
    exit(0)
    