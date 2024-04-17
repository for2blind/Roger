import pandas as pd
from datetime import datetime
import os, sys, subprocess, math
import numpy as np
import logging, time
from cachetools import TTLCache, cached
import asyncio
from collections import defaultdict
import multiprocessing

current_path = os.path.dirname(os.path.abspath(__file__))
os.chdir(current_path)

logging.basicConfig(filename="cv_wula.log", level=logging.DEBUG)
GATEWAY = "http://172.169.8.253:31112"
namespace = "cdgp"

exp = "E1_1"
function_model_size = pd.read_csv(
    "../../benchmark/function_model_size.csv", header=0
)

def extract_prefix_and_number(key):
    parts = key.split("-")
    prefix = "-".join(parts[:-1])
    number = int(parts[-1])
    return prefix, number


url_list = {
    "OPT-66B": "http://172.169.8.253:31112/function/opt-66b-submod-0-param-8.cdgp",
    "BERT-21B": "http://172.169.8.253:31112/function/bert-21b-submod-0-param-8.cdgp",
    "LLAMA-7B": "http://172.169.8.253:31112/function/llama-7b-submod-0-param-7.cdgp",
    "BERT-QA": "http://172.169.8.253:31112/function/bert-qa.cdgp",
    "Resnet-50": "http://172.169.8.253:31112/function/resnet-50.cdgp",
    "Resnet-152": "http://172.169.8.253:31112/function/resnet-152.cdgp",
}


def build_url_list(wrkname):
    return url_list


def get_slo(benchmark):
    slo_list = {
        "OPT-66B": 5,
        "BERT-21B": 5,
        "LLAMA-7B": 5,
        "BERT-QA": 1,
        "Resnet-50": 1,
        "Resnet-152": 1,
    }
    return slo_list[benchmark]


def generate_gamma_distribution(t: float, mu: float, sigma: float):
    beta = sigma**2 / mu
    alpha = mu / beta
    n = int(math.ceil(t / mu))
    s = t * 2
    while s > t * 1.5:
        dist = np.random.gamma(alpha, beta, n)
        for i in range(1, n):
            dist[i] += dist[i - 1]
        s = dist[-1]
    return dist


def to_file(distFile: str, dist: list[float]):
    os.makedirs(os.path.dirname(distFile), exist_ok=True)
    with open(distFile, "w+") as f:
        for d in dist:
            f.write(f"{d}\n")


async def run_http_request_with_wula(
    wrkname: str,
    benchmark: str,
    scheduler: str,
    slo_do: str,
    start_time: str,
    url: str,
    synctime: str,
):
    resp_path = f"../metrics/wula/{exp}/{wrkname}/{scheduler}/{start_time}/"
    os.makedirs(os.path.dirname(resp_path), exist_ok=True)
    cvd_path = f"../workloads/Azure/"
    request_cmd = f"../workloads/wula -name {benchmark} -dist AF21-0.5h -dir {cvd_path} -dest {resp_path}/{benchmark}.csv -url {url} -SLO {slo_do} -synctime {synctime}"
    print(request_cmd)
    logging.debug(request_cmd)
    process = await asyncio.create_subprocess_shell(request_cmd)
    await process.wait()


async def main():
    benchmark_entry = build_url_list(wrkname)
    benchmarks = list(benchmark_entry.keys())

    tasks = []
    synctime = int((time.time() + 10) * 1000)
    for benchmark, url in benchmark_entry.items():
        print(benchmark)
        slo_do = get_slo(benchmark)
        tasks.append(
            run_http_request_with_wula(
                wrkname, benchmark, scheduler, slo_do, start_time, url, synctime
            )
        )
    await asyncio.gather(*tasks)
    await asyncio.sleep(10)


if __name__ == "__main__":
    start_time = time.time().__int__()
    default_values = ["cv_wula.py", "ME", "none", "dasheng", "10", start_time, exp]
    args = sys.argv
    if len(args) > 1:
        args = sys.argv
    else:
        args = default_values
    if len(args) == 0:
        print(
            "Usage: python3 cv_wula.py  <wrkname> <scaling> <scheduler> <slo> <start_time> <exp>"
        )
        # exit(1)
    wrkname = args[1]
    scaling = args[2]
    scheduler = args[3]
    slo = args[4]
    start_time = args[5]
    exp = args[6]
    print(args)
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    loop.run_until_complete(main())  # <-- Use this instead
    loop.close()  # <-- Stop the event loop
    exit(0)
