from prometheus_api_client import PrometheusConnect
import datetime, time
import pandas as pd
from kubernetes import client, config, watch

# read json config file from 'gss.json'
import json
import os, re
import sys, argparse

config.load_kube_config()
import pytz
import glob

os.chdir(os.path.dirname(os.path.abspath(__file__)))
v1 = client.CoreV1Api()
v1_app = client.AppsV1Api()
api = client.CustomObjectsApi()
# v1beta1 = client.ExtensionsV1beta1Api()

ret = v1.list_node(watch=False)
node_ip = {}
for i in ret.items:
    node_ip[i.status.addresses[0].address] = i.metadata.name


class MetricCollector:
    def __init__(self):
        self.prom = PrometheusConnect(url="http://172.169.8.253:30500")
        self.columns = ["timestamp", "value", "metrics", "instance", "ip"]
        self.columns_gpu = [
            "timestamp",
            "value",
            "metrics",
            "instance",
            "gpu",
            "modelName",
        ]
        self.columns_pod = ["timestamp", "value", "metrics", "function"]
        self.mdict_node = {
            "node_gpu_util": "DCGM_FI_DEV_GPU_UTIL",
            "gpu_DRAM_activated": "DCGM_FI_PROF_DRAM_ACTIVE",
            "gpu_tensor_pipe_active": "DCGM_FI_PROF_PIPE_TENSOR_ACTIVE",
            "gpu_men_util": "DCGM_FI_DEV_MEM_COPY_UTIL",
        }
        self.mdict_pod = {
            "pod_cpu_util": 'avg(rate(container_cpu_usage_seconds_total{namespace="cdgp"} [30s])) by(pod) /avg(kube_pod_container_resource_limits{resource="cpu",namespace="cdgp"}) by (pod) *100',
            "pod_mem_usage": "avg(container_memory_rss+container_memory_cache+container_memory_usage_bytes+container_memory_working_set_bytes{namespace='cdgp'}) by (pod)/1024/1024",
        }
        self.sum_metrics = {
            "active_gpu_num": "count(DCGM_FI_DEV_GPU_UTIL > 0)",
            "energy_consumption": "sum(rate(DCGM_FI_DEV_POWER_USAGE[60s]))",
        }
        self.deploy_metrics = {
            "replica_num": 'sum (kube_deployment_status_replicas_available{namespace="cdgp"})  by (deployment) ',
        }

    def collect_pod_metrics(self, start_time, end_time):
        start_time = datetime.datetime.fromtimestamp(start_time, tz=pytz.UTC)
        end_time = datetime.datetime.fromtimestamp(end_time, tz=pytz.UTC)
        Mdata_pod = pd.DataFrame(columns=self.columns_pod)
        for metric_name in self.mdict_pod.keys():
            query_result = self.prom.custom_query_range(
                self.mdict_pod[metric_name],
                end_time=end_time,
                start_time=start_time,
                step="5s",
            )
            for m in query_result:
                data_t = pd.json_normalize(m)
                dk_tmp = pd.DataFrame(columns=self.columns_pod)
                for i, r in data_t.iterrows():
                    df_values = pd.DataFrame(
                        r["values"], columns=["timestamp", "value"]
                    )
                    df_values["pod"] = r["metric.pod"]
                    dk_tmp = pd.concat([dk_tmp, df_values], axis=0)
                dk_tmp["metrics"] = metric_name
                Mdata_pod = pd.concat([Mdata_pod, dk_tmp], axis=0)
            print(f"pod : {metric_name}")
        return Mdata_pod

    def collect_deploy_metrics(self, start_time, end_time):
        start_time = datetime.datetime.fromtimestamp(start_time, tz=pytz.UTC)
        end_time = datetime.datetime.fromtimestamp(end_time, tz=pytz.UTC)
        Mdata_deployment = pd.DataFrame(columns=self.columns_pod)
        for metric_name in self.deploy_metrics.keys():
            query_result = self.prom.custom_query_range(
                self.deploy_metrics[metric_name],
                end_time=end_time,
                start_time=start_time,
                step="5s",
            )
            for m in query_result:
                data_t = pd.json_normalize(m)
                dk_tmp = pd.DataFrame(columns=self.columns_pod)
                for i, r in data_t.iterrows():
                    df_values = pd.DataFrame(
                        r["values"], columns=["timestamp", "value"]
                    )
                    df_values["deployment"] = r["metric.deployment"]
                    dk_tmp = pd.concat([dk_tmp, df_values], axis=0)
                dk_tmp["metrics"] = metric_name
                Mdata_deployment = pd.concat([Mdata_deployment, dk_tmp], axis=0)
            print(f"deployment : {metric_name}")
        return Mdata_deployment

    def collect_sum_metrics(self, start_time, end_time):
        start_time = datetime.datetime.fromtimestamp(start_time, tz=pytz.UTC)
        end_time = datetime.datetime.fromtimestamp(end_time, tz=pytz.UTC)
        Mdata_sum = pd.DataFrame()
        for metric_name in self.sum_metrics.keys():
            query_result = self.prom.custom_query_range(
                self.sum_metrics[metric_name],
                end_time=end_time,
                start_time=start_time,
                step="5s",
            )
            for m in query_result:
                data_t = pd.json_normalize(m)
                dk_tmp = pd.DataFrame()
                for i, r in data_t.iterrows():
                    df_values = pd.DataFrame(
                        r["values"], columns=["timestamp", "value"]
                    )
                    # df_values['deployment'] = r['metric.deployment']
                    dk_tmp = pd.concat([dk_tmp, df_values], axis=0)
                dk_tmp["metrics"] = metric_name
                Mdata_sum = pd.concat([Mdata_sum, dk_tmp], axis=0)
            print(f"sum : {metric_name}")
        return Mdata_sum

    def get_node_metrics(self, start_time, end_time):
        start_time = datetime.datetime.fromtimestamp(start_time, tz=pytz.UTC)
        end_time = datetime.datetime.fromtimestamp(end_time, tz=pytz.UTC)
        Mdata_node = pd.DataFrame(columns=self.columns)
        Mdata_GPU = pd.DataFrame(columns=self.columns)
        for metric_name in self.mdict_node.keys():
            query_result = self.prom.custom_query_range(
                self.mdict_node[metric_name],
                end_time=end_time,
                start_time=start_time,
                step="5s",
            )
            if "gpu" in metric_name:
                for m in query_result:
                    data_t = pd.json_normalize(m)
                    dk_tmp = pd.DataFrame(columns=self.columns_gpu)
                    for i, r in data_t.iterrows():
                        df_values = pd.DataFrame(
                            r["values"], columns=["timestamp", "value"]
                        )
                        df_values["gpu"] = r["metric.gpu"]
                        df_values["modelName"] = r["metric.modelName"]
                        df_values["instance"] = r["metric.kubernetes_node"]
                        dk_tmp = pd.concat([dk_tmp, df_values], axis=0)
                    dk_tmp["metrics"] = metric_name
                    Mdata_GPU = pd.concat([Mdata_GPU, dk_tmp], axis=0)
            else:
                for m in query_result:
                    data_t = pd.json_normalize(m)
                    dk_tmp = pd.DataFrame(columns=self.columns)
                    for i, r in data_t.iterrows():
                        df_values = pd.DataFrame(
                            r["values"], columns=["timestamp", "value"]
                        )
                        dk_tmp = pd.concat([dk_tmp, df_values], axis=0)
                    dk_tmp["metrics"] = metric_name
                    dk_tmp["ip"] = m["metric"]["instance"].split(":")[0]
                    Mdata_node = pd.concat([Mdata_node, dk_tmp], axis=0)
                    Mdata_node["instance"] = Mdata_node["ip"].apply(
                        lambda x: node_ip[x]
                    )
            print(f"node : {metric_name}")
        return Mdata_node, Mdata_GPU

    def collect_all_metrics(self, start_time, end_time):
        node_perf, gpu_perf = self.get_node_metrics(start_time, end_time)
        return node_perf, gpu_perf


class OpenFaasCollector:
    def __init__(self, fun_name):
        self.prom = PrometheusConnect(
            url="http://172.169.8.253:31113/", disable_ssl=False
        )
        # irate(gateway_functions_seconds_sum{function_name=~'bert-21b-submod-.*-latency-10.cdgp'}[60s])
        self.sum_metrics = {
            # 'latency':f'avg by (function_name, code)(rate(gateway_functions_seconds_sum[30s]) / rate(gateway_functions_seconds_count[30s]))',
            "execution_time": f'(irate(gateway_functions_seconds_sum{{function_name=~"{fun_name}"}}[60s]) / irate(gateway_functions_seconds_count{{function_name=~"{fun_name}"}}[60s]))',
            "response": f'irate(gateway_function_invocation_total{{function_name=~"{fun_name}"}}[60s])',
            "scale": f'sum(gateway_service_count{{function_name=~"{fun_name}"}}) by (function_name)/6',
            "rps": f'sum(irate(gateway_function_invocation_total{{function_name=~"{fun_name}"}}[60s])) by (function_name)',
            "queue": f'sum by (function_name) (gateway_function_invocation_started{{function_name=~"{fun_name}"}}) - sum by (function_name) (gateway_function_invocation_total{{function_name=~"{fun_name}"}})',
            "invalid": f'sum by (function_name)(irate(gateway_function_invocation_total{{function_name=~"{fun_name}",code!="200"}}[60s]))',
            "goodput": f'sum by (function_name)(irate(gateway_function_invocation_total{{function_name=~"{fun_name}",code="200"}}[60s]))',
        }
        self.invoke_count_metrics = {
            "invoke_count": f'sum by (function_name) (gateway_function_invocation_total{{function_name=~"{fun_name}"}})',
        }

        print(f"openfaas : {fun_name}")

    def collect_invoke_count(self, start_time, end_time):
        dk = pd.DataFrame(
            columns=["timestamp", "value", "function", "metrics", "time_type"]
        )

        for time_type, query_time in zip(["start", "end"], [start_time, end_time]):
            for metric_name in self.invoke_count_metrics.keys():
                query_result = self.prom.custom_query(
                    self.invoke_count_metrics[metric_name],
                    params={"time": query_time},
                )

                for m in query_result:
                    timestamp = m["value"][0]
                    value = m["value"][1]
                    function_name = m["metric"]["function_name"]

                    df_values = pd.DataFrame(
                        {
                            "timestamp": [timestamp],
                            "value": [value],
                            "function": [function_name],
                            "metrics": [metric_name],
                            "time_type": [time_type],
                        }
                    )

                    dk = pd.concat([dk, df_values], axis=0, ignore_index=True)
                    print(f"openfaas invoke count : {function_name} {metric_name}")
        return dk

    def collect_sum_metrics(self, start_time, end_time):
        start_time = datetime.datetime.fromtimestamp(start_time, tz=pytz.UTC)
        end_time = datetime.datetime.fromtimestamp(end_time, tz=pytz.UTC)
        dk = pd.DataFrame()
        for metric_name in self.sum_metrics.keys():
            # print(self.sum_metrics[metric_name])
            query_result = self.prom.custom_query_range(
                self.sum_metrics[metric_name],
                end_time=end_time,
                start_time=start_time,
                step="5s",
            )
            for m in query_result:
                data_t = pd.json_normalize(m)
                for i, r in data_t.iterrows():
                    df_values = pd.DataFrame(
                        r["values"], columns=["timestamp", "value"]
                    )
                    df_values["function"] = r["metric.function_name"]
                    df_values["metrics"] = metric_name
                    dk = pd.concat([dk, df_values], axis=0)
            print(f"openfaas : {fun_name} {metric_name}")
        return dk


def get_csv_files_in_bottom_directory(root_path):
    csv_files = []
    for root, dirs, files in os.walk(root_path):
        if not dirs:
            csv_files.extend(glob.glob(os.path.join(root, "*.csv")))
    return csv_files


if __name__ == "__main__":
    # uuid,model,benchmark,concurrency,scaling,scheduler,workload,start_time,end_time,slo,collected
    # ee7713db-16bc-4a1a-873d-67d47b780901,LATENCY,baseline,baseline,1682668681,1682669882,10,0
    columns = [
        "uuid",
        "wrkname",
        "scaling",
        "scheduler",
        "start_time",
        "end_time",
        "slo",
        "collected",
        "csv",
    ]
    exp = "E2_final"
    record_csv = f"/home/pengshijie/roger/evaluation/metrics/evaluation_record/evaluation_record_{exp}.csv"
    record = pd.read_csv(record_csv, header=0)
    mc = MetricCollector()
    interval = 10
    wula_base_path = f"/home/pengshijie/roger/evaluation/metrics/wula/{exp}/"
    for i, r in record.iterrows():
        # if int(r['collected']) == 1:
        #     continue
        # if r['scaling'] != 'cdgp-x':
        #     continue
        # print(r['start_time'],r['end_time'])
        start_time_rc = r["start_time"]
        print(
            f'collecting {exp} {r["wrkname"]}, {r["start_time"]}, {r["end_time"]}, total time: {(r["end_time"]-r["start_time"])/3600}'
        )
        evaluation_id = r["uuid"]
        wrkname = r["wrkname"]
        # if wrkname!='inferline':
        #     continue
        scheduler = r["scheduler"]
        csv = r["csv"]
        collect_record = pd.DataFrame(columns=["uuid", "cv_mu_duration", "collected"])

        for csv_file in get_csv_files_in_bottom_directory(csv):
            file_name = os.path.basename(csv_file)
            cv_mu_duration = csv_file.split("/")[-2]
            # cv_value = cv_mu_duration.split("_")[0]
            # cv_mu_duration='2.0_1_600'
            # cv_value=2
            # wrkname = csv_file.split("/")[-5]
            prom_data_path = f"/home/pengshijie/roger/evaluation/metrics/system/{exp}/{wrkname}/{scheduler}/{start_time_rc}/"
            if not os.path.exists(prom_data_path):
                os.makedirs(prom_data_path)

            df = pd.read_csv(csv_file)
            df["RequestTime"] = df["RequestTime"].astype(int)
            df["ResponseTime"] = df["ResponseTime"].astype(int)
            max_request_time = df["RequestTime"].max()
            min_request_time = df["RequestTime"].min()
            max_response_time = df["ResponseTime"].max()
            min_response_time = df["ResponseTime"].min()

            start_time_r = min(min_request_time, min_response_time) / 1e9
            end_time_r = max(max_request_time, max_response_time) / 1e9

            start_time = start_time_r - interval
            end_time = end_time_r + interval
            if (
                not collect_record[
                    (collect_record["uuid"] == evaluation_id)
                    & (collect_record["cv_mu_duration"] == cv_mu_duration)
                ]["collected"]
                .eq(1)
                .any()
            ):
                node_perf_all, gpu_perf_all = mc.collect_all_metrics(
                    start_time=start_time, end_time=end_time
                )
                gpu_perf_all["evaluation_id"] = evaluation_id
                gpu_perf_all["cv_mu_duration"] = cv_mu_duration
                gpu_perf_all.to_csv(
                    f"{prom_data_path}/../gpu_{evaluation_id}_{cv_mu_duration}.csv",
                    header=False,
                )

                pod_metrics_all = mc.collect_pod_metrics(
                    start_time=start_time, end_time=end_time
                )
                pod_metrics_all["evaluation_id"] = evaluation_id
                pod_metrics_all["cv_mu_duration"] = cv_mu_duration
                pod_metrics_all.to_csv(
                    f"{prom_data_path}/../pod_{evaluation_id}_{cv_mu_duration}.csv",
                    header=False,
                )

                sum_metrics_all = mc.collect_sum_metrics(
                    start_time=start_time, end_time=end_time
                )
                sum_metrics_all["evaluation_id"] = evaluation_id
                sum_metrics_all["cv_mu_duration"] = cv_mu_duration
                sum_metrics_all.to_csv(
                    f"{prom_data_path}/../sum_{evaluation_id}_{cv_mu_duration}.csv",
                    header=False,
                )
                collect_record = collect_record._append(
                    {
                        "uuid": evaluation_id,
                        "cv_mu_duration": cv_mu_duration,
                        "collected": 1,
                    },
                    ignore_index=True,
                )

            fun_url = df["URL"].unique().tolist()
            fun_name_list = [
                re.sub(r"-(\d+)-", r"-.*-", s.split("/")[-1]) for s in fun_url
            ]
            fun_name_list = list(set(fun_name_list))
            for fun_name in fun_name_list:
                openfaas_prom = OpenFaasCollector(fun_name)
                scale_df = openfaas_prom.collect_sum_metrics(
                    start_time=start_time, end_time=end_time
                )
                scale_df["evaluation_id"] = evaluation_id
                scale_df["cv_mu_duration"] = cv_mu_duration
                scale_df.to_csv(
                    f"{prom_data_path}/openfaas_{evaluation_id}.csv",
                    header=False,
                    mode="a",
                )
                invoke_df = openfaas_prom.collect_invoke_count(
                    start_time=start_time, end_time=end_time
                )
                invoke_df["evaluation_id"] = evaluation_id
                invoke_df["cv_mu_duration"] = cv_mu_duration
                invoke_df.to_csv(
                    f"{prom_data_path}/invoke_{evaluation_id}.csv",
                    header=False,
                    mode="a",
                )

        record.loc[i, "collected"] = 1
        record.to_csv(record_csv, header=True, index=False)
