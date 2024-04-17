import base64
import yaml, os
os.chdir(os.path.dirname(os.path.abspath(__file__)))
namespace = "cdgp"


schduler_ip = "172.16.101.223"
port = "9008"
url = f"https://{schduler_ip}:{port}/mutate"

# schduler_ip = "172.16.101.8"
# port = "9010"
# url = f"https://{schduler_ip}:{port}/mutate"


with open(f'{namespace}_openssl.cnf', 'w') as f:
    f.write(f"""
[req]
distinguished_name=req
[SAN]
subjectAltName=IP:{schduler_ip}
""")

# Generating the certificate and the key
os.system(f"openssl genrsa -out {namespace}_server.key 2048")
os.system(f"openssl req -new -x509 -sha256 -key {namespace}_server.key -out {namespace}_server.crt -days 3650 -subj /CN={schduler_ip} -extensions SAN -config '{namespace}_openssl.cnf'")

# Encoding the certificate
with open(f"{namespace}_server.crt", "rb") as cert_file:
    cert_base64 = base64.b64encode(cert_file.read()).decode()

# Reading the yaml file
with open(f"webhook.yml", 'r') as stream:
    try:
        yaml_content = yaml.safe_load(stream)
    except yaml.YAMLError as exc:
        print(exc)

new_metadata_name = f'{namespace}-webhook'
# Replacing {cert_base64}
if 'metadata' in yaml_content and 'name' in yaml_content['metadata']:
    yaml_content['metadata']['name'] = new_metadata_name
if isinstance(yaml_content, list):
    for section in yaml_content:
        if isinstance(section, dict) and 'webhooks' in section:
            for webhook in section['webhooks']:
                if 'clientConfig' in webhook and 'caBundle' in webhook['clientConfig']:
                    webhook['clientConfig']['caBundle'] = cert_base64
                    webhook['clientConfig']['url'] = url

                    webhook['name'] = f'{namespace}.siat.ac.cn'
                    matchLabels = f'{namespace}webhook'
                    if 'namespaceSelector' in webhook and isinstance(webhook['namespaceSelector'], dict):
                        if 'matchLabels' not in webhook['namespaceSelector']:
                            webhook['namespaceSelector']['matchLabels'] = {}
                        webhook['namespaceSelector']['matchLabels'][matchLabels] = 'enabled'
                        del webhook['namespaceSelector']['matchLabels']['webhook']

elif isinstance(yaml_content, dict):
    if 'webhooks' in yaml_content:
        for webhook in yaml_content['webhooks']:
            if 'clientConfig' in webhook and 'caBundle' in webhook['clientConfig']:
                webhook['clientConfig']['caBundle'] = cert_base64
                webhook['clientConfig']['url'] = url

                webhook['name'] = f'{namespace}.siat.ac.cn'
                matchLabels = f'{namespace}webhook'
                if 'namespaceSelector' in webhook and isinstance(webhook['namespaceSelector'], dict):
                    if 'matchLabels' not in webhook['namespaceSelector']:
                        webhook['namespaceSelector']['matchLabels'] = {}
                    webhook['namespaceSelector']['matchLabels'][matchLabels] = 'enabled'
                    del webhook['namespaceSelector']['matchLabels']['webhook']


# Writing back to yaml
with open(f"webhook-{namespace}.yml", 'w') as outfile:
    yaml.dump(yaml_content, outfile)

os.system(f"kubectl create ns {namespace}")
os.system(f'kubectl annotate namespace/{namespace} openfaas="1"')
os.system(f"kubectl delete -f webhook-{namespace}.yml")

os.system(f"kubectl apply -f webhook-{namespace}.yml")
os.system(f"kubectl label namespaces {namespace} {namespace}webhook=enabled --overwrite=true")
