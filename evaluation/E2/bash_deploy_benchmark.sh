cd ../../benchmark/GPT/PARAM-8
faas-cli deploy -f config.yml --gateway=http://172.169.8.253:31112 --namespace=cdgp --env retry_check='true' --env infer_device='cuda'
cd ../../benchmark/BERT/PARAM-8
faas-cli deploy -f config.yml --gateway=http://172.169.8.253:31112 --namespace=cdgp --env retry_check='true' --env infer_device='cuda'
cd ../../benchmark/LLAMA/PARAM-7
faas-cli deploy -f config.yml --gateway=http://172.169.8.253:31112 --namespace=cdgp --env retry_check='true' --env infer_device='cuda'
cd ../../benchmark/stream/
faas-cli deploy -f config.yml --gateway=http://172.169.8.253:31112 --namespace=cdgp --filter bert-qa --env retry_check='false' --env infer_device='cuda'
cd ../../benchmark/stream/
faas-cli deploy -f config.yml --gateway=http://172.169.8.253:31112 --namespace=cdgp --filter resnet-50 --env retry_check='false' --env infer_device='cuda'
cd ../../benchmark/stream/
faas-cli deploy -f config.yml --gateway=http://172.169.8.253:31112 --namespace=cdgp --filter resnet-152 --env retry_check='false' --env infer_device='cuda'