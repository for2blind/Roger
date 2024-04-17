cd ../../benchmark/GPT/PARAM-8
faas-cli delete -f config.yml --gateway=http://172.169.8.253:31112 --namespace=cdgp
cd ../../benchmark/BERT/PARAM-8
faas-cli delete -f config.yml --gateway=http://172.169.8.253:31112 --namespace=cdgp
cd ../../benchmark/LLAMA/PARAM-7
faas-cli delete -f config.yml --gateway=http://172.169.8.253:31112 --namespace=cdgp
cd ../../benchmark/stream/
faas-cli delete -f config.yml --gateway=http://172.169.8.253:31112 --namespace=cdgp --filter bert-qa
cd ../../benchmark/stream/
faas-cli delete -f config.yml --gateway=http://172.169.8.253:31112 --namespace=cdgp --filter resnet-50
cd ../../benchmark/stream/
faas-cli delete -f config.yml --gateway=http://172.169.8.253:31112 --namespace=cdgp --filter resnet-152