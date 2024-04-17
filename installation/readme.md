# Installation
 Before starting, you should already have a Kubernetes cluster that meets the dependencies. If you have Kubernetes installation questions, please refer to https://kubernetes.io/docs/setup/
# OpenFaaS
```shell
curl -fsSL -o get_helm.sh https://raw.githubusercontent.com/helm/helm/main/scripts/get-helm-3
chmod 700 get_helm.sh
./get_helm.sh
helm repo add openfaas https://openfaas.github.io/faas-netes
helm install openfaas openfaas/openfaas --version 14.2.34 
curl -sSL https://cli.openfaas.com | sudo -E sh
```
 For more information, please refer to https://artifacthub.io/packages/helm/openfaas/openfaas

# Docker
```shell
# Add Docker's official GPG key:
sudo apt-get update
sudo apt-get install ca-certificates curl
sudo install -m 0755 -d /etc/apt/keyrings
sudo curl -fsSL https://download.docker.com/linux/ubuntu/gpg -o /etc/apt/keyrings/docker.asc
sudo chmod a+r /etc/apt/keyrings/docker.asc

# Add the repository to Apt sources:
echo \
  "deb [arch=$(dpkg --print-architecture) signed-by=/etc/apt/keyrings/docker.asc] https://download.docker.com/linux/ubuntu \
  $(. /etc/os-release && echo "$VERSION_CODENAME") stable" | \
  sudo tee /etc/apt/sources.list.d/docker.list > /dev/null
sudo apt-get update
# Install the Docker packages.
sudo apt-get install docker-ce docker-ce-cli containerd.io docker-buildx-plugin docker-compose-plugin
# Verify that the Docker Engine installation is successful by running the hello-world image.
sudo docker run hello-world
```
 For more information, please refer to https://docs.docker.com/engine/install/ubuntu/

# requirements
```shell
bash install_requirements.sh
```

