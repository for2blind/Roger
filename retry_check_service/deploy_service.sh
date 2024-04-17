docker build -t retry_check_service .
docker tag retry_check_service:latest
docker push retry_check_service:latest
kubectl deploy -f roger/retry_check_service/deploy.yaml 