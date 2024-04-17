FROM python:3.10

# Install additional packages for debugging and editing
RUN apt-get update && apt-get install -y vim procps

RUN mkdir /app
WORKDIR /app

COPY ./requirements.txt requirements.txt 
COPY ./retry_check_service.py retry_check_service.py

RUN pip install -r requirements.txt --proxy=http://172.20.110.146:10081

# Start the service
CMD ["python", "retry_check_service.py"]

# Keep the container running
ENTRYPOINT ["/bin/bash", "-c", "trap : TERM INT; sleep infinity & wait"]

#tag:k.harbor.siat.ac.cn/cc/roger_retry:v1.1

