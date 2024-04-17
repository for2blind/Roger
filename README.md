# Roger
## introduction
[Roger](https://github.com/for2blind/Roger) is a fault-tolerant pipeline parallel inference system designed to handle large-scale model inference in distributed environments. It optimizes global decisions based on request distribution and system load during failure stages.
## installation
Roger is deployed on a Kubernetes cluster. Initially, when an inference request arrives, it is routed by the OpenFaaS gateway to aspecific pipeline.  The [installation](https://github.com/for2blind/Roger/tree/main/installation) shows the instruction to install the environment.
## benchmark
We provide some [benchmark](https://github.com/for2blind/Roger/tree/main/benchmark) for evaluating our system. The Benchmark shows the instruction to build and deploy several serverless workflows.
## system
Intelligent scaler and scheduler for model pipelines.The details can be found in [system](https://github.com/for2blind/Roger/tree/main/system).
## plot
Script for Plotting Evaluation Experiment Result Data.The details can be found in [plot](https://github.com/for2blind/Roger/tree/main/plot).
## retry_check_service
The primary core component of Roger is a service module responsible for determining whether to retry decisions when errors occur in the pipeline.The details can be found in [retry_check_service](https://github.com/for2blind/Roger/tree/main/retry_check_service).
## evaluation
This section encompasses the experimental evaluation of Roger.The details can be found in [evaluation](https://github.com/for2blind/Roger/tree/main/evaluation).