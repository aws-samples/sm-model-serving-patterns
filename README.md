# SageMaker Model Serving Patterns

## 1. Multi-container Endpoint (Korean NLP models)

### Multi-container Endpoint Overview
SageMaker 멀티 컨테이너 엔드포인트를 사용하면 서로 다른 serving 스택(예: 모델 서버, 머신 러닝 프레임워크, 프레임워크 버전, 알고리즘 등)에 구축된 여러 추론 컨테이너를 하나의 엔드포인트에서 실행하고 독립적으로 각 추론 컨테이너를 호출할 수 있습니다. 

- 인스턴스의 전체 수용량을 포화시킬 정도의 트래픽이 없는 경우에 여러 모델(예: Object Detection, Named Entity Recognition)을 서빙
- A/B 테스트와 같은 시나리오에서 서로 다른 프레임워크 버전(예: TensorFlow 1.x vs. TensorFlow 2.x)에서 실행되는 유사한 아키텍처의 비교

### Hands-on Lab
- [Module 1](multi-container-endpoint/1_local_endpoint.ipynb): Check Inference Results & Local Mode Deployment (Optional) 
- [Module 2](multi-container-endpoint/2_multi-container-endpoint.ipynb): Multi-container Endpoint Deployment

### References
- AWS AIML Blog: https://aws.amazon.com/ko/blogs/machine-learning/deploy-multiple-serving-containers-on-a-single-instance-using-amazon-sagemaker-multi-container-endpoints/
- Developer Guide: https://docs.aws.amazon.com/sagemaker/latest/dg/multi-container-endpoints.html
  
<br>  

## 2. Asynchronous Inference (Instance Segmentation)

### Asynchronous Inference Overview

SageMaker 비동기(asynchronous) 추론 엔드포인트는 처리 시간이 수 분 이상 걸릴 수 있는 대규모 입력 페이로드를 사용하는 유즈케이스에 적합합니다.
AsyncInvokeEndpoint API를 호출하면 Amazon S3에 payload를 배치하고 별도의 관리형 queue에 입력 데이터에 대한 요청을 포인터로 전송합니다. 호스팅 인스턴스는 포인터를 사용하여 payload 데이터를 다운로드하고 추론 컨테이너를 통해 모델 추론 결괏값을 계산하여 S3에 저장합니다. 이 때, 선택적으로 SNS를 통해 추론이 완료되었다는 알림을 받을 수 있습니다. 대표적인 특장점들은 아래와 같습니다.

- Larger payload: 실시간 엔드포인트에서 허용되는 최대 6MB payload보다 훨씬 큰 payload를 제공하며, 최대 1GB까지 가능합니다. (참조: https://docs.aws.amazon.com/general/latest/gr/sagemaker.html)
- Longer Model Execution Time: 실시간 엔드포인트의 최대 60초 모델 호출 시간 제약이 없습니다.
- Autoscaling down to 0 instance: 트래픽이 없을 때 인스턴스 개수를 0개까지 축소 가능하기에 비용을 절감할 수 있습니다.

### Hands-on Lab
- [Module 1](async-inference-cv/async-inference-cv.ipynb): Amazon SageMaker Asynchronous Endpoints - Computer Vision

### References

- AWS AIML Blog: https://aws.amazon.com/ko/blogs/machine-learning/run-computer-vision-inference-on-large-videos-with-amazon-sagemaker-asynchronous-endpoints/
- GitHub: https://github.com/aws-samples/amazon-sagemaker-asynchronous-inference-computer-vision
- Developer Guide: https://docs.aws.amazon.com/sagemaker/latest/dg/async-inference.html


<br>  

## 3. Lambda Serverless Inference

### Overview

re:Invent 2020에 소개된 Lambda 컨테이너 기능 지원으로 기존 Lambda에서 수행하기 어려웠던 대용량 머신 모델에 대한 추론을 보다 수월하게 실행할 수 있게 되었습니다. Lambda 컨테이너 이미지를 Amazon ECR(Amazon Elastic Container Registry)에 푸시하였다면 Lambda 함수를 생성하여 직접 컨테이너 이미지를 배포하거나 SageMaker의 API 호출로 Serverless endpoint를 쉽게 배포할 수 있습니다.
  
### Hands-on Lab
- [Module (Computer Vision)](lambda-serverless-inference/cv/README.md): Deploy Lambda Serverless endpoint - Object Detection (YOLO-v3)
- [Module (Korean Natural Language Processing)](lambda-serverless-inference/kornlp/README.md): Deploy Lambda Serverless endpoint - Korean NLP (Sentiment Classification for Naver Movie corpus)

### References

- AWS Lambda의 새로운 기능 — 컨테이너 이미지 지원: https://aws.amazon.com/ko/blogs/korea/new-for-aws-lambda-container-image-support/
- Lambda Serverless Inference: https://sagemaker.readthedocs.io/en/stable/overview.html?highlight=lambdamodel#serverless-inference
- AWS Builders Online - AWS Lambda 컨테이너 이미지 서비스 활용하기 (김태수 SA): https://www.youtube.com/watch?v=tTg9Lp7Sqok
  

## 4. SageMaker Serverless Inference

### Overview

Amazon SageMaker Serverless Inference는 re:Invent 2021에 런칭된 신규 추론 옵션으로 호스팅 인프라 관리에 대한 부담 없이 머신 러닝을 모델을 쉽게 배포하고 확장할 수 있도록 제작된 신규 추론 옵션입니다. SageMaker Serverless Inference는 컴퓨팅 리소스를 자동으로 시작하고 트래픽에 따라 자동으로 스케일 인/아웃을 수행하므로 인스턴스 유형을 선택하거나 스케일링 정책을 관리할 필요가 없습니다. 따라서, 트래픽 급증 사이에 유휴 기간이 있고 콜드 스타트를 허용할 수 있는 워크로드에 이상적입니다.

### Hands-on Lab
- [Module (Korean Natural Language Processing)](serverless-inference): Deploy SageMaker Serverless endpoint - Korean NLP downstream tasks

## License Summary

This sample code is provided under the MIT-0 license. See the LICENSE file.