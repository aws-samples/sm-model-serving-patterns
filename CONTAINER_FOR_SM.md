# Containers for Amazon SageMaker Hosting

## Overview
단일 모델을 소규모 서비스로 배포 시에는 여러 모듈을 구성할 필요 없이 하나의 모듈 안에서 필요한 로직을 구성해도 무방합니다. 여러 종류의 모델들을 프로덕션 환경에서 배포 시,추론 환경을 안정적으로 빌드해야 함은 물론이고 각 모델의 프레임워크 종류, 프레임워크 버전 및 종속성을 고려해야 합니다. 또한, 동일한 시스템에서 실행되는 여러 모델들이 한정된 리소스를 두고 경쟁할 수 있으며, 특정 모델에서 오류 발생 시 여러 호스팅 모델들의 성능을 저하시킬 수 있습니다.

마이크로서비스 구조는 각 모듈을 독립된 형태로 구성하기 때문에 각 모듈의 관리가 쉽고 다양한 형태의 모델에 빠르게 대응할 수 있다는 장점이 있습니다. 도커(Docker)로 대표되는 컨테이너화 기술은 가상 머신과 달리 공통 운영 제체를 공유하면서 여러 모듈들에게 독립된 환경을 제공함으로써 유지 보수가 용이합니다.

Amazon SageMaker는 모델 훈련 및 배포 시에 도커 컨테이너를 자유롭게 사용할 수 있습니다.

## Algorithm Containers
ㅇ
SageMaker에서 제공하고 있는 17가지의 빌트인 알고리즘은 훈련 및 배포에 필요한 코드가 사전 패키징되어 있기에 별도의 코드를 작성할 필요가 없습니다. 배포 작업(deployment job)을 시작하면 SageMaker는 ECR에서 빌트인 추론 컨테이너를 가져오고 S3에서 훈련된 모델을 로드합니다.
S3에 저장된 모델은 SageMaker 훈련 작업(training job)을 런칭하여 훈련된 모델 아티팩트(model.tar.gz)를 의미하며, XGBoost와 BlazingText는 오픈소스 라이브러리와 호환되므로 온프렘에서 훈련한 모델을 model.tar.gz로 아카이빙하여 S3에 업로드하는 방식도 가능합니다.

## Managed Framework Containers 
TBA

## Bring Your Own Container(BYOC)
TBA

## AWS Marketplace Container
TBA