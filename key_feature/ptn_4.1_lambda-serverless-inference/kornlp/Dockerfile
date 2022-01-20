FROM python:3.9.1-slim-buster AS custom-build-image

RUN apt-get update && apt-get upgrade -y && mkdir -p /app
WORKDIR /app

COPY requirements.txt ./
COPY model-nsmc ./model-nsmc
COPY app.py ./

RUN python -m pip install -r requirements.txt -t . && python -m pip install awslambdaric --target .


# Stage 2 최종 이미지
FROM python:3.9.1-slim-buster
WORKDIR /app

# 최종 이미지에 State 1에서 빌드된 종속성 결과물 추가하기
COPY --from=custom-build-image /app /app
# 로컬 테스트를 위한 AWS Lambda Runtime Interface Emulator 추가
ADD https://github.com/aws/aws-lambda-runtime-interface-emulator/releases/latest/download/aws-lambda-rie /usr/bin/aws-lambda-rie
RUN chmod 755 /usr/bin/aws-lambda-rie
COPY entry_script.sh /
RUN chmod +x /entry_script.sh
ENTRYPOINT [ "/entry_script.sh" ]
CMD [ "app.lambda_handler" ]