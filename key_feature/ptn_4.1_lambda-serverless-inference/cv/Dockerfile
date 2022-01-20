# Stage 1 lambda insight extensions
# FROM public.ecr.aws/serverless/extensions/lambda-insights:12 AS lambda-insights

# Stage 2
FROM python:3.9.1-slim-buster AS build-image

RUN apt-get update
RUN apt-get upgrade -y
RUN apt-get install ffmpeg libsm6 libxext6  -y
RUN mkdir -p /app
WORKDIR /app

COPY requirements.txt ./
RUN python -m pip install -r requirements.txt -t .
RUN python -m pip install awslambdaric --target .

# COPY --from=lambda-insights /opt /opt
COPY yolov3.cfg yolov3.weights coco.names samples/remote-control.jpeg  ./
COPY app.py ./

# RIC 엔트리포인트 세팅
ENTRYPOINT [ "/usr/local/bin/python", "-m", "awslambdaric" ]
CMD ["app.lambda_handler"]