#!/bin/bash

set -e

git clone https://github.com/tensorflow/models.git

source activate tensorflow_p36

cd /home/ec2-user/SageMaker/tfs-workshop/files/models/research
protoc object_detection/protos/*.proto --python_out=.

python setup.py build
python setup.py install

export PYTHONPATH=/home/ec2-user/SageMaker/tfs-workshop/files/models/research

source deactivate tensorflow_p36