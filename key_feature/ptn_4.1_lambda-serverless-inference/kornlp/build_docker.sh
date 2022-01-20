algorithm_name=$1

echo ==-------- Settings ---------==
echo $algorithm_name

if [ "$algorithm_name" == "" ]
then
    echo "Usage: $0 <image-name>"
    exit 1
fi

# Copy models to current directory
cp -r ../../multi-container-endpoint/model-nsmc ./model-nsmc

account=$(aws sts get-caller-identity --query Account --output text)

# Get the region defined in the current configuration 
region=$(aws configure get region)
#region=${region:-us-east-1}

fullname="${account}.dkr.ecr.${region}.amazonaws.com/${algorithm_name}:latest"

echo ==-------- Create ECR ---------==
# If the repository doesn't exist in ECR, create it.
aws ecr describe-repositories --repository-names "${algorithm_name}" > /dev/null 2>&1

if [ $? -ne 0 ]
then
    aws ecr create-repository --repository-name "${algorithm_name}" \
    --image-scanning-configuration scanOnPush=true \
    --region "${region}" > /dev/null
fi

# Get the login command from ECR and execute it directly
aws ecr get-login-password --region ${region}|docker login --username AWS --password-stdin ${fullname}

echo ==-------- Build Docker Image ---------==
# Build the docker image locally with the image name and then push it to ECR
# with the full name.
docker build -f Dockerfile -t ${algorithm_name} .
docker tag ${algorithm_name} ${fullname}

echo Local Docker Image : ${algorithm_name}
echo ECR Docker Image : ${fullname}

echo ==-------- Push Docker Image to ECR ---------==
docker push ${fullname}

echo == -------- Testing Docker Image. Open a new terminal and test the lambda function with test_lambda.sh or postman. --------==
docker run --rm -p 9000:8080 ${algorithm_name}:latest