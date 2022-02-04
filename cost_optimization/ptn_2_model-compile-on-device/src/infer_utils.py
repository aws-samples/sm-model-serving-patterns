import numpy as np
import time
import json
import boto3
import sagemaker
from sagemaker.utils import name_from_base


def get_classes(train_path):
    #https://github.com/pytorch/vision/blob/50d9dc5f5af89e607100cee9aa34cfda67e627fb/torchvision/datasets/folder.py#L114
    classes = [d.name for d in os.scandir(train_path) if d.is_dir()]
    classes.sort()
    classes_dict = {i:c for i, c in enumerate(classes)}
    return classes, classes_dict
    
    
def save_classes_dict(classes_dict, filename='classes_dict.json'):
    with open(filename, "w") as fp:
        json.dump(classes_dict, fp) 
        
        
def load_classes_dict(filename):
    with open(filename, 'r') as f:
        classes_dict = json.load(f)
        
    classes_dict = {int(k):v for k, v in classes_dict.items()}
    return classes_dict


def get_inference(img_path, predictor, classes_dict, show_img=True):
    with open(img_path, mode='rb') as file:
        payload = bytearray(file.read())

    response = predictor.predict(payload)
    result = json.loads(response.decode())
    pred_cls_idx, pred_cls_str, prob = parse_result(result, classes_dict, img_path, show_img)
    
    return pred_cls_idx, pred_cls_str, prob 


def parse_result(result, classes_dict, img_path=None, show_img=True):
    pred_cls_idx = np.argmax(result)
    pred_cls_str = classes_dict[pred_cls_idx]
    prob = np.amax(result)*100
    
    if show_img:
        import cv2
        import matplotlib.pyplot as plt
        im = cv2.imread(img_path, cv2.COLOR_BGR2RGB)
        font = cv2.FONT_HERSHEY_COMPLEX
        cv2.putText(im, f'{pred_cls_str} {prob:.2f}%', (10,40), font, 0.7, (255, 255, 0), 2, cv2.LINE_AA)
        plt.figure(figsize=(10, 10))
        plt.imshow(im[:,:,::-1])    

    return pred_cls_idx, pred_cls_str, prob


def compile_model_for_jetson(role, bucket, target_device='jetson-nano',
                             dataset_dir=None, framework='PYTORCH', 
                             trt_ver='7.1.3', cuda_ver='10.2', gpu_code='sm_53',
                             base_model_name='model', img_size=224, use_gpu=True):
    if dataset_dir is None:
        print("[INFO] The dataset prefix of the s3 bucket is automatically assigned as 'modelzoo'.")
        dataset_dir = 'modelzoo'
        
    sm_client = boto3.client('sagemaker')
    sess = sagemaker.Session()
    region = sess.boto_region_name
    target_device_ = target_device.replace('_', '-')

    if use_gpu:
        compilation_job_name = name_from_base(f'{target_device_}-{base_model_name}-gpu-pytorch')
    else:
        compilation_job_name = name_from_base(f'{target_device_}-{base_model_name}-cpu-pytorch')        
    
    s3_compiled_model_path = 's3://{}/{}/{}/neo-output'.format(bucket, dataset_dir, compilation_job_name)
    key_prefix = f'{dataset_dir}/{compilation_job_name}/model'
    s3_model_path = sess.upload_data(path='model.tar.gz', key_prefix=key_prefix)

    # Configuration
    if use_gpu:
        input_config = {
            'S3Uri': s3_model_path,
            'DataInputConfig': f'{{"input0": [1,3,{img_size},{img_size}]}}',
            'Framework': framework,
        }
        output_config = {
            'S3OutputLocation': s3_compiled_model_path,
            'TargetPlatform': { 
                'Os': 'LINUX', 
                'Arch': 'ARM64', # change this to X86_64 if you need
                'Accelerator': 'NVIDIA'  # comment this if you don't have an Nvidia GPU
            },        
            # Jetson Xavier: sm_72; Jetson Nano: sm_53
            'CompilerOptions': f'{{"trt-ver": "{trt_ver}", "cuda-ver": "{cuda_ver}", "gpu-code": "{gpu_code}"}}' # Jetpack 4.5.1            
        }
    else:
        input_config = {
            'S3Uri': s3_model_path,
            'DataInputConfig': f'{{"input0": [1,3,{img_size},{img_size}]}}',
            'Framework': framework,
        }
        output_config = {
            'S3OutputLocation': s3_compiled_model_path,
            'TargetPlatform': { 
                'Os': 'LINUX', 
                'Arch': 'ARM64', # change this to X86_64 if you need
            },        
        }        
        
    # Create Compilation job    
    compilation_response = sm_client.create_compilation_job(
        CompilationJobName=compilation_job_name,
        RoleArn=role,
        InputConfig=input_config,
        OutputConfig=output_config,
        StoppingCondition={ 'MaxRuntimeInSeconds': 900 }
    )

    return {
        'response': compilation_response,
        'job_name': compilation_job_name, 
        's3_compiled_model_path': s3_compiled_model_path, 
        's3_model_path': s3_model_path
    }


def compile_model_for_cloud(role, bucket, target_device, 
                            dataset_dir=None, framework='PYTORCH', framework_version='1.8',
                            base_model_name='model', img_size=224):    
    valid_target_device = ['ml_m4', 'ml_m5', 
                           'ml_c4', 'ml_c5', 
                           'ml_p2', 'ml_p3', 'ml_g4dn', 
                           'ml_inf1', 'ml_eia2']
    
    if not target_device in valid_target_device:
        print('[ERROR] Please use valid target device!')
        return
    
    if dataset_dir is None:
        print("[INFO] The dataset prefix of the s3 bucket is automatically assigned as 'modelzoo'.")
        dataset_dir = 'modelzoo'
            
    sm_client = boto3.client('sagemaker')
    sess = sagemaker.Session()
    region = sess.boto_region_name
    target_device_ = target_device.replace('_', '-')

    compilation_job_name = name_from_base(f'{target_device_}-{base_model_name}-pytorch')        
    
    s3_compiled_model_path = 's3://{}/{}/{}/neo-output'.format(bucket, dataset_dir, compilation_job_name)
    key_prefix = f'{dataset_dir}/{compilation_job_name}/model'
    s3_model_path = sess.upload_data(path='model.tar.gz', key_prefix=key_prefix)


    input_config = {
        'S3Uri': s3_model_path,
        'DataInputConfig': f'{{"input0": [1,3,{img_size},{img_size}]}}',
        'Framework': framework,
        'FrameworkVersion': framework_version
    }
    output_config = {
        'TargetDevice': target_device,        
        'S3OutputLocation': s3_compiled_model_path,    
    }        

    # Create Compilation job    
    compilation_response = sm_client.create_compilation_job(
        CompilationJobName=compilation_job_name,
        RoleArn=role,
        InputConfig=input_config,
        OutputConfig=output_config,
        StoppingCondition={ 'MaxRuntimeInSeconds': 900 }
    )
    
    return {
        'response': compilation_response,
        'job_name': compilation_job_name, 
        's3_compiled_model_path': s3_compiled_model_path, 
        's3_model_path': s3_model_path
    }