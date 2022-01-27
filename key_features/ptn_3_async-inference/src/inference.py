
import cv2
import json
import torch
import torchvision.transforms as transforms
from six import BytesIO
import io
import numpy as np
import tempfile

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def video2frame(file_path, frame_width, frame_height, interval):
    """
    Extract frame from video by interval
    :param video_src_path: video src path
    :param video:　video file name
    :param frame_width:　frame width
    :param frame_height:　frame height
    :param interval:　interval for frame to extract
    :return:　list of numpy.ndarray 
    """
    video_frames = []
    cap = cv2.VideoCapture(file_path)
    frame_index = 0
    frame_count = 0
    if cap.isOpened():
        success = True
    else:
        success = False
        print("Read failed!")

    while success:
        success, frame = cap.read()
        if frame_index % interval == 0:
            print("---> Reading the %d frame:" % frame_index, success)
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            resize_frame = cv2.resize(
                frame, (frame_width, frame_height), interpolation=cv2.INTER_AREA
            )
            video_frames.append(resize_frame)
            frame_count += 1

        frame_index += 1

    cap.release()
    print(f'Total frames={frame_index}, Number of extracted frames={frame_count}')
    return video_frames


def model_fn(model_dir):
    '''
    Loads the model into memory from storage and return the model.
    '''    
    model = torch.load(model_dir + '/model.pth', map_location=torch.device(device))
    model = model.eval()
    return model


def input_fn(request_body, request_content_type=None):
    frame_width = 256
    frame_height = 256
    interval = 30
    print("content_type=")
    print(request_content_type)
    
    f = io.BytesIO(request_body)
    with tempfile.NamedTemporaryFile(delete=False) as tfile:
        tfile.write(f.read())
        filename = tfile.name
    
    video_frames = video2frame(filename, frame_width, frame_height, interval)  
    return video_frames


def predict_fn(video_frames, model):
    transform = transforms.Compose([
        transforms.Lambda(lambda video_frames: torch.stack([transforms.ToTensor()(frame) for frame in video_frames])) # returns        a 4D tensor
    ])
    image_tensors = transform(video_frames).to(device)
    
    with torch.no_grad():
        output = model(image_tensors)
    return output


def output_fn(output_batch, accept='application/json'):
    res = []
    
    print(f'output list length={len(output_batch)}')
    for output in output_batch:
        boxes = output['boxes'].detach().cpu().numpy()
        labels = output['labels'].detach().cpu().numpy()
        scores = output['scores'].detach().cpu().numpy()
        masks = output['masks'].detach().cpu().numpy()
        masks = np.squeeze(masks.transpose(2,3,0,1)) # 4D(batch x 1 height x width) to 3D(height x width x batch)
        
        res.append({
            'boxes': boxes.tolist(),
            'labels': labels.tolist(),
            'scores': scores.tolist(),
            'masks': masks.tolist()     
        })
    
    return json.dumps(res)
