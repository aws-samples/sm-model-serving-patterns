
import io
import json
import logging
import os
import pickle

import numpy as np
import torch
import torchvision.transforms as transforms
from PIL import Image  # Training container doesn't have this package

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)
    
    
def transform_fn(model, payload, request_content_type='application/octet-stream', 
                 response_content_type='application/json'):

    logger.info('Invoking user-defined transform function')

    if request_content_type != 'application/octet-stream':
        raise RuntimeError(
            'Content type must be application/octet-stream. Provided: {0}'.format(request_content_type))

    # preprocess
    decoded = Image.open(io.BytesIO(payload))
    preprocess = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[
                0.485, 0.456, 0.406], std=[
                0.229, 0.224, 0.225]),
    ])
    normalized = preprocess(decoded)
    batchified = normalized.unsqueeze(0)

    # predict
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    batchified = batchified.to(device)
    result = model.forward(batchified)

    # Softmax (assumes batch size 1)
    result = np.squeeze(result.detach().cpu().numpy())
    result_exp = np.exp(result - np.max(result))
    result = result_exp / np.sum(result_exp)

    response_body = json.dumps(result.tolist())

    return response_body, response_content_type
