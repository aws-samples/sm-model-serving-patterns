
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
    
# To use new EIA inference API, customer should use attach_eia(model, eia_ordinal_number)
VERSIONS_USE_NEW_API = ["1.5.1"]

def model_fn(model_dir):
    try:
        loaded_model = torch.jit.load("model.pth", map_location=torch.device("cpu"))
        if torch.__version__ in VERSIONS_USE_NEW_API:
            import torcheia

            loaded_model = loaded_model.eval()
            loaded_model = torcheia.jit.attach_eia(loaded_model, 0)
        return loaded_model
    except Exception as e:
        logger.exception(f"Exception in model fn {e}")
        return None

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
    # With EI, client instance should be CPU for cost-efficiency. Subgraphs with unsupported arguments run locally. Server runs with CUDA
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    batchified = batchified.to(device)
    
    # Please make sure model is loaded to cpu and has been eval(), in this example, we have done this step in model_fn()
    with torch.no_grad():
        if torch.__version__ in VERSIONS_USE_NEW_API:
            # Please make sure torcheia has been imported
            import torcheia

            # We need to set the profiling executor for EIA
            torch._C._jit_set_profiling_executor(False)
            with torch.jit.optimized_execution(True):
                result =  model.forward(batchified)
        # Set the target device to the accelerator ordinal
        else:
            with torch.jit.optimized_execution(True, {"target_device": "eia:0"}):
                result = model(batchified)

    # Softmax (assumes batch size 1)
    result = np.squeeze(result.detach().cpu().numpy())
    result_exp = np.exp(result - np.max(result))
    result = result_exp / np.sum(result_exp)

    response_body = json.dumps(result.tolist())

    return response_body, response_content_type
