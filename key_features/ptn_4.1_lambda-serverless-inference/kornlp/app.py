import json
import sys
import logging
import torch
from torch import nn
from transformers import ElectraConfig
from transformers import ElectraModel, AutoTokenizer, ElectraTokenizer, ElectraForSequenceClassification
logger = logging.getLogger(__name__)

model_path = 'model-nsmc'
max_seq_length = 128
classes = ['Neg', 'Pos']
tokenizer = AutoTokenizer.from_pretrained(f"{model_path}/vocab")
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Huggingface pre-trained model: 'monologg/koelectra-small-v3-discriminator'
def model_fn(model_path):
    config = ElectraConfig.from_json_file(f'{model_path}/config.json')
    model = ElectraForSequenceClassification.from_pretrained(f'{model_path}/model.pth', config=config)
    model.to(device)
    return model

model = model_fn(model_path)


def input_fn(input_data, content_type="application/json"): 

    text = input_data["text"]
    logger.info("input text: {}".format(text))          
    encode_plus_token = tokenizer.encode_plus(
        text,
        max_length=max_seq_length,
        add_special_tokens=True,
        return_token_type_ids=False,
        padding="max_length",
        return_attention_mask=True,
        return_tensors="pt",
        truncation=True,
    )
        
    return encode_plus_token


def predict_fn(data, model):

    data = data.to(device)
    output = model(**data)

    softmax_fn = nn.Softmax(dim=1)
    softmax_output = softmax_fn(output[0])
    _, prediction = torch.max(softmax_output, dim=1)

    predicted_class_idx = prediction.item()
    predicted_class = classes[predicted_class_idx]
    score = softmax_output[0][predicted_class_idx]
    logger.info("predicted_class: {}".format(predicted_class))

    prediction_dict = {}
    prediction_dict["predicted_label"] = predicted_class
    prediction_dict['score'] = score.cpu().detach().numpy().tolist()

    return prediction_dict

def output_fn(outputs, accept="application/json"):
    return {
        'statusCode': 200,
        'body': json.dumps(outputs),
        'headers': {
            'Content-Type': accept,
            'Access-Control-Allow-Origin': '*'
        }        
    }


def lambda_handler(event, context):
    print('lambda')
    print(event.get('body'))
    body = json.loads(event.get('body'))

    features = input_fn(body)
    preds = predict_fn(features, model)
    outputs = output_fn(preds)
    return outputs