import os
import json
import torch
from torch import nn
from transformers import AutoTokenizer, AutoModelForSequenceClassification, AutoConfig
os.environ["TOKENIZERS_PARALLELISM"] = "false"

JSON_CONTENT_TYPE = 'application/json'

max_seq_length = 128
classes = ['not paraphrase', 'paraphrase']
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model_id = 'bert-base-cased-finetuned-mrpc'

def model_fn(model_dir):

    tokenizer_init = AutoTokenizer.from_pretrained(model_id)
    model = AutoModelForSequenceClassification.from_pretrained(model_id).eval().to(device)
    
    return (model, tokenizer_init)


def input_fn(serialized_input_data, content_type=JSON_CONTENT_TYPE):
    if content_type == JSON_CONTENT_TYPE:
        input_data = json.loads(serialized_input_data)
        return input_data
    else:
        raise Exception('Requested unsupported ContentType in Accept: ' + content_type)
        return
    

def predict_fn(input_data, models):

    model, tokenizer = models
    sequence_0 = input_data[0] 
    sequence_1 = input_data[1]

    paraphrase = tokenizer.encode_plus(
        sequence_0,
        sequence_1,
        max_length=max_seq_length,
        padding='max_length',
        truncation=True,
        return_tensors='pt'
    ).to(device)

    
    # Convert example inputs to a format that is compatible with TorchScript tracing
    example_inputs = paraphrase['input_ids'], paraphrase['attention_mask'], paraphrase['token_type_ids']

    with torch.no_grad():
        logits = model(*example_inputs)[0]

    softmax_fn = nn.Softmax(dim=1)
    softmax_output = softmax_fn(logits)[0]
    pred_idx = softmax_output.argmax().item()
    pred_class = classes[pred_idx]
    score = softmax_output[pred_idx].item()

    out_str = f'pred_idx={pred_idx}, pred_class={pred_class}, prob={score:.5f}'
    
    return out_str


def output_fn(prediction_output, accept=JSON_CONTENT_TYPE):
    if accept == JSON_CONTENT_TYPE:
        return json.dumps(prediction_output), accept
    
    raise Exception('Requested unsupported ContentType in Accept: ' + accept)