import json
import sys
import logging
import torch
from torch import nn
from transformers import PreTrainedTokenizerFast, BartForConditionalGeneration

logging.basicConfig(
    level=logging.INFO, 
    format='[{%(filename)s:%(lineno)d} %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(filename='tmp.log'),
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger(__name__)

tokenizer = PreTrainedTokenizerFast.from_pretrained("ainize/kobart-news")
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def model_fn(model_path=None):
    model = BartForConditionalGeneration.from_pretrained("ainize/kobart-news")
    model.to(device)
    return model


def transform_fn(model, input_data, content_type="application/jsonlines", accept="application/jsonlines"): 
    data_str = input_data.decode("utf-8")
    jsonlines = data_str.split("\n")

    predicted = []
    
    for jsonline in jsonlines:
        text = json.loads(jsonline)["text"][0]
        logger.info("input text: {}".format(text))  
        
        input_ids = tokenizer.encode(text, return_tensors="pt")
        input_ids = input_ids.to(device)        
        # Generate Summary Text Ids
        summary_text_ids = model.generate(
            input_ids=input_ids,
            bos_token_id=model.config.bos_token_id,
            eos_token_id=model.config.eos_token_id,
            length_penalty=2.0,
            max_length=512,
            min_length=32,
            num_beams=4,
        )        
        
        # Decoding Text
        summary_outputs = tokenizer.decode(summary_text_ids[0], skip_special_tokens=True)
        logger.info("summary_outputs: {}".format(summary_outputs))        
        
        prediction_dict = {}
        prediction_dict["summary"] = summary_outputs

        jsonline = json.dumps(prediction_dict)
        predicted.append(jsonline)

    predicted_jsonlines = "\n".join(predicted)
    return predicted_jsonlines