import torch
import intel_extension_for_pytorch as ipex 
# ipex.enable_auto_mixed_precision(mixed_dtype = torch.bfloat16) 

from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
from transformers import logging
logging.set_verbosity_error()

from fastapi import FastAPI

from typing import List, Optional
from pydantic import BaseModel

GRAMMAR_CHECKPOINT = "vennify/t5-base-grammar-correction"
TINYROBERTA_CHECKPOINT = "deepset/tinyroberta-squad2"

class InferenceModel:
    def __init__(self, checkpoint: str, quantize: bool = False):
        torch.set_num_threads(1)
        torch.set_grad_enabled(False)

        self.tokenizer = AutoTokenizer.from_pretrained(checkpoint)
        # model = AutoModelForQuestionAnswering.from_pretrained(checkpoint)
        model = AutoModelForSeq2SeqLM.from_pretrained(checkpoint)
        model = model.to(memory_format=torch.channels_last)
        if quantize: model = torch.quantization.quantize_dynamic(model, {torch.nn.Linear}, dtype=torch.qint8)
        model = ipex.optimize(model)
        #model = torch.jit.script(model)

        self.model = model

    def predict(self, sentence: str):
        text = ''
        with torch.no_grad():
            input_ids = self.tokenizer.encode(sentence, truncation=True, padding=True, return_tensors='pt')
            preds = self.model.generate(
                input_ids,
                max_length=128,
                do_sample=True,
                top_k=50,
                top_p=0.95,
                early_stopping=True,
                num_return_sequences=1
            )
            for pred in preds:
                text += self.tokenizer.decode(pred, skip_special_tokens=True)
        return text
            
    # def predict(self, context: str, question: str):
    #     with torch.no_grad():
    #         encoding = self.tokenizer.encode_plus(question, context)
    #         input_ids, attention_mask = encoding["input_ids"], encoding["attention_mask"]
    #         outputs = model(torch.tensor([input_ids]), attention_mask=torch.tensor([attention_mask]))
    #         ans_tokens = input_ids[torch.argmax(outputs.start_logits) : torch.argmax(outputs.end_logits)+1]
    #         answer_tokens = self.tokenizer.convert_ids_to_tokens(ans_tokens , skip_special_tokens=True)
    #         answer_tokens_to_string = self.tokenizer.convert_tokens_to_string(answer_tokens)
    #     return answer_tokens_to_string

class SimpleMessage(BaseModel):
    sentence: str
    #context: Optional[str] = 'test'
    #question: Optional[str] = 'test'

# model = InferenceModel(TINYROBERTA_CHECKPOINT)
model = InferenceModel(GRAMMAR_CHECKPOINT)

app = FastAPI()

@app.get("/")
def run_prediction():
    prediction = model.predict("He are moving here!")
    return {'prediction': prediction}

@app.post('/prediction')
async def run_prediction(message: SimpleMessage):
    prediction = model.predict(message.sentence)
    return {"prediction": prediction}

@app.get("/ping")
def ping_check():
    return {"ping": "pong"}


