import torch
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, AutoModelForQuestionAnswering

def load_s2s_model(model_name):
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
    return model, tokenizer

def load_qa_model(model_name):
    model = AutoModelForQuestionAnswering.from_pretrained(model_name)
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    return model, tokenizer

def grammar_correct(sentence, model, tokenizer, prefix='gec:'):
    sentence = prefix + sentence 
    input_ids = tokenizer.encode(sentence, return_tensors="pt")
    preds = model.generate(
        input_ids,
        max_length=128,
        do_sample=True,
        top_k=50,
        top_p=0.95,
        early_stopping=True,
        num_return_sequences=1
        )

    corrected = set()
    for pred in preds:
        corrected.add(tokenizer.decode(pred, skip_special_tokens=True).strip())
    return corrected

def qa_inference(model, tokenizer, question, context):
    """
    Make a Question-Answering inference step by passing model, tokenizer and question and context.

    question: str
    context: str
    """
    encoding = tokenizer.encode_plus(question, context)
    input_ids, attention_mask = encoding["input_ids"], encoding["attention_mask"]
    outputs = model(torch.tensor([input_ids]), attention_mask=torch.tensor([attention_mask]))
    ans_tokens = input_ids[torch.argmax(outputs.start_logits) : torch.argmax(outputs.end_logits)+1]
    answer_tokens = tokenizer.convert_ids_to_tokens(ans_tokens , skip_special_tokens=True)
    answer_tokens_to_string = tokenizer.convert_tokens_to_string(answer_tokens)
    return answer_tokens_to_string
