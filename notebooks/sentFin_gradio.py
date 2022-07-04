import re

import nltk
nltk.download('punkt')
from sumy.parsers.html import HtmlParser
from sumy.nlp.tokenizers import Tokenizer
from sumy.summarizers.lsa import LsaSummarizer as Summarizer
from sumy.nlp.stemmers import Stemmer
from sumy.utils import get_stop_words

from transformers import BartForConditionalGeneration, BartTokenizerFast, BertForSequenceClassification, BertTokenizerFast, AutoTokenizer, AutoModelForTokenClassification
from transformers import pipeline 

import gradio as gr

import spacy
nlp = spacy.load('en_core_web_sm')
nlp.add_pipe('sentencizer')

def split_in_sentences(text):
    doc = nlp(text)
    return [str(sent).strip() for sent in doc.sents]

checkpoint_summary = "facebook/bart-large-cnn"
checkpoint_sentiment = "yiyanghkust/finbert-tone"

### Abstractive Summary
tokenizer = BartTokenizerFast.from_pretrained(checkpoint_summary)
model = BartForConditionalGeneration.from_pretrained(checkpoint_summary)

### Sentiment Analysis
finbert = BertForSequenceClassification.from_pretrained(checkpoint_sentiment, num_labels=3)
tokenizer_sentiment = BertTokenizerFast.from_pretrained(checkpoint_sentiment)
nlp_sentiment = pipeline("sentiment-analysis", model=finbert, tokenizer=tokenizer_sentiment)


# ### NER
# tokenizer_nem = AutoTokenizer.from_pretrained("dslim/bert-base-NER")
# model_nem = AutoModelForTokenClassification.from_pretrained("dslim/bert-base-NER")
# nlp_nem = pipeline("ner", model=model_nem, tokenizer=tokenizer_nem)

model_dict = {
  'model': model, 
  'max_length': 512,
  'min_length': 120
}

tokenizer_dict = {
  'tokenizer': tokenizer, 
  'max_length': 1024
}

LANGUAGE = "english"
SENTENCE_COUNT = 15

def get_extractive_summary_from_url(url: str) -> str:
  parser = HtmlParser.from_url(url, Tokenizer(LANGUAGE))
  stemmer = Stemmer(LANGUAGE)
  summarizer = Summarizer(stemmer)
  summarizer.stop_words = get_stop_words(LANGUAGE)
  extractive_summary_from_url = ' '.join([sent._text for sent in summarizer(parser.document, SENTENCE_COUNT)])
  return extractive_summary_from_url


def get_summary(text_content):
  # text_content = get_extractive_summary(text_content, EXTRACTED_ARTICLE_SENTENCES_LEN)
  tokenizer = tokenizer_dict['tokenizer']
  model = model_dict['model']

  inputs = tokenizer(text_content, max_length=tokenizer_dict['max_length'], truncation=True, return_tensors="pt")
  outputs = model.generate(
      inputs["input_ids"], max_length=model_dict['max_length'], min_length=model_dict['min_length'], 
  )

  summarized_text = tokenizer.decode(outputs[0])
  match = re.search(r"<s>(.*)</s>", summarized_text)
  if match is not None: summarized_text = match.group(1)

  return summarized_text.replace('<s>', '').replace('</s>', '') 

def get_sentiment(text):
    preds = nlp_sentiment(text)
    return preds[0]['label']

##Company Extraction    
def fin_ner(text):
    api = gr.Interface.load("dslim/bert-base-NER", src='models')
    replaced_spans = api(text)
    return replaced_spans    

demo = gr.Blocks()

with demo:
    gr.Markdown("## Financial Analyst AI")
    gr.Markdown("This project provides an AI financial tool for summarizing financial articles and computing sentiment analysis.")
    with gr.Row():
        with gr.Column():
            url = gr.inputs.Textbox(default="https://news.iobanker.com/2022/07/01/bitcoin-will-see-long-bear-market-says-trader-with-btc-price-stuck-at-19k/")
            with gr.Row():
                b1 = gr.Button("Extractive Summarize Text")
                extractive_summary = gr.Textbox()
                b1.click(get_extractive_summary_from_url, inputs=url, outputs=extractive_summary)
            with gr.Row():
                b2 = gr.Button("Abstractive Summarize Text")
                abstractive_summary = gr.Textbox()
                b2.click(get_summary, inputs=extractive_summary, outputs=abstractive_summary)
            with gr.Row():
                b3 = gr.Button("Classify Financial Tone")
                label = gr.Label()
                b3.click(get_sentiment, inputs=abstractive_summary, outputs=label) 
            with gr.Row():
                b4 = gr.Button("Identify Companies & Locations")
                replaced_spans = gr.HighlightedText()
                b4.click(fin_ner, inputs=abstractive_summary, outputs=replaced_spans)
    
demo.launch(share=True)