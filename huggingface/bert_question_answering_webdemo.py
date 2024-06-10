import streamlit as st
from transformers import AutoTokenizer, AutoModelForQuestionAnswering
import torch

# Load the model and tokenizer from Hugging Face
@st.cache_resource
def load_model():
    model_name = "../huggingface_models/distilbert-base-cased-distilled-squad"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForQuestionAnswering.from_pretrained(model_name)
    return tokenizer, model

tokenizer, model = load_model()

st.title("Question Answering with Hugging Face")

question = st.text_input("Enter your question:")
context = st.text_area("Enter the context:", "My name is Sylvain and I work at Hugging Face in Brooklyn")

if st.button("Get Answer"):
    if question and context:
        inputs = tokenizer(question, context, return_tensors='pt')
        with torch.no_grad():
            outputs = model(**inputs)
        answer_start = torch.argmax(outputs.start_logits)
        answer_end = torch.argmax(outputs.end_logits) + 1
        answer = tokenizer.convert_tokens_to_string(tokenizer.convert_ids_to_tokens(inputs.input_ids[0][answer_start:answer_end]))
        st.write("Answer:", answer)
    else:
        st.write("Please enter both a question and a context.")
