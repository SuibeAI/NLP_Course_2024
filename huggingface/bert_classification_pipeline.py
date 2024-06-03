from transformers import AutoTokenizer, AutoModelForSequenceClassification

# Specify the path to your local model folder
local_model_path = "../huggingface_models/bert-base-cased"

# Load the tokenizer from the local model path
tokenizer = AutoTokenizer.from_pretrained(local_model_path)

# Define the sentence to be tokenized
sentence = "Alan likes transformer"

# Tokenize the sentence
tokens = tokenizer.tokenize(sentence)
print("Tokens:", tokens)

# Add special tokens for BERT input
tokens = ['[CLS]'] + tokens + ['[SEP]']

# Convert tokens to their respective IDs
token_ids = tokenizer.convert_tokens_to_ids(tokens)
print("Token IDs:", token_ids)

# Load the model from the local model path
model = AutoModelForSequenceClassification.from_pretrained(local_model_path)

# Prepare the inputs for the model
inputs = tokenizer(sentence, return_tensors='pt')
print("Model Inputs:", inputs)



# Get the embeddings from the embedding layer
embeddings = model.bert.embeddings(inputs['input_ids'])
print("Embeddings:", embeddings.shape)

# Pass the embeddings through the model to get the outputs
outputs = model.bert(**inputs)
print("Backbone Outputs:", outputs.last_hidden_state.shape)

# Pass the outputs through the classification head (using the [CLS] token's hidden state)
logits = model.classifier(outputs.last_hidden_state[:, 0])
print("model.classifier", model.classifier)
print("Logits:", logits)
