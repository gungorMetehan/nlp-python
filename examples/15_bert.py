# installs required libraries for NLP models and tensor computation
pip install transformers torch

# imports PyTorch for running the model and handling tensors
import torch

# imports BERT tokenizer (text to tokens) and masked language model
from transformers import BertTokenizer, BertForMaskedLM

# defines which pretrained BERT model will be used
m_name = "bert-base-uncased"

# loads the tokenizer corresponding to the selected BERT model
tokenizer = BertTokenizer.from_pretrained(m_name)

# loads the pretrained BERT model for masked language modeling
model = BertForMaskedLM.from_pretrained(m_name)

# example sentence containing a [MASK] token for prediction
text = "I am going to [MASK] next year."

# converts the input text into PyTorch tensors (token IDs and attention mask)
inputs = tokenizer(text, return_tensors = "pt")

# runs the model without tracking gradients (inference mode)
with torch.no_grad():
    # feeds the tokenized input into the BERT model
    outputs = model(**inputs)

# finds the position index of the [MASK] token in the input sequence
mask_index = torch.where(inputs["input_ids"] == tokenizer.mask_token_id)[1]

# retrieves the raw prediction scores (logits) from the model output
logits = outputs.logits

# extracts logits corresponding only to the [MASK] position
mask_logits = logits[0, mask_index, :]

# selects the top 5 most likely tokens for the masked position
top_tokens = torch.topk(mask_logits, k = 5, dim = 1).indices[0]

# prints a header for the predicted tokens
print("Predicted tokens for [MASK]:")

# decodes and prints each predicted token as readable text
for token in top_tokens:
    print(tokenizer.decode([token.item()]))
