import torch  # PyTorch tensor operations
from transformers import AutoTokenizer, AutoModelForQuestionAnswering  # Hugging Face QA utilities

# pretrained QA model identifier
model_name = "deepset/roberta-base-squad2"

# load corresponding tokenizer
tokenizer = AutoTokenizer.from_pretrained(model_name)

# load question answering model
model = AutoModelForQuestionAnswering.from_pretrained(model_name)

# set model to inference mode
model.eval()

# predict extractive answer span (an user-defined function)
def pred_answer(context, question, max_length = 512, null_threshold = 0.0):
    
    # tokenize question and context
    encoding = tokenizer(
        question,
        context,
        return_tensors = "pt",
        truncation = True,
        max_length = max_length
    )
    
    # extract model input tensors
    input_ids = encoding["input_ids"]
    attention_mask = encoding["attention_mask"]

    # disable gradient computation
    with torch.no_grad():
        outputs = model(
            input_ids = input_ids,
            attention_mask = attention_mask
        )

    # retrieve start and end logits
    start_logits = outputs.start_logits
    end_logits = outputs.end_logits

    # compute joint start-end scores
    seq_len = start_logits.size(1)
    joint_scores = start_logits.unsqueeze(2) + end_logits.unsqueeze(1)

    # mask invalid span positions
    mask = torch.triu(torch.ones(seq_len, seq_len), diagonal = 0).bool()
    joint_scores[0][~mask] = -1e9

    # select best scoring span
    best_start, best_end = torch.argmax(joint_scores[0]).div(seq_len, rounding_mode = "floor"), \
                           torch.argmax(joint_scores[0]) % seq_len

    # compute no-answer confidence score
    null_score = start_logits[0, 0] + end_logits[0, 0]
    best_score = joint_scores[0, best_start, best_end]

    if best_score < null_score + null_threshold:
        return "No answer found."

    # decode predicted answer tokens
    answer_ids = input_ids[0][best_start: best_end + 1]
    answer = tokenizer.decode(answer_ids, skip_special_tokens = True)

    return answer

# example question and context
question = "Which employee becomes the regional manager of the Scranton branch near the end of the series?"

context = """
The Office is an American mockumentary-style television series that aired on NBC from 2005 to 2013.
It depicts the everyday work lives of employees at the Scranton, Pennsylvania branch of the fictional Dunder Mifflin Paper Company.

For most of the series, the Scranton branch is managed by Michael Scott, whose management style is often inappropriate but well-meaning.
After Michael leaves the company to move to Colorado with Holly Flax, the branch goes through a period of leadership instability.

During this time, several employees either temporarily assume managerial responsibilities or are considered for the role.
Dwight Schrute briefly becomes acting manager but is removed due to his extreme behavior.
Andy Bernard later becomes the official regional manager, though his tenure is marked by frequent absences and declining performance.

Near the end of the series, corporate leadership makes a final decision regarding long-term management of the Scranton branch.
Ultimately, Dwight Schrute is promoted and becomes the permanent regional manager, fulfilling his long-standing ambition.
"""

answer = pred_answer(context, question)

print("Question:", question)
print("Answer:", answer)