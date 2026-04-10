from transformers import pipeline  # import NLP pipeline

# create summarization pipeline
summarizer = pipeline("summarization")

text = """
NLP, or Natural Language Processing, is a field of artificial intelligence that focuses on 
enabling computers to understand, interpret, and generate human language. It combines techniques from 
computer science, linguistics, and machine learning to process text and speech in meaningful ways. 
NLP allows machines to perform tasks such as translation, sentiment analysis, summarization, 
and question answering. Modern NLP systems learn patterns from large datasets using statistical and 
deep learning models. A key challenge in NLP is dealing with ambiguity, context, and 
the variability of human language. Despite these difficulties, NLP has become a core technology behind 
many everyday digital tools and continues to advance rapidly.
"""

# summarization with sampling
summary1 = summarizer(text, max_length = 100, min_length = 30, do_sample = True)
print(summary1[0]["summary_text"])

# deterministic summarization output
summary2 = summarizer(text, max_length = 100, min_length = 30, do_sample = False)
print(summary2[0]["summary_text"])
