from transformers import pipeline

summarizer = pipeline("summarization", model="facebook/bart-large-cnn")

text = """
Artificial intelligence is transforming many industries including healthcare,
finance, and transportation. Machine learning models can analyze large amounts
of data and discover patterns that humans may miss. In natural language
processing, AI systems can understand and generate human language, enabling
applications such as chatbots, translation systems, and text summarization.
Text summarization is particularly useful because it allows large documents
to be condensed into shorter summaries while preserving the main ideas.
"""

summary = summarizer(text, max_length=60, min_length=20, do_sample=False)
print(summary[0]["summary_text"])