from transformers import pipeline

summarizer = pipeline(
    "summarization",
    model="./fine_tuned_bart",
    tokenizer="./fine_tuned_bart"
)

text = """
Artificial intelligence is helping automate repetitive tasks
and improve productivity across multiple industries.
"""

summary = summarizer(text, max_length=50, min_length=15, do_sample=False)
print(summary[0]["summary_text"])