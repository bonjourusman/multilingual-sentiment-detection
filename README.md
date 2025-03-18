# multilingual-sentiment-analysis

**Finetuning an LLM for multilingual sentiment analysis.**

In this project, I have fine-tuned a pre-trained LLM (ChatGPT-2) to perform sentiment analysis across a variety of languages.

**Dataset:**
`multilingual-amazon-review-sentiment-processed`

**Source:**
https://huggingface.co/datasets/hungnm/multilingual-amazon-review-sentiment-processed

**Dataset Attributes used:**
`text` : customer reviews in English, Japanese, German, French, Spanish, and Chinese
`label` : binary sentiment used as ground truth

Because of computational constraints, only a subset of the available data has been used for finetuning. Based on the accuracy metrics observed during training, I suggest using GPUs to process a larger corpus of data for better model performance.

## A word about sentiment analysis using LLMs

Establishing 'ground truth' can be a challenge for sentiment analysis tasks. Sarcasm, cultural differences, and language modalities can make it difficult to establish whether the sentiment is positive or negative, especially to what extent, e.g. on a scale of 1 to 10.

Buscemi & Proverbio^{[1]}^ observed that the same input in different languages (having translated using native speakers) had significant variance in sentiment scores when evaluated by both LLMs and humans.

This arises a few pertinent questions:

_Should an LLM behave similarly to humans and account for such inherent variances that exist among languages and their speakers?_

_Given the cultural and linguistic biases, should we 'adjust' or normalize the sentiment score to establish an unbiased global ground truth?_

In the case of LLMs, this variance is observed, not only across languages, but also among the various LLMs. Some models tend to have consistently positive scores while others show a more balanced distribution.

Either way, careful considerations are required when applying these findings to real-world applications especially if there are agentic actions triggered (AI or humans) based on the sentiment analysis outputs. For example, if Spanish users have a tendency towards lower sentiment scores, a company shouldn't be compelled to make unnecessary promotional offers or take corrective actions to boost their sentiment scores, when comparing performance across user groups.

**References:**

[1] Buscemi, A., & Proverbio, D. (2024). Chatgpt vs gemini vs llama on multilingual sentiment analysis. _arXiv preprint arXiv:2402.01715_.
