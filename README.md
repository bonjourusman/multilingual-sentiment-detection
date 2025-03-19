# multilingual-sentiment-detection

# Finetuning an LLM for multilingual sentiment detection.

In this project, I have fine-tuned a pre-trained LLM (ChatGPT-2 by OpenAI) to perform sentiment detection across a variety of languages.

**Dataset:**
`multilingual-amazon-review-sentiment-processed`

**Source:**
https://huggingface.co/datasets/hungnm/multilingual-amazon-review-sentiment-processed

**Dataset Attributes used:**
`text` : customer reviews in English, Japanese, German, French, Spanish, and Chinese
`label` : binary sentiment used as ground truth

## Model Training

I have used the transformer model named `openai-community/gpt2` with a classification head to perform binary classification of sentiments, with `0` as NEGATIVE and `1` as POSITIVE.

While GPT-3 and GPT-4 are more powerful and pretrained to be more effective on sentiment detection tasks, GPT-2 is not. Moreover, GPT-3 and GPT-4 come with a significant cost overhead. Therefore, by fine-tuning GPT-2, we can use it for sentiment detection at little to no cost and achieve very good performance.

The dataset used for finetuning is preprocessed and split into training, validation, and test sets having 1.33M, 47.5k, and 47.5k records respectively.

Because of computational constraints, only a subset of the downloaded data has been used for finetuning. Based on the accuracy metrics observed during training, I suggest using GPUs to process a larger corpus of data for better model performance. For instance, when finetuning with only 1,000 examples on a CPU, I observed accuracy scores above 0.75 in the later epochs.

To re-train, update the `load_and_prepare_data()` function in `helpers.py` and then run `train.py` to generate a new fine-tuned model using the training dataset and test its performance using the validation or test datasets.

Also, when running this on cloud, data download isn't required. One can simply load the Dataset from the source and commence model training.

## API

The repo also contains code for a local API created to call the finetuned model. Using the `analyze` endpoint, the API returns the sentiment, i.e. POSITIVE or NEGATIVE, for a given input text.

The API and model can be packaged into a docker container and easily deployed to production on a cloud platform of choice, e.g. as a web service.

## A word about sentiments and LLMs

Establishing 'ground truth' can be a challenge for sentiment analysis tasks. Sarcasm, cultural differences, and language modalities can make it difficult to establish whether the sentiment is positive or negative, especially to what extent, e.g. on a scale of 1 to 10.

Buscemi & Proverbio<sup>[1]</sup> observed that the same input in different languages (having translated using native speakers) had significant variance in sentiment scores when evaluated by both LLMs and humans.

This arises a few pertinent questions:

_Should an LLM behave similarly to humans and account for such inherent variances that exist among languages and their speakers?_

_Given the cultural and linguistic biases, should we 'adjust' or normalize the sentiment score to establish an unbiased global ground truth?_

In the case of LLMs, this variance is observed, not only across languages, but also among the various LLMs. Some models tend to have consistently positive scores while others show a more balanced distribution.

Nevertheless, careful consideration is required when applying these findings to real-world applications especially if there are agentic actions triggered (AI or humans) based on the sentiment analysis outputs. For example, if Spanish users have a tendency towards lower sentiment scores, a company shouldn't be compelled to make unnecessary promotional offers or take corrective actions to boost their sentiment scores, when comparing performance across other user groups.

**References:**

[1] Buscemi, A., & Proverbio, D. (2024). Chatgpt vs gemini vs llama on multilingual sentiment analysis. _arXiv preprint arXiv:2402.01715_.
