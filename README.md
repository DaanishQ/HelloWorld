---

# **Pre-Training Transformers Models in PyTorch with Hugging Face Transformers**

Train 67 transformers models on your specific dataset.

---

**Note:** I've structured this tutorial notebook similarly to my other ones intentionally, aiming to maintain consistency for readers.

---

This notebook serves the purpose of training transformers models using [Huggingface](https://huggingface.co/transformers/) with your custom dataset.

With the AutoClasses feature, we can utilize the code across a wide range of transformers models!

This notebook is designed to:

- Utilize a pretrained transformers model and fine-tune it on your custom dataset.
- Train a transformer model from scratch on a custom dataset. This necessitates a pre-trained tokenizer. If one isn't provided, this notebook defaults to using the pretrained tokenizer.

This notebook draws heavy inspiration from the Hugging Face script used for training language models: [transformers/tree/master/examples/language-modeling](https://github.com/huggingface/transformers/tree/master/examples/language-modeling). I've essentially adapted that script to function smoothly within a notebook, with numerous comments for clarity.

---

## **What Prerequisites Do I Need for This Notebook?**

Given that I'm using PyTorch for fine-tuning our transformers models, any familiarity with PyTorch is beneficial.

A basic understanding of the [transformers](https://github.com/huggingface/transformers) library would also be advantageous.

In this notebook, I'm using raw text data to train / fine-tune transformers models (when using a pretrained model, I refer to this as "extended pretraining" since I 'continue' the original training of the model on a custom dataset). Since we're not engaged in classification, labeled data isn't necessary. The Transformers library manages the text files similarly to the original implementation of each model.

---

## **How Can I Utilize This Notebook?**

As with any project, I've designed this notebook with reusability in mind. It retrieves the custom dataset from `.txt` files. Since the dataset isn't contained within a single `.txt` file, I've created a custom function, `movie_reviews_to_file`, which reads the dataset and generates the `text` file. The method of loading the `.txt` files can be readily adapted for any other dataset.

The only modifications required to use your own dataset will be in the paths provided to the training `.txt` file and evaluation `.txt` file.

All modifiable parameters are located under the **Parameters Setup** section. Each parameter is thoroughly commented and structured for ease of understanding.

---

## **Which transformers Models Are Compatible with This Notebook?**

While many individuals may primarily use it for BERT, it's essential to know which other transformer model architectures are compatible with this code. Since the notebook's name is **Training Transformers**, it should function with more than one type of transformer.

I've tested this notebook across all the pretrained models available on Hugging Face Transformer. This way, you'll know in advance if the model you plan to use works seamlessly with this code, requiring no modifications.

You can find the list of pretrained transformers models that work with this notebook under `Training Transformers with Pytorch`. A total of **67 models succeeded** 😄, while 39 models encountered failures 😢 when used with this notebook. *Remember, these are pretrained models fine-tuned on a custom dataset.*

---

## **Dataset**

This notebook focuses on pretraining transformers on a custom dataset. Specifically, I'll utilize the well-known movies reviews positive-negative labeled [Large Movie Review Dataset](https://ai.stanford.edu/~amaas/data/sentiment/).

Here's the description provided on the Stanford website:

*This dataset is for binary sentiment classification, containing substantially more data than previous benchmark datasets. We provide a set of 25,000 highly polar movie reviews for training and 25,000 for testing. There is additional unlabeled data available as well. Raw text and already processed bag of words formats are provided. Refer to the README file contained in the release for more details.*

**Why this dataset?** I believe it's an easily understandable and usable dataset for classification. Sentiment data is always engaging to work with.

---

## **Coding**

Now, let's dive into some coding! We'll walk through each code cell in the notebook, describing its purpose, the code within it, and when relevant, showcasing the output.

I've formatted this guide to be easy to follow along, should you decide to run each code cell in your own Python notebook.

When I learn from a tutorial, I always aim to replicate the results. Having the code alongside the explanations makes it easier to follow along.



# Fine-tune Transformers in PyTorch Using Hugging Face Transformers

> Fine-tune transformers models for classification tasks

## Overview

This guide demonstrates how to fine-tune a pre-trained transformers model for classification tasks using PyTorch and the Hugging Face Transformers library. The tutorial focuses on code implementation and adaptability to various needs.

We leverage the [AutoClasses](https://huggingface.co/transformers/model_doc/auto.html) feature from the [transformers](https://github.com/huggingface/transformers) library by [Hugging Face](https://huggingface.co/), allowing automatic detection of model configuration, tokenizer, and architecture based solely on the model's name.

## Prerequisites

This notebook is accessible with basic Python coding knowledge. Familiarity with PyTorch and the transformers library is beneficial.

## How to Utilize This Notebook

This notebook is designed for reusability. Loading the dataset into the PyTorch class follows a standard procedure and can be adapted for other datasets.

To use your own dataset, modify the dataset loading section within the PyTorch **Dataset** class. The **DataLoader** returns batch inputs in a dictionary format suitable for direct feeding into the model.

Basic parameters such as `epochs`, `batch_size`, `max_length`, `model_name_or_path`, and `labels_ids` are defined for ease of configuration.

## Supported Transformers Models

This notebook accommodates various transformer models beyond BERT for classification tasks. Compatibility was verified across a wide range of models available on Hugging Face Transformer.

Out of the tested models, **73** were successfully integrated, while **33** exhibited compatibility issues.

## Dataset

The notebook focuses on fine-tuning transformers for a binary classification task using the Large Movie Review Dataset for sentiment analysis. This dataset comprises 25,000 polar movie reviews for training and testing.

According to the Stanford website:

*This dataset offers substantial data for binary sentiment classification, surpassing previous benchmark datasets. It includes 25,000 highly polar movie reviews for training and an equivalent set for testing. Additional unlabeled data is also provided.*

**Rationale**: The Large Movie Review Dataset is selected for its simplicity, familiarity, and suitability for classification tasks. Sentiment analysis adds to its appeal.
