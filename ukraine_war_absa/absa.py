"""Slightly adapted version of code for my Master's thesis:
https://github.com/felixrech/PC-AI_analysis/blob/main/07_sentiment/sentiment/utils/data.py
"""

import os
import numpy as np
import pandas as pd
import itertools as it
import more_itertools as mit
from functools import partial

import torch
import transformers
from torch.nn.functional import softmax
from transformers import (
    AutoTokenizer as Tokenizer,
    AutoModelForSequenceClassification as Model,
)


os.environ["CUDA_LAUNCH_BLOCKING"] = "1"
DEFAULT_CHUNK_SIZE = 4


def _absa_single(
    text: list[str],
    tokenizer: transformers.DebertaV2TokenizerFast,
    model: transformers.DebertaV2ForSequenceClassification,
) -> np.ndarray:
    """Compute aspect-based sentiment analysis.

    Parameters
    ----------
    texts : list[str]
        List of strings of mentions in the correct format, e.g.
        "[CLS]I love pizza.[SEP]pizza[SEP]".
    tokenizer : transformers.DebertaV2TokenizerFast
        Tokenizer matching the model
    model : transformers.DebertaV2ForSequenceClassification
        ABSA model matching the tokenizer.

    Returns
    -------
    np.ndarray
        [n, 3]-dimensional arrays with "negative", "neutral", and "positive" columns.
    """
    # Tokenize input and copy to GPU
    tokens = tokenizer(text, padding=True, return_tensors="pt")
    tokens = tokens.to("cuda")

    # Predict and softmax to get final answer
    predictions = model(**tokens)
    results_tensor = softmax(predictions.logits, dim=1)
    results = results_tensor.to("cpu").detach().numpy()

    # Release the GPU memory the computations take up
    del tokens, predictions, results_tensor
    torch.cuda.empty_cache()

    return results


def absa(text: list[str], chunk_size: int = DEFAULT_CHUNK_SIZE) -> np.ndarray:
    """Compute aspect-based sentiment analysis and return it as a numpy array.

    Parameters
    ----------
    texts : list[str]
        List of strings of mentions in the correct format, e.g.
        "[CLS]I love pizza.[SEP]pizza[SEP]".
    chunk_size : int, optional
        Chunk size (to adapt to available GPU memory), by default DEFAULT_CHUNK_SIZE.

    Returns
    -------
    np.ndarray
        [n, 3]-dimensional arrays with "negative", "neutral", and "positive" columns.
    """
    # Initialize the model
    model_name = "yangheng/deberta-v3-base-absa-v1.1"
    tokenizer: transformers.DebertaV2TokenizerFast = Tokenizer.from_pretrained(
        model_name
    )  # type: ignore
    model = Model.from_pretrained(model_name).to("cuda")

    # Compute the result in small chunks to fit into GPU memory
    results = np.vstack(
        list(
            it.chain(
                *map(
                    partial(_absa_single, tokenizer=tokenizer, model=model),
                    mit.chunked(text, chunk_size),
                )
            )
        )
    )

    # Release the GPU memory the model takes up
    del model
    torch.cuda.empty_cache()
    return results


def absa_df(texts: list[str], chunk_size: int = DEFAULT_CHUNK_SIZE) -> pd.DataFrame:
    """Compute aspect-based sentiment analysis and return it as a dataframe.

    Parameters
    ----------
    texts : list[str]
        List of strings of mentions in the correct format, e.g.
        "[CLS]I love pizza.[SEP]pizza[SEP]".
    chunk_size : int, optional
        Chunk size (to adapt to available GPU memory), by default DEFAULT_CHUNK_SIZE.

    Returns
    -------
    pd.DataFrame
        Dataframe with "negative", "neutral", and "positive" columns.
    """
    results = absa(texts, chunk_size=chunk_size)

    return pd.DataFrame(results, columns=["negative", "neutral", "positive"])


def add_sentiment_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Add aspect-based sentiment analysis columns to any rows of a dataframe with
    non-null "sentiment_string"

    Parameters
    ----------
    df : pd.DataFrame
        Dataframe with "sentiment_string" and "sentence_index" columns.

    Returns
    -------
    pd.DataFrame
        Dataframe with added "negative", "neutral", "positive", and "sentiment" columns.
    """
    mentions = df.query("sentiment_string.notnull()").copy().reset_index(drop=True)
    others = df.query("sentiment_string.isnull()").copy()

    results = absa_df(mentions["sentiment_string"].to_list())

    results_df = pd.concat((mentions, results), axis=1)
    results_df["sentiment"] = pd.Series(
        np.argmax(results_df[["negative", "neutral", "positive"]].values, axis=1)
    ).map({0: "negative", 1: "neutral", 2: "positive"})

    return (
        pd.concat((results_df, others), ignore_index=True, axis=0)
        .sort_values("id")
        .reset_index(drop=True)
    )
