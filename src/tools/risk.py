from textblob import TextBlob
from typing import List, Union, Dict, Any
import numpy as np


def sentiment_score(texts: Union[str, List[str]]) -> float:
    """
    Calculate the average sentiment score for a single text or a list of texts using TextBlob.
    The sentiment score ranges from -1.0 (most negative) to 1.0 (most positive).
    
    Args:
        texts (Union[str, List[str]]): A single text string or a list of text strings to analyze
        
    Returns:
        float: Average sentiment score for the given text(s)
        
    Example:
        >>> sentiment_score("Great video!")
        0.8
        >>> comments = ["Great video!", "This was terrible", "I learned a lot"]
        >>> sentiment_score(comments)
        -0.06666666666666667
    """
    # Normalize input to list
    if isinstance(texts, str):
        texts = [texts]
    if not texts:
        raise ValueError("Input text or list of texts cannot be empty")
        
    # Calculate sentiment polarity for each text
    sentiments = [TextBlob(text).sentiment.polarity for text in texts]
    
    # Compute and return the mean sentiment score
    return float(np.mean(sentiments))