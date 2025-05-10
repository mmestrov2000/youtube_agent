from textblob import TextBlob
from typing import List, Dict, Any
import numpy as np

def sentiment_score(comments: List[str]) -> float:
    """
    Calculate the average sentiment score for a list of comments using TextBlob.
    The sentiment score ranges from -1.0 (most negative) to 1.0 (most positive).
    
    Args:
        comments (List[str]): List of comment strings to analyze
        
    Returns:
        float: Average sentiment score across all comments
        
    Example:
        >>> comments = ["Great video!", "This was terrible", "I learned a lot"]
        >>> score = sentiment_score(comments)
        >>> print(f"Average sentiment score: {score}")
    """
    if not comments:
        raise ValueError("Comments list cannot be empty")
        
    # Calculate sentiment for each comment
    sentiments = [TextBlob(comment).sentiment.polarity for comment in comments]
    
    # Return the mean sentiment score
    return float(np.mean(sentiments))