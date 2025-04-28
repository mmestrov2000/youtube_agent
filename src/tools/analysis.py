import numpy as np
from typing import List, Tuple
from scipy import stats

def predict_next_video_views(historical_views: List[int], confidence_level: float = 0.90) -> int:
    """
    Predict the minimum view count for the next video with a given confidence level.
    Uses a one-sided confidence interval to predict the lower bound of expected views.
    
    Args:
        historical_views (List[int]): List of view counts from previous videos
        confidence_level (float): Confidence level for the prediction (default: 0.90 for 90%)
        
    Returns:
        int: The minimum expected view count with the specified confidence level
        
    Example:
        >>> views = [10000, 12000, 15000, 11000, 13000]
        >>> min_views = predict_next_video_views(views)
        >>> print(f"We are 90% confident the next video will get at least {min_views} views")
    """
    if not historical_views:
        raise ValueError("Historical views list cannot be empty")
    
    # Convert to numpy array for calculations
    views = np.array(historical_views)
    
    # Calculate mean and standard error
    mean_views = np.mean(views)
    std_error = np.std(views, ddof=1) / np.sqrt(len(views))
    
    # Calculate the t-score for the given confidence level
    # We use a one-sided interval, so we use 1 - (1 - confidence_level)
    t_score = stats.t.ppf(confidence_level, df=len(views) - 1)
    
    # Calculate the lower bound of the confidence interval
    lower_bound = mean_views - t_score * std_error
    
    # Round to nearest integer since views are whole numbers
    return int(round(lower_bound))
