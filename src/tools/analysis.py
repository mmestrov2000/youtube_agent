import numpy as np
from typing import List, Tuple, Literal
from scipy import stats

def predict_next_video_views(
    historical_views: List[int],
    confidence_level: float = 0.90,
    interval_type: Literal["lower", "upper", "two-sided"] = "two-sided"
) -> Tuple[float, float]:
    """
    Predict a one‑ or two‑sided confidence interval for the next video’s view count,
    assuming a log‑normal model.

    Args:
        historical_views (List[int]): Past view counts (must be >0)
        confidence_level (float): Coverage for the bound(s) (default 0.90)
        interval_type (str): 
            - "lower"      → one‑sided lower bound (L, ∞)
            - "upper"      → one‑sided upper bound (-∞, U)
            - "two-sided" → central interval (L, U)

    Returns:
        Tuple[float, float]:
            Depending on `interval_type`:
              - "lower":      (L, ∞)  with  P(X ≥ L) = confidence_level
              - "upper":      (-∞, U) with  P(X ≤ U) = confidence_level
              - "two-sided": (L, U)  with  P(L ≤ X ≤ U) = confidence_level

    Raises:
        ValueError: if list is empty, contains non‑positive values, or invalid `interval_type`
    """
    if not historical_views:
        raise ValueError("Historical views list cannot be empty")
    views = np.array(historical_views, dtype=float)
    if np.any(views <= 0):
        raise ValueError("All view counts must be positive to fit a log‑normal")

    # Fit a log‑normal: returns (shape, loc, scale)
    shape, loc, scale = stats.lognorm.fit(views, floc=0)

    alpha = 1.0 - confidence_level

    if interval_type == "lower":
        # one‑sided lower: find the α‑quantile so P(X ≥ L)=confidence_level
        L = stats.lognorm.ppf(alpha, shape, loc=loc, scale=scale)
        return float(L), float("inf")

    elif interval_type == "upper":
        # one‑sided upper: find the confidence_level‑quantile so P(X ≤ U)=confidence_level
        U = stats.lognorm.ppf(confidence_level, shape, loc=loc, scale=scale)
        return float("-inf"), float(U)

    elif interval_type == "two-sided":
        # central interval: cut off α/2 in each tail
        lower_q = stats.lognorm.ppf(alpha / 2, shape, loc=loc, scale=scale)
        upper_q = stats.lognorm.ppf(1 - alpha / 2, shape, loc=loc, scale=scale)
        print(float(lower_q), float(upper_q))
        return float(lower_q), float(upper_q)