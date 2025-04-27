import logging
import requests
from io import BytesIO
from PIL import Image
import torch
import numpy as np
from transformers import CLIPProcessor, CLIPModel
import torch.nn.functional as F

# ─── Logging setup ─────────────────────────────────────────────────────────────
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class ThumbnailAnalyzer:
    def __init__(
        self,
        clip_model_name: str = "openai/clip-vit-large-patch14",
        temperature: float = 0.07,
        scale: float = 5.0
    ):
        """
        Args:
            clip_model_name: HuggingFace model for CLIP
            temperature: softmax temperature for similarity logits
            scale: multiplier applied before sigmoid normalization
        """
        logger.info(f"Loading CLIP model {clip_model_name}…")
        self.model = CLIPModel.from_pretrained(clip_model_name)
        self.processor = CLIPProcessor.from_pretrained(clip_model_name)
        self.temperature = temperature
        self.scale = scale

        # prompts covering design, clarity, emotion, composition
        self.positive_prompts = [
            "eye-catching thumbnail",
            "bold, vibrant colors",
            "clear, readable text",
            "prominent faces",
            "professional design"
        ]
        self.negative_prompts = [
            "blurry or out of focus",
            "dark or underexposed",
            "dull colors",
            "small or unreadable text",
            "cluttered layout"
        ]

    def _download_image(self, url: str) -> Image.Image:
        resp = requests.get(url, timeout=5)
        resp.raise_for_status()
        return Image.open(BytesIO(resp.content)).convert("RGB")

    def score_thumbnail(self, thumbnail_url: str) -> float:
        """
        Compute a 0–1 score for how “attractive” a thumbnail is.
        
        Steps:
         1. Download + preprocess image
         2. Encode image & prompts with CLIP
         3. Compute similarity logits with temperature
         4. Take mean positive minus mean negative
         5. Normalize via sigmoid(scale * diff)
        
        Returns:
            A float in [0,1], higher is better.
        """
        try:
            logger.info(f"Scoring thumbnail: {thumbnail_url}")
            img = self._download_image(thumbnail_url)

            texts = self.positive_prompts + self.negative_prompts
            inputs = self.processor(
                images=img,
                text=texts,
                return_tensors="pt",
                padding=True
            )
            
            # Get and normalize features
            img_feats = self.model.get_image_features(inputs["pixel_values"])
            txt_feats = self.model.get_text_features(inputs["input_ids"])
            img_feats = img_feats / img_feats.norm(dim=-1, keepdim=True)
            txt_feats = txt_feats / txt_feats.norm(dim=-1, keepdim=True)

            # similarity logits
            logits = (img_feats @ txt_feats.T) / self.temperature  # shape (1, N_prompts)
            logits = logits.squeeze(0)  # shape (N_prompts,)

            # Debug: log a few values
            for p, score in zip(texts, logits.tolist()):
                logger.debug(f"  ‘{p}’: {score:.3f}")

            n_pos = len(self.positive_prompts)
            pos_mean = logits[:n_pos].mean()
            neg_mean = logits[n_pos:].mean()
            diff = pos_mean - neg_mean

            # sigmoid normalization
            score = torch.sigmoid(diff * self.scale).item()
            logger.info(f"Thumbnail score → {score:.4f}")
            return float(score)

        except Exception as e:
            logger.error(f"Failed to score thumbnail: {e}")
            return 0.0