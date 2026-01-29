#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Multimodal encoders for image and text feature extraction using Qwen2.5-VL.
"""

import os
import numpy as np
from abc import ABC, abstractmethod
from typing import List, Protocol, Optional, Union
from PIL import Image

import torch
from transformers import Qwen2_5_VLForConditionalGeneration, AutoProcessor
from qwen_vl_utils import process_vision_info
from qwen3_vl_embedding import Qwen3VLEmbedder


class MultimodalEncoder(Protocol):
    """Protocol for multimodal encoders."""

    @abstractmethod
    def encode_image(self, pil_images: List[Image.Image]) -> np.ndarray:
        """Encode a batch of PIL images to embeddings."""
        pass

    @abstractmethod
    def encode_text(self, texts: List[str]) -> np.ndarray:
        """Encode a batch of text strings to embeddings."""
        pass


class DummyMultimodalEncoder:
    """
    A dummy multimodal encoder using random vectors for testing.
    """

    def __init__(self, image_dim: int = 512, text_dim: int = 512, seed: int = 42):
        self.image_dim = image_dim
        self.text_dim = text_dim
        self.rng = np.random.default_rng(seed)

    def encode_image(self, pil_images: List[Image.Image]) -> np.ndarray:
        batch_size = len(pil_images)
        return self.rng.normal(size=(batch_size, self.image_dim)).astype("float32")

    def encode_text(self, texts: List[str]) -> np.ndarray:
        batch_size = len(texts)
        return self.rng.normal(size=(batch_size, self.text_dim)).astype("float32")


class Qwen25VLEncoder:
    """
    Real Qwen2.5-VL multimodal encoder for production use.
    """

    def __init__(self, model_path: str, device: str = "cuda"):
        """
        Initialize Qwen2.5-VL encoder.

        Args:
            model_path: Path to the Qwen2.5-VL model (e.g., "Qwen/Qwen2.5-VL-3B-Instruct")
            device: Device to run on ('auto', 'cpu', 'cuda', etc.)
        """
        self.model_path = model_path
        self.device = device

        print(f"Loading Qwen2.5-VL model from {model_path}...")
        
        # Load model using Qwen2_5_VL class
        # flash_attn is recommended if available
        self.model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
            model_path,
            torch_dtype=torch.bfloat16,
            #attn_implementation="flash_attention_2", # Use "eager" if flash attn not installed
            attn_implementation="eager",
            device_map=device,
        )
        
        # Load processor
        self.processor = AutoProcessor.from_pretrained(model_path)

        # Set model to evaluation mode
        self.model.eval()
        if hasattr(self.model.config, "hidden_size"):
            # 标准 HuggingFace 模型
            self.hidden_dim = self.model.config.hidden_size
        elif hasattr(self.model.config, "text_config") and hasattr(self.model.config.text_config, "hidden_size"):
            # Qwen 系列通常在这里
            self.hidden_dim = self.model.config.text_config.hidden_size
        else:
            # 最后的保底：直接获取 Embedding 层的维度
            self.hidden_dim = self.model.get_input_embeddings().embedding_dim

        # Get hidden dimension (typically 4096 for 7B, 2048/2560 for 3B)
        #self.hidden_dim = self.model.config.hidden_size
        print(f"Model loaded successfully. Hidden dimension: {self.hidden_dim}")

    def _get_mean_pooling(self, hidden_states, attention_mask):
        """
        Perform mean pooling on hidden states, ignoring padded tokens.
        
        Args:
            hidden_states: [B, seq_len, hidden_dim]
            attention_mask: [B, seq_len] (1 for valid, 0 for pad)
        """
        # Expand attention mask to [B, seq_len, 1]
        input_mask_expanded = attention_mask.unsqueeze(-1).expand(hidden_states.size()).float()
        
        # Sum hidden states of valid tokens
        sum_embeddings = torch.sum(hidden_states * input_mask_expanded, 1)
        
        # Count valid tokens (clamp to avoid division by zero)
        sum_mask = torch.clamp(input_mask_expanded.sum(1), min=1e-9)
        
        return sum_embeddings / sum_mask

    @torch.no_grad()
    def encode_image(self, pil_images: List[Image.Image]) -> np.ndarray:
        """
        Encode images using Qwen2.5-VL.
        
        Note: Since Qwen2.5-VL is a causal LM, we encode the image by asking it to 
        'describe the image' (or just feeding the image token) and taking the mean of the output states.
        """
        if not pil_images:
            return np.empty((0, self.hidden_dim), dtype=np.float32)

        # Prepare messages
        messages = []
        for img in pil_images:
            messages.append([
                {
                    "role": "user",
                    "content": [
                        {"type": "image", "image": img},
                        {"type": "text", "text": "Describe this image."} # Prompt to encourage visual processing
                    ],
                }
            ])

        # Prepare inputs using qwen_vl_utils logic implicit in processor
        texts = [
            self.processor.apply_chat_template(msg, tokenize=False, add_generation_prompt=True)
            for msg in messages
        ]
        
        image_inputs, video_inputs = process_vision_info(messages)
        
        inputs = self.processor(
            text=texts,
            images=image_inputs,
            videos=video_inputs,
            padding=True,
            return_tensors="pt",
        ).to(self.model.device)

        # Forward pass
        outputs = self.model(**inputs, output_hidden_states=True)
        
        # Extract last hidden state: [B, seq_len, hidden_dim]
        last_hidden_state = outputs.hidden_states[-1]
        
        # Use mean pooling with attention mask to ignore padding
        embeddings = self._get_mean_pooling(last_hidden_state, inputs.attention_mask)
        
        return embeddings.cpu().float().numpy()

    @torch.no_grad()
    def encode_text(self, texts: List[str]) -> np.ndarray:
        """
        Encode text using Qwen2.5-VL.
        """
        if not texts:
            return np.empty((0, self.hidden_dim), dtype=np.float32)

        # Prepare messages
        messages = []
        for text in texts:
            messages.append([
                {"role": "user", "content": text}
            ])

        # Convert to chat format
        text_inputs = [
            self.processor.apply_chat_template(msg, tokenize=False, add_generation_prompt=True)
            for msg in messages
        ]
        
        # Tokenize
        inputs = self.processor(
            text=text_inputs,
            padding=True,
            return_tensors="pt",
        ).to(self.model.device)

        # Forward pass
        outputs = self.model(**inputs, output_hidden_states=True)
        
        # Extract last hidden state
        last_hidden_state = outputs.hidden_states[-1]
        
        # Mean pooling
        embeddings = self._get_mean_pooling(last_hidden_state, inputs.attention_mask)

        return embeddings.cpu().float().numpy()


class Qwen3VLEmbeddingEncoder:
    """
    Qwen3-VL-Embedding encoder wrapper for your pipeline.
    Produces ONE embedding per sample from (image, text) together.
    """

    def __init__(
        self,
        model_path: str,
        instruction: str = "Represent the user's input.",
        torch_dtype: torch.dtype = torch.float16,
        attn_implementation: str = "eager",  # if installed: "flash_attention_2"
        max_length: int = 8192,
        normalize: bool = True,
        min_pixels: int = 4 * (16*2) * (16*2),
        max_pixels: int = 1800 * (16*2) * (16*2),
        total_pixels: int = 10 * (768 * (16*2) * (16*2)),
    ):
        self.instruction = instruction
        self.normalize = normalize

        # Qwen3VLEmbedder accepts **kwargs passed to from_pretrained
        # so torch_dtype / attn_implementation will work here.
        self.embedder = Qwen3VLEmbedder(
            model_name_or_path=model_path,
            max_length=max_length,
            min_pixels=min_pixels,
            max_pixels=max_pixels,
            total_pixels=total_pixels,
            default_instruction=instruction,
            torch_dtype=torch_dtype,
            attn_implementation=attn_implementation,
        )

    @torch.no_grad()
    def encode_fused(
        self,
        images: List[Union[str, Image.Image]],
        texts: List[str],
        instructions: Optional[List[str]] = None,
    ) -> torch.Tensor:
        """
        Returns: torch.Tensor [B, D] on GPU
        """
        assert len(images) == len(texts)

        inputs = []
        for i, (img, txt) in enumerate(zip(images, texts)):
            ins = self.instruction if instructions is None else instructions[i]
            item = {
                "text": txt,
                "image": img,              # can be PIL.Image or file path
                "instruction": ins,
            }
            inputs.append(item)

        # returns torch.Tensor on GPU
        return self.embedder.process(inputs, normalize=self.normalize)