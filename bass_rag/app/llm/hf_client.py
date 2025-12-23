"""HuggingFace Transformers LLM client with inputs_embeds support."""
import torch
import numpy as np
from typing import List, Dict, Optional
from transformers import AutoTokenizer, AutoModelForCausalLM, AutoConfig
from app.config import Config
from app.llm.base import LLMClient
from app.utils.logger import get_logger

logger = get_logger(__name__)


class HuggingFaceLLMClient(LLMClient):
    """
    HuggingFace Transformers LLM client with inputs_embeds support.
    
    This client supports TokenProjector by accepting pre-computed embeddings
    and using inputs_embeds instead of tokenized input, which can significantly
    reduce latency by skipping tokenization.
    """
    
    def __init__(
        self,
        model_name: str = None,
        device: str = None,
        torch_dtype: torch.dtype = None,
        use_token_projector: bool = True
    ):
        """
        Initialize HuggingFace LLM client.
        
        Args:
            model_name: HuggingFace model name (e.g., "google/gemma-2-27b-it")
            device: Device to use ("cuda", "cpu", or "auto")
            torch_dtype: Data type for model (torch.float16, torch.bfloat16, etc.)
            use_token_projector: Whether to use TokenProjector for inputs_embeds
        """
        self.model_name = model_name or Config.HF_MODEL_NAME
        self.use_token_projector = use_token_projector
        
        # Auto-detect device
        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
        self.device = device
        
        # Auto-detect dtype
        if torch_dtype is None:
            if device == "cuda":
                # Use bfloat16 for better compatibility with modern GPUs
                torch_dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
            else:
                torch_dtype = torch.float32
        self.torch_dtype = torch_dtype
        
        logger.info(f"Loading HuggingFace model: {self.model_name}")
        logger.info(f"Device: {self.device}, Dtype: {self.torch_dtype}")
        
        # Load tokenizer
        try:
            self.tokenizer = AutoTokenizer.from_pretrained(
                self.model_name,
                trust_remote_code=Config.HF_TRUST_REMOTE_CODE
            )
            # Set pad token if not set
            if self.tokenizer.pad_token is None:
                self.tokenizer.pad_token = self.tokenizer.eos_token
        except Exception as e:
            logger.error(f"Failed to load tokenizer: {e}")
            raise
        
        # Load model
        try:
            self.model = AutoModelForCausalLM.from_pretrained(
                self.model_name,
                torch_dtype=self.torch_dtype,
                device_map="auto" if device == "cuda" else None,
                trust_remote_code=Config.HF_TRUST_REMOTE_CODE,
                low_cpu_mem_usage=True
            )
            if device == "cpu":
                self.model = self.model.to(device)
            self.model.eval()  # Set to evaluation mode
            logger.info(f"Model loaded successfully on {self.device}")
        except Exception as e:
            logger.error(f"Failed to load model: {e}")
            raise
        
        # Get model's embedding dimension
        try:
            config = AutoConfig.from_pretrained(self.model_name, trust_remote_code=Config.HF_TRUST_REMOTE_CODE)
            self.embedding_dim = config.vocab_size if hasattr(config, 'vocab_size') else None
            # Try to get actual embedding dimension from model
            if hasattr(self.model, 'get_input_embeddings'):
                embedding_layer = self.model.get_input_embeddings()
                if hasattr(embedding_layer, 'embedding_dim'):
                    self.embedding_dim = embedding_layer.embedding_dim
                elif hasattr(embedding_layer, 'weight'):
                    self.embedding_dim = embedding_layer.weight.shape[1]
            logger.info(f"Model embedding dimension: {self.embedding_dim}")
        except Exception as e:
            logger.warning(f"Could not determine embedding dimension: {e}")
            self.embedding_dim = None
    
    async def chat(
        self,
        system_prompt: str,
        messages: List[Dict[str, str]],
        max_tokens: int = 512,
        temperature: float = 0.2,
        inputs_embeds: Optional[np.ndarray] = None
    ) -> str:
        """
        Generate response using HuggingFace model.
        
        Args:
            system_prompt: System prompt
            messages: List of message dicts with 'role' and 'content'
            max_tokens: Maximum tokens to generate
            temperature: Sampling temperature
            inputs_embeds: Optional pre-computed embeddings (from TokenProjector)
            
        Returns:
            Generated text response
        """
        # Build prompt from messages
        prompt_text = self._build_prompt(system_prompt, messages)
        
        # Use inputs_embeds if provided (TokenProjector path)
        if inputs_embeds is not None and self.use_token_projector:
            return await self._generate_with_embeds(
                prompt_text=prompt_text,
                inputs_embeds=inputs_embeds,
                max_tokens=max_tokens,
                temperature=temperature
            )
        else:
            # Fallback to tokenized input (standard path)
            return await self._generate_with_tokens(
                prompt_text=prompt_text,
                max_tokens=max_tokens,
                temperature=temperature
            )
    
    def _build_prompt(self, system_prompt: str, messages: List[Dict[str, str]]) -> str:
        """Build prompt text from system prompt and messages."""
        prompt_parts = []
        
        if system_prompt:
            prompt_parts.append(f"System: {system_prompt}\n")
        
        for msg in messages:
            role = msg.get("role", "user")
            content = msg.get("content", "")
            if role == "user":
                prompt_parts.append(f"User: {content}\n")
            elif role == "assistant":
                prompt_parts.append(f"Assistant: {content}\n")
        
        prompt_parts.append("Assistant: ")
        return "".join(prompt_parts)
    
    async def _generate_with_embeds(
        self,
        prompt_text: str,
        inputs_embeds: np.ndarray,
        max_tokens: int,
        temperature: float
    ) -> str:
        """
        Generate using pre-computed embeddings (TokenProjector path).
        
        This skips tokenization and directly uses embeddings, which can be
        significantly faster for long contexts.
        """
        try:
            # Convert numpy to torch
            if isinstance(inputs_embeds, np.ndarray):
                inputs_embeds = torch.from_numpy(inputs_embeds).float()
            
            # Validate and reshape inputs_embeds
            original_shape = inputs_embeds.shape
            
            # Ensure correct shape: [batch_size, seq_len, embedding_dim]
            if inputs_embeds.dim() == 1:
                # Single embedding vector: [embedding_dim] -> [1, 1, embedding_dim]
                inputs_embeds = inputs_embeds.unsqueeze(0).unsqueeze(0)
                logger.debug(f"Reshaped 1D embedding: {original_shape} -> {inputs_embeds.shape}")
            elif inputs_embeds.dim() == 2:
                # [seq_len, embedding_dim] or [embedding_dim, seq_len]
                if inputs_embeds.shape[0] == self.embedding_dim:
                    # [embedding_dim, seq_len] -> [1, seq_len, embedding_dim]
                    inputs_embeds = inputs_embeds.T.unsqueeze(0)
                else:
                    # [seq_len, embedding_dim] -> [1, seq_len, embedding_dim]
                    inputs_embeds = inputs_embeds.unsqueeze(0)
                logger.debug(f"Reshaped 2D embedding: {original_shape} -> {inputs_embeds.shape}")
            elif inputs_embeds.dim() == 3:
                # Already [batch_size, seq_len, embedding_dim]
                if inputs_embeds.shape[0] != 1:
                    logger.warning(f"Expected batch_size=1, got {inputs_embeds.shape[0]}. Using first batch.")
                    inputs_embeds = inputs_embeds[0:1]
            else:
                raise ValueError(f"Invalid inputs_embeds dimension: {inputs_embeds.dim()}")
            
            # Validate embedding dimension
            if self.embedding_dim is not None:
                actual_embedding_dim = inputs_embeds.shape[-1]
                if actual_embedding_dim != self.embedding_dim:
                    logger.warning(
                        f"Embedding dimension mismatch: expected {self.embedding_dim}, "
                        f"got {actual_embedding_dim}. This may cause errors."
                    )
                    # Try to handle dimension mismatch by padding or truncating
                    if actual_embedding_dim < self.embedding_dim:
                        # Pad with zeros
                        padding = torch.zeros(
                            inputs_embeds.shape[0],
                            inputs_embeds.shape[1],
                            self.embedding_dim - actual_embedding_dim,
                            device=inputs_embeds.device
                        )
                        inputs_embeds = torch.cat([inputs_embeds, padding], dim=-1)
                        logger.debug(f"Padded embedding to match model dimension: {inputs_embeds.shape}")
                    else:
                        # Truncate
                        inputs_embeds = inputs_embeds[..., :self.embedding_dim]
                        logger.debug(f"Truncated embedding to match model dimension: {inputs_embeds.shape}")
            
            # Move to device
            inputs_embeds = inputs_embeds.to(self.device)
            
            # Get sequence length
            seq_len = inputs_embeds.shape[1]
            
            # Create attention mask (all tokens are valid for now)
            # In future, we could mark padding tokens as 0
            attention_mask = torch.ones(
                (inputs_embeds.shape[0], seq_len),
                dtype=torch.long,
                device=self.device
            )
            
            logger.debug(
                f"Generating with inputs_embeds: shape={inputs_embeds.shape}, "
                f"seq_len={seq_len}, max_tokens={max_tokens}"
            )
            
            # Generate with inputs_embeds
            with torch.no_grad():
                outputs = self.model.generate(
                    inputs_embeds=inputs_embeds,
                    attention_mask=attention_mask,
                    max_new_tokens=max_tokens,
                    temperature=temperature,
                    do_sample=temperature > 0,
                    pad_token_id=self.tokenizer.pad_token_id,
                    eos_token_id=self.tokenizer.eos_token_id
                )
            
            # Decode only the generated part (skip input)
            # outputs shape: [batch_size, total_length]
            # total_length = seq_len (input) + generated_tokens
            generated_ids = outputs[0][seq_len:]
            response = self.tokenizer.decode(generated_ids, skip_special_tokens=True)
            
            logger.debug(
                f"Generated {len(generated_ids)} tokens using inputs_embeds. "
                f"Response length: {len(response)} chars"
            )
            return response.strip()
            
        except Exception as e:
            logger.warning(
                f"Generation with inputs_embeds failed: {e}. "
                f"Falling back to tokenized input. Original shape: {inputs_embeds.shape if 'inputs_embeds' in locals() else 'unknown'}"
            )
            import traceback
            logger.debug(f"Traceback: {traceback.format_exc()}")
            # Fallback to tokenized input
            return await self._generate_with_tokens(prompt_text, max_tokens, temperature)
    
    async def _generate_with_tokens(
        self,
        prompt_text: str,
        max_tokens: int,
        temperature: float
    ) -> str:
        """Generate using tokenized input (standard path)."""
        try:
            # Tokenize input
            inputs = self.tokenizer(
                prompt_text,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=2048  # Limit input length
            ).to(self.device)
            
            # Generate
            with torch.no_grad():
                outputs = self.model.generate(
                    **inputs,
                    max_new_tokens=max_tokens,
                    temperature=temperature,
                    do_sample=temperature > 0,
                    pad_token_id=self.tokenizer.pad_token_id,
                    eos_token_id=self.tokenizer.eos_token_id
                )
            
            # Decode only the generated part
            input_length = inputs["input_ids"].shape[1]
            generated_ids = outputs[0][input_length:]
            response = self.tokenizer.decode(generated_ids, skip_special_tokens=True)
            
            logger.debug(f"Generated {len(generated_ids)} tokens using tokenized input")
            return response.strip()
            
        except Exception as e:
            logger.error(f"Generation failed: {e}")
            raise RuntimeError(f"HuggingFace generation error: {e}")

