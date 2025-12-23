"""Configuration management for RAG server."""
import os
import re
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()


class Config:
    """Application configuration."""
    # Paths (can be overridden by environment variables)
    ROOT = Path(os.getenv("RAG_ROOT_DIR", "/home/keti_spark1/j309"))
    DATA_DIR = Path(os.getenv("RAG_DATA_DIR", str(ROOT / "data")))
    INDEX_DIR = Path(os.getenv("RAG_INDEX_DIR", str(ROOT / "index")))
    
    OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
    OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "gemma3:27b")
    EMBED_MODEL_NAME = os.getenv("EMBED_MODEL_NAME", "sentence-transformers/all-MiniLM-L6-v2")
    
    # LLM Backend selection
    LLM_BACKEND = os.getenv("LLM_BACKEND", "ollama").lower()  # "ollama" or "huggingface"
    
    # HuggingFace settings
    HF_MODEL_NAME = os.getenv("HF_MODEL_NAME", "google/gemma-3-27b")  # HuggingFace model name (matches Ollama's gemma3:27b)
    HF_DEVICE = os.getenv("HF_DEVICE", "auto")  # "auto", "cuda", or "cpu"
    HF_TRUST_REMOTE_CODE = os.getenv("HF_TRUST_REMOTE_CODE", "false").lower() == "true"
    
    CHUNK_SIZE = int(os.getenv("CHUNK_SIZE", "400"))
    CHUNK_OVERLAP = int(os.getenv("CHUNK_OVERLAP", "100"))
    TOP_K_DEFAULT = int(os.getenv("TOP_K_DEFAULT", "5"))
    
    # Performance optimization settings
    USE_RERANKER = os.getenv("USE_RERANKER", "true").lower() == "true"
    RERANKER_MIN_CANDIDATES = int(os.getenv("RERANKER_MIN_CANDIDATES", "5"))  # Only rerank if more than this
    MAX_QUERY_EXPANSIONS = int(os.getenv("MAX_QUERY_EXPANSIONS", "3"))  # Reduced from 5
    SEARCH_K_MULTIPLIER = int(os.getenv("SEARCH_K_MULTIPLIER", "2"))  # Reduced from 3
    
    # Re-retrieval settings (optimized)
    USE_RERETRIEVAL = os.getenv("USE_RERETRIEVAL", "true").lower() == "true"  # Enable/disable re-retrieval
    RERETRIEVAL_MIN_CONFIDENCE = float(os.getenv("RERETRIEVAL_MIN_CONFIDENCE", "0.3"))  # Only re-retrieve if answer confidence is low
    RERETRIEVAL_MAX_KEYWORDS = int(os.getenv("RERETRIEVAL_MAX_KEYWORDS", "3"))  # Max keywords to extract
    RERETRIEVAL_SEARCH_K = int(os.getenv("RERETRIEVAL_SEARCH_K", "5"))  # Small search range for re-retrieval
    
    # REFRAG settings
    CHUNK_TOKEN_SIZE = int(os.getenv("CHUNK_TOKEN_SIZE", "256"))
    MAX_RETRIEVED_CHUNKS = int(os.getenv("MAX_RETRIEVED_CHUNKS", "32"))
    MAX_EXPANDED_CHUNKS = int(os.getenv("MAX_EXPANDED_CHUNKS", "5"))
    # REFRAG-style context window management
    CTX_MAX_TOKENS = int(os.getenv("CTX_MAX_TOKENS", "2048"))  # Max context tokens (REFRAG-style)
    EXPAND_FRAC = float(os.getenv("EXPAND_FRAC", "0.25"))  # Fraction of chunks to expand (REFRAG-style)
    
    # Phase 3: TokenProjector settings
    USE_TOKEN_PROJECTOR = os.getenv("USE_TOKEN_PROJECTOR", "false").lower() == "true"  # Enable TokenProjector
    ENCODER_EMBEDDING_DIM = int(os.getenv("ENCODER_EMBEDDING_DIM", "384"))  # sentence-transformers dimension
    DECODER_EMBEDDING_DIM = int(os.getenv("DECODER_EMBEDDING_DIM", "4096"))  # LLM embedding dimension (estimated)
    TOKEN_PROJECTOR_PATH = os.getenv("TOKEN_PROJECTOR_PATH", None)  # Path to pretrained projector (optional)
    
    # Security settings
    MAX_UPLOAD_SIZE = int(os.getenv("MAX_UPLOAD_SIZE", "52428800"))  # 50MB in bytes
    ALLOWED_FILE_EXTENSIONS = os.getenv("ALLOWED_FILE_EXTENSIONS", ".txt,.md,.csv").split(",")
    
    # Index update optimization settings
    INDEX_SAVE_DEBOUNCE_SECONDS = float(os.getenv("INDEX_SAVE_DEBOUNCE_SECONDS", "5.0"))  # Delay before saving (debounce)
    INDEX_SAVE_BATCH_SIZE = int(os.getenv("INDEX_SAVE_BATCH_SIZE", "10"))  # Save after N updates
    INDEX_SAVE_BACKGROUND = os.getenv("INDEX_SAVE_BACKGROUND", "true").lower() == "true"  # Save in background
    
    # Caching settings
    ENABLE_EMBEDDING_CACHE = os.getenv("ENABLE_EMBEDDING_CACHE", "true").lower() == "true"
    EMBEDDING_CACHE_SIZE = int(os.getenv("EMBEDDING_CACHE_SIZE", "1000"))  # Max cached embeddings
    ENABLE_QUERY_EXPANSION_CACHE = os.getenv("ENABLE_QUERY_EXPANSION_CACHE", "true").lower() == "true"
    QUERY_EXPANSION_CACHE_SIZE = int(os.getenv("QUERY_EXPANSION_CACHE_SIZE", "500"))  # Max cached query expansions
    
    INDEX_DIR.mkdir(parents=True, exist_ok=True)
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    
    @staticmethod
    def validate_config():
        """
        Validate configuration values.
        
        Raises:
            ValueError: If configuration is invalid
        """
        errors = []
        
        # Validate OLLAMA_BASE_URL format
        url_pattern = re.compile(r'^https?://[^\s/$.?#].[^\s]*$')
        if not url_pattern.match(Config.OLLAMA_BASE_URL):
            errors.append(f"Invalid OLLAMA_BASE_URL format: {Config.OLLAMA_BASE_URL}")
        
        # Extract port from URL if present
        port_match = re.search(r':(\d+)', Config.OLLAMA_BASE_URL)
        if port_match:
            port = int(port_match.group(1))
            if not (1 <= port <= 65535):
                errors.append(f"Invalid port in OLLAMA_BASE_URL: {port}")
        
        # Validate paths exist
        if not Config.ROOT.exists():
            errors.append(f"ROOT directory does not exist: {Config.ROOT}")
        
        if not Config.DATA_DIR.exists():
            errors.append(f"DATA_DIR does not exist: {Config.DATA_DIR}")
        
        if not Config.INDEX_DIR.exists():
            errors.append(f"INDEX_DIR does not exist: {Config.INDEX_DIR}")
        
        # Validate MAX_UPLOAD_SIZE
        if Config.MAX_UPLOAD_SIZE <= 0:
            errors.append(f"MAX_UPLOAD_SIZE must be positive: {Config.MAX_UPLOAD_SIZE}")
        
        # Validate ALLOWED_FILE_EXTENSIONS
        if not Config.ALLOWED_FILE_EXTENSIONS:
            errors.append("ALLOWED_FILE_EXTENSIONS cannot be empty")
        
        if errors:
            raise ValueError(f"Configuration validation failed:\n" + "\n".join(f"  - {e}" for e in errors))

