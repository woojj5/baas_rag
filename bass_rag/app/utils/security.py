"""Security utilities for file handling and error sanitization."""
from pathlib import Path
from typing import Tuple
import re


def sanitize_filename(filename: str) -> str:
    """
    Sanitize filename to prevent path traversal attacks.
    
    Args:
        filename: Original filename
    
    Returns:
        Sanitized filename
    """
    # Remove path separators and parent directory references
    filename = filename.replace('/', '').replace('\\', '')
    filename = filename.replace('..', '')
    
    # Remove any remaining dangerous characters
    filename = re.sub(r'[<>:"|?*]', '', filename)
    
    # Limit length
    if len(filename) > 255:
        filename = filename[:255]
    
    return filename


def validate_save_path(base_dir: Path, filename: str) -> Tuple[Path, bool]:
    """
    Validate if a file can be safely saved within a base directory.
    
    Args:
        base_dir: Base directory path
        filename: Filename to save
    
    Returns:
        Tuple of (safe_path, is_valid)
    """
    try:
        # Sanitize filename
        safe_filename = sanitize_filename(filename)
        
        # Resolve paths to prevent path traversal
        base_dir = base_dir.resolve()
        safe_path = (base_dir / safe_filename).resolve()
        
        # Check if resolved path is still within base directory
        if not str(safe_path).startswith(str(base_dir)):
            return safe_path, False
        
        return safe_path, True
    except Exception:
        return Path(filename), False


def sanitize_error_message(e: Exception) -> str:
    """
    Sanitize error message to remove sensitive information.
    
    Args:
        e: Exception object
    
    Returns:
        Sanitized error message
    """
    error_msg = str(e)
    
    # Remove potential sensitive information patterns
    # File paths (keep only filename)
    error_msg = re.sub(r'/[^\s]+/([^/\s]+)', r'\1', error_msg)
    
    # URLs with credentials
    error_msg = re.sub(r'://[^:]+:[^@]+@', r'://***:***@', error_msg)
    
    # API keys or tokens (long alphanumeric strings)
    error_msg = re.sub(r'\b[a-zA-Z0-9]{32,}\b', '***', error_msg)
    
    # Email addresses
    error_msg = re.sub(r'\b[\w.-]+@[\w.-]+\.\w+\b', '***@***', error_msg)
    
    return error_msg

