"""TokenProjector: 인코더 임베딩 → 디코더 임베딩 공간 투영."""
import numpy as np
import torch
import torch.nn as nn
from typing import Optional, Union
from app.config import Config
from app.utils.logger import get_logger

logger = get_logger(__name__)


class TokenProjector(nn.Module):
    """
    TokenProjector: 인코더 임베딩을 디코더 임베딩 공간으로 투영.
    
    REFRAG 논문의 핵심 아이디어:
    - Sentence-transformers (인코더) 임베딩을 LLM (디코더) 임베딩 공간으로 변환
    - inputs_embeds를 사용하여 토큰화 단계를 건너뛰고 직접 임베딩 입력
    - 토큰화 오버헤드 제거로 5-10배 속도 향상 가능
    """
    
    def __init__(
        self,
        encoder_dim: int = 384,  # sentence-transformers 기본 차원
        decoder_dim: int = 4096,  # gemma3:27b 임베딩 차원 (추정)
        use_pretrained: bool = False,
        projector_path: Optional[str] = None
    ):
        """
        Args:
            encoder_dim: 인코더 임베딩 차원 (sentence-transformers)
            decoder_dim: 디코더 임베딩 차원 (LLM 모델)
            use_pretrained: 사전 학습된 projector 사용 여부
            projector_path: 사전 학습된 projector 경로
        """
        super().__init__()
        self.encoder_dim = encoder_dim
        self.decoder_dim = decoder_dim
        
        # 선형 투영 레이어 (간단한 구현)
        # 실제로는 학습된 projector가 필요하지만, 여기서는 기본 구조만 구현
        self.projector = nn.Linear(encoder_dim, decoder_dim, bias=False)
        
        # 사전 학습된 projector 로드 (있는 경우)
        if use_pretrained and projector_path:
            try:
                self.load_state_dict(torch.load(projector_path, map_location='cpu'))
                logger.info(f"Loaded pretrained projector from {projector_path}")
            except Exception as e:
                logger.warning(f"Failed to load pretrained projector: {e}. Using random initialization.")
        
        # Xavier 초기화 (더 나은 수렴)
        nn.init.xavier_uniform_(self.projector.weight)
    
    def forward(self, encoder_embeddings: Union[np.ndarray, torch.Tensor]) -> torch.Tensor:
        """
        인코더 임베딩을 디코더 임베딩 공간으로 투영.
        
        Args:
            encoder_embeddings: 인코더 임베딩 (shape: [batch_size, encoder_dim] 또는 [encoder_dim])
            
        Returns:
            디코더 임베딩 (shape: [batch_size, decoder_dim] 또는 [decoder_dim])
        """
        # numpy → torch 변환
        if isinstance(encoder_embeddings, np.ndarray):
            encoder_embeddings = torch.from_numpy(encoder_embeddings).float()
        
        # 1D → 2D 변환 (필요한 경우)
        if encoder_embeddings.dim() == 1:
            encoder_embeddings = encoder_embeddings.unsqueeze(0)
        
        # 투영
        decoder_embeddings = self.projector(encoder_embeddings)
        
        # L2 정규화 (임베딩 공간 일관성 유지)
        decoder_embeddings = nn.functional.normalize(decoder_embeddings, p=2, dim=-1)
        
        return decoder_embeddings
    
    def project_batch(self, encoder_embeddings: np.ndarray) -> np.ndarray:
        """
        배치 인코더 임베딩을 디코더 임베딩으로 투영 (numpy 입출력).
        
        Args:
            encoder_embeddings: 인코더 임베딩 배열 (shape: [batch_size, encoder_dim])
            
        Returns:
            디코더 임베딩 배열 (shape: [batch_size, decoder_dim])
        """
        self.eval()  # 평가 모드
        with torch.no_grad():
            decoder_embeddings = self.forward(encoder_embeddings)
            return decoder_embeddings.cpu().numpy()
    
    def save(self, path: str):
        """Projector 가중치 저장."""
        torch.save(self.state_dict(), path)
        logger.info(f"Saved TokenProjector to {path}")
    
    def load(self, path: str):
        """Projector 가중치 로드."""
        self.load_state_dict(torch.load(path, map_location='cpu'))
        logger.info(f"Loaded TokenProjector from {path}")


# 전역 TokenProjector 인스턴스 (lazy initialization)
_token_projector: Optional[TokenProjector] = None


def get_token_projector(
    encoder_dim: int = 384,
    decoder_dim: int = 4096,
    use_pretrained: bool = False,
    projector_path: Optional[str] = None
) -> TokenProjector:
    """
    전역 TokenProjector 인스턴스 가져오기 (싱글톤 패턴).
    
    Args:
        encoder_dim: 인코더 임베딩 차원
        decoder_dim: 디코더 임베딩 차원
        use_pretrained: 사전 학습된 projector 사용 여부
        projector_path: 사전 학습된 projector 경로
        
    Returns:
        TokenProjector 인스턴스
    """
    global _token_projector
    
    if _token_projector is None:
        _token_projector = TokenProjector(
            encoder_dim=encoder_dim,
            decoder_dim=decoder_dim,
            use_pretrained=use_pretrained,
            projector_path=projector_path
        )
        logger.info(f"Initialized TokenProjector: {encoder_dim} → {decoder_dim}")
    
    return _token_projector


def project_embeddings(
    encoder_embeddings: np.ndarray,
    encoder_dim: int = 384,
    decoder_dim: int = 4096
) -> np.ndarray:
    """
    편의 함수: 인코더 임베딩을 디코더 임베딩으로 투영.
    
    Args:
        encoder_embeddings: 인코더 임베딩 배열
        encoder_dim: 인코더 임베딩 차원
        decoder_dim: 디코더 임베딩 차원
        
    Returns:
        디코더 임베딩 배열
    """
    projector = get_token_projector(encoder_dim=encoder_dim, decoder_dim=decoder_dim)
    return projector.project_batch(encoder_embeddings)

