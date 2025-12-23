"""Optimized index saving utilities with batch and debounce support."""
import asyncio
import time
from typing import Optional
import faiss
import numpy as np
from app.config import Config
from app.rag_index import save_index
from app.utils.logger import get_logger

logger = get_logger(__name__)


class IndexSaveManager:
    """Manages optimized index saving with debounce and batch support."""
    
    def __init__(self):
        self.pending_task: Optional[asyncio.Task] = None
        self.last_update_time: float = 0.0
        self.update_count: int = 0
        self.lock = asyncio.Lock()
        self.pending_embeddings: Optional[np.ndarray] = None
        self.pending_passages: Optional[list] = None
    
    async def schedule_save(
        self,
        faiss_index: faiss.Index,
        passages: list,
        new_embeddings: Optional[np.ndarray] = None
    ):
        """
        Schedule index save with debounce and batch optimization.
        
        Args:
            faiss_index: FAISS index to save
            passages: List of passages
            new_embeddings: Optional new embeddings to append
        """
        async with self.lock:
            self.update_count += 1
            self.last_update_time = time.time()
            
            # Store current state for saving
            self.pending_passages = passages.copy() if passages else None
            
            # If we have new embeddings, we need to load and append to existing
            # For simplicity, we'll save the full index (FAISS doesn't support incremental save easily)
            
            # Cancel existing task if any
            if self.pending_task and not self.pending_task.done():
                self.pending_task.cancel()
            
            # Check if we should save immediately (batch size reached)
            if self.update_count >= Config.INDEX_SAVE_BATCH_SIZE:
                logger.info(f"Batch size reached ({self.update_count} updates) - saving immediately")
                await self._save_now(faiss_index, passages, new_embeddings)
                self.update_count = 0
                return
            
            # Schedule debounced save
            if Config.INDEX_SAVE_BACKGROUND:
                self.pending_task = asyncio.create_task(
                    self._debounced_save(faiss_index, passages, new_embeddings)
                )
            else:
                # Immediate save (no background)
                await self._save_now(faiss_index, passages, new_embeddings)
                self.update_count = 0
    
    async def _debounced_save(
        self,
        faiss_index: faiss.Index,
        passages: list,
        new_embeddings: Optional[np.ndarray] = None
    ):
        """Debounced save - waits for quiet period before saving."""
        try:
            await asyncio.sleep(Config.INDEX_SAVE_DEBOUNCE_SECONDS)
            
            # Check if there were more updates during the wait
            async with self.lock:
                if time.time() - self.last_update_time < Config.INDEX_SAVE_DEBOUNCE_SECONDS:
                    # More updates came in, reschedule
                    logger.debug("More updates detected - rescheduling save")
                    return
                
                # Save now
                await self._save_now(faiss_index, self.pending_passages or passages, new_embeddings)
                self.update_count = 0
        except asyncio.CancelledError:
            logger.debug("Index save task cancelled (new update came in)")
        except Exception as e:
            logger.error(f"Error in debounced save: {e}")
    
    async def _save_now(
        self,
        faiss_index: faiss.Index,
        passages: list,
        new_embeddings: Optional[np.ndarray] = None
    ):
        """Save index immediately."""
        try:
            logger.info(f"Saving index ({len(passages)} passages)")
            
            # For now, we save the full index
            # Note: FAISS doesn't easily support incremental saves, so we save the full index
            # The optimization here is in the timing (debounce/batch), not the save itself
            
            if Config.INDEX_SAVE_BACKGROUND:
                # Run in executor to avoid blocking
                loop = asyncio.get_event_loop()
                await loop.run_in_executor(
                    None,
                    save_index,
                    faiss_index,
                    passages,
                    None,  # doc_id_to_original
                    new_embeddings  # embeddings (if provided)
                )
            else:
                # Synchronous save
                save_index(faiss_index, passages, None, new_embeddings)
            
            logger.info("Index saved successfully")
        except Exception as e:
            logger.error(f"Error saving index: {e}")
            raise
    
    async def force_save(
        self,
        faiss_index: faiss.Index,
        passages: list,
        new_embeddings: Optional[np.ndarray] = None
    ):
        """Force immediate save (e.g., on shutdown)."""
        async with self.lock:
            if self.pending_task and not self.pending_task.done():
                self.pending_task.cancel()
            await self._save_now(faiss_index, passages, new_embeddings)
            self.update_count = 0


# Global instance
index_save_manager = IndexSaveManager()

