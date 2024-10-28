import torch
from torch.utils.data import Dataset
from transformers import AutoTokenizer
from datasets import load_dataset
from typing import Optional
import logging

logger = logging.getLogger(__name__)

class TinyStoriesDataset(Dataset):
    """Dataset wrapper for Tiny Stories."""
    
    def __init__(
        self,
        split: str,
        tokenizer: AutoTokenizer,
        max_length: int = 512,
        subset_size: Optional[int] = None
    ):
        self.tokenizer = tokenizer
        self.max_length = max_length
        
        logger.info(f"Loading TinyStories dataset split: {split}")
        
        # Load Tiny Stories dataset
        try:
            self.dataset = load_dataset("roneneldan/TinyStories", split=split)
            
            # Take subset if specified
            if subset_size is not None:
                original_size = len(self.dataset)
                self.dataset = self.dataset.select(range(min(subset_size, original_size)))
                logger.info(f"Using {len(self.dataset)} examples from {original_size} total")
            else:
                logger.info(f"Using full dataset of {len(self.dataset)} examples")
                
        except Exception as e:
            logger.error(f"Failed to load dataset: {str(e)}")
            raise
            
    def __len__(self):
        return len(self.dataset)
        
    def __getitem__(self, idx):
        # Get story text
        text = self.dataset[idx]['text']
        
        # Tokenize with special tokens
        try:
            inputs = self.tokenizer(
                text,
                max_length=self.max_length,
                truncation=True,
                padding='max_length',
                return_tensors='pt'
            )
            
            # Prepare inputs and labels for causal LM training
            input_ids = inputs['input_ids'].squeeze(0)
            attention_mask = inputs['attention_mask'].squeeze(0)
            labels = input_ids.clone()
            
            # Mask out padding tokens in labels
            labels[attention_mask == 0] = -100
            
            return {
                'input_ids': input_ids,
                'attention_mask': attention_mask,
                'labels': labels
            }
            
        except Exception as e:
            logger.error(f"Failed to process example {idx}: {str(e)}")
            raise