from datasets import load_dataset, Dataset
from transformers import AutoTokenizer
from torch.utils.data import DataLoader, IterableDataset
from typing import Optional, List, Dict, Tuple
import torch
from loguru import logger

class SWESmithDataset:
    """
    SWE-bench SWE-smith dataset for code bug fixing.
    
    Structure:
    - problem_statement: Description of the bug/issue
    - patch: Git diff with the solution
    - test_patch: Tests to verify the fix
    """
    
    def __init__(self, config):
        self.config = config
        self.logger = logger
        
    def load_dataset(self) -> Dataset:
        """Load SWE-smith dataset from Hugging Face"""
        self.logger.info(f"Loading dataset: {self.config.data.dataset_name}")
        
        dataset = load_dataset(
            self.config.data.dataset_name,
            split=self.config.data.split,
            cache_dir=self.config.data.cache_dir,
        )
        
        self.logger.info(f"Dataset loaded. Size: {len(dataset)}")
        return dataset
    
    def preprocess_example(
        self, 
        example: Dict,
        tokenizer: AutoTokenizer
    ) -> Dict[str, torch.Tensor]:
        """
        Convert SWE-smith example to training format.
        
        Format: [PROBLEM] issue description [SOLUTION] code fix
        """
        problem = example.get('problem_statement', '')
        solution = example.get('patch', '')
        
        # Truncate long prompts
        if len(problem) > self.config.data.max_prompt_length * 4:
            problem = problem[:self.config.data.max_prompt_length * 4]
        
        # Create prompt-completion pair
        prompt = f"Fix this bug:\n{problem}\n\nSolution:"
        completion = f"\n{solution}"
        
        # Tokenize prompt
        prompt_tokens = tokenizer.encode(prompt, add_special_tokens=False)
        completion_tokens = tokenizer.encode(completion, add_special_tokens=False)
        
        # Combine
        input_ids = prompt_tokens + completion_tokens
        
        # Truncate to max length
        max_len = self.config.data.max_seq_length
        if len(input_ids) > max_len:
            input_ids = input_ids[:max_len]
        
        # Create loss mask (only compute loss on completion)
        loss_mask = [False] * len(prompt_tokens) + [True] * (len(input_ids) - len(prompt_tokens))
        loss_mask = loss_mask[:max_len]
        
        return {
            'input_ids': input_ids,
            'loss_mask': loss_mask,
            'problem': problem[:200],
            'solution_length': len(completion_tokens),
        }
    
    def prepare_dataset(self, tokenizer: AutoTokenizer) -> Dataset:
        """Prepare dataset for training"""
        dataset = self.load_dataset()
        
        self.logger.info("Preprocessing dataset...")
        
        def preprocess_fn(examples):
            batch_data = {
                'input_ids': [],
                'loss_mask': [],
            }
            
            for idx in range(len(examples['problem_statement'])):
                example = {
                    'problem_statement': examples['problem_statement'][idx],
                    'patch': examples.get('patch', [''])[idx] if 'patch' in examples else '',
                }
                
                try:
                    processed = self.preprocess_example(example, tokenizer)
                    batch_data['input_ids'].append(processed['input_ids'])
                    batch_data['loss_mask'].append(processed['loss_mask'])
                except Exception as e:
                    self.logger.warning(f"Error processing example {idx}: {e}")
                    continue
            
            return batch_data
        
        processed_dataset = dataset.map(
            preprocess_fn,
            batched=True,
            batch_size=100,
            remove_columns=dataset.column_names,
        )
        
        self.logger.info(f"Dataset preprocessed. Size: {len(processed_dataset)}")
        return processed_dataset


class CoconutDataCollator:
    """Collate function for COCONUT training"""
    
    def __init__(self, tokenizer, max_seq_length: int = 1024):
        self.tokenizer = tokenizer
        self.max_seq_length = max_seq_length
        self.logger = logger
    
    def __call__(self, batch: List[Dict]) -> Dict[str, torch.Tensor]:
        """Collate batch for training"""
        input_ids_list = []
        loss_mask_list = []
        
        for example in batch:
            input_ids = example['input_ids']
            loss_mask = example['loss_mask']
            
            # Pad or truncate
            if len(input_ids) < self.max_seq_length:
                pad_len = self.max_seq_length - len(input_ids)
                input_ids = input_ids + [self.tokenizer.pad_token_id] * pad_len
                loss_mask = loss_mask + [False] * pad_len
            else:
                input_ids = input_ids[:self.max_seq_length]
                loss_mask = loss_mask[:self.max_seq_length]
            
            input_ids_list.append(input_ids)
            loss_mask_list.append(loss_mask)
        
        return {
            'input_ids': torch.tensor(input_ids_list, dtype=torch.long),
            'loss_mask': torch.tensor(loss_mask_list, dtype=torch.bool),
            'attention_mask': torch.ones_like(torch.tensor(input_ids_list, dtype=torch.long)),
        }


def get_dataloader(
    config,
    tokenizer: AutoTokenizer,
    stage: int = 0,
) -> DataLoader:
    """Get DataLoader for COCONUT training at given stage"""
    logger.info(f"Loading data for stage {stage}")
    
    dataset_loader = SWESmithDataset(config)
    dataset = dataset_loader.prepare_dataset(tokenizer)
    
    # Split for stages (simulate different difficulty)
    samples_per_stage = len(dataset) // (config.training.num_stages + 1)
    stage_dataset = dataset.select(
        range(stage * samples_per_stage, (stage + 1) * samples_per_stage)
    )
    
    collator = CoconutDataCollator(tokenizer, config.data.max_seq_length)
    
    dataloader = DataLoader(
        stage_dataset,
        batch_size=config.data.batch_size,
        collate_fn=collator,
        num_workers=config.data.num_workers,
        shuffle=True,
    )
    
    logger.info(f"DataLoader created. Samples: {len(stage_dataset)}, Batch size: {config.data.batch_size}")
    return dataloader
