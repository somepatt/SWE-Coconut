"""
Evaluation script for COCONUT model
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from loguru import logger

def generate_solution(
    model,
    tokenizer,
    problem: str,
    max_length: int = 512,
):
    """Generate solution for given problem"""
    
    prompt = f"Fix this bug:\n{problem}\n\nSolution:"
    
    inputs = tokenizer(prompt, return_tensors="pt").to("cuda")
    
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_length=max_length,
            num_beams=3,
            early_stopping=True,
            temperature=0.7,
        )
    
    solution = tokenizer.decode(outputs[0], skip_special_tokens=True)
    return solution

def main():
    # Load model
    model_dir = "./outputs/final_model"
    model = AutoModelForCausalLM.from_pretrained(model_dir)
    tokenizer = AutoTokenizer.from_pretrained(model_dir)
    
    model = model.to("cuda")
    model.eval()
    
    # Test
    problem = "The function returns None instead of an empty list"
    
    solution = generate_solution(model, tokenizer, problem)
    
    logger.info(f"Problem: {problem}")
    logger.info(f"Solution: {solution}")

if __name__ == "__main__":
    main()
