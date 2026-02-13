"""
Inference script for Auto-CoT demo construction and evaluation.
Handles both XRC-AutoCoT (proposed) and CW-AutoCoT (baseline).
"""

import os
import sys
import json
import re
from typing import List, Dict, Tuple, Optional
from collections import Counter
import numpy as np
from tqdm import tqdm
import wandb
from omegaconf import DictConfig, OmegaConf

from src.preprocess import load_math_dataset, cluster_questions


def call_llm(
    prompt: str,
    model_name: str,
    provider: str,
    temperature: float,
    max_tokens: int,
    api_base: Optional[str] = None
) -> str:
    """Call LLM API and return response."""
    if provider == 'openai':
        from openai import OpenAI
        api_key = os.environ.get('OPENAI_API_KEY')
        if not api_key:
            raise ValueError("OPENAI_API_KEY environment variable not set")
        
        client = OpenAI(api_key=api_key, base_url=api_base)
        response = client.chat.completions.create(
            model=model_name,
            messages=[{"role": "user", "content": prompt}],
            temperature=temperature,
            max_tokens=max_tokens
        )
        return response.choices[0].message.content
    else:
        raise ValueError(f"Unknown provider: {provider}")


def extract_final_answer(response: str) -> Optional[str]:
    """Extract final answer from LLM response."""
    # Look for "Final answer: X" pattern
    match = re.search(r'Final answer:\s*([^\n]+)', response, re.IGNORECASE)
    if match:
        answer = match.group(1).strip()
        # Clean up common formatting
        answer = answer.replace('$', '').replace(',', '').strip()
        return answer
    return None


def normalize_answer(answer: str) -> str:
    """Normalize answer for comparison."""
    if answer is None:
        return ""
    # Remove whitespace, currency symbols, commas
    normalized = str(answer).strip().replace('$', '').replace(',', '').lower()
    return normalized


def construct_demos_xrc(
    demo_pool: List[Dict],
    cluster_candidates: List[List[int]],
    cfg: DictConfig,
    mode: str
) -> List[Dict]:
    """
    Construct demonstrations using XRC-AutoCoT method.
    
    Args:
        demo_pool: Full pool of candidate questions
        cluster_candidates: List of candidate indices per cluster
        cfg: Configuration
        mode: 'sanity_check' or 'main'
        
    Returns:
        List of selected demonstrations with reliability scores
    """
    method = cfg.method
    model = cfg.model
    
    n_samples_a = 2 if mode == 'sanity_check' else method.n_samples_mode_a
    n_samples_b = 2 if mode == 'sanity_check' else method.n_samples_mode_b
    
    selected_demos = []
    
    for cluster_id, candidate_indices in enumerate(tqdm(cluster_candidates, desc="Processing clusters")):
        if not candidate_indices:
            continue
        
        best_candidate = None
        best_reliability = -1
        
        for candidate_idx in candidate_indices:
            question = demo_pool[candidate_idx]['question']
            
            # Sample with Mode A (Forward CoT)
            prompt_a = f"{question}\n\n{method.mode_a_prompt}"
            answers_a = []
            for _ in range(n_samples_a):
                response = call_llm(
                    prompt_a,
                    model.name,
                    model.provider,
                    method.temperature,
                    model.max_tokens,
                    model.api_base
                )
                answer = extract_final_answer(response)
                if answer:
                    answers_a.append(normalize_answer(answer))
            
            # Sample with Mode B (Verification/backward check)
            prompt_b = f"{question}\n\n{method.mode_b_prompt}"
            answers_b = []
            for _ in range(n_samples_b):
                response = call_llm(
                    prompt_b,
                    model.name,
                    model.provider,
                    method.temperature,
                    model.max_tokens,
                    model.api_base
                )
                answer = extract_final_answer(response)
                if answer:
                    answers_b.append(normalize_answer(answer))
            
            if not answers_a or not answers_b:
                continue
            
            # Compute per-mode distributions
            counter_a = Counter(answers_a)
            counter_b = Counter(answers_b)
            
            total_a = len(answers_a)
            total_b = len(answers_b)
            
            prob_a = {ans: count / total_a for ans, count in counter_a.items()}
            prob_b = {ans: count / total_b for ans, count in counter_b.items()}
            
            # Get modal answers
            modal_a = counter_a.most_common(1)[0][0]
            modal_b = counter_b.most_common(1)[0][0]
            
            # Check cross-mode agreement
            if modal_a != modal_b:
                continue
            
            # Compute cross-reasoning reliability r(q) = max_a p_A(a) * p_B(a)
            all_answers = set(prob_a.keys()) | set(prob_b.keys())
            r = max(prob_a.get(ans, 0) * prob_b.get(ans, 0) for ans in all_answers)
            
            # Accept if r >= threshold
            if r >= method.tau_xrc:
                if r > best_reliability:
                    best_reliability = r
                    # Generate one final reasoning chain with mode A for the demo
                    final_prompt = f"{question}\n\n{method.mode_a_prompt}"
                    final_response = call_llm(
                        final_prompt,
                        model.name,
                        model.provider,
                        0.0,  # Deterministic for final demo
                        model.max_tokens,
                        model.api_base
                    )
                    best_candidate = {
                        'question': question,
                        'reasoning': final_response,
                        'answer': modal_a,
                        'reliability': r,
                        'cluster_id': cluster_id
                    }
        
        if best_candidate:
            selected_demos.append(best_candidate)
    
    # Sort by reliability (descending) if configured
    if cfg.inference.order_by_reliability:
        selected_demos.sort(key=lambda x: x['reliability'], reverse=True)
    
    return selected_demos[:cfg.inference.max_demos]


def construct_demos_cw(
    demo_pool: List[Dict],
    cluster_candidates: List[List[int]],
    cfg: DictConfig,
    mode: str
) -> List[Dict]:
    """
    Construct demonstrations using CW-AutoCoT baseline method.
    
    Args:
        demo_pool: Full pool of candidate questions
        cluster_candidates: List of candidate indices per cluster
        cfg: Configuration
        mode: 'sanity_check' or 'main'
        
    Returns:
        List of selected demonstrations with reliability scores
    """
    method = cfg.method
    model = cfg.model
    
    n_samples = 2 if mode == 'sanity_check' else method.n_samples
    
    selected_demos = []
    
    for cluster_id, candidate_indices in enumerate(tqdm(cluster_candidates, desc="Processing clusters")):
        if not candidate_indices:
            continue
        
        best_candidate = None
        best_probability = -1
        
        for candidate_idx in candidate_indices:
            question = demo_pool[candidate_idx]['question']
            
            # Sample with single prompt
            prompt = f"{question}\n\n{method.prompt}"
            answers = []
            for _ in range(n_samples):
                response = call_llm(
                    prompt,
                    model.name,
                    model.provider,
                    method.temperature,
                    model.max_tokens,
                    model.api_base
                )
                answer = extract_final_answer(response)
                if answer:
                    answers.append(normalize_answer(answer))
            
            if not answers:
                continue
            
            # Compute modal answer probability
            counter = Counter(answers)
            total = len(answers)
            modal_answer = counter.most_common(1)[0][0]
            modal_count = counter[modal_answer]
            p = modal_count / total
            
            # Accept if p >= threshold
            if p >= method.tau_cw:
                if p > best_probability:
                    best_probability = p
                    # Generate final reasoning chain
                    final_response = call_llm(
                        prompt,
                        model.name,
                        model.provider,
                        0.0,  # Deterministic
                        model.max_tokens,
                        model.api_base
                    )
                    best_candidate = {
                        'question': question,
                        'reasoning': final_response,
                        'answer': modal_answer,
                        'reliability': p,
                        'cluster_id': cluster_id
                    }
        
        if best_candidate:
            selected_demos.append(best_candidate)
    
    # Sort by reliability (descending) if configured
    if cfg.inference.order_by_reliability:
        selected_demos.sort(key=lambda x: x['reliability'], reverse=True)
    
    return selected_demos[:cfg.inference.max_demos]


def evaluate_with_demos(
    test_set: List[Dict],
    demos: List[Dict],
    cfg: DictConfig,
    mode: str
) -> Tuple[float, List[Dict]]:
    """
    Evaluate test set using constructed demonstrations.
    
    Args:
        test_set: Test questions with ground truth answers
        demos: Selected demonstrations
        cfg: Configuration
        mode: 'sanity_check' or 'main'
        
    Returns:
        (accuracy, predictions) where predictions is list of dicts with results
    """
    model = cfg.model
    
    # Build prompt with demonstrations
    demo_text = "\n\n---\n\n".join([
        f"Question: {d['question']}\n{d['reasoning']}"
        for d in demos
    ])
    
    # Evaluate test set
    n_test = 5 if mode == 'sanity_check' else len(test_set)
    test_subset = test_set[:n_test]
    
    predictions = []
    correct = 0
    
    for item in tqdm(test_subset, desc="Evaluating"):
        question = item['question']
        ground_truth = normalize_answer(item['answer'])
        
        # Build full prompt
        if demos:
            full_prompt = f"{demo_text}\n\n---\n\nQuestion: {question}\n\nThink step by step to solve this problem. End your response with 'Final answer: [your numeric answer]'."
        else:
            full_prompt = f"Question: {question}\n\nThink step by step to solve this problem. End your response with 'Final answer: [your numeric answer]'."
        
        # Get prediction
        response = call_llm(
            full_prompt,
            model.name,
            model.provider,
            model.temperature_eval,
            model.max_tokens,
            model.api_base
        )
        
        predicted_answer = extract_final_answer(response)
        predicted_normalized = normalize_answer(predicted_answer) if predicted_answer else ""
        
        is_correct = (predicted_normalized == ground_truth)
        if is_correct:
            correct += 1
        
        predictions.append({
            'question': question,
            'ground_truth': ground_truth,
            'predicted': predicted_normalized,
            'correct': is_correct,
            'response': response
        })
    
    accuracy = correct / len(predictions) if predictions else 0.0
    return accuracy, predictions


def run_inference(cfg: DictConfig):
    """Main inference function."""
    mode = cfg.mode
    
    # Initialize WandB
    if cfg.wandb.mode != 'disabled':
        wandb_project = f"{cfg.wandb.project}-sanity" if mode == 'sanity_check' else cfg.wandb.project
        wandb.init(
            entity=cfg.wandb.entity,
            project=wandb_project,
            id=cfg.run.run_id,
            config=OmegaConf.to_container(cfg, resolve=True),
            resume="allow"
        )
        print(f"WandB run URL: {wandb.run.get_url()}")
    
    # Load dataset
    print(f"Loading dataset: {cfg.dataset.name}")
    demo_pool, test_set = load_math_dataset(
        cfg.dataset.name,
        cfg.cache_dir,
        n_pool=cfg.method.n_pool if mode != 'sanity_check' else 50,
        n_test=cfg.dataset.n_test,
        seed=42
    )
    print(f"Demo pool size: {len(demo_pool)}, Test set size: {len(test_set)}")
    
    # Cluster questions
    print("Clustering questions...")
    questions = [item['question'] for item in demo_pool]
    cluster_candidates = cluster_questions(
        questions,
        n_clusters=cfg.method.n_clusters,
        candidates_per_cluster=cfg.method.candidates_per_cluster,
        cache_dir=cfg.cache_dir,
        seed=42
    )
    
    # Construct demonstrations
    print(f"Constructing demonstrations with {cfg.method.name}...")
    if cfg.method.name == 'xrc_autocot':
        demos = construct_demos_xrc(demo_pool, cluster_candidates, cfg, mode)
    elif cfg.method.name == 'cw_autocot':
        demos = construct_demos_cw(demo_pool, cluster_candidates, cfg, mode)
    else:
        raise ValueError(f"Unknown method: {cfg.method.name}")
    
    print(f"Selected {len(demos)} demonstrations")
    
    # Save demos
    results_dir = os.path.join(cfg.results_dir, cfg.run.run_id)
    os.makedirs(results_dir, exist_ok=True)
    
    with open(os.path.join(results_dir, 'demos.json'), 'w') as f:
        json.dump(demos, f, indent=2)
    
    # Log demo info to WandB
    if cfg.wandb.mode != 'disabled':
        wandb.log({
            'n_demos': len(demos),
            'avg_reliability': np.mean([d['reliability'] for d in demos]) if demos else 0.0
        })
    
    # Evaluate
    print("Evaluating on test set...")
    accuracy, predictions = evaluate_with_demos(test_set, demos, cfg, mode)
    
    print(f"Accuracy: {accuracy:.4f}")
    
    # Save predictions
    with open(os.path.join(results_dir, 'predictions.json'), 'w') as f:
        json.dump(predictions, f, indent=2)
    
    # Save metrics
    metrics = {
        'accuracy': accuracy,
        'n_demos': len(demos),
        'n_test': len(predictions)
    }
    
    with open(os.path.join(results_dir, 'metrics.json'), 'w') as f:
        json.dump(metrics, f, indent=2)
    
    # Log to WandB
    if cfg.wandb.mode != 'disabled':
        wandb.log({'accuracy': accuracy, 'n_test': len(predictions)})
        wandb.summary.update(metrics)
        wandb.finish()
    
    # Sanity validation
    if mode == 'sanity_check':
        perform_sanity_validation(predictions, demos)
    
    return accuracy


def perform_sanity_validation(predictions: List[Dict], demos: List[Dict]):
    """Perform sanity validation checks and print verdict."""
    n_samples = len(predictions)
    n_correct = sum(1 for p in predictions if p['correct'])
    accuracy = n_correct / n_samples if n_samples > 0 else 0.0
    
    # Check conditions
    all_valid = all(p['predicted'] != '' for p in predictions)
    not_all_same = len(set(p['predicted'] for p in predictions)) > 1
    has_demos = len(demos) > 0
    
    # Print summary
    print(f"SANITY_VALIDATION_SUMMARY: {{\"samples\":{n_samples}, \"correct\":{n_correct}, \"accuracy\":{accuracy:.4f}, \"n_demos\":{len(demos)}, \"all_valid\":{all_valid}, \"not_all_same\":{not_all_same}}}")
    
    # Determine pass/fail
    if n_samples < 5:
        print("SANITY_VALIDATION: FAIL reason=insufficient_samples")
    elif not all_valid:
        print("SANITY_VALIDATION: FAIL reason=invalid_outputs")
    elif not not_all_same:
        print("SANITY_VALIDATION: FAIL reason=all_identical_outputs")
    elif not has_demos:
        print("SANITY_VALIDATION: FAIL reason=no_demos_constructed")
    else:
        print("SANITY_VALIDATION: PASS")
