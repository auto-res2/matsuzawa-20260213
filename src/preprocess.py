"""
Dataset loading and diversity clustering for Auto-CoT methods.
"""

import os
import random
from typing import List, Dict, Tuple
import numpy as np
from datasets import load_dataset
from sentence_transformers import SentenceTransformer
from sklearn.cluster import KMeans
from tqdm import tqdm


def load_math_dataset(dataset_name: str, cache_dir: str, n_pool: int = 1000, n_test: int = 200, seed: int = 42) -> Tuple[List[Dict], List[Dict]]:
    """
    Load math word problem dataset and split into demo pool and test set.
    
    Args:
        dataset_name: 'gsm8k' or 'aqua'
        cache_dir: Cache directory for datasets
        n_pool: Number of samples for demo pool (from training split)
        n_test: Number of samples for test set
        seed: Random seed
        
    Returns:
        (demo_pool, test_set) where each is a list of dicts with 'question' and 'answer' keys
    """
    random.seed(seed)
    np.random.seed(seed)
    
    if dataset_name.lower() == 'gsm8k':
        dataset = load_dataset('gsm8k', 'main', cache_dir=cache_dir)
        train_data = dataset['train']
        test_data = dataset['test']
        
        # Extract questions and answers
        demo_pool = []
        for item in train_data:
            question = item['question']
            # Extract numeric answer from "#### number" format
            answer_text = item['answer']
            answer_parts = answer_text.split('####')
            if len(answer_parts) == 2:
                answer = answer_parts[1].strip().replace(',', '')
            else:
                answer = answer_text.strip()
            demo_pool.append({'question': question, 'answer': answer})
        
        # Sample from demo pool
        if len(demo_pool) > n_pool:
            demo_pool = random.sample(demo_pool, n_pool)
        
        # Create test set
        test_set = []
        for item in test_data:
            question = item['question']
            answer_text = item['answer']
            answer_parts = answer_text.split('####')
            if len(answer_parts) == 2:
                answer = answer_parts[1].strip().replace(',', '')
            else:
                answer = answer_text.strip()
            test_set.append({'question': question, 'answer': answer})
        
        if len(test_set) > n_test:
            test_set = random.sample(test_set, n_test)
            
    elif dataset_name.lower() == 'aqua':
        dataset = load_dataset('aqua_rat', 'raw', cache_dir=cache_dir)
        train_data = dataset['train']
        test_data = dataset['test']
        
        # Extract questions and answers
        demo_pool = []
        for item in train_data:
            question = item['question']
            # AQUA has multiple choice, correct answer is in 'correct' field
            options = item['options']
            correct = item['correct']
            
            # Build question with options
            question_with_options = question + '\n' + '\n'.join(options)
            demo_pool.append({'question': question_with_options, 'answer': correct})
        
        if len(demo_pool) > n_pool:
            demo_pool = random.sample(demo_pool, n_pool)
        
        test_set = []
        for item in test_data:
            question = item['question']
            options = item['options']
            correct = item['correct']
            question_with_options = question + '\n' + '\n'.join(options)
            test_set.append({'question': question_with_options, 'answer': correct})
        
        if len(test_set) > n_test:
            test_set = random.sample(test_set, n_test)
    
    else:
        raise ValueError(f"Unknown dataset: {dataset_name}")
    
    return demo_pool, test_set


def cluster_questions(
    questions: List[str],
    n_clusters: int,
    candidates_per_cluster: int,
    cache_dir: str,
    seed: int = 42
) -> List[List[int]]:
    """
    Cluster questions by semantic similarity and return top-M candidates per cluster.
    
    Args:
        questions: List of question strings
        n_clusters: Number of clusters
        candidates_per_cluster: Number of candidates to select per cluster (closest to centroid)
        cache_dir: Cache directory for sentence transformer model
        seed: Random seed
        
    Returns:
        List of lists, where each inner list contains indices of candidate questions for that cluster
    """
    random.seed(seed)
    np.random.seed(seed)
    
    # Load sentence transformer model
    model = SentenceTransformer('all-MiniLM-L6-v2', cache_folder=cache_dir)
    
    # Encode questions
    print(f"Encoding {len(questions)} questions...")
    embeddings = model.encode(questions, show_progress_bar=True, convert_to_numpy=True)
    
    # K-means clustering
    print(f"Clustering into {n_clusters} clusters...")
    kmeans = KMeans(n_clusters=n_clusters, random_state=seed, n_init=10)
    labels = kmeans.fit_predict(embeddings)
    centroids = kmeans.cluster_centers_
    
    # For each cluster, find top-M candidates closest to centroid
    cluster_candidates = []
    for cluster_id in range(n_clusters):
        # Get all question indices in this cluster
        cluster_indices = np.where(labels == cluster_id)[0]
        
        if len(cluster_indices) == 0:
            cluster_candidates.append([])
            continue
        
        # Compute distances to centroid
        cluster_embeddings = embeddings[cluster_indices]
        centroid = centroids[cluster_id]
        distances = np.linalg.norm(cluster_embeddings - centroid, axis=1)
        
        # Sort by distance and take top-M
        sorted_indices = np.argsort(distances)
        top_m = sorted_indices[:min(candidates_per_cluster, len(cluster_indices))]
        candidate_indices = cluster_indices[top_m].tolist()
        
        cluster_candidates.append(candidate_indices)
    
    return cluster_candidates
