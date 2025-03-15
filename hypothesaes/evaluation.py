"""Methods for evaluating hypotheses on reference sets and real datasets."""

import numpy as np
import pandas as pd
from typing import List, Dict, Tuple, Optional
from sklearn.metrics import roc_auc_score, average_precision_score, f1_score
from scipy.stats import pearsonr, ttest_ind
from scipy.optimize import linear_sum_assignment
import statsmodels.api as sm
from tqdm.auto import tqdm

from .llm_api import get_completion
from .utils import load_prompt

def compute_pairwise_correlation_matrix(
    reference_hypotheses: Dict[str, np.ndarray],
    predicted_hypotheses: Dict[str, np.ndarray]
) -> Dict[Tuple[str, str], float]:
    """
    Compute pairwise correlations between reference and predicted hypotheses.
    
    Args:
        reference_hypotheses: Dict mapping hypothesis text to binary vector
        predicted_hypotheses: Dict mapping hypothesis text to binary vector
    """
    corr_dict = {}
    for ref_hyp in reference_hypotheses.keys():
        for pred_hyp in predicted_hypotheses.keys():
            corr_dict[(ref_hyp, pred_hyp)] = pearsonr(
                reference_hypotheses[ref_hyp], 
                predicted_hypotheses[pred_hyp]
            )[0]
    return corr_dict

def hungarian_matching_algorithm(
    hypothesis_list_1: List[str],
    hypothesis_list_2: List[str],
    similarity_scores: Dict[Tuple[str, str], float]
) -> List[Tuple[str, str, float]]:
    """
    Find optimal matching between two sets of hypotheses.
    
    Returns list of (hyp1, hyp2, similarity_score) tuples.
    """
    N = len(hypothesis_list_1)
    assert N == len(hypothesis_list_2)
    
    # Convert dict to matrix
    sim_matrix = np.zeros((N, N))
    for i, hyp_i in enumerate(hypothesis_list_1):
        for j, hyp_j in enumerate(hypothesis_list_2):
            sim_matrix[i,j] = similarity_scores[(hyp_i, hyp_j)]
    
    # Hungarian algorithm
    row_ind, col_ind = linear_sum_assignment(-sim_matrix)  # Negative for maximization
    
    # Create matched pairs
    matches = []
    for i, j in zip(row_ind, col_ind):
        matches.append((
            hypothesis_list_1[i],
            hypothesis_list_2[j],
            sim_matrix[i, j]
        ))
    
    return matches

def evaluate_predicate_surface_similarity(
    predicate1: str,
    predicate2: str,
    model: str = "gpt-4-turbo-preview",
    temperature: float = 0.7,
    n_samples: int = 5
) -> float:
    """Evaluate surface similarity between two predicates using LLM."""
    prompt = load_prompt("surface-similarity")
    scores = []
    
    for _ in range(n_samples):
        response = get_completion(
            prompt=prompt.format(text_a=predicate1, text_b=predicate2),
            model=model,
            temperature=temperature,
            max_tokens=2
        )
        
        response = response.strip().lower()
        if response.startswith("yes"):
            scores.append(1.0)
        elif response.startswith("related"):
            scores.append(0.5)
        elif response.startswith("no"):
            scores.append(0.0)
    
    return sum(scores) / len(scores)

def compute_hypothesis_separation_scores(
    hypothesis_annotations: Dict[str, np.ndarray],
    y_true: np.ndarray
) -> Dict[str, Tuple[float, float]]:
    """
    The separation score is defined as the difference in mean of the target variable between the items that have and do not have the hypothesis concept.
    Compute effect size and p-value for each hypothesis.
    
    Returns dict mapping hypothesis to (effect_size, p_value).
    """
    results = {}
    for hypothesis, annotations in hypothesis_annotations.items():
        if -1 in annotations:
            # For pairwise data we want to compute E[Y | A == 1] + E[1-Y | A == -1]
            pos_mean = 0.5*(np.mean(y_true[annotations == 1]) + np.mean(1 - y_true[annotations == -1]))
        else:
            pos_mean = np.mean(y_true[annotations == 1])

        neg_mean = np.mean(y_true[annotations == 0])
        effect_size = pos_mean - neg_mean
        
        # T-test between groups
        pos_vals = np.concatenate([y_true[annotations == 1], -1*y_true[annotations == -1]])
        neg_vals = y_true[annotations == 0]
        _, p_value = ttest_ind(pos_vals, neg_vals)
        
        results[hypothesis] = (effect_size, p_value)
    
    return results

def compute_ols_metrics(
    hypothesis_annotations: Dict[str, np.ndarray],
    y_true: np.ndarray,
    classification: bool = False,
    print_summary: bool = False
) -> Tuple[Dict[str, float], Dict[str, Tuple[float, float]]]:
    """
    Compute OLS/Logistic regression metrics for hypotheses.
    
    Returns:
        Tuple of (metrics dict, coefficient/p-value dict)
    """
    hypotheses = list(hypothesis_annotations.keys())
    X = np.array([hypothesis_annotations[h] for h in hypotheses]).T
    X = sm.add_constant(X)
    
    if classification:
        model = sm.Logit(y_true, X)
    else:
        model = sm.OLS(y_true, X)
    
    try:
        results = model.fit()
    except Exception as e:
        print(f"Error fitting model: {e}, trying OLS instead")
        model = sm.OLS(y_true, X)
        results = model.fit()
    
    if print_summary:
        print(results.summary())
    
    # Get metrics
    metrics = {}
    y_pred = results.predict(X)
    
    if classification:
        metrics.update({
            'auroc': roc_auc_score(y_true, y_pred),
            'auprc': average_precision_score(y_true, y_pred),
        })
        if hasattr(results, 'prsquared'):
            metrics['r2'] = results.prsquared
    else:
        r, _ = pearsonr(y_true, y_pred)
        metrics['r2'] = r**2
    
    # Get coefficients and p-values
    coefs_pvals = {
        hyp: (coef, pval) 
        for hyp, coef, pval in zip(hypotheses, results.params[1:], results.pvalues[1:])
    }
    
    return metrics, coefs_pvals

def score_hypotheses(
    hypothesis_annotations: Dict[str, np.ndarray],
    y_true: np.ndarray,
    classification: bool = False,
    corrected_pval_threshold: float = 0.1,
    print_summary: bool = False,
) -> Tuple[Dict[str, float], pd.DataFrame]:
    """
    Evaluate hypotheses on real dataset.
    
    Returns:
        Tuple of (metrics dict, hypothesis details DataFrame)
    """
    # Get OLS metrics
    metrics, coefs_pvals = compute_ols_metrics(
        hypothesis_annotations=hypothesis_annotations,
        y_true=y_true,
        classification=classification,
        print_summary=print_summary
    )

    # Compute count of significant hypotheses after Bonferroni correction
    corrected_p_value_threshold = corrected_pval_threshold / len(hypothesis_annotations)
    significant_hypotheses = [hyp for hyp, (_, pval) in coefs_pvals.items() if pval < corrected_p_value_threshold]
    metrics['Significant'] = (len(significant_hypotheses), len(hypothesis_annotations), corrected_p_value_threshold)
    
    # Create base dataframe
    hypothesis_df = pd.DataFrame({
        'hypothesis': list(coefs_pvals.keys()),
        'regression_coef': [coef for coef, _ in coefs_pvals.values()],
        'regression_pval': [pval for _, pval in coefs_pvals.values()]
    })
    
    # Add feature prevalence
    hypothesis_df['feature_prevalence'] = [
        np.mean(hypothesis_annotations[h] != 0)
        for h in hypothesis_df['hypothesis']
    ]
    
    # Add separation scores
    separation_scores = compute_hypothesis_separation_scores(
        hypothesis_annotations=hypothesis_annotations,
        y_true=y_true
    )
    hypothesis_df['separation_score'] = [
        separation_scores[h][0] for h in hypothesis_df['hypothesis']
    ]
    hypothesis_df['separation_pval'] = [
        separation_scores[h][1] for h in hypothesis_df['hypothesis']
    ]
    
    # Sort by coefficient magnitude
    hypothesis_df = hypothesis_df[['hypothesis', 'separation_score', 'separation_pval', 'regression_coef', 'regression_pval', 'feature_prevalence']]
    hypothesis_df = hypothesis_df.sort_values('separation_score', ascending=False)
    
    return metrics, hypothesis_df