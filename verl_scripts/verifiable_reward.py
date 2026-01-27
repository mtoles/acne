"""
Verifiable reward function for GRPO training.
Implement your reward logic here.
"""


def compute_score(data_source, solution_str, ground_truth, extra_info=None):
    """
    Reward function that computes scores for generated solutions.
    
    Args:
        data_source: The dataset name/source identifier
        solution_str: The generated solution string
        ground_truth: The ground truth answer
        extra_info: Optional additional information dict
    
    Returns:
        float: Reward score for the solution (typically 0.0 to 1.0)
    """
    # TODO: Implement your reward function logic here
    # For now, return zero rewards as placeholder
    return 0.0

