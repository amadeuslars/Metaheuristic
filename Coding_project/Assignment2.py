import numpy as np
from Utils import *
from joblib import Parallel, delayed

problem = load_problem("Data/Call_7_Vehicle_3.txt")
compatibility_list = precompute_compatibility(problem)

def evaluate_solution(solution, problem):
    """
    Evaluate a single solution: first check feasibility, and if feasible, compute its cost.
    Returns a tuple: (feasible (bool), cost (float), solution).
    """
    feas, reason = feasibility_check(solution, problem)
    if feas:
        cost = cost_function(solution, problem)
    else:
        cost = np.inf  # assign a very high cost if not feasible
    return feas, cost, solution

def blind_random_search(problem, compatibility_list, max_iter, n_jobs=-1):
    """
    Generate a batch of random solutions and evaluate them in parallel.
    
    Parameters:
      problem: the problem instance (dictionary).
      compatibility_list: precomputed lookup of compatible vehicles.
      max_iter: number of solutions to generate.
      n_jobs: number of parallel jobs (-1 uses all available cores).
      
    Returns:
      best_solution: the feasible solution with the lowest cost (or None if none feasible).
      best_cost: cost of the best solution (or np.inf if none feasible).
      errors: number of infeasible solutions encountered.
    """
    # Generate a list of new random solutions
    solutions = [random_solution(problem, compatibility_list) for _ in range(max_iter)]
    
    # Evaluate all solutions in parallel.
    results = Parallel(n_jobs=n_jobs)(
        delayed(evaluate_solution)(sol, problem) for sol in solutions
    )
    
    # Count infeasible solutions
    errors = sum(1 for feas, cost, sol in results if not feas)
    
    # Select feasible solutions only
    feasible_results = [res for res in results if res[0]]
    if not feasible_results:
        return None, np.inf, errors
    
    # Choose the feasible solution with the minimum cost.
    best = min(feasible_results, key=lambda x: x[1])

    return best[2], best[1], errors


