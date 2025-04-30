from typing import List
from collections import defaultdict
import logging
from random import randint, random, choice, seed, choices
from timeit import default_timer as timer
from copy import deepcopy

from Utils2 import *

def alter_solution_4steven(problem, current_solution: List[int]) -> List[int]:
	""" A combination of removing n random calls and inserting those greedily"""
	""" Functions to choose from
	removed_solution, to_remove, removed_from = remove_random_call(current_solution, problem, number_to_remove)
	removed_solution, to_remove, removed_from = remove_highest_cost_call(current_solution, problem, number_to_remove)
	removed_solution, to_remove, removed_from = remove_dummy_call(current_solution, problem, number_to_remove)
	solution = insert_regretk(removed_solution, problem, to_remove, 2)
	solution = insert_greedy(removed_solution, problem, to_remove)
	# solution = insert_back_to_dummy(removed_solution, problem, to_remove)"""
	removed_solution, to_remove, removed_from = remove_random_call(current_solution, problem, randint(1,3))
	solution = insert_greedy(removed_solution, problem, to_remove, removed_from)
	return solution

def alter_solution_5jackie(problem, current_solution: List[int]) -> List[int]:
	""" A combination of removing n highest cost calls and inserting them with regretk"""

	removed_solution, to_remove, removed_from = remove_highest_cost_call(current_solution, problem, randint(1,3))
	solution = insert_regretk(removed_solution, problem, to_remove, removed_from, 2)
	return solution

def alter_solution_6sebastian(problem, current_solution: List[int]) -> List[int]:
	""" A combination of removing n dummy calls and inserting them greedily"""

	removed_solution, to_remove, removed_from = remove_dummy_call(current_solution, problem, randint(1,3))
	solution = insert_greedy(removed_solution, problem, to_remove, removed_from)
	return solution

def alter_solution_7steinar(problem, current_solution: List[int]) -> List[int]:
	""" A combination of removing n random calls and inserting them with regretk"""

	removed_solution, to_remove, removed_from = remove_random_call(current_solution, problem, randint(1,3))
	solution = insert_regretk(removed_solution, problem, to_remove, removed_from, 2)
	return solution

def alter_solution_8stian(problem, current_solution: List[int]) -> List[int]:
	""" A combination of removing n highest cost calls and inserting them greedily"""

	removed_solution, to_remove, removed_from = remove_highest_cost_call(current_solution, problem, randint(1,3))
	solution = insert_greedy(removed_solution, problem, to_remove, removed_from)
	return solution

def alter_solution_9karina(problem, current_solution: List[int]) -> List[int]:
	""" A combination of removing n dummy calls and inserting them with regretk"""

	removed_solution, to_remove, removed_from = remove_dummy_call(current_solution, problem, randint(1,3))
	solution = insert_regretk(removed_solution, problem, to_remove, removed_from, 2)
	return solution

def adaptive_algorithm(problem, init_sol, num_of_iterations: int = 10000, allowed_neighbours: list = [4, 5, 6, 7, 8, 9], file_num = None, statistics=False):
	""" Adaptive algorithm inspired from simulated annealing and Ahmeds slides 12-20"""
	logging.info(f"Start adaptive algorithm with neighbour(s) {allowed_neighbours}")

	if statistics:
		import matplotlib.pyplot as plt
		from matplotlib.pyplot import figure

	# Dictionary of past probabilities
	prob_hist = defaultdict(lambda: list())
	prob_hist["y"].append(0)

	# Best solution (starts as initial)
	best_sol = deepcopy(init_sol)
	s = deepcopy(init_sol)

	cost = cost_function(init_sol, problem)
	best_cost = cost
	cost_s = cost
	iterations_since_best_found = 0
	last_iteration_found_best = 0

	# Save original cost
	orig_cost = cost

	# Parameters
	r = 0.2
	param_debug_iterations = 1000
	param_escape_value = 100
	param_weight_change = 200

	# Dictionary of weights
	weights = dict()
	probabilities = list()
	for neighbour in allowed_neighbours:
		weight_val = 1/len(allowed_neighbours)
		weights[neighbour] = weight_val
		probabilities.append(weight_val)
		prob_hist[f"x{neighbour}"].append(weight_val)
	score_sums = defaultdict(lambda: 0)
	neighbour_used_counter = defaultdict(lambda: 0)

	# Found solutions
	found_sol = set()

	w = 0
	while w < num_of_iterations:
		if w%param_weight_change == 0:
			# Initialise set of neighbours not yet used
			not_used_yet = set(allowed_neighbours)

		if w%param_debug_iterations == 0:
			logging.info(f"Iteration num: {w}")

		new_score_val = 0
		# +1 found a new solution not explored yet
		# +2 found better than current
		# +4 found new best

		if iterations_since_best_found > param_escape_value:
			s, cost_s, is_new_best = escape_algorithm(problem=problem, current_solution=deepcopy(s), allowed_neighbours=[4, 7], best_sol_cost=best_cost, cost_s=cost_s) # alternate operator

			if is_new_best:
				best_sol = deepcopy(s)
				best_cost = cost_s
				last_iteration_found_best = w

			iterations_since_best_found = 0
		
		s2 = deepcopy(s)
		print(s2)
		# Choose a neighbour function
		neighbourfunc_id = choices(allowed_neighbours, probabilities, k=1)[0]

		# Use functions not used yet
		if w%param_weight_change > param_weight_change*(3/4):
			if len(not_used_yet) > 0:
				neighbourfunc_id = choice(list(not_used_yet))
		
		neighbour_used_counter[neighbourfunc_id] += 1
		not_used_yet = not_used_yet.difference({neighbourfunc_id})

		# Apply neighbouring function
		if neighbourfunc_id == 4:
			s2 = alter_solution_4steven(problem, deepcopy(s2))
		elif neighbourfunc_id == 5:
			s2 = alter_solution_5jackie(problem, deepcopy(s2))
		elif neighbourfunc_id == 6:
			s2 = alter_solution_6sebastian(problem, deepcopy(s2))
		elif neighbourfunc_id == 7:
			s2 = alter_solution_7steinar(problem, deepcopy(s2))
		elif neighbourfunc_id == 8:
			s2 = alter_solution_8stian(problem, deepcopy(s2))
		elif neighbourfunc_id == 9:
			s2 = alter_solution_9karina(problem, deepcopy(s2))

		feasiblity, _ = feasibility_check(deepcopy(s2), problem)

		updated_value = False
		if feasiblity:
			new_cost = cost_function(deepcopy(s2), problem)

			if new_cost < best_cost:
				new_score_val = 4
				best_sol = deepcopy(s2)
				best_cost = new_cost
				updated_value = True
				s = deepcopy(s2)
				cost_s = new_cost
				last_iteration_found_best = w
			
			elif new_cost < cost_s:
				new_score_val = 2
				s = deepcopy(s2)
				cost_s = new_cost
			
			elif random() < 0.2:
				s = deepcopy(s2)
				cost_s = new_cost
			
			hashed_sol = solution_to_hashable_tuple_2d(s2)
			if hashed_sol not in found_sol:
				new_score_val = 1
				found_sol.add(hashed_sol)

		if updated_value:
			iterations_since_best_found = 0
		else:
			iterations_since_best_found += 1
		
		w += 1
		# Update scores
		score_sums[neighbourfunc_id] += new_score_val
		if w%param_weight_change == 0:
			# update_parameters
			probabilities = []
			for neighbour in allowed_neighbours:
				new_weight = weights[neighbour] * (1-r) + r * (score_sums[neighbour]/neighbour_used_counter[neighbour])
				weights[neighbour] = new_weight
				score_sums[neighbour] = 0
				neighbour_used_counter[neighbour] = 0
				probabilities.append(new_weight)
			prob_hist["y"].append(w)

			sum_prob = sum(probabilities)
			for idx, el in enumerate(probabilities):
				probabilities[idx] = el/sum_prob
				prob_hist[f"x{idx+4}"].append(el/sum_prob)
			
			logging.debug(f"New weights: {probabilities}")

	if statistics:
		plt.figure(figsize=(20, 10))
		
		plt.axvline(x=last_iteration_found_best, color='b', label="best")
		y = prob_hist["y"]
		for k, v in prob_hist.items():
			if k != "y":
				label = k[1:]
				plt.plot(y, v, label = label)

		plt.legend()
		plt.savefig(f"./tempdata/weights{file_num}.png")
	logging.info(f"Last iteration with new best: {last_iteration_found_best}")

	# Check if everything is valid
	sol_correct_format, was_valid = return_output_solution(best_sol, problem)

	if not was_valid:
		sol_old_format = split_a_list_at_zeros(sol_correct_format)
		best_cost = cost_function(sol_old_format, problem)
	
	improvement = round(100*(orig_cost-best_cost)/orig_cost, 2)
	logging.info(f"Final probabilities: {list(map(lambda x: round(x, ndigits=2), probabilities))}")
	logging.debug(f"Original cost: {orig_cost}")
	logging.debug(f"New cost: {best_cost}")
	logging.debug(f"Improvement: {improvement}%")

	return sol_correct_format, best_cost, improvement

def escape_algorithm(problem, current_solution, allowed_neighbours, best_sol_cost, cost_s, num_iterations=20):
    """ This is the escape algorithm to get out of a local minimum"""
    found_new_feasible_solution = False
    iteration_num = 0
    probabilities = [1] * len(allowed_neighbours)
    new_cost = cost_s  # Initialize new_cost with current cost
    
    while iteration_num < num_iterations and not found_new_feasible_solution:
        iteration_num += 1

        # Choose a neighbour function
        neighbourfunc_id = choices(allowed_neighbours, probabilities, k=1)[0]

        # Apply neighbouring function
        if neighbourfunc_id == 4:
            s2 = alter_solution_4steven(problem, current_solution)
        elif neighbourfunc_id == 5:
            s2 = alter_solution_5jackie(problem, current_solution)
        elif neighbourfunc_id == 6:
            s2 = alter_solution_6sebastian(problem, current_solution)
        elif neighbourfunc_id == 7:
            s2 = alter_solution_7steinar(problem, current_solution)
        elif neighbourfunc_id == 8:
            s2 = alter_solution_8stian(problem, current_solution)
        elif neighbourfunc_id == 9:
            s2 = alter_solution_9karina(problem, current_solution)

        feasiblity, _ = feasibility_check(s2, problem)

        if feasiblity:
            new_cost = cost_function(s2, problem)
            current_solution = s2
            found_new_feasible_solution = True

            if new_cost < best_sol_cost:
                return current_solution, new_cost, True

    # If we exit the loop without finding a feasible solution
    if not found_new_feasible_solution:
        return current_solution, cost_s, False
    
    return current_solution, new_cost, False

problem = load_problem2('Code/Data/Call_18_Vehicle_5.txt')
init_sol = initial_solution(problem)
solution, best_cost, improvement = adaptive_algorithm(problem, init_sol, 10000,[4, 5, 6, 7, 8, 9], file_num = None, statistics=False)
solution = [int(i) for i in solution]
print(solution)
print(best_cost)
print(improvement)