print("Starting adaptive algorithm")
start_time = time.time()
best_solution, best_cost, history, probability_history, weight_iteration_points = Adaptive_algorithm(
    problem=problem, initial_solution=initial_solution_generator(problem), max_iter=10000)
end_time = time.time()
best_solution = [int(x) for x in best_solution]
print(f"Final cost: {best_cost}, Time: {end_time - start_time:.2f}s")
print(f"Feasibility: {feasibility_check(best_solution, problem)[0]}")
print(f"Solution: {best_solution}")

# Plot operator weights
plot_operator_weights(probability_history, weight_iteration_points)