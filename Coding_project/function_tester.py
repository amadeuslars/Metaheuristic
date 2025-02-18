from Utils import *
import time
from Assignment2 import *
from Assignment3 import *
import csv  # For writing CSV files

filename = [
    "Data/Call_7_Vehicle_3.txt",
    "Data/Call_18_Vehicle_5.txt",
    "Data/Call_35_Vehicle_7.txt",
    "Data/Call_80_Vehicle_20.txt",
    "Data/Call_130_Vehicle_40.txt",
    "Data/Call_300_Vehicle_90.txt"
]


best_solutions = [[] for _ in range(len(filename))]
best_score = [[] for _ in range(len(filename))]
times = [[] for _ in range(len(filename))]
error_score = [[] for _ in range(len(filename))]

# Run loop for all files and record runtime, bestscore, errors
def run():
    print("Start")
    # Here, we loop through a subset of the files if needed.
    for j in range(len(filename)):
        problem = load_problem(filename[j])
        initial_solution = initial_solution_generator(problem)
        compatibility_list = precompute_compatibility(problem)
        print("initial_solution:", initial_solution)
        for i in range(1):
            start_time = time.time()
            # best_solution, best_cost, errors = blind_random_search(problem, compatibility_list, 10_000)
            best_solution, best_cost, errors = local_search(problem, initial_solution, 10_000)
            end_time = time.time()
            runtime = end_time - start_time
            best_solutions[j].append(best_solution)
            best_score[j].append(best_cost)   
            times[j].append(runtime)
            error_score[j].append(errors/10_000)
    [[float(value) for value in sublist] for sublist in best_score]

run()


def save_to_file():
    # Now save the data to a CSV file.
    output_filename = "results2.csv"
    with open(output_filename, "w", newline="") as csvfile:
        writer = csv.writer(csvfile)
        # Write header row.
        writer.writerow(["File", "Best Score", "Average Time", "Average Score", "Best Solution", "Errors"])
        # For each file, compute summary statistics and write a row.
        for i in range(len(best_score)):
            if not best_score[i]:
                continue  # Skip if no data for this file.
            avg_score = sum(best_score[i]) / len(best_score[i])
            avg_time = sum(times[i]) / len(times[i])
            min_score = min(best_score[i])
            # Find the index of the best score.
            best_index = best_score[i].index(min_score)
            best_solution = best_solutions[i][best_index]
            # Convert the best solution to a string for CSV storage.
            best_solution_str = str(best_solution)
            writer.writerow([filename[i], min_score, avg_time, avg_score, best_solution_str, error_score[i]])

save_to_file()
