from Utils import *
import time
from Assignment2 import *
from Assignment3 import *
from Assignment4 import *
import csv  # For writing CSV files
import os
from datetime import datetime
from Assignment5 import *

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
initial_solutions = [[] for _ in range(len(filename))]

# Run loop for all files and record runtime, bestscore, errors
def run(func):
    print("-" * 50)
    print("Start")
    print("-" * 50)
    
    # For each problem file
    for j in range(2):
        problem = load_problem(filename[j])
        initial_solution = initial_solution_generator(problem)
        initial_solutions[j].append(initial_solution)
        print(f'\nThe initial cost of file {filename[j]} is:', cost_function(initial_solution, problem),'\n')
        compatibility_table = precompute_compatibility(problem)
        
        # Run multiple times for this file
        for i in range(10):
            start_time = time.time()
            if func == "simulated_annealing":
                best_solution, best_cost = simulated_annealing2(problem, initial_solution, compatibility_table, 10_000)
            elif func == "blind_random_search":
                best_solution, best_cost = blind_random_search(problem, compatibility_list, 10_000)
            elif func == "local_search":
                best_solution, best_cost = local_search(problem, initial_solution, compatibility_table, 10_000)
            elif func == "Adaptive_algorithm":
                best_solution, best_cost, _, _, _ = Adaptive_algorithm(initial_solution, problem, 10_000)
            end_time = time.time()
            runtime = end_time - start_time
            
            best_solutions[j].append(best_solution)
            best_score[j].append(best_cost)   
            times[j].append(runtime)
        
        # Save after completing all runs for this file
        print(f"Completed all runs for {filename[j]}")
        save_to_file(func)
    [[float(value) for value in sublist] for sublist in best_score]


def save_to_file(algorithm_name="Unknown"):
    # Use a consistent filename
    output_filename = "results_history.csv"
    
    # Check if file exists to determine if we need to write headers
    file_exists = os.path.isfile(output_filename)
    
    # Get current date and time
    current_date = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    
    # Open file in append mode
    with open(output_filename, "a", newline="") as csvfile:
        writer = csv.writer(csvfile)
        
        # Write header row only if the file is new
        if not file_exists:
            writer.writerow([
                "Algorithm", "File", "Best Score", "Average Score", "Average Time", 
                "Best Solution", 'Improvement'
            ])
        
        # Add a separator row with the date
        writer.writerow([""])  # Blank row for spacing
        writer.writerow([f"--- NEW RUN: {current_date} ---"] + [""] * 9)  # Date separator with empty cells to fill row
        writer.writerow([""])  # Blank row for spacing
        
        # For each file, compute summary statistics and write a row
        for i in range(len(best_score)):
            if not best_score[i]:
                continue  # Skip if no data for this file
                
            # Calculate metrics
            num_rounds = len(best_score[i])
            avg_score = sum(best_score[i]) / num_rounds
            avg_time = round(sum(times[i]) / num_rounds, 3)
            min_score = min(best_score[i])
            improvement = (min_score - cost_function(initial_solutions[i][0], load_problem(filename[i]))) / cost_function(initial_solutions[i][0], load_problem(filename[i]))
            
            # Find the index of the best score
            best_index = best_score[i].index(min_score)
            best_solution = best_solutions[i][best_index]
            
            # Convert the best solution to a string for CSV storage
            best_solution_str = str(best_solution)
            
            # Write row with all the information
            writer.writerow([
                algorithm_name,
                filename[i],
                min_score,
                avg_score,
                avg_time,
                best_solution_str,
                improvement
            ])
            
            # Add a blank row after each file's data for spacing
            writer.writerow([""])
        
        # Add a final separator row
        writer.writerow([""] * 10)  # Full row of empty cells for spacing
    
    print(f"Results appended to {output_filename}")

algorithm_name = "Adaptive_algorithm"  # Change this to the name of the algorithm
run(algorithm_name)
