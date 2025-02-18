from Utils import *
from joblib import Parallel, delayed

# 1-reinsertion function
import random

# To test the reinsert function, we need a problem and an initial solution.
problem = load_problem("Data/Call_300_Vehicle_90.txt")

def reinsert(solution):
    """
    Given a solution list like:
    [1, 1, 0, 2, 3, 2, 3, 0, 4, 4, 0, 7, 7]
    where each nonzero number represents a call (the first occurrence is the pickup
    and the second is the delivery) and 0 marks the beginning of a new car's route,
    this function randomly removes one call pair from one car and reinserts it
    into a different car's route at two random positions (pickup comes before delivery).
    """
    # ---------------------------
    # 1. Parse the solution into segments (car routes)
    segments = []
    current_segment = []
    for call in solution:
        if call == 0:
            segments.append(current_segment)
            current_segment = []
        else:
            current_segment.append(call)
    segments.append(current_segment)  # Append the last segment

    # ---------------------------
    # 2. Identify all call pairs by scanning each segment.
    #    We expect a call (nonzero number) to appear exactly twice in a segment.
    valid_calls = []  # each element is a tuple (segment_index, call)
    for seg_idx, seg in enumerate(segments):
        counts = {}
        for call in seg:
            counts[call] = counts.get(call, 0) + 1
        for call, count in counts.items():
            if count == 2:
                valid_calls.append((seg_idx, call))
    
    # If no valid call pair is found, return the original solution.
    if not valid_calls:
        print("No valid call pair found to move.")
        return solution

    # ---------------------------
    # 3. Randomly choose one call pair from the valid ones and remove the call from the vehicle
    src_segment_index, call_to_move = random.choice(valid_calls)
    src_seg = segments[src_segment_index]
    src_seg = [x for x in src_seg if x != call_to_move]
    segments[src_segment_index] = src_seg
    

    # ---------------------------
    # 4. Choose a destination segment that is different from the source.
    possible_destinations = list(range(len(segments)))
    dest_segment_index = random.choice(possible_destinations)
    dest_seg = segments[dest_segment_index]


    # ---------------------------
    # 5. Choose two random insertion positions in the destination segment.
    # There are len(dest_seg)+1 possible insertion positions.
    L = len(dest_seg)
    # Pick two distinct positions from the available insertion slots.
    if L == 0:
        insertion_positions = [0, 0]
        pickup_index, delivery_index = insertion_positions
        # To avoid index shifting issues, insert in descending order.
        dest_seg.insert(delivery_index, call_to_move)  # delivery
        dest_seg.insert(pickup_index, call_to_move)      # pickup
    else:
        insertion_positions = random.sample(range(L + 1), 2)
        insertion_positions.sort()  # first will be for the pickup, second for the delivery.
        pickup_index, delivery_index = insertion_positions

        # To avoid index shifting issues, insert in descending order.
        dest_seg.insert(delivery_index, call_to_move)  # delivery
        dest_seg.insert(pickup_index, call_to_move)      # pickup

    segments[dest_segment_index] = dest_seg

    # ---------------------------
    # 6. Reassemble the global solution list.
    # Insert a 0 between each segment.
    new_solution = []
    for i, seg in enumerate(segments):
        new_solution.extend(seg)
        if i < len(segments) - 1:
            new_solution.append(0)
    
    return new_solution

# def reinsert(solution):
#     """
#     Apply the 1-reinsert heuristic: Select a random pickup-delivery pair,
#     remove it from its current vehicle, and reinsert it into a different vehicle.
#     """
#     # Find the vehicles by splitting at zeros (vehicle boundaries)
#     vehicles = []
#     current_vehicle = []
    
#     for item in solution:
#         if item == 0:
#             vehicles.append(current_vehicle)
#             current_vehicle = []
#         else:
#             current_vehicle.append(item)
    
#     if current_vehicle:
#         vehicles.append(current_vehicle)
    
#     # Randomly select a vehicle and a random pair (pickup and its corresponding delivery)
#     vehicle_idx = random.randint(0, len(vehicles) - 1)
#     vehicle = vehicles[vehicle_idx]
    
#     # Make sure the vehicle has at least one pair to reinsert
#     if len(vehicle) < 2:
#         return solution
    
#     # Randomly pick a pickup-delivery pair (pickup and its corresponding delivery)
#     pair_idx = random.randint(0, len(vehicle) // 2 - 1)
#     pickup = vehicle[2 * pair_idx]
#     delivery = vehicle[2 * pair_idx + 1]
    
#     # Remove the pickup-delivery pair from the vehicle
#     vehicle.pop(2 * pair_idx)  # Remove pickup
#     vehicle.pop(2 * pair_idx)  # Remove delivery
    
#     # Randomly choose a new vehicle (must be a different vehicle)
#     new_vehicle_idx = random.randint(0, len(vehicles) - 1)
#     while new_vehicle_idx == vehicle_idx:
#         new_vehicle_idx = random.randint(0, len(vehicles) - 1)
    
#     new_vehicle = vehicles[new_vehicle_idx]
    
#     new_position = random.randint(0, len(new_vehicle))  # Pick a random position
#     new_vehicle.insert(new_position, pickup)
#     random_delivery = random.randint(new_position + 1, len(new_vehicle))
#     new_vehicle.insert(random_delivery, delivery)
    
#     # Rebuild the solution with the modified vehicles
#     modified_solution = []
#     for i, v in enumerate(vehicles):
#         modified_solution.extend(v)
#         if i < len(vehicles) - 1:
#             modified_solution.append(0)  # Add boundary between vehicles
    
#     return modified_solution

def local_search(problem, initial_solution, max_iter):
    print(initial_solution)
    errors = 0
    cost = cost_function(initial_solution, problem)
    best_solution = initial_solution
    

    for i in range(max_iter):
        new_solution = reinsert(best_solution)
        if feasibility_check(new_solution, problem)[0]:
            new_cost = cost_function(new_solution, problem)
            if new_cost < cost:
                best_solution = new_solution
                cost = new_cost
                print("Best solution:",best_solution, 'best cost:', cost)
        else:
            errors += 1

    return best_solution, cost, errors

# best_solution, cost, errors = local_search(problem, initial_solution_generator(problem), 10000)
# print(best_solution, cost, errors)


# Simmulated annealing function
def simulated_annealing(problem, initial_solution, max_iter):
    Tf = 0.1
    incumbent = initial_solution
    best_solution = initial_solution
    w = []
    for _ in range(100):
        new_solution = reinsert(incumbent)
        E = cost_function(new_solution, problem) - cost_function(incumbent, problem)
        if feasibility_check(new_solution, problem)[0]:
            if E < 0:
                incumbent = new_solution
                if cost_function(incumbent, problem) < cost_function(best_solution, problem):
                    best_solution = incumbent
            else:
                p = 0.8
                if np.random.rand() < p:
                    incumbent = new_solution
                w.append(E)
    
    delta_avg = np.mean(w)
    T_0 = -delta_avg / np.log(0.8)
    alfa = (Tf / T_0) ** (1 / 9900)
    T = T_0
    for _ in range(100, max_iter):
        new_solution = reinsert(incumbent)
        E = cost_function(new_solution, problem) - cost_function(incumbent, problem)
        if feasibility_check(new_solution, problem)[0]:
            if E < 0:
                incumbent = new_solution
                if cost_function(incumbent, problem) < cost_function(best_solution, problem):
                    best_solution = incumbent
            else:
                p = np.exp(-E / T)
                if np.random.rand() < p:
                    incumbent = new_solution
        T = alfa * T
    return best_solution, cost_function(best_solution, problem)

# best_solution, cost = simulated_annealing(problem, initial_solution_generator(problem), 10000)
# print(best_solution, cost)