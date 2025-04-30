import numpy as np
import random
from early_code.Utils import *
import time
from copy import deepcopy
import matplotlib.pyplot as plt
from copy import deepcopy   

# Global variables for normalization factors
max_travel_cost = 0
biggest_time_window_diff = 0
biggest_cargo_size_diff = 0

# Load problem and precompute compatibility
problem_file = "Data/Call_18_Vehicle_5.txt"
problem = load_problem(problem_file)
comatibility_table = precompute_compatibility(problem)
route_cost_cache = {}

# Create a function that claculates for the problem, all the relatedness factors
# These factors are problem specific, so when the related removal is called
# one can just look up for the call chosen, the relatedness factors 


def calculate_normalization_factors(problem):
    global max_travel_cost, biggest_time_window_diff, biggest_cargo_size_diff
    
    n_vehicles = problem['n_vehicles']
    n_calls = problem['n_calls']
    cargo = problem['Cargo']
    travel_cost = problem['TravelCost']
    
    # 1. Maximum Travel Cost
    max_travel_cost = 0
    for v in range(n_vehicles):
        vehicle_max = np.max(travel_cost[v])
        if vehicle_max > max_travel_cost:
            max_travel_cost = vehicle_max
    
    # 2. Biggest Time Window Difference
    pickup_early = cargo[:, 4]
    pickup_late = cargo[:, 5]
    delivery_early = cargo[:, 6]
    delivery_late = cargo[:, 7]
    
    # Maximum possible difference for pickup and delivery time windows
    max_pickup_diff = 0
    max_delivery_diff = 0
    
    for i in range(n_calls):
        for j in range(n_calls):
            if i != j:
                pickup_diff = abs(pickup_early[i] - pickup_early[j]) + abs(pickup_late[i] - pickup_late[j])
                delivery_diff = abs(delivery_early[i] - delivery_early[j]) + abs(delivery_late[i] - delivery_late[j])
                
                max_pickup_diff = max(max_pickup_diff, pickup_diff)
                max_delivery_diff = max(max_delivery_diff, delivery_diff)
    
    biggest_time_window_diff = max(max_pickup_diff, max_delivery_diff)
    
    # 3. Biggest Cargo Size Difference
    cargo_sizes = cargo[:, 2]
    biggest_cargo_size_diff = 0
    
    for i in range(n_calls):
        for j in range(n_calls):
            if i != j:
                size_diff = abs(cargo_sizes[i] - cargo_sizes[j])
                biggest_cargo_size_diff = max(biggest_cargo_size_diff, size_diff)
    
    # print(f"Normalization factors calculated:")
    # print(f"- Max travel cost: {max_travel_cost}")
    # print(f"- Biggest time window diff: {biggest_time_window_diff}")
    # print(f"- Biggest cargo size diff: {biggest_cargo_size_diff}")
    
    return max_travel_cost, biggest_time_window_diff, biggest_cargo_size_diff

max_travel_cost, biggest_time_window_diff, biggest_cargo_size_diff = calculate_normalization_factors(problem)

def get_valid_calls(vehicles):
    valid_calls = []
    
    for v_idx, vehicle in enumerate(vehicles):
        call_counts = {}
        for call in vehicle:
            call_counts[call] = call_counts.get(call, 0) + 1
        
        for call, count in call_counts.items():
            if count == 2:  
                valid_calls.append((v_idx, call))
    
    return valid_calls

# def cost(vehicles, problem):
#     total_cost = 0
#     for v_idx, vehicle in enumerate(vehicles):
#         total_cost += simplified_cost_function(vehicle, v_idx, problem)
#     return total_cost

def cost(vehicles, problem):
    return cost_function(reassemble_solution(vehicles), problem)

# def feasibility(vehicles, problem):
#     for v_idx, vehicle in enumerate(vehicles):
#         if feasibility_check_vehicle(vehicle, v_idx, problem)[0] == False:
#             return False
#     return True

def feasibility(vehicles, problem):
    return feasibility_check(reassemble_solution(vehicles), problem)[0]

def parse_solution_to_vehicles(solution):
    vehicles = []
    current_vehicle = []
    
    for call in solution:
        if call == 0:
            vehicles.append(current_vehicle)
            current_vehicle = []
        else:
            current_vehicle.append(call)
            
    vehicles.append(current_vehicle)  # Append the last segment (dummy vehicle)
    return vehicles

def reassemble_solution(vehicles):
    solution = []
    for i, vehicle in enumerate(vehicles):
        solution.extend(vehicle)
        if i < len(vehicles) - 1:  # Don't add a zero after the last vehicle
            solution.append(0)
    return solution

def update_operator_weights(operators):
    r = 0.7
    for op in operators:
        # Calculate base success rate
        successes = op['successes']
        success_bonus = 1 * successes
        
        # Calculate best solution bonus (if tracked)
        best_solutions_found = op.get('best_improvements', 0)
        best_solution_bonus = 4 * best_solutions_found 

        # imporvement bonus
        improvements = op.get('improvements', 0)
        imporvements_bonus = 2 * improvements

        pi = (
            success_bonus +                         
            best_solution_bonus + 
            imporvements_bonus
        )
        
        op_old_weight = op['weight']
        new_weight = max(0.1, op_old_weight *(1-r) + r * pi/max(1, op['attempts']))
        op['weight'] =  new_weight
        
        # Reset statistics for next period
        op['successes'] = 0
        op['attempts'] = 1  # Start with 1 to avoid division by zero
        op['total_improvement'] = 0
        op['improvements'] = 0
        op['best_improvements'] = 0  # Reset best solution counter
        op['new_solutions'] = 0  # Reset new solution counter

def simplified_cost_function(vehicle, v_idx, problem):
    n_vehicles = problem['n_vehicles']
    

    if v_idx == n_vehicles:
        # Dummy vehicle - penalty costs for unserved calls
        segment_cost = 0
        for call in sorted(set(vehicle)):
            segment_cost += problem['Cargo'][call - 1][3]
        return segment_cost
    
    segment_cost = 0
    if not vehicle:
        return 0

    # unique calls and their count to ensure we're processing pickup/delivery pairs
    unique_calls = {}
    for call in vehicle:
        unique_calls[call] = unique_calls.get(call, 0) + 1
    
    valid_vehicle_calls = [call for call, count in unique_calls.items() if count == 2]
    
    if not valid_vehicle_calls:
        return 0

    # Cost function logic
    call_indices = np.array([i for i in range(len(vehicle)) if vehicle[i] in valid_vehicle_calls]) 
    calls = np.array([vehicle[i] for i in call_indices]) - 1  # Convert to 0-indexed
    
    # Sorting the calls for proper processing
    sortRout = np.sort(calls, kind='mergesort')
    I = np.argsort(calls, kind='mergesort')
    Indx = np.argsort(I, kind='mergesort')
    
    # Getting port indices (pickup and delivery locations)
    PortIndex = problem['Cargo'][sortRout, 1].astype(int)
    PortIndex[::2] = problem['Cargo'][sortRout[::2], 0]
    PortIndex = PortIndex[Indx] - 1
    
    # First travel cost from vehicle origin to first pickup
    if len(PortIndex) > 0:
        first_call = calls[0]
        FirstVisitCost = problem['FirstTravelCost'][v_idx][int(problem['Cargo'][first_call, 0] - 1)]
        segment_cost += FirstVisitCost
        
        # Travel costs between consecutive nodes
        if len(PortIndex) > 1:
            for i in range(len(PortIndex) - 1):
                segment_cost += problem['TravelCost'][v_idx][PortIndex[i]][PortIndex[i+1]]
        
        # Port costs (loading/unloading)
        for call in valid_vehicle_calls:
            segment_cost += problem['PortCost'][v_idx][call - 1]

    return segment_cost

def remove_calls(vehicles, calls_to_remove):
    """Remove specified calls from vehicles and return list of removed calls"""
    removed_calls = []
    for v_idx, call in calls_to_remove:
        vehicles[v_idx] = [c for c in vehicles[v_idx] if c != call]
        removed_calls.append(call)
    return removed_calls

def build_call_info(removed_calls, compatibility_table, n_vehicles):
    """Build call info with compatible vehicles for each call"""
    call_info = []
    for call in removed_calls:
        compatible_vehicles = []
        for v in range(n_vehicles + 1):
            if compatibility_table[call - 1, v] == 1:
                compatible_vehicles.append(v)
        call_info.append((call, compatible_vehicles))
    return call_info

def calculate_tw_overlap(tw1, tw2):
    """Calculate time window overlap as a percentage"""
    early1, late1 = tw1
    early2, late2 = tw2
    
    # No overlap
    if late1 < early2 or late2 < early1:
        return 0
    
    # Calculate overlap
    overlap = min(late1, late2) - max(early1, early2)
    span1 = late1 - early1
    span2 = late2 - early2
    
    # Normalize by the smallest time window
    return overlap / min(span1, span2) if min(span1, span2) > 0 else 0

def calculate_tardiness(vehicles, problem):
    n_vehicles = problem['n_vehicles']
    call_tardiness = {}  # Maps call_id to tardiness value
    
    for v_idx in range(n_vehicles):  
        vehicle = vehicles[v_idx]
        if not vehicle:
            continue
            
        call_counts = {}
        for call in vehicle:
            call_counts[call] = call_counts.get(call, 0) + 1
        
        valid_calls = [call for call, count in call_counts.items() if count == 2]
        if not valid_calls:
            continue
            
        valid_indices = [i for i, call in enumerate(vehicle) if call in valid_calls]
        valid_route = [vehicle[i] for i in valid_indices]
        
        
        current_time = 0
        call_positions = {}
        
        for pos, call in enumerate(valid_route):
            if call not in call_positions:
                # First position is pickup
                call_positions[call] = [pos]
            else:
                # Second position is delivery
                call_positions[call].append(pos)
        
        sorted_calls = np.array([call for call in valid_calls]) - 1  # Convert to 0-indexed
        
        # port indices
        port_indices = []
        for i, call in enumerate(valid_route):
            call_idx = call - 1  # Converting to 0-indexed
            if i == call_positions[call][0]:  # Pickup
                port_indices.append(problem['Cargo'][call_idx, 0] - 1)  # Origin port
            else:  # Delivery
                port_indices.append(problem['Cargo'][call_idx, 1] - 1)  # Destination port
        
        # travel times
        if port_indices:
            # First leg from vehicle start to first port
            first_call = valid_route[0] - 1  # 0-indexed
            first_port = problem['Cargo'][first_call, 0] - 1  # Origin port of first call (0-indexed) 
            current_time = problem['FirstTravelTime'][v_idx, first_port]
            
            for i in range(len(valid_route)):
                call = valid_route[i] - 1  # 0-indexed
                is_pickup = (i == call_positions[valid_route[i]][0])
                
                # time window
                if is_pickup:
                    earliest = problem['Cargo'][call, 4]
                    latest = problem['Cargo'][call, 5]
                    load_time = problem['LoadingTime'][v_idx, call]
                else:
                    earliest = problem['Cargo'][call, 6]
                    latest = problem['Cargo'][call, 7]
                    load_time = problem['UnloadingTime'][v_idx, call]
                
                # arrive time
                arrive_time = max(current_time, earliest)
                
                # tardiness
                tardiness = max(0, arrive_time - latest)
                
                # Updating record of worst tardiness for this call
                if call+1 not in call_tardiness or tardiness > call_tardiness[call+1]:
                    call_tardiness[call+1] = tardiness
                
                # Updating current time for next node
                current_time = arrive_time + load_time
                
                # Adding travel time to next node if not last
                if i < len(valid_route) - 1:
                    next_call = valid_route[i+1] - 1
                    next_is_pickup = (i+1 == call_positions[valid_route[i+1]][0])
                    
                    if next_is_pickup:
                        next_port = problem['Cargo'][next_call, 0] - 1
                    else:
                        next_port = problem['Cargo'][next_call, 1] - 1
                        
                    current_time += problem['TravelTime'][v_idx, port_indices[i], next_port]
    
    return call_tardiness

def escape(current, best_solution, best_cost, problem, compatibility_table):
    escape_solution = [vehicle.copy() for vehicle in current]
    
    updated_best_solution = best_solution
    updated_best_cost = best_cost
    found_new_best = False
    
    
    # Try up to 20 different perturbations
    attempts = 0
    
    while attempts < 15: 
        attempts += 1
       
        strategy = random.choice([
            random_greedy 
            ])
        
        k = random.randint(round(0.4 * problem['n_calls']), round(0.7 * problem['n_calls']))
         
        perturbed_solution = strategy(escape_solution, problem, compatibility_table, k)
        
        # Check feasibility
        is_feasible = feasibility(perturbed_solution, problem)
        if is_feasible:
            perturbed_cost = cost(perturbed_solution, problem)
            
            # Always accept the feasible solution for future escape iterations
            escape_solution = [vehicle.copy() for vehicle in perturbed_solution]
            # print(f"Escaped to new solution with cost {perturbed_cost} using {selected_op['name']}")
            
            # Check if we found a new global best
            if perturbed_cost < updated_best_cost:
                # Create a deep copy to prevent modification
                updated_best_solution = [vehicle.copy() for vehicle in perturbed_solution]
                updated_best_cost = perturbed_cost
                found_new_best = True
        else:
            escape_solution = [vehicle.copy() for vehicle in current]
    # Also deep copy the escape solution before returning
    
    return escape_solution, updated_best_solution, updated_best_cost, found_new_best

# removal functions
def worst_removal(vehicles, problem, compatibility_table, k):
    n_vehicles = problem['n_vehicles']
    valid_calls = get_valid_calls(vehicles)
    flattening_factor = 0.5

    if not valid_calls:
        return vehicles, []
    
    # Calculate the cost contribution of each call
    call_costs = []
    
    for v_idx, call in valid_calls:
        # Skip dummy vehicle calls for cost calculation (they're just penalties)
        if v_idx == n_vehicles:
            # Use the penalty cost directly
            penalty_cost = problem['Cargo'][call-1][3]
            call_costs.append((v_idx, call, penalty_cost))
            continue
        
        # Calculate cost with the call
        vehicle = vehicles[v_idx]
        current_cost = simplified_cost_function(vehicle, v_idx, problem)
        
        # Calculate cost without the call
        vehicle_without_call = [c for c in vehicle if c != call]
        cost_without_call = simplified_cost_function(vehicle_without_call, v_idx, problem)
        
        # The cost contribution is the difference
        cost_contribution = current_cost - cost_without_call
        
        # If removing a call improves cost, prioritize it highly
        if cost_contribution <= 0:
            # Small positive value to avoid negative weights in selection
            cost_contribution = 0.1
            
        call_costs.append((v_idx, call, cost_contribution))
    
    # Sort by cost contribution (highest first)
    call_costs.sort(key=lambda x: x[2], reverse=True)
    
    # Calculate selection probabilities using exponential weighting
    total_cost = sum(cost for _, _, cost in call_costs)
    if total_cost > 0:
        noisy_costs = [cost * (1 + random.uniform(-flattening_factor, flattening_factor)) 
                  for _, _, cost in call_costs]
        weights = [cost/sum(noisy_costs) for cost in noisy_costs]
        probs = [w/sum(weights) for w in weights]
    else:
        # Equal probabilities if all costs are 0
        probs = [1.0/len(call_costs)] * len(call_costs)
    
    # Use weighted random selection to choose k calls
    k = min(k, len(valid_calls))
    indices_to_remove = np.random.choice(
        len(call_costs), 
        size=k, 
        replace=False, 
        p=probs
    )
    
    calls_to_remove = [call_costs[i][:2] for i in indices_to_remove]
    
    # Remove the selected calls
    removed_calls = remove_calls(vehicles, calls_to_remove)
    
    # Build call info for removed calls
    call_info = build_call_info(removed_calls, compatibility_table, n_vehicles)
    
    return vehicles, call_info

def related_removal(vehicles, problem, compatibility_table, k):
    n_vehicles = problem['n_vehicles']

    valid_calls = get_valid_calls(vehicles)

    if not valid_calls:
        return vehicles, []
    
    seed_idx = random.randint(0, len(valid_calls) - 1)
    seed_v_idx, seed_call = valid_calls[seed_idx]

    # Seed vehicle data
    seed_data = problem['Cargo'][seed_call - 1]
    seed_pickup = seed_data[0] - 1
    seed_delivery = seed_data[1] - 1

    # Time Window data
    seed_pickup_tw_early = problem['Cargo'][seed_call -1, 4]
    seed_pickup_tw_late = problem['Cargo'][seed_call - 1, 5]
    seed_delivery_tw_early = problem['Cargo'][seed_call - 1, 6]
    seed_delivery_tw_late = problem['Cargo'][seed_call - 1, 7]

    # Cargo size
    seed_cargo_size = seed_data[2]

    # Calculate relatedness
    relatedness = []
    for v_idx, vehicle in enumerate(vehicles):
        call_counts = {}
        for call in vehicle:
            call_counts[call] = call_counts.get(call, 0) + 1
        
        for call, count in call_counts.items():
            if count == 2 and (v_idx, call) != (seed_v_idx, seed_call):
                call_data = problem['Cargo'][call - 1]
                call_pickup = int(call_data[0] - 1)
                call_delivery = int(call_data[1] - 1)

                # Travel Cost similarity
                travel_cost = problem['TravelCost'][0][seed_pickup][call_pickup] # First vehicles cost as reference

                # Time Window similarity for pickup
                call_pickup_tw_early = call_data[4]
                call_pickup_tw_late = call_data[5]
                pickup_tw_diff = abs(seed_pickup_tw_early - call_pickup_tw_early) + abs(seed_pickup_tw_late - call_pickup_tw_late)

                # Time Window similarity for delivery
                call_delivery_tw_early = call_data[6]
                call_delivery_tw_late = call_data[7]
                delivery_tw_diff = abs(seed_delivery_tw_early - call_delivery_tw_early) + abs(seed_delivery_tw_late - call_delivery_tw_late)

                # Cargo size similarity
                call_cargo_size = call_data[2]
                cargo_size_diff = abs(seed_cargo_size - call_cargo_size)

                # Normalizing the factors
                norm_travel_cost = min(1.0, travel_cost / max_travel_cost)
                max_tw_diff = biggest_time_window_diff
                norm_pickup_tw = min(1.0, pickup_tw_diff / max_tw_diff)
                norm_delivery_tw = min(1.0, delivery_tw_diff / max_tw_diff)
                max_cargo_size_diff = biggest_cargo_size_diff
                norm_cargo_size = min(1.0, cargo_size_diff / max_cargo_size_diff)

                # Weighted sum
                relatedness_score = (
                    0.4 * norm_travel_cost + 
                    0.25 * norm_pickup_tw + 
                    0.25 * norm_delivery_tw + 
                    0.1 * norm_cargo_size
                )

                relatedness.append((v_idx, call, relatedness_score))

    # NEW: Weighted probabilistic selection based on relatedness scores
    if not relatedness:
        # If no other calls found, just return the seed call
        calls_to_remove = [(seed_v_idx, seed_call)]
    else:
        # Sort by relatedness score (lower score = more related)            
        relatedness.sort(key=lambda x: x[2])
        
        # Number of calls to select (excluding seed call)
        num_to_select = min(k-1, len(relatedness))
        
        if num_to_select > 0:
            # Extract scores and invert them (since lower score is better)
            scores = [1.0 - rel[2] for rel in relatedness]
            
            # Normalize scores to sum to 1
            total_score = sum(scores)
            if total_score > 0:
                weights = [score/total_score for score in scores]
                
                # Apply exponential weighting to emphasize differences
                exp_weights = [np.exp(2*w) for w in weights]
                total_exp_weight = sum(exp_weights)
                probs = [w/total_exp_weight for w in exp_weights]
                
                # Select calls using weighted random selection
                indices = np.random.choice(
                    len(relatedness), 
                    size=num_to_select,
                    replace=False, 
                    p=probs
                )
                
                # Create calls to remove list (seed call + selected related calls)
                related_calls = [(relatedness[i][0], relatedness[i][1]) for i in indices]
                calls_to_remove = [(seed_v_idx, seed_call)] + related_calls
            else:
                # Fallback to random selection if all scores are 0
                indices = np.random.choice(len(relatedness), size=num_to_select, replace=False)
                related_calls = [(relatedness[i][0], relatedness[i][1]) for i in indices]
                calls_to_remove = [(seed_v_idx, seed_call)] + related_calls
        else:
            # Just the seed call
            calls_to_remove = [(seed_v_idx, seed_call)]
    
    # Remove the selected calls
    removed_calls = remove_calls(vehicles, calls_to_remove)

    # Get compatible vehicles for each call
    call_info = build_call_info(removed_calls, compatibility_table, n_vehicles)

    return vehicles, call_info

def random_removal(vehicles, problem, compatibility_table, k):
    n_vehicles = problem['n_vehicles']
    valid_calls = get_valid_calls(vehicles)
    
    if not valid_calls:
        return vehicles, []
    
    # Separate calls by vehicle type (dummy vs regular)
    dummy_calls = [(v_idx, call) for v_idx, call in valid_calls if v_idx == n_vehicles]
    regular_calls = [(v_idx, call) for v_idx, call in valid_calls if v_idx != n_vehicles]
    
    # Determine how many calls to remove
    k = min(k, len(valid_calls))
    calls_to_remove = []
    
    # If there are calls in the dummy vehicle
    if dummy_calls:
        # Higher probability (e.g., 70%) to remove from dummy if available
        dummy_probability = 0.7
        
        for _ in range(k):
            if dummy_calls and (random.random() < dummy_probability or not regular_calls):
                # Select from dummy vehicle
                selected_call = random.choice(dummy_calls)
                dummy_calls.remove(selected_call)
                calls_to_remove.append(selected_call)
            elif regular_calls:
                # Select from regular vehicles
                selected_call = random.choice(regular_calls)
                regular_calls.remove(selected_call)
                calls_to_remove.append(selected_call)
            else:
                # Both lists are empty, break
                break
    else:
        # No calls in dummy, use standard random selection
        calls_to_remove = random.sample(valid_calls, k)
    
    # Remove the selected calls
    removed_calls = remove_calls(vehicles, calls_to_remove)
    
    # Build call info for removed calls
    call_info = build_call_info(removed_calls, compatibility_table, n_vehicles)
    
    return vehicles, call_info

def dummy_removal(vehicles, problem, compatibility_table, k):
    n_vehicles = problem['n_vehicles']

    valid_calls = get_valid_calls(vehicles)

    if not valid_calls:
        return vehicles, []
    
    num_calls_in_dummy = len([call for _, call in valid_calls if _ == n_vehicles])
    k = min(k, num_calls_in_dummy)
    
    # Remove calls from the dummy vehicle
    dummy_calls = [(n_vehicles, call) for _, call in valid_calls if _ == n_vehicles]
    calls_to_remove = random.sample(dummy_calls, k)

    removed_calls = remove_calls(vehicles, calls_to_remove)

    call_info = build_call_info(removed_calls, compatibility_table, n_vehicles)

    return vehicles, call_info

def tardiness_removal(vehicles, problem, compatibility_table, k):
   
    n_vehicles = problem['n_vehicles']
    
    # Get valid calls (only those assigned to real vehicles, not dummy)
    valid_calls = []
    for v_idx, vehicle in enumerate(vehicles):
        if v_idx == n_vehicles:  # Skip dummy vehicle
            continue
            
        call_counts = {}
        for call in vehicle:
            call_counts[call] = call_counts.get(call, 0) + 1
            
        for call, count in call_counts.items():
            if count == 2:  # Only consider complete pickup-delivery pairs
                valid_calls.append((v_idx, call))
    
    if not valid_calls:
        return vehicles, []
    
    # Calculate tardiness for each call
    call_tardiness = calculate_tardiness(vehicles, problem)
    
    # If no tardiness found, use random removal instead
    if not call_tardiness or all(t == 0 for t in call_tardiness.values()):
        return random_removal(vehicles, problem, compatibility_table, k)
    
    # Create tardiness-based weights for calls in regular vehicles
    tardiness_weights = []
    calls_to_weight = []
    
    for v_idx, call in valid_calls:
        if call in call_tardiness:
            tardiness = call_tardiness[call]
            if tardiness > 0:
                tardiness_weights.append(tardiness)
                calls_to_weight.append((v_idx, call))
    
    # If no calls with tardiness, use random removal
    if not tardiness_weights:
        return random_removal(vehicles, problem, compatibility_table, k)
    
    # Normalize and apply exponential weighting to create probability distribution
    max_tardiness = max(tardiness_weights)
    norm_weights = [w/max_tardiness for w in tardiness_weights]
    
    # Apply exponential weighting to emphasize differences
    exp_weights = [np.exp(3*w) for w in norm_weights]  # Adjust the multiplier (3) to change emphasis
    
    # Create probability distribution
    total_weight = sum(exp_weights)
    probs = [w/total_weight for w in exp_weights]
    
    # Select k calls based on the probability distribution
    k = min(k, len(calls_to_weight))
    selected_indices = np.random.choice(
        len(calls_to_weight), 
        size=k, 
        replace=False, 
        p=probs
    )
    
    calls_to_remove = [calls_to_weight[i] for i in selected_indices]
    
    # Remove the selected calls
    removed_calls = remove_calls(vehicles, calls_to_remove)
    
    # Build call info for removed calls
    call_info = build_call_info(removed_calls, compatibility_table, n_vehicles)
    
    return vehicles, call_info

def remove_long_wait_calls(solution, problem, wait_threshold=60):

    
    num_vehicles = problem['n_vehicles']
    Cargo = problem['Cargo']
    TravelTime = problem['TravelTime']
    FirstTravelTime = problem['FirstTravelTime']
    VesselCapacity = problem['VesselCapacity']
    LoadingTime = problem['LoadingTime']
    UnloadingTime = problem['UnloadingTime']
    
    # Make a copy of the solution to modify
    new_solution = solution.copy()
    print("New solution before processing:", new_solution)
    
    # Ensure the solution retains all zeros and ends with a zero
    if new_solution[-1] != 0:
        new_solution = np.append(new_solution, 0)
    print("New solution after ensuring ending zero:", new_solution)
    
    # Find indices of zeros (vehicle separators)
    zero_indices = np.array(np.where(new_solution == 0)[0], dtype=int)
    print("Zero indices:", zero_indices)
    
    # List to store calls to remove
    calls_to_remove = []
    print("Calls to remove:", calls_to_remove)
    
    # Process each vehicle's route
    temp_idx = 0
    for i in range(num_vehicles):
        print(f"Processing vehicle {i + 1}:")

        # Extract the current vehicle's plan
        if i < len(zero_indices) - 1:
            current_plan = new_solution[temp_idx:zero_indices[i + 1]]
        else:
            current_plan = new_solution[temp_idx:]  # Handle case where zero_indices[i + 1] is out of bounds
        print("Current plan before processing:", current_plan)

        # Skip empty plans
        if len(current_plan) == 0:
            temp_idx = zero_indices[i + 1] + 1 if i < len(zero_indices) - 1 else len(new_solution)
            continue

        current_plan = current_plan - 1  # Adjust indices
        print("Current plan after adjusting indices:", current_plan)

        temp_idx = zero_indices[i + 1] + 1 if i < len(zero_indices) - 1 else len(new_solution)
        
        if len(current_plan) > 0:
            # Calculate waiting times for this vehicle
            current_time = 0
            
            # Sort the route for processing
            sort_route = np.sort(current_plan, kind='mergesort')
            idx = np.argsort(current_plan, kind='mergesort')
            indx = np.argsort(idx, kind='mergesort')
            
            # Get port indices for each call
            port_indices = Cargo[sort_route, 1].astype(int)
            port_indices[::2] = Cargo[sort_route[::2], 0]
            port_indices = port_indices[indx] - 1
            
            # Get time windows for each call
            time_windows = np.zeros((2, len(current_plan)))
            time_windows[0] = Cargo[sort_route, 6]  # Lower bound
            time_windows[0, ::2] = Cargo[sort_route[::2], 4]  # Pickup lower bounds
            time_windows[1] = Cargo[sort_route, 7]  # Upper bound
            time_windows[1, ::2] = Cargo[sort_route[::2], 5]  # Pickup upper bounds
            time_windows = time_windows[:, indx]
            
            # Get loading/unloading times
            lu_time = UnloadingTime[i, sort_route]
            lu_time[::2] = LoadingTime[i, sort_route[::2]]
            lu_time = lu_time[indx]
            
            # Calculate travel times
            diag = TravelTime[i, port_indices[:-1], port_indices[1:]] if len(port_indices) > 1 else np.array([])
            first_visit_time = FirstTravelTime[i, int(Cargo[current_plan[0], 0] - 1)]
            route_travel_time = np.hstack((first_visit_time, diag.flatten()))
            
            # Calculate arrival and waiting times
            for j in range(len(current_plan)):
                # Arrival time at next location
                arrival_time = current_time + route_travel_time[j]
                # Calculate waiting time
                waiting_time = max(0, time_windows[0, j] - arrival_time)
                
                # If waiting time exceeds threshold, mark call for removal
                if waiting_time > wait_threshold:
                    # Convert back to original call index (add 1)
                    call_to_remove = current_plan[j] + 1
                    calls_to_remove.append(call_to_remove)
                
                # Update current time
                service_start_time = max(arrival_time, time_windows[0, j])
                current_time = service_start_time + lu_time[j]
    
    # Remove marked calls from the solution
    for call in calls_to_remove:
        # Remove both pickup and delivery for this call
        new_solution = new_solution[new_solution != call]
    
    # Ensure the solution still has vehicle separators
    if len(new_solution) == 0 or new_solution[-1] != 0:
        new_solution = np.append(new_solution, [0])
    
    return new_solution, calls_to_remove


# insertion functions
def random_insertion(vehicles, call_info, problem):
    n_vehicles = problem['n_vehicles']
    best_cost = np.inf
    best_vehicles = vehicles

    for call, compatible_vehicles in call_info:
        inserted = False

        v_idx = random.choice(compatible_vehicles)
        vehicle = vehicles[v_idx]
        for p_idx in range(len(vehicle) + 1):
            for d_idx in range(p_idx + 1, len(vehicle) + 2):
                temp_vehicle = vehicle.copy()
                temp_vehicle.insert(p_idx, call)
                temp_vehicle.insert(d_idx, call)
                is_feas, _ = feasibility_check_vehicle(temp_vehicle, v_idx, problem)
                if is_feas:
                    vehicles[v_idx] = temp_vehicle
                    inserted = True
                    current_cost = cost(vehicles, problem)
                    if current_cost < best_cost:
                        best_cost = current_cost
                        best_vehicles = vehicles
            if inserted:
                break

        # If we couldn't insert feasibly in any regular vehicle, use dummy
        if not inserted:
            vehicles[n_vehicles].append(call)
            vehicles[n_vehicles].append(call)
            vehicles[n_vehicles].sort()
            best_vehicles = vehicles

    return best_vehicles

def regret_insertion(vehicles, call_info, problem):
    n_vehicles = problem['n_vehicles']
    
    # If no calls to insert, return immediately
    if not call_info:
        return vehicles
    
    # Precompute call sizes
    call_sizes = {call: problem['Cargo'][call-1][2] for call, _ in call_info}
    
    # Calculate regret values for all calls
    regret_data = []
    
    for call_idx, (call, compatible_vehicles) in enumerate(call_info):
        # Regular compatible vehicles (excluding dummy)
        regular_vehicles = [v for v in compatible_vehicles if v < n_vehicles]
        
        # Track best insertion positions across all vehicles
        insertion_costs = []
        
        # Special case for dummy vehicle
        if n_vehicles in compatible_vehicles:
            dummy_delta = problem['Cargo'][call-1][3]  # Penalty cost
            insertion_costs.append((n_vehicles, None, None, dummy_delta))
        
        # Try each regular vehicle
        for v_idx in regular_vehicles:
            vehicle = vehicles[v_idx]
            
            # Skip if call is too big for vehicle
            if call_sizes[call] > problem['VesselCapacity'][v_idx]:
                continue
            
            # Calculate base cost for this vehicle
            base_cost = simplified_cost_function(vehicle, v_idx, problem)
            
            # Try all insertion positions
            vehicle_len = len(vehicle)
            
            for p_idx in range(vehicle_len + 1):
                for d_idx in range(p_idx + 1, vehicle_len + 2):
                    # Create temporary vehicle
                    temp_vehicle = vehicle.copy()
                    temp_vehicle.insert(p_idx, call)
                    
                    # Adjust delivery index
                    d_idx_adjusted = d_idx if d_idx <= p_idx else d_idx + 1
                    temp_vehicle.insert(d_idx_adjusted, call)
                    
                    is_feas, _ = feasibility_check_vehicle(temp_vehicle, v_idx, problem)
                    if is_feas:
                        # Evaluate cost delta for this vehicle only
                        new_cost = simplified_cost_function(temp_vehicle, v_idx, problem)
                        delta_cost = new_cost - base_cost
                        
                        insertion_costs.append((v_idx, p_idx, d_idx, delta_cost))
        
        # Sort insertions by cost
        insertion_costs.sort(key=lambda x: x[3])
        
        # Store all insertion options, and calculate regret value
        if len(insertion_costs) >= 2:
            best_cost = insertion_costs[0][3]
            second_best_cost = insertion_costs[1][3]
            regret_value = second_best_cost - best_cost
            
            # Store for later selection - now keeping all insertion costs
            regret_data.append({
                'call_idx': call_idx,
                'regret': regret_value,
                'insertion_options': insertion_costs[:3]  # Keep top 3 positions
            })
        elif len(insertion_costs) == 1:
            # Only one feasible insertion
            regret_data.append({
                'call_idx': call_idx,
                'regret': 50_000,  # Reduced from 100_000
                'insertion_options': insertion_costs
            })
        else:
            # No feasible insertions, must go to dummy
            regret_data.append({
                'call_idx': call_idx,
                'regret': 0,
                'insertion_options': [(n_vehicles, None, None, problem['Cargo'][call-1][3])]
            })
    
    # Process calls in order of regret (with probabilistic selection)
    remaining_calls = set(range(len(call_info)))
    
    while remaining_calls:
        # Calculate selection probabilities based on regret values
        candidates = [rd for rd in regret_data if rd['call_idx'] in remaining_calls]
        
        # Extract regret values and apply REDUCED exponential weighting
        regret_values = [rd['regret'] for rd in candidates]
        
        # Normalize regret values to avoid numerical issues
        max_regret = max(regret_values) if regret_values else 1
        # Fix: Check if max_regret is 0 to avoid division by zero
        if max_regret == 0:
            # If all regret values are 0, use equal probabilities
            probs = [1.0/len(candidates)] * len(candidates)
        else:
            # Normal case - proceed with normalization
            norm_regrets = [r/max_regret for r in regret_values]
            
            # Apply REDUCED exponential weighting (from 3 to 2)
            exp_regrets = [np.exp(1.5*r) * (1 + random.uniform(-0.2, 0.2)) for r in norm_regrets]
            
            # Convert to probabilities
            total = sum(exp_regrets)
            if total > 0:
                probs = [er/total for er in exp_regrets]
            else:
                probs = [1.0/len(candidates)] * len(candidates)
        
        # Select a call with weighted probabilities
        selected_idx = np.random.choice(len(candidates), p=probs)
        selected = candidates[selected_idx]
        
        # Get the call and all insertion options
        call_idx = selected['call_idx']
        call, compatible_vehicles = call_info[call_idx]
        insertion_options = selected['insertion_options']
        regret_value = selected['regret']
        

        if len(insertion_options) >= 2:
            # Define fixed probabilities for different positions
            fixed_probs = [0.35, 0.3, 0.25, 0.1]  # Best, second best, third best
            
            # Create actual probabilities based on available options
            position_probs = []
            
            # Determine how many positions we have and distribute probabilities
            if len(insertion_options) <= 3:
                # We have 2 or 3 positions - redistribute the probabilities proportionally
                available_probs = fixed_probs[:len(insertion_options)]
                total = sum(available_probs)
                position_probs = [p/total for p in available_probs]
            else:
                # We have more than 3 positions
                # First 3 positions get their fixed probabilities
                position_probs = fixed_probs.copy()
                
                # Remaining positions share the remaining 0.1 probability
                remaining_pos = len(insertion_options) - 3
                if remaining_pos > 0:
                    remaining_prob = 0.1 / remaining_pos
                    position_probs.extend([remaining_prob] * remaining_pos)
            
            # Select position
            position_idx = np.random.choice(len(insertion_options), p=position_probs)
            v_idx, p_idx, d_idx, _ = insertion_options[position_idx]
        else:
            # Only one option
            v_idx, p_idx, d_idx, _ = insertion_options[0]
        
        # Insert the call
        if v_idx == n_vehicles:
            # Insert in dummy vehicle
            vehicles[n_vehicles].append(call)
            vehicles[n_vehicles].append(call)
        else:
            # Insert in regular vehicle
            vehicle = vehicles[v_idx]
            vehicle.insert(p_idx, call)
            
            # Adjust delivery index
            d_idx_adjusted = d_idx if d_idx <= p_idx else d_idx + 1
            vehicle.insert(d_idx_adjusted, call)
        
        # Remove this call from remaining calls
        remaining_calls.remove(call_idx)

        # vehicles[n_vehicles].sort()

    return vehicles

def greedy_insertion(vehicles, call_info, problem):
    n_vehicles = problem['n_vehicles']
    
    # Precompute call sizes
    call_sizes = {call: problem['Cargo'][call-1][2] for call, _ in call_info}
    
    # Process each call
    for call, compatible_vehicles in call_info:
        # Get regular compatible vehicles (excluding dummy)
        regular_vehicles = [v for v in compatible_vehicles if v < n_vehicles]
        
        # Randomly select vehicles to try
        vehicles_to_try = []
        if len(regular_vehicles) <= 5:
            vehicles_to_try = regular_vehicles
        else:
            vehicles_to_try = regular_vehicles
            
        # Always include dummy vehicle as fallback
        if n_vehicles in compatible_vehicles:
            vehicles_to_try.append(n_vehicles)
        
        # If no compatible vehicles, skip this call
        if not vehicles_to_try:
            continue
            
        # NEW: Track pool of good insertion positions
        insertion_pool = []
        best_delta = float('inf')
        
        # Try each vehicle
        for v_idx in vehicles_to_try:
            vehicle = vehicles[v_idx]
            
            # Special case for dummy vehicle
            if v_idx == n_vehicles:
                # Calculate penalty cost for adding this call to dummy
                dummy_delta = problem['Cargo'][call-1][3]
                insertion_pool.append({
                    'v_idx': n_vehicles,
                    'p_idx': len(vehicle),
                    'd_idx': len(vehicle) + 1,
                    'delta': dummy_delta
                })
                if dummy_delta < best_delta:
                    best_delta = dummy_delta
                continue
            
            # Skip if call is too big for vehicle
            if call_sizes[call] > problem['VesselCapacity'][v_idx]:
                continue
            
            # Calculate base cost for this vehicle
            base_cost = simplified_cost_function(vehicle, v_idx, problem)
            
            # Try ALL insertion positions
            vehicle_len = len(vehicle)
            
            for p_idx in range(vehicle_len + 1):
                for d_idx in range(p_idx + 1, vehicle_len + 2):
                    # Create temporary vehicle
                    temp_vehicle = vehicle.copy()
                    temp_vehicle.insert(p_idx, call)
                    
                    # Adjust delivery index if needed
                    d_idx_adjusted = d_idx if d_idx <= p_idx else d_idx + 1
                    temp_vehicle.insert(d_idx_adjusted, call)
                    
                    is_feas, reason = feasibility_check_vehicle(temp_vehicle, v_idx, problem)
                    if is_feas:
                        # Evaluate cost delta for this vehicle only
                        new_cost = simplified_cost_function(temp_vehicle, v_idx, problem)
                        delta_cost = new_cost - base_cost
                        
                        # NEW: Add position to pool if it's good enough
                        # Keep positions that are within 200% of the best so far
                        if delta_cost < 2.0 * best_delta:
                            insertion_pool.append({
                                'v_idx': v_idx,
                                'p_idx': p_idx,
                                'd_idx': d_idx,
                                'delta': delta_cost
                            })
                            
                        # Still track the absolute best for threshold comparison
                        if delta_cost < best_delta:
                            best_delta = delta_cost
        
        # NEW: Select from pool using weighted probabilities
        if insertion_pool:
            # Sort by cost (lowest first)
            insertion_pool.sort(key=lambda x: x['delta'])
            
            # Calculate weights using exponential decay
            # - Best position gets highest weight
            # - Weight decreases exponentially for worse positions
            deltas = [pos['delta'] for pos in insertion_pool]
            
            # Normalize deltas relative to the best one
            if best_delta > 0:
                norm_deltas = [d/best_delta for d in deltas]
            else:
                # Handle case where best delta is 0 or negative
                min_delta = min(0.1, min(deltas))
                norm_deltas = [(d - min_delta + 0.1) for d in deltas]
            
            # Apply exponential weighting (smaller value = better position)
            weights = [np.exp(-2 * nd) for nd in norm_deltas]
            
            # Convert to probabilities
            total_weight = sum(weights)
            probs = [w/total_weight for w in weights]
            
            # Select position using weighted probabilities
            selected_idx = np.random.choice(len(insertion_pool), p=probs)
            selected = insertion_pool[selected_idx]
            
            # Insert the call at the selected position
            v_idx = selected['v_idx']
            p_idx = selected['p_idx']
            d_idx = selected['d_idx']
            
            if v_idx == n_vehicles:
                vehicles[n_vehicles].append(call)
                vehicles[n_vehicles].append(call)
            else:
                vehicle = vehicles[v_idx]
                vehicle.insert(p_idx, call)
                
                # When we insert at p_idx, positions after shift by 1
                d_idx_adjusted = d_idx
                if d_idx > p_idx:
                    d_idx_adjusted = d_idx + 1
                
                vehicle.insert(d_idx_adjusted, call)
        else:
            # Fallback to dummy if no feasible insertion found
            if n_vehicles in compatible_vehicles:
                vehicles[n_vehicles].append(call)
                vehicles[n_vehicles].append(call)

    return vehicles

def dummy_insertion(vehicles, call_info, problem):
    n_vehicles = problem['n_vehicles']

    # Instead of using insertion heuristics, just put all calls in the dummy vehicle
    for call, _ in call_info:
        vehicles[n_vehicles].append(call)
        vehicles[n_vehicles].append(call)

    vehicles[n_vehicles].sort()
    # Reassemble and return the solution
    return vehicles


# Operators
def random_random(vehicles, problem, compatibility_table, k):    
    vehicles, call_info = random_removal(vehicles, problem, compatibility_table, k)
    vehicles = random_insertion(vehicles, call_info, problem)
    return vehicles

def related_regret(vehicles, problem, compatibility_table, k):
    vehicles, call_info = related_removal(vehicles, problem, compatibility_table, k)
    vehicles = regret_insertion(vehicles, call_info, problem)
    return vehicles

def worst_regret(vehicles, problem, compatibility_table, k):
    vehicles, call_info = worst_removal(vehicles, problem, compatibility_table, k)
    vehicles = regret_insertion(vehicles, call_info, problem)
    return vehicles

def related_greedy(vehicles, problem, compatibility_table, k):
    vehicles, call_info = related_removal(vehicles, problem, compatibility_table, k)
    vehicles = greedy_insertion(vehicles, call_info, problem)
    return vehicles

def random_regret(vehicles, problem, compatibility_table, k):
    vehicles, call_info = random_removal(vehicles, problem, compatibility_table, k)
    vehicles = regret_insertion(vehicles, call_info, problem)
    return vehicles

def random_greedy(vehicles, problem, compatibility_table, k):
    vehicles, call_info = random_removal(vehicles, problem, compatibility_table, k)
    vehicles = greedy_insertion(vehicles, call_info, problem)
    return vehicles

def worst_greedy(vehicles, problem, compatibility_table, k):
    vehicles, call_info = worst_removal(vehicles, problem, compatibility_table, k)
    vehicles = greedy_insertion(vehicles, call_info, problem)
    return vehicles

def random_dummy(vehicles, problem, compatibility_table, k):
    vehicles, call_info = random_removal(vehicles, problem, compatibility_table, k)
    vehicles = dummy_insertion(vehicles, call_info, problem)
    return vehicles

def tardiness_greedy(vehicles, problem, compatibility_table, k):
    vehicles, call_info = tardiness_removal(vehicles, problem, compatibility_table, k)
    vehicles = greedy_insertion(vehicles, call_info, problem)
    return vehicles

def tardiness_dummy(vehicles, problem, compatibility_table, k):
    vehicles, call_info = tardiness_removal(vehicles, problem, compatibility_table, k)
    vehicles = dummy_insertion(vehicles, call_info, problem)
    return vehicles

def tardiness_regret(vehicles, problem, compatibility_table, k):
    vehicles, call_info = tardiness_removal(vehicles, problem, compatibility_table, k)
    vehicles = regret_insertion(vehicles, call_info, problem)
    return vehicles

def dummy_greedy(vehicles, problem, compatibility_table, k):
    vehicles, call_info = dummy_removal(vehicles, problem, compatibility_table, k)
    vehicles = greedy_insertion(vehicles, call_info, problem)
    return vehicles

    
    # Rest of function remains the same...

def dummy_regret(vehicles, problem, compatibility_table, k):
    vehicles, call_info = dummy_removal(vehicles, problem, compatibility_table, k)
    vehicles = regret_insertion(vehicles, call_info, problem)
    return vehicles

def long_wait_greedy(vehicles, problem, compatibility_table, k):
    """Remove calls with excessive waiting times and reinsert using greedy insertion."""
    vehicles, removed_call_info = remove_long_wait_calls(reassemble_solution(vehicles), problem, wait_threshold=30)
    vehicles = greedy_insertion(vehicles, removed_call_info, problem)
    return vehicles

# Algorithm
def adaptive_algorithm(initial_solution, problem, neighbours, compatibility_table, max_iter):
    # Initialize calls to vehicles as list of lists
    vehicles = parse_solution_to_vehicles(initial_solution)
    best_solution = [vehicle.copy() for vehicle in vehicles]  # Deep copy
    current_solution = [vehicle.copy() for vehicle in vehicles]  # Deep copy
    best_cost_value = cost(best_solution, problem)
    current_cost_value = best_cost_value 
    feasibles = 0
    
    # History tracking variables
    best_iteration = 0

    history = {
        "iteration": [],
        "best_cost": [],
        "current_cost": [],
        "acceptance_rate": [], 
        "operator_used": [], 
        "delta_value": []
    }
    
    # Track operator weights over time
    iteration_points = []
    operator_weight_history = {op.__name__: [] for op in neighbours}
    operator_selection_counts = {op.__name__: 0 for op in neighbours}
    accepted_solutions = 0
    total_evaluations = 0
    
   
    # Initialize operator objects
    operators = [{"name": op.__name__, "func": op, "weight": 1.0, 
                 "successes": 0, "attempts": 1, "best_improvements": 0, 'improvements': 0} for op in neighbours]
    
    # Main search loop
    iters_since_improvement = 0
    
    for iteration in range(max_iter-200):
        escape_threshold = 1000
        # Status update every 1000 iterations
        if iteration % 1000 == 0:
            print(f'Iteration: {iteration} | Current: {current_cost_value:.2f} | Best: {best_cost_value:.2f}')

        

        # Call escape when stuck
        if iters_since_improvement == escape_threshold:
            print(f"Escape triggered at iteration {iteration} after {iters_since_improvement} iterations without improvement")
            
            
            current_solution, best_solution, best_cost_value, found = escape(
                current_solution, best_solution, best_cost_value, problem, compatibility_table
            )
            
            if found:       
                best_solution = [vehicle.copy() for vehicle in best_solution]
                
            else:
                current_cost_value = cost(current_solution, problem)
            iters_since_improvement = 0
        
        # Update operator weights
        if iteration % 100 == 0 and iteration > 0:
            update_operator_weights(operators)
            # Record operator weights every 100 iterations
            iteration_points.append(iteration)
            for op in operators:
                operator_weight_history[op["name"]].append(op["weight"])
            
        # Select operator and apply
        total_weight = sum(op["weight"] for op in operators)
        probs = [op["weight"]/total_weight for op in operators]
        selected_op_idx = np.random.choice(len(operators), p=probs)
        selected_op = operators[selected_op_idx]
        operator_selection_counts[selected_op["name"]] += 1
        
        q = random.randint(1, round(0.4*problem['n_calls']))
        new_solution = [vehicle.copy() for vehicle in current_solution]
        new_solution = selected_op["func"](new_solution, problem, compatibility_table, q)
        operators[selected_op_idx]["attempts"] += 1
        total_evaluations += 1

        # Check feasibility and decide whether to accept
        if feasibility(new_solution, problem):
            feasibles += 1

            current_cost_value = cost(current_solution, problem)
            new_cost_value = cost(new_solution, problem)
            E = new_cost_value - current_cost_value
            history["delta_value"].append(E)

            
            if E < 0:  
                operators[selected_op_idx]["improvements"] += 1
                accepted_solutions += 1
                current_solution = [vehicle.copy() for vehicle in new_solution]
                current_cost_value = new_cost_value

                # Check if we've found a new best solution
                if current_cost_value < best_cost_value:
                    best_solution = [vehicle.copy() for vehicle in current_solution]
                    best_cost_value = current_cost_value
                    best_iteration = iteration
                    iters_since_improvement = 0
                    operators[selected_op_idx]["best_improvements"] = operators[selected_op_idx].get("best_improvements", 0) + 1
                    print(f"New best solution found at iteration {iteration}: {best_cost_value:.2f} using {selected_op['name']}")
                        
                else:
                    iters_since_improvement += 1
            elif E > 0:  # Worse solution - accept with probability
                D = 0.2 * (((max_iter- 200) - iteration) / (max_iter - 200)) * best_cost_value
                criteria = best_cost_value + D
               
                if current_cost_value < criteria:
                    current_solution = new_solution
                    current_cost_value = cost(current_solution, problem)
                    accepted_solutions += 1
                        
                
                iters_since_improvement += 1
        else:
            iters_since_improvement += 1
            history["delta_value"].append(0)
                

        # Record history
        history["iteration"].append(iteration)
        history["best_cost"].append(best_cost_value)
        history["current_cost"].append(current_cost_value)
        history["acceptance_rate"].append(accepted_solutions / (total_evaluations or 1))
        history["operator_used"].append(selected_op["name"])
   
    
    print(f'infeasible solutions: {9800 - feasibles}')
    # Return the solution and additional data
    return (
        reassemble_solution(best_solution), 
        best_cost_value,
        best_iteration, 
        history,
        operator_weight_history,
        iteration_points
    )

# Plots

def plot_acceptance_probability(history, problem_name):
    
    plt.figure(figsize=(12, 7))
    
    # Extract iteration data
    iterations = history["iteration"]
    temps = history["temperature"]
    current_costs = history["current_cost"]
    
    # Find iterations where solution quality worsened
    worse_iterations = []
    acceptance_probs = []
    delta_values = []
    
    # Calculate delta E and acceptance probability for each iteration
    for i in range(1, len(iterations)):
        delta_E = current_costs[i] - current_costs[i-1]
        
        # Only include iterations where solution quality worsened (delta > 0)
        if delta_E > 0:
            iter_num = iterations[i]
            temperature = temps[i]
            
            # Calculate acceptance probability using Metropolis criterion
            acceptance_prob = np.exp(-delta_E / temperature)
            
            worse_iterations.append(iter_num)
            acceptance_probs.append(acceptance_prob)
            delta_values.append(delta_E)
    
    # Create the scatter plot with point size varying by delta
    normalized_delta = np.array(delta_values) / max(delta_values) if delta_values else []
    sizes = 20 + 100 * normalized_delta if len(normalized_delta) > 0 else 20
    
    # Plot with a colormap representing iteration progression
    sc = plt.scatter(worse_iterations, acceptance_probs, 
                    c=worse_iterations, 
                    cmap='viridis', 
                    alpha=0.7,
                    s=sizes)
    
    plt.colorbar(sc, label='Iteration')
    plt.xlabel('Iteration')
    plt.ylabel('Acceptance Probability')
    plt.title(f'Acceptance Probability for Worse Solutions - {problem_name}')
    plt.grid(True, alpha=0.3)
    
    if isinstance(problem_name, str):
        safe_name = problem_name.split('/')[-1].replace('.txt', '')
    else:
        safe_name = str(problem_name).replace('/', '_')
    
    plt.tight_layout()
    plt.savefig(f'acceptance_probability_{safe_name.replace(" ", "_")}.png')
    plt.show()

def plot_solution_progress(history, best_iteration, problem_name):
    plt.figure(figsize=(12, 8))
    
    # Extract data
    iterations = history["iteration"]
    current_costs = history["current_cost"]
    best_costs = history["best_cost"]
    
    # Plot current cost (with light color)
    plt.plot(iterations, current_costs, 'lightblue', alpha=0.6, label='Current Solution')
    
    # Plot best cost (with darker color)
    plt.plot(iterations, best_costs, 'darkblue', linewidth=2, label='Best Solution')
    
    # Mark the best solution
    best_cost = min(best_costs)
    
    # Add vertical line at best iteration
    plt.axvline(x=best_iteration, color='red', linestyle='--', alpha=0.7, 
                label=f'Best Solution Found at Iteration {best_iteration}')
    
    # Add a marker for the best solution
    plt.plot(best_iteration, best_cost, 'ro', markersize=10)
    plt.annotate(f'Best: {best_cost:.2f}',
                xy=(best_iteration, best_cost),
                xytext=(best_iteration+len(iterations)*0.02, best_cost*0.95),
                arrowprops=dict(arrowstyle='->'))
    
    # Add a text box with solution statistics
    textstr = '\n'.join((
        f'Best cost: {best_cost:.2f}',
        f'Found at iteration: {best_iteration}/{max(iterations) + 1}'
    ))
    
    props = dict(boxstyle='round', facecolor='wheat', alpha=0.4)
    plt.text(0.02, 0.02, textstr, transform=plt.gca().transAxes, fontsize=10,
            verticalalignment='bottom', horizontalalignment='left', bbox=props)
    
    # Add grid and labels
    plt.grid(True, alpha=0.3)
    plt.xlabel('Iteration')
    plt.ylabel('Solution Cost')
    plt.title(f'Solution Progress - {problem_name}')
    plt.legend(loc='upper right')
    
    # Set y-axis scale to focus on relevant cost range
    min_cost = min(best_costs) * 0.95
    max_cost = max(current_costs[:len(current_costs)//10]) * 1.05  # Focus on first 10% for max
    plt.ylim(min_cost, max_cost)
    
    if isinstance(problem_name, str):
        safe_name = problem_name.split('/')[-1].replace('.txt', '')
    else:
        safe_name = str(problem_name).replace('/', '_')

    plt.tight_layout()
    plt.savefig(f'solution_progress_{safe_name.replace(" ", "_")}.png')
    plt.show()

def plot_operator_deltas(history, neighbours, problem_name):
   
    if "operator_used" not in history:
        print("Error: No operator usage data in history. Please add operator tracking to adaptive_algorithm.")
        return
    
    # Get operator names
    operator_names = [op.__name__ for op in neighbours]
    
    # Create a separate plot for each operator
    for op_name in operator_names:
        plt.figure(figsize=(10, 6))
        
        # Find iterations where this operator was used
        iterations = []
        deltas = []
        
        for i, used_op in enumerate(history["operator_used"]):
            if used_op == op_name:
                iterations.append(history["iteration"][i])
                deltas.append(history["delta_value"][i])
        
        if not iterations:
            plt.close()
            continue
        
        # Create scatter plot - green for improvements, red for degradations
        colors = ['green' if d <= 0 else 'red' for d in deltas]
        plt.scatter(iterations, deltas, c=colors, alpha=0.7, s=30)
        
        # Add horizontal line at zero
        plt.axhline(y=0, color='black', linestyle='-', alpha=0.3)
        
        # Add labels and title
        plt.xlabel('Iteration')
        plt.ylabel('Delta Value')
        plt.title(f'Delta Values for {op_name}')
        plt.grid(True, alpha=0.3)
        
        if isinstance(problem_name, str):
            safe_name = problem_name.split('/')[-1].replace('.txt', '')
        else:
            safe_name = str(problem_name).replace('/', '_')

        plt.tight_layout()
        plt.savefig(f'operator_delta_{op_name}_{safe_name}.png')
        plt.show()

def plot_operator_probabilities(operator_weight_history, iteration_points, problem_name):
    
    # Create figure with adjusted size ratio to accommodate legend
    plt.figure(figsize=(14, 8))
    
    # Create subplot with adjusted position to use most of the figure
    ax = plt.subplot(111)
    
    # Calculate segment numbers (iterations / 100)
    segments = [i//100 for i in iteration_points]
    
    # Calculate selection probabilities at each segment
    prob_history = {}
    
    # For each segment, calculate the probability for each operator
    for i in range(len(segments)):
        segment_weights = {}
        total_weight = 0
        
        for op_name, weights in operator_weight_history.items():
            if i < len(weights):  # Ensure the segment exists
                segment_weights[op_name] = weights[i]
                total_weight += weights[i]
        
        # Calculate probabilities
        for op_name, weight in segment_weights.items():
            if op_name not in prob_history:
                prob_history[op_name] = []
            prob_history[op_name].append(weight / total_weight if total_weight > 0 else 0)
    
    # Define a color cycle for operators
    colors = plt.cm.tab10.colors
    
    # Plot each operator's probabilities with step-wise changes
    for i, (operator_name, probs) in enumerate(prob_history.items()):
        color = colors[i % len(colors)]
        
        # Plot with step lines to show constant value between updates
        ax.step(segments, probs, where='post', 
                linewidth=2, label=operator_name, color=color)
        
        # Add markers at each update point
        ax.plot(segments, probs, 'o', markersize=4, color=color)
    
    # Add labels and title
    ax.set_xlabel('Segment Number (Every 100 Iterations)')
    ax.set_ylabel('Selection Probability')
    
    # Extract just the filename part for the title
    if isinstance(problem_name, str) and '/' in problem_name:
        problem_title = problem_name.split('/')[-1].replace('.txt', '')
    else:
        problem_title = problem_name
        
    ax.set_title(f'Operator Selection Probabilities - {problem_title}')
    
    # Add grid
    ax.grid(True, alpha=0.3)
    
    # Find the operator with highest final probability
    final_probs = {op: probs[-1] if probs else 0 for op, probs in prob_history.items()}
    if final_probs:
        best_op = max(final_probs.items(), key=lambda x: x[1])[0]
        best_prob = final_probs[best_op]
        
        # Annotate the most probable operator at the end
        ax.annotate(f'Highest prob: {best_prob:.2f}',
                   xy=(segments[-1], best_prob),
                   xytext=(segments[-1]-5, best_prob+0.05),
                   arrowprops=dict(arrowstyle='->'))
    
    # Either place legend at bottom horizontally
    # ax.legend(loc='upper center', bbox_to_anchor=(0.5, -0.12), ncol=4, fontsize=9)
    
    # OR move legend to the right but adjust the plot area
    # Shrink current axis by 20% to make room for legend
    box = ax.get_position()
    ax.set_position([box.x0, box.y0, box.width * 0.8, box.height])
    
    # Put a legend to the right of the current axis
    ax.legend(loc='center left', bbox_to_anchor=(1, 0.5), fontsize=9)
    
    # Extract just the filename part for the saved file
    if isinstance(problem_name, str):
        safe_name = problem_name.split('/')[-1].replace('.txt', '')
    else:
        safe_name = str(problem_name).replace('/', '_')
    
    plt.savefig(f'operator_probabilities_{safe_name}.png', bbox_inches='tight')
    plt.show()

# Main code
neighbours = [  
                long_wait_greedy,
                ]
              

print("Starting adaptive algorithm")
start_time = time.time()
best_solution, best_cost, best_iteration, history, op_weight_history, iter_points = adaptive_algorithm(
    initial_solution_generator(problem), problem, neighbours, compatibility_table=comatibility_table, max_iter=10000)
end_time = time.time()
best_solution = [int(x) for x in best_solution]
print(f'best iteration: {best_iteration}')
print(f"Final cost: {best_cost}, Time: {end_time - start_time:.2f}s")
print(f"Feasibility: {feasibility_check(best_solution, problem)[0]}")
print(f"Solution: {best_solution}")

# # # Plot operator weights
# plot_temperature(history, problem_file)
# plot_acceptance_probability(history, problem_file)
# plot_solution_progress(history, best_iteration, problem_file)
# plot_operator_deltas(history, neighbours, problem_file)
# plot_operator_probabilities(op_weight_history, iter_points, problem_file)

# def long_wait_removal(solution, problem, wait_threshold=30):
#     """
#     Remove calls with excessive waiting time.
    
#     Args:
#         solution: Either a flat solution array or a list of vehicle routes
#         problem: Problem dictionary
#         wait_threshold: Maximum acceptable waiting time (minutes)
        
#     Returns:
#         vehicles: Modified list of vehicle routes
#         removed_calls: List of removed calls
#     """
#     # Check if input is a flat solution and convert if needed
#     if isinstance(solution, list) and any(isinstance(item, int) for item in solution):
#         vehicles = []
#         current_vehicle = []
        
#         for call in solution:
#             if call == 0:
#                 vehicles.append(current_vehicle)
#                 current_vehicle = []
#             else:
#                 current_vehicle.append(call)
                
#         vehicles.append(current_vehicle) 

#     else:
#         vehicles = solution  # Already in vehicles format
    
#     n_vehicles = problem['n_vehicles']
#     calls_with_wait = []
    
#     # Analyze each regular vehicle route (not dummy)
#     for v_idx in range(n_vehicles):
#         vehicle = vehicles[v_idx]
        
#         # Skip empty vehicles
#         if not vehicle:
#             continue
            
#         # Map positions to calls
#         call_positions = {}
#         for pos, call in enumerate(vehicle):
#             if call not in call_positions:
#                 call_positions[call] = [pos]  # First occurrence (pickup)
#             else:
#                 call_positions[call].append(pos)  # Second occurrence (delivery)
        
#         # For calculating waiting times, track vehicle's schedule
#         current_time = 0
#         current_location = None
#         processed_calls = set()
        
#         # Process the route in sequence
#         for pos, call in enumerate(vehicle):
#             is_pickup = call not in processed_calls
#             call_idx = call - 1  # Convert to 0-indexed for problem data
            
#             # Determine location (pickup or delivery node)
#             if is_pickup:
#                 location = int(problem['Cargo'][call_idx, 0]) - 1  # Origin port (0-indexed)
#                 earliest_time = problem['Cargo'][call_idx, 4]      # Pickup earliest time
#                 latest_time = problem['Cargo'][call_idx, 5]        # Pickup latest time
#                 service_time = problem['LoadingTime'][v_idx, call_idx]
#                 processed_calls.add(call)
#             else:
#                 location = int(problem['Cargo'][call_idx, 1]) - 1  # Destination port (0-indexed)
#                 earliest_time = problem['Cargo'][call_idx, 6]      # Delivery earliest time
#                 latest_time = problem['Cargo'][call_idx, 7]        # Delivery latest time
#                 service_time = problem['UnloadingTime'][v_idx, call_idx]
            
#             # Calculate travel time to this location
#             if current_location is None:
#                 # First location in route
#                 travel_time = problem['FirstTravelTime'][v_idx, location]
#                 arrival_time = travel_time
#             else:
#                 travel_time = problem['TravelTime'][v_idx, current_location, location]
#                 arrival_time = current_time + travel_time
            
#             # Calculate waiting time
#             waiting_time = max(0, earliest_time - arrival_time)
            
#             # If waiting time exceeds threshold, mark call for removal
#             if waiting_time > wait_threshold:
#                 calls_with_wait.append((v_idx, call, waiting_time))
            
#             # Update current state
#             service_start = max(arrival_time, earliest_time)
#             current_time = service_start + service_time
#             current_location = location
    
#     # Sort calls by waiting time (longest first)
#     calls_with_wait.sort(key=lambda x: x[2], reverse=True)
    
#     # Remove the selected calls
#     removed_calls = []
#     for v_idx, call, _ in calls_with_wait:
#         # Find positions of the call
#         pickup_pos = delivery_pos = None
#         for pos, c in enumerate(vehicles[v_idx]):
#             if c == call:
#                 if pickup_pos is None:
#                     pickup_pos = pos
#                 else:
#                     delivery_pos = pos
#                     break
        
#         # Remove calls in reverse order to avoid position issues
#         if pickup_pos is not None and delivery_pos is not None:
#             # Remove delivery first (higher index)
#             vehicles[v_idx].pop(delivery_pos)
#             # Then remove pickup
#             vehicles[v_idx].pop(pickup_pos)
#             removed_calls.append(call)
    
#     return vehicles, removed_calls

# def remove_random_call(solution, problem):
#     # Convert the solution into a list of vehicle routes
#     vehicle_routes = []
#     current_route = []
#     for call in solution:
#         if call == 0:
#             vehicle_routes.append(current_route)
#             current_route = []
#         else:
#             current_route.append(call)
#     if current_route:
#         vehicle_routes.append(current_route)

#     # Debugging: Check vehicle routes
#     print(f"Vehicle routes before removing a call: {vehicle_routes}")

#     # Randomly select a vehicle and a call to remove
#     non_empty_routes = [i for i in range(len(vehicle_routes)) if vehicle_routes[i]]
#     if not non_empty_routes:
#         print("No non-empty routes available. Returning original solution.")
#         return solution  # Return the original solution if no calls can be removed

#     vehicle_idx = random.choice(non_empty_routes)
#     call_idx = random.choice(range(len(vehicle_routes[vehicle_idx])))
#     call_removed = vehicle_routes[vehicle_idx].pop(call_idx)

#     # Ensure the call is removed twice (pickup and delivery)
#     vehicle_routes[vehicle_idx] = [c for c in vehicle_routes[vehicle_idx] if c != call_removed]

#     # Debugging: Check removed call
#     print(f"Removed call: {call_removed}")
#     print(f"Vehicle routes after removing the call: {vehicle_routes}")

#     # Flatten the vehicle routes back into a single solution list
#     new_solution = []
#     for route in vehicle_routes:
#         new_solution.extend(route)
#         new_solution.append(0)

#     # Remove the last 0
#     if new_solution[-1] == 0:
#         new_solution.pop()

#     return new_solution

# def greedy_insert_call(solution, problem, removed_call):
#     """
#     Insert a removed call into the best feasible position by evaluating cost.
    
#     Args:
#         solution: Current solution without the removed call
#         problem: Problem definition containing constraints and parameters
#         removed_call: Call ID to insert
        
#     Returns:
#         The best feasible solution after inserting the removed call
#     """
#     import numpy as np
    
#     # Convert the solution into a list of vehicle routes
#     if isinstance(solution, np.ndarray):
#         solution = solution.tolist()
    
#     vehicle_routes = []
#     current_route = []
#     for call in solution:
#         if call == 0:
#             vehicle_routes.append(current_route)
#             current_route = []
#         else:
#             current_route.append(call)
#     if current_route:
#         vehicle_routes.append(current_route)
    
#     # Ensure we have the correct number of vehicle routes
#     while len(vehicle_routes) < problem.get('n_vehicles', 1):
#         vehicle_routes.append([])
    
#     # Add one more for the dummy vehicle if needed
#     if len(vehicle_routes) <= problem.get('n_vehicles', 1):
#         vehicle_routes.append([])
    
#     # Track best solution and its cost
#     best_solution = None
#     best_cost = float('inf')
    
#     # Try inserting the removed call into each vehicle route
#     for vehicle_idx, route in enumerate(vehicle_routes):
#         # Check if vehicle capacity is sufficient for this call
#         if vehicle_idx < problem['n_vehicles']:  # Only check real vehicles
#             call_size = problem['Cargo'][removed_call-1][2]
#             vehicle_capacity = problem['VesselCapacity'][vehicle_idx]
#             if call_size > vehicle_capacity:
#                 continue  # Skip if call is too big for this vehicle
        
#         # Try all possible insertion positions for pickup and delivery
#         for pickup_pos in range(len(route) + 1):
#             for delivery_pos in range(pickup_pos + 1, len(route) + 2):
#                 # Create a new route with the pickup and delivery nodes inserted
#                 new_route = route.copy()
#                 new_route.insert(pickup_pos, removed_call)
                
#                 # After inserting pickup, delivery position may need adjustment
#                 delivery_pos_adjusted = delivery_pos
#                 if delivery_pos > pickup_pos:
#                     delivery_pos_adjusted += 1  # Adjust for the inserted pickup
                
#                 new_route.insert(delivery_pos_adjusted, removed_call)
                
#                 # Create copy of all routes
#                 new_vehicle_routes = [r.copy() for r in vehicle_routes]
#                 new_vehicle_routes[vehicle_idx] = new_route
                
#                 # Reconstruct the full solution
#                 new_solution = []
#                 for i, r in enumerate(new_vehicle_routes):
#                     new_solution.extend(r)
#                     if i < len(new_vehicle_routes) - 1:
#                         new_solution.append(0)
                
#                 # Convert for evaluation
#                 new_solution_np = np.array(new_solution)
                
#                 try:
#                     # Check feasibility
#                     is_feasible = feasibility_check(new_solution_np, problem)[0]
#                     if not is_feasible:
#                         continue
                    
#                     # Evaluate cost
#                     current_cost = cost_function(new_solution_np, problem)
#                     if current_cost < best_cost:
#                         best_solution = new_solution
#                         best_cost = current_cost
#                 except Exception as e:
#                     # Skip invalid solutions
#                     continue
    
#     # If no valid insertion was found, add to dummy vehicle
#     if best_solution is None:
#         # Add to dummy vehicle (last one)
#         dummy_route = vehicle_routes[-1].copy()
#         dummy_route.append(removed_call)
#         dummy_route.append(removed_call)
        
#         # Create new vehicle routes with updated dummy
#         new_vehicle_routes = [r.copy() for r in vehicle_routes]
#         new_vehicle_routes[-1] = dummy_route
        
#         # Reconstruct solution
#         best_solution = []
#         for i, r in enumerate(new_vehicle_routes):
#             best_solution.extend(r)
#             if i < len(new_vehicle_routes) - 1:
#                 best_solution.append(0)
    
#     return np.array(best_solution) if best_solution is not None else np.array(solution)

# new, removed = long_wait_removal([4, 4, 2, 2, 0, 7, 7, 0, 1, 5, 5, 3, 3, 1, 0, 6, 6], problem, 60)
# print(new, removed)
# new = reassemble_solution(new)

# print(greedy_insert_call(new, problem, removed[0]))