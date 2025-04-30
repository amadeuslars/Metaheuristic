from Utils2 import *
from early_code.Utils import feasibility_check2, load_problem

import random
import time
import numpy as np
import matplotlib.pyplot as plt

filenames = [
    "Data/Call_7_Vehicle_3.txt",
    "Data/Call_18_Vehicle_5.txt",
    "Data/Call_35_Vehicle_7.txt",
    "Data/Call_80_Vehicle_20.txt",
    "Data/Call_130_Vehicle_40.txt",
    ]

# General functions
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

def cost(vehicles, problem):
    return cost_function(vehicles, problem)

def feasibility(vehicles, problem):
    return feasibility_check(vehicles, problem)

def escape(current, best_solution, best_cost, problem):
    """
    Attempts to escape a local optimum by applying strong perturbations.
    Tracks and returns the best feasible solution found during the escape attempts.
    Also updates the global best if a new one is found.
    """
    updated_best_solution = [v.copy() for v in best_solution] # Keep track of global best
    updated_best_cost = best_cost
    found_new_global_best = False

    # --- Track the best solution found specifically within this escape phase ---
    best_in_escape_solution = [v.copy() for v in current] # Start with current
    best_in_escape_cost = cost(best_in_escape_solution, problem) # Cost of current

    attempts = 0
    num_feas = 0
    max_attempts = 20 # Increased attempts
    min_feas_needed = 3 # Try to find at least a few feasible options

    # Define perturbation operators to potentially use
    perturbation_operators = [
        (random_removal, greedy_insertion),
        (worst_removal, regret_insertion), # Add another pair for diversity
        (segment_removal, greedy_insertion) # Add segment removal
    ]

    while attempts < max_attempts and num_feas < min_feas_needed:
        attempts += 1
        temp_escape_solution = [v.copy() for v in current] # Start perturbation from current

        # --- Select perturbation operator ---
        # Simple random choice for now
        remove_op, insert_op = random.choice(perturbation_operators)
        # print(f"  Escape attempt {attempts}: Using {remove_op.__name__} + {insert_op.__name__}")

        # --- Apply perturbation ---
        # Determine k (consider a potentially larger range for escape)
        min_k = max(2, round(0.25 * problem['n_calls']))
        max_k = max(min_k, round(0.60 * problem['n_calls'])) # Wider range, up to 60%
        if problem['n_calls'] >= min_k:
             k = random.randint(min_k, max_k)
        else:
             k = problem['n_calls']

        if k <= 0: continue # Skip if no calls to remove

        try:
            # Apply removal
            # Handle operators needing relatedness matrix (though not used in default list)
            if 'related' in remove_op.__name__ or 'shaw' in remove_op.__name__:
                 # Need relatedness_matrix if these are used
                 # relatedness_matrix = calculate_relatedness(problem) # Or pass it in
                 # temp_escape_solution, removed_calls = remove_op(temp_escape_solution, problem, k, relatedness_matrix)
                 pass # Avoid using related/shaw for now unless matrix is passed
            elif 'historical' in remove_op.__name__:
                 # Need call_blame if this is used
                 # call_blame = {} # Or pass it in
                 # temp_escape_solution, removed_calls = remove_op(temp_escape_solution, problem, k, call_blame)
                 pass # Avoid using historical for now unless blame is passed
            elif 'segment' in remove_op.__name__ and 'insertion' in remove_op.__name__: # Special case for segment_removal_for_segment_insertion
                 temp_escape_solution, removed_calls, _ = remove_op(temp_escape_solution, problem, k) # Ignore removed segment list here
            else:
                 temp_escape_solution, removed_calls = remove_op(temp_escape_solution, problem, k)

            # Apply insertion
            # Handle operators needing relatedness matrix
            if 'related' in insert_op.__name__ or 'shaw' in insert_op.__name__:
                 # temp_escape_solution = insert_op(temp_escape_solution, removed_calls, problem, relatedness_matrix)
                 pass
            else:
                 temp_escape_solution = insert_op(temp_escape_solution, removed_calls, problem)

        except Exception as e:
            print(f"  Error during escape perturbation ({remove_op.__name__}/{insert_op.__name__}): {e}")
            continue # Skip this attempt

        # --- Check feasibility and cost ---
        is_feasible, details = feasibility(temp_escape_solution, problem)
        if is_feasible:
            num_feas += 1
            perturbed_cost = cost(temp_escape_solution, problem)

            # --- Update best solution found *within this escape phase* ---
            if perturbed_cost < best_in_escape_cost:
                best_in_escape_solution = [v.copy() for v in temp_escape_solution]
                best_in_escape_cost = perturbed_cost
                # print(f"    Found better escape solution: {best_in_escape_cost:.2f}")

            # --- Check if we found a new *global* best ---
            if perturbed_cost < updated_best_cost:
                updated_best_solution = [v.copy() for v in temp_escape_solution]
                updated_best_cost = perturbed_cost
                found_new_global_best = True
                # print(f"    *** Found new GLOBAL best during escape: {updated_best_cost:.2f} ***")
        # else:
            # print(f"  Escape attempt {attempts} infeasible: {details}")


    # Return the best solution found during the escape attempts,
    # the (potentially updated) global best solution and cost,
    # and the flag indicating if a new global best was found.
    # The main loop will use 'best_in_escape_solution' as its new 'current_solution'.
    # print(f"  Escape finished. Best escape cost: {best_in_escape_cost:.2f}. Found global best: {found_new_global_best}")
    return best_in_escape_solution, updated_best_solution, updated_best_cost, found_new_global_best

def update_operator_weights(operators):
    """
    Update operator weights based on performance in the last period,
    with smoother updates and adjusted scoring.
    """
    r = 0.1

    for op in operators:
        # Use max(1, ...) to avoid division by zero if an operator wasn't used
        attempts = max(1, op['attempts'])

        # --- Score Components from the last period ---
        # Found solution better than the previous one (E < 0)
        found_improving = op['improvements']
        # Found a new global best solution
        found_new_best = op.get('best_improvements', 0)

        # --- Define Score based on achievements (Tune coefficients here) ---
    
        # Moderate reward for finding an improving solution
        score_improving = 3 * found_improving
        # High reward for finding a new best solution
        score_best = 10.0 * found_new_best

        # Total score for the period
        pi = score_improving + score_best

        # Normalize score by attempts for the period
        period_score = pi / attempts

        # --- Update weight using exponential smoothing ---
        op_old_weight = op['weight']
        # new_weight = old_weight * (1-r) + period_score * r
        new_weight = op_old_weight * (1 - r) + period_score * r

        # Ensure a minimum weight to prevent operators from dying out completely
        op['weight'] = max(0.1, new_weight)

        # --- Reset statistics for the next period ---
        op['successes'] = 0
        op['attempts'] = 0 # Reset to 0, increment when selected in the main loop
        op['improvements'] = 0
        op['best_improvements'] = 0

def build_call_info(removed_calls, n_vehicles, problem):
    call_info = []
    for call in removed_calls:
        # Find compatible vehicles by checking which vehicle_calls lists contain this call
        compatible_vehicles = []
        
        # Always include dummy vehicle
        compatible_vehicles.append(n_vehicles)
        
        # Check each regular vehicle (1-indexed in vehicle_calls)
        for v_idx in range(1, n_vehicles + 1):
            # If call is in the list of calls compatible with this vehicle
            if call in problem['vehicle_calls'][v_idx]:
                compatible_vehicles.append(v_idx - 1)  # Convert to 0-indexed
        
        call_info.append((call, compatible_vehicles))
    return call_info

def get_calls(vehicles):
    call_locations = []
    
    for v_idx, vehicle in enumerate(vehicles):
        call_counts = {}
        for call in vehicle:
            call_counts[call] = call_counts.get(call, 0) + 1
        
        for call, count in call_counts.items():
            if count == 2:  
                call_locations.append((v_idx, call))
    
    return call_locations

def calculate_relatedness(problem):
    n_calls = problem['n_calls']
    n_vehicles = problem['n_vehicles']
    call_info = problem['call_info']
    
    # Initialize relatedness matrix
    relatedness_matrix = np.zeros((n_calls, n_calls))
    
    # Find normalization factors
    # 1. Maximum travel cost
    max_travel_cost = 0
    for v_idx in range(1, n_vehicles + 1):  # Vehicle indices are 1-based
        for i in range(n_calls):
            origin_i = call_info[i][1]  # Origin node of call i
            dest_i = call_info[i][2]    # Destination node of call i
            
            for j in range(n_calls):
                origin_j = call_info[j][1]  # Origin node of call j
                dest_j = call_info[j][2]    # Destination node of call j
                
                # Travel cost from origin i to origin j
                travel_cost_oo = problem['travel_time_cost'].get((v_idx, origin_i, origin_j), [0, 0])[1]
                # Travel cost from origin i to destination j
                travel_cost_od = problem['travel_time_cost'].get((v_idx, origin_i, dest_j), [0, 0])[1]
                # Travel cost from destination i to origin j
                travel_cost_do = problem['travel_time_cost'].get((v_idx, dest_i, origin_j), [0, 0])[1]
                # Travel cost from destination i to destination j
                travel_cost_dd = problem['travel_time_cost'].get((v_idx, dest_i, dest_j), [0, 0])[1]
                
                max_cost = max(travel_cost_oo, travel_cost_od, travel_cost_do, travel_cost_dd)
                max_travel_cost = max(max_travel_cost, max_cost)
    
    # 2. Biggest time window difference
    max_pickup_tw_diff = 0
    max_delivery_tw_diff = 0
    
    for i in range(n_calls):
        pickup_early_i = call_info[i][5]    # Earliest pickup for call i
        pickup_late_i = call_info[i][6]     # Latest pickup for call i
        delivery_early_i = call_info[i][7]  # Earliest delivery for call i
        delivery_late_i = call_info[i][8]   # Latest delivery for call i
        
        for j in range(n_calls):
            pickup_early_j = call_info[j][5]    # Earliest pickup for call j
            pickup_late_j = call_info[j][6]     # Latest pickup for call j
            delivery_early_j = call_info[j][7]  # Earliest delivery for call j
            delivery_late_j = call_info[j][8]   # Latest delivery for call j
            
            pickup_tw_diff = abs(pickup_early_i - pickup_early_j) + abs(pickup_late_i - pickup_late_j)
            delivery_tw_diff = abs(delivery_early_i - delivery_early_j) + abs(delivery_late_i - delivery_late_j)
            
            max_pickup_tw_diff = max(max_pickup_tw_diff, pickup_tw_diff)
            max_delivery_tw_diff = max(max_delivery_tw_diff, delivery_tw_diff)
    
    max_tw_diff = max(max_pickup_tw_diff, max_delivery_tw_diff)
    
    # 3. Biggest cargo size difference
    max_cargo_size_diff = 0
    
    for i in range(n_calls):
        cargo_size_i = call_info[i][3]  # Size of call i
        
        for j in range(n_calls):
            cargo_size_j = call_info[j][3]  # Size of call j
            cargo_size_diff = abs(cargo_size_i - cargo_size_j)
            max_cargo_size_diff = max(max_cargo_size_diff, cargo_size_diff)
    
    # Calculate relatedness scores for each pair
    for i in range(n_calls):
        origin_i = call_info[i][1]      # Origin node of call i
        dest_i = call_info[i][2]        # Destination node of call i
        cargo_size_i = call_info[i][3]  # Size of call i
        pickup_early_i = call_info[i][5]    # Earliest pickup for call i
        pickup_late_i = call_info[i][6]     # Latest pickup for call i
        delivery_early_i = call_info[i][7]  # Earliest delivery for call i
        delivery_late_i = call_info[i][8]   # Latest delivery for call i
        
        for j in range(n_calls):
            if i == j:
                # A call is perfectly related to itself
                relatedness_matrix[i, j] = 1.0
                continue
                
            origin_j = call_info[j][1]      # Origin node of call j
            dest_j = call_info[j][2]        # Destination node of call j
            cargo_size_j = call_info[j][3]  # Size of call j
            pickup_early_j = call_info[j][5]    # Earliest pickup for call j
            pickup_late_j = call_info[j][6]     # Latest pickup for call j
            delivery_early_j = call_info[j][7]  # Earliest delivery for call j
            delivery_late_j = call_info[j][8]   # Latest delivery for call j
            
            # 1. Travel cost (use average from all vehicles)
            avg_travel_cost = 0
            count = 0
            
            for v_idx in range(1, n_vehicles + 1):
                # Check if both calls can be transported by this vehicle
                if i+1 in problem['vehicle_calls'][v_idx] and j+1 in problem['vehicle_calls'][v_idx]:
                    # Travel cost from origin i to origin j
                    oo_key = (v_idx, origin_i, origin_j)
                    travel_cost_oo = problem['travel_time_cost'].get(oo_key, [0, 0])[1]
                    
                    # Travel cost from origin i to destination j
                    od_key = (v_idx, origin_i, dest_j)
                    travel_cost_od = problem['travel_time_cost'].get(od_key, [0, 0])[1]
                    
                    # Travel cost from destination i to origin j
                    do_key = (v_idx, dest_i, origin_j)
                    travel_cost_do = problem['travel_time_cost'].get(do_key, [0, 0])[1]
                    
                    # Travel cost from destination i to destination j
                    dd_key = (v_idx, dest_i, dest_j)
                    travel_cost_dd = problem['travel_time_cost'].get(dd_key, [0, 0])[1]
                    
                    # Average travel cost between these calls for this vehicle
                    vehicle_avg = (travel_cost_oo + travel_cost_od + travel_cost_do + travel_cost_dd) / 4
                    avg_travel_cost += vehicle_avg
                    count += 1
            
            # If no compatible vehicles, use maximum travel cost as a penalty
            if count == 0:
                avg_travel_cost = max_travel_cost
            else:
                avg_travel_cost /= count
            
            # Invert the normalized costs to get relatedness (lower cost = higher relatedness)
            norm_travel_cost = 1 - (min(1.0, avg_travel_cost / max_travel_cost) if max_travel_cost > 0 else 0)
            
            # 2. Time window differences - normalize and invert (smaller difference = higher relatedness)
            pickup_tw_diff = abs(pickup_early_i - pickup_early_j) + abs(pickup_late_i - pickup_late_j)
            delivery_tw_diff = abs(delivery_early_i - delivery_early_j) + abs(delivery_late_i - delivery_late_j)
            
            norm_pickup_tw = 1 - (min(1.0, pickup_tw_diff / max_tw_diff) if max_tw_diff > 0 else 0)
            norm_delivery_tw = 1 - (min(1.0, delivery_tw_diff / max_tw_diff) if max_tw_diff > 0 else 0)
            
            # 3. Cargo size difference - normalize and invert (smaller difference = higher relatedness)
            cargo_size_diff = abs(cargo_size_i - cargo_size_j)
            norm_cargo_size = 1 - (min(1.0, cargo_size_diff / max_cargo_size_diff) if max_cargo_size_diff > 0 else 0)
            
            # Calculate final relatedness score using provided weights
            relatedness_score = (
                0.4 * norm_travel_cost + 
                0.2 * norm_pickup_tw + 
                0.2 * norm_delivery_tw + 
                0.2 * norm_cargo_size
            )
            
            relatedness_matrix[i, j] = relatedness_score
    
    return relatedness_matrix

# Removal functions
def worst_removal(vehicles, problem, k):
    n_vehicles = problem['n_vehicles']
   
    # Find all calls currently assigned to vehicles
    call_locations = []
    
    # Iterate through all vehicles and their calls
    for v_idx, vehicle in enumerate(vehicles):
        call_counts = {}
        for pos, call in enumerate(vehicle):
            if call not in call_counts:
                call_counts[call] = [pos]  # First occurrence (pickup)
            else:
                call_counts[call].append(pos)  # Second occurrence (delivery)
        
        # Add only complete pickup-delivery pairs
        for call, positions in call_counts.items():
            if len(positions) == 2:  # Both pickup and delivery
                call_locations.append((v_idx, call, positions[0], positions[1]))
    
    # Calculate cost contribution of each call
    call_costs = []
    
    for v_idx, call, pickup_pos, delivery_pos in call_locations:
        # Skip dummy vehicle calls for detailed cost calculation - use penalty directly
        if v_idx == n_vehicles:
            # Get penalty cost from call info
            penalty_cost = problem['call_info'][call-1][4]  # Penalty is at index 4
            call_costs.append((v_idx, call, penalty_cost))
            continue
        
        # Calculate cost with the call
        vehicle = vehicles[v_idx]
        # Utils2 cost_helper expects 1-indexed vehicle number
        current_cost = cost_helper(vehicle, problem, v_idx+1)
        
        # Calculate cost without the call
        vehicle_without_call = [c for c in vehicle if c != call]
        # Utils2 cost_helper expects 1-indexed vehicle number
        cost_without_call = cost_helper(vehicle_without_call, problem, v_idx+1)
        
        # The cost contribution is the difference
        cost_contribution = current_cost - cost_without_call
        
        # If removing a call improves cost, prioritize it highly
        if cost_contribution <= 0:
            # Small positive value to avoid negative weights in selection
            cost_contribution = 0.1
            
        call_costs.append((v_idx, call, cost_contribution))
    
    # Nothing to remove
    if not call_costs:
        return vehicles, []
    
    # Sort by cost contribution (highest first)
    call_costs.sort(key=lambda x: x[2], reverse=True)
    
    # Calculate selection probabilities using exponential weighting
    total_cost = sum(cost for _, _, cost in call_costs)
    if total_cost > 0:
        weights = [cost/total_cost for _, _, cost in call_costs]
        total_weight = sum(weights)
        probs = [w/total_weight for w in weights]
    else:
        # Equal probabilities if all costs are 0
        probs = [1.0/len(call_costs)] * len(call_costs)
    
    # Use weighted random selection to choose k calls
    k = min(k, len(call_costs))
    indices_to_remove = np.random.choice(
        len(call_costs), 
        size=k, 
        replace=False, 
        p=probs
    )
    
    calls_to_remove = [call_costs[i][:2] for i in indices_to_remove]
    
    # Remove the selected calls (delivery first, then pickup to avoid index shifts)
    removed_calls = []
    
    for v_idx, call in calls_to_remove:
        # Find positions of the call
        pickup_pos = delivery_pos = None
        for pos, c in enumerate(vehicles[v_idx]):
            if c == call:
                if pickup_pos is None:
                    pickup_pos = pos
                else:
                    delivery_pos = pos
                    break
        
        if pickup_pos is not None and delivery_pos is not None:
            # Remove delivery first (higher index)
            vehicles[v_idx].pop(delivery_pos)
            # Then remove pickup
            vehicles[v_idx].pop(pickup_pos)
            removed_calls.append(call)

    call_info = build_call_info(removed_calls, n_vehicles, problem)
    
    return vehicles, call_info

def random_removal(vehicles, problem, k):
    n_vehicles = problem['n_vehicles']
    
    # Get all calls with their vehicle indices
    call_locations = []
    
    # Find all calls and their vehicle locations
    for v_idx, vehicle in enumerate(vehicles):
        call_counts = {}
        for pos, call in enumerate(vehicle):
            if call not in call_counts:
                call_counts[call] = [pos]  # First occurrence (pickup)
            else:
                call_counts[call].append(pos)  # Second occurrence (delivery)
        
        # Add only complete pickup-delivery pairs
        for call, positions in call_counts.items():
            if len(positions) == 2:  # Both pickup and delivery
                call_locations.append((v_idx, call))
    
    # Separate dummy and regular calls
    dummy_calls = [(v_idx, call) for v_idx, call in call_locations if v_idx == n_vehicles]
    regular_calls = [(v_idx, call) for v_idx, call in call_locations if v_idx != n_vehicles]
    
    # Determine how many calls to remove
    k = min(k, len(call_locations))
    calls_to_remove = []
    
    # If there are calls in the dummy vehicle
    if dummy_calls:
        # Higher probability (e.g., 70%) to remove from dummy if available
        dummy_probability = 0.7
        
        for _ in range(k):
            if dummy_calls and (random.random() < dummy_probability or not regular_calls):
                # Select from dummy vehicle
                selected_idx = random.randrange(len(dummy_calls))
                selected_location = dummy_calls.pop(selected_idx)
                calls_to_remove.append(selected_location)
            elif regular_calls:
                # Select from regular vehicles
                selected_idx = random.randrange(len(regular_calls))
                selected_location = regular_calls.pop(selected_idx)
                calls_to_remove.append(selected_location)
            else:
                # Both lists are empty, break
                break
    else:
        # No calls in dummy, use standard random selection
        num_to_select = min(k, len(regular_calls))
        calls_to_remove = random.sample(regular_calls, num_to_select)
    
    # Remove the selected calls
    removed_calls = []
    
    for v_idx, call in calls_to_remove:
        # Find positions of the call
        pickup_pos = delivery_pos = None
        for pos, c in enumerate(vehicles[v_idx]):
            if c == call:
                if pickup_pos is None:
                    pickup_pos = pos
                else:
                    delivery_pos = pos
                    break
        
        if pickup_pos is not None and delivery_pos is not None:
            # Remove delivery first (higher index)
            vehicles[v_idx].pop(delivery_pos)
            # Then remove pickup
            vehicles[v_idx].pop(pickup_pos)
            removed_calls.append(call)
    
    # Build call info for removed calls
    call_info = build_call_info(removed_calls, n_vehicles, problem)
    
    return vehicles, call_info

def related_removal(vehicles, problem, k, relatedness_matrix):
    
    n_vehicles = problem['n_vehicles']
    
    # Get all calls currently in solution
    call_locations = []
    for v_idx, vehicle in enumerate(vehicles):
        call_counts = {}
        for pos, call in enumerate(vehicle):
            if call not in call_counts:
                call_counts[call] = [pos]  # First occurrence (pickup)
            else:
                call_counts[call].append(pos)  # Second occurrence (delivery)
        
        # Add only complete pickup-delivery pairs
        for call, positions in call_counts.items():
            if len(positions) == 2:  # Both pickup and delivery
                call_locations.append((v_idx, call, positions[0], positions[1]))
    
    # If no calls, return
    if not call_locations:
        return vehicles, []
    
    # Map actual calls to their indices in the relatedness matrix
    # Matrix is 0-indexed but calls are 1-indexed
    call_to_index = {call: call-1 for _, call, _, _ in call_locations}
    
    # Select a random first call to remove as the seed
    seed_location = random.choice(call_locations)
    seed_v_idx, seed_call, _, _ = seed_location
    
    # Keep track of selected calls
    selected_calls = [(seed_v_idx, seed_call)]
    remaining_calls = [(v_idx, call) for v_idx, call, _, _ in call_locations 
                      if call != seed_call or v_idx != seed_v_idx]
    
    # Select k-1 more calls with weighted probabilistic selection
    for _ in range(min(k-1, len(remaining_calls))):
        if not remaining_calls:
            break
        
        # Calculate relatedness scores for all remaining calls
        relatedness_scores = []
        
        for v_idx, call in remaining_calls:
            call_idx = call_to_index[call]
            
            # Find highest relatedness to any already selected call
            max_relatedness = -1
            for selected_v_idx, selected_call in selected_calls:
                selected_idx = call_to_index[selected_call]
                # Use relatedness score directly - higher means more related
                relatedness = relatedness_matrix[call_idx, selected_idx]
                max_relatedness = max(max_relatedness, relatedness)
            
            relatedness_scores.append(max_relatedness)
        
        # Apply exponential weighting to emphasize differences in relatedness
        # Higher alpha = more deterministic (favors higher scores more strongly)
        alpha = 2
        weights = [score ** alpha for score in relatedness_scores]
        
        # Handle the case where all weights are zero
        if sum(weights) <= 0:
            weights = [1.0] * len(remaining_calls)
            
        # Normalize to get probabilities
        total_weight = sum(weights)
        probabilities = [w / total_weight for w in weights]
        
        # Select next call based on probabilities
        selected_idx = np.random.choice(len(remaining_calls), p=probabilities)
        selected_call = remaining_calls.pop(selected_idx)
        selected_calls.append(selected_call)
    
    # Remove the selected calls (delivery first, then pickup to avoid index shifts)
    removed_calls = []
    
    for v_idx, call in selected_calls:
        # Find positions of the call
        pickup_pos = delivery_pos = None
        for pos, c in enumerate(vehicles[v_idx]):
            if c == call:
                if pickup_pos is None:
                    pickup_pos = pos
                else:
                    delivery_pos = pos
                    break
        
        if pickup_pos is not None and delivery_pos is not None:
            # Remove delivery first (higher index)
            vehicles[v_idx].pop(delivery_pos)
            # Then remove pickup
            vehicles[v_idx].pop(pickup_pos)
            removed_calls.append(call)
    
    # Build call info for removed calls
    call_info = build_call_info(removed_calls, n_vehicles, problem)
    
    return vehicles, call_info

def segment_removal(vehicles, problem, k):
    """
    Remove a segment of consecutive calls from a route, probabilistically
    selecting the segment based on its cost contribution (higher cost = higher probability).
    """
    n_vehicles = problem['n_vehicles']
    potential_segments = [] # Store segment info

    # --- Evaluate potential segments across all valid vehicles ---
    for v_idx, vehicle in enumerate(vehicles):
        # Only consider non-dummy vehicles with enough calls for a segment
        if v_idx == n_vehicles or len(vehicle) < 4: # Need at least 2 calls (4 entries)
            continue

        original_vehicle_cost = cost_helper(vehicle, problem, v_idx + 1)
        if original_vehicle_cost == float('inf'): # Skip infeasible vehicles
             continue

        # Find positions of all calls
        call_positions = {}
        for pos, call in enumerate(vehicle):
            if call not in call_positions:
                call_positions[call] = [pos]
            else:
                call_positions[call].append(pos)

        # Get complete calls sorted by pickup position
        complete_calls = []
        for call, positions in call_positions.items():
            if len(positions) == 2:
                complete_calls.append({"id": call, "p_pos": positions[0], "d_pos": positions[1]})
        if len(complete_calls) < 2: continue # Need at least 2 calls for a segment
        complete_calls.sort(key=lambda x: x["p_pos"])

        # Iterate through possible start calls
        for start_call_idx in range(len(complete_calls)):
            # Iterate through possible segment lengths (2 to k)
            lower_bound_len = 2
            for seg_len in range(lower_bound_len, min(k, len(complete_calls) - start_call_idx) + 1):
                end_call_idx = start_call_idx + seg_len - 1

                # Define the segment boundaries in the original vehicle list
                segment_start_pos = complete_calls[start_call_idx]["p_pos"]
                segment_end_pos_exclusive = complete_calls[end_call_idx]["d_pos"] + 1
                # segment_list = vehicle[segment_start_pos:segment_end_pos_exclusive] # Not needed for return
                calls_in_segment_set = set(call["id"] for call in complete_calls[start_call_idx : end_call_idx + 1])

                # Create vehicle without this segment
                vehicle_without_segment = [c for c in vehicle if c not in calls_in_segment_set]

                # Check feasibility of remaining route
                is_feas_remaining, _ = feasibility_helper(vehicle_without_segment, problem, v_idx + 1)
                if not is_feas_remaining:
                    continue # Removing this segment makes the rest infeasible

                cost_without_segment = cost_helper(vehicle_without_segment, problem, v_idx + 1)
                cost_contribution = original_vehicle_cost - cost_without_segment

                # Store potential segment info (only need v_idx and call IDs)
                potential_segments.append({
                    "v_idx": v_idx,
                    "cost": cost_contribution,
                    "calls": list(calls_in_segment_set) # Store the call IDs removed
                })

    # --- Select a segment to remove ---
    if not potential_segments:
        # Fallback if no valid segments found
        return random_removal(vehicles, problem, k)

    # Use probabilistic selection based on cost contribution
    costs = np.array([seg['cost'] for seg in potential_segments])
    min_cost = np.min(costs)
    if min_cost <= 0:
        costs += abs(min_cost) + 1 # Shift costs to be positive

    total_cost = np.sum(costs)
    if total_cost > 0:
        probs = costs / total_cost
        probs /= np.sum(probs) # Ensure sum to 1
    else:
        probs = np.ones(len(potential_segments)) / len(potential_segments)

    # Choose a segment index
    chosen_idx = np.random.choice(len(potential_segments), p=probs)
    segment_to_remove = potential_segments[chosen_idx]

    # --- Perform the removal ---
    v_idx_to_modify = segment_to_remove["v_idx"]
    calls_removed_set = set(segment_to_remove["calls"])
    vehicles[v_idx_to_modify] = [c for c in vehicles[v_idx_to_modify] if c not in calls_removed_set]

    removed_call_ids = segment_to_remove["calls"]

    # Build call_info needed for insertion operators
    call_info = build_call_info(removed_call_ids, n_vehicles, problem)

    return vehicles, call_info

def segment_removal_for_segment_insertion(vehicles, problem, k):
    n_vehicles = problem['n_vehicles']
    potential_segments = [] # Store (v_idx, start_pos, end_pos_exclusive, cost_contribution, segment_list)

    # --- Evaluate potential segments across all valid vehicles ---
    for v_idx, vehicle in enumerate(vehicles):
        # Only consider non-dummy vehicles with enough calls for a segment
        if v_idx == n_vehicles or len(vehicle) < 4:
            continue

        original_vehicle_cost = cost_helper(vehicle, problem, v_idx + 1)
        if original_vehicle_cost == float('inf'): # Skip infeasible vehicles
             continue

        # Find positions of all calls
        call_positions = {}
        for pos, call in enumerate(vehicle):
            if call not in call_positions:
                call_positions[call] = [pos]
            else:
                call_positions[call].append(pos)

        # Get complete calls sorted by pickup position
        complete_calls = []
        for call, positions in call_positions.items():
            if len(positions) == 2:
                complete_calls.append({"id": call, "p_pos": positions[0], "d_pos": positions[1]})
        if len(complete_calls) < 2: continue # Need at least 2 calls for a segment
        complete_calls.sort(key=lambda x: x["p_pos"])

        # Iterate through possible start calls
        for start_call_idx in range(len(complete_calls)):
            # Iterate through possible segment lengths (2 to k)
            # Ensure lower bound is at least 2
            lower_bound_len = 2
            for seg_len in range(lower_bound_len, min(k, len(complete_calls) - start_call_idx) + 1):
                end_call_idx = start_call_idx + seg_len - 1

                # Define the segment boundaries in the original vehicle list
                segment_start_pos = complete_calls[start_call_idx]["p_pos"]
                segment_end_pos_exclusive = complete_calls[end_call_idx]["d_pos"] + 1
                segment_list = vehicle[segment_start_pos:segment_end_pos_exclusive]
                calls_in_segment_set = set(call["id"] for call in complete_calls[start_call_idx : end_call_idx + 1])

                # Create vehicle without this segment
                vehicle_without_segment = [c for c in vehicle if c not in calls_in_segment_set]

                # Check feasibility of remaining route (important!)
                is_feas_remaining, _ = feasibility_helper(vehicle_without_segment, problem, v_idx + 1)
                if not is_feas_remaining:
                    continue # Removing this segment makes the rest infeasible

                cost_without_segment = cost_helper(vehicle_without_segment, problem, v_idx + 1)
                cost_contribution = original_vehicle_cost - cost_without_segment

                # Store potential segment info
                potential_segments.append({
                    "v_idx": v_idx,
                    "start": segment_start_pos,
                    "end": segment_end_pos_exclusive,
                    "cost": cost_contribution,
                    "segment": segment_list,
                    "calls": list(calls_in_segment_set) # Store the call IDs removed
                })

    # --- Select a segment to remove ---
    if not potential_segments:
        # Fallback if no valid segments found
        vehicles, call_info = random_removal(vehicles, problem, k)
        return vehicles, call_info, []

    # Use probabilistic selection based on cost contribution (higher cost = higher probability)
    costs = np.array([seg['cost'] for seg in potential_segments])
    # Handle non-positive costs - give them a small chance
    min_cost = np.min(costs)
    if min_cost <= 0:
        costs += abs(min_cost) + 1 # Shift all costs to be positive

    total_cost = np.sum(costs)
    if total_cost > 0:
        probs = costs / total_cost
        # Ensure probabilities sum to 1
        probs /= np.sum(probs)
    else:
        # Equal probability if all adjusted costs are somehow zero
        probs = np.ones(len(potential_segments)) / len(potential_segments)

    # Choose a segment index
    chosen_idx = np.random.choice(len(potential_segments), p=probs)
    segment_to_remove = potential_segments[chosen_idx]

    # --- Perform the removal ---
    v_idx_to_modify = segment_to_remove["v_idx"]
    calls_removed_set = set(segment_to_remove["calls"])
    vehicles[v_idx_to_modify] = [c for c in vehicles[v_idx_to_modify] if c not in calls_removed_set]

    removed_segment_list = segment_to_remove["segment"]
    removed_call_ids = segment_to_remove["calls"]

    # Build call_info needed for the fallback greedy insertion
    call_info = build_call_info(removed_call_ids, n_vehicles, problem)

    return vehicles, call_info, removed_segment_list

def shaw_removal(vehicles, problem, k, relatedness_matrix, shaw_p=3):
    """
    Remove k related calls using the Shaw Removal heuristic.
    Relates candidate calls to the set of already removed calls.
    Uses a determinism parameter 'shaw_p' to control randomness.

    Args:
        vehicles: List of vehicle routes.
        problem: Problem dictionary.
        k: Number of calls to remove.
        relatedness_matrix: Pre-computed relatedness scores between calls (0-indexed).
        shaw_p: Determinism parameter (higher value = more deterministic).

    Returns:
        vehicles: Modified vehicle routes.
        call_info: Call information for removed calls.
    """
    n_vehicles = problem['n_vehicles']

    # 1. Get all calls currently in solution (including vehicle index)
    call_locations = []
    for v_idx, vehicle in enumerate(vehicles):
        call_counts = {}
        for pos, call in enumerate(vehicle):
            if call not in call_counts:
                call_counts[call] = [pos]  # Pickup
            else:
                call_counts[call].append(pos)  # Delivery
        for call, positions in call_counts.items():
            if len(positions) == 2:  # Complete pair
                call_locations.append((v_idx, call))

    if not call_locations:
        return vehicles, [] # Nothing to remove

    # Map call IDs (1-based) to matrix indices (0-based)
    call_to_index = {call: call - 1 for _, call in call_locations}

    # 2. Select a random first call to remove (seed)
    seed_location_idx = random.randrange(len(call_locations))
    seed_v_idx, seed_call = call_locations.pop(seed_location_idx) # Remove seed from available calls

    removed_set = [(seed_v_idx, seed_call)] # Store (v_idx, call_id) of removed calls
    remaining_calls = call_locations # Calls still available to be removed

    # 3. Iteratively select k-1 more calls
    for _ in range(min(k - 1, len(remaining_calls))):
        if not remaining_calls:
            break

        # Calculate relatedness of each remaining call to the *set* of already removed calls
        candidate_scores = []
        for cand_v_idx, cand_call in remaining_calls:
            cand_idx = call_to_index[cand_call]
            total_relatedness = 0
            for rem_v_idx, rem_call in removed_set:
                rem_idx = call_to_index[rem_call]
                # Use pre-calculated relatedness (higher score = more related)
                total_relatedness += relatedness_matrix[cand_idx, rem_idx]

            # Average relatedness to the removed set
            avg_relatedness = total_relatedness / len(removed_set) if removed_set else 0
            candidate_scores.append(avg_relatedness)

        # Select the next call probabilistically based on relatedness
        # Use determinism parameter shaw_p
        weights = [score ** shaw_p for score in candidate_scores]

        # Handle cases with zero weights
        if sum(weights) <= 0:
            # If all scores are 0 or negative, use uniform probability
            probabilities = [1.0 / len(remaining_calls)] * len(remaining_calls)
        else:
            total_weight = sum(weights)
            probabilities = [w / total_weight for w in weights]

        # Choose index based on probability
        chosen_idx = np.random.choice(len(remaining_calls), p=probabilities)

        # Move chosen call from remaining to removed set
        chosen_v_idx, chosen_call = remaining_calls.pop(chosen_idx)
        removed_set.append((chosen_v_idx, chosen_call))

    # 4. Physically remove the selected calls from the vehicles
    removed_call_ids = []
    calls_to_remove_dict = {} # Group by vehicle index for efficient removal
    for v_idx, call in calls_to_remove_dict:
        if v_idx not in calls_to_remove_dict:
            calls_to_remove_dict[v_idx] = set()
        calls_to_remove_dict[v_idx].add(call)
        removed_call_ids.append(call)

    for v_idx, calls in calls_to_remove_dict.items():
        original_vehicle = vehicles[v_idx]
        vehicles[v_idx] = [c for c in original_vehicle if c not in calls]

    # 5. Build call info for the removed calls
    call_info = build_call_info(removed_call_ids, n_vehicles, problem)

    return vehicles, call_info

def historical_removal(vehicles, problem, k, call_blame):
    """
    Removes k calls based on their historical 'blame' score.
    Calls with higher blame scores are more likely to be removed.

    Args:
        vehicles: List of vehicle routes.
        problem: Problem dictionary.
        k: Number of calls to remove.
        call_blame (dict): Dictionary mapping call_id to its blame score.

    Returns:
        vehicles: Modified vehicle routes.
        call_info: Call information for removed calls.
    """
    n_vehicles = problem['n_vehicles']

    # 1. Get all calls currently in solution (including vehicle index)
    call_locations = get_calls(vehicles) # Use helper function

    if not call_locations:
        return vehicles, [] # Nothing to remove

    # 2. Get blame scores for current calls
    calls_with_blame = []
    for v_idx, call_id in call_locations:
        blame_score = call_blame.get(call_id, 0) # Default to 0 if no blame recorded
        calls_with_blame.append({'v_idx': v_idx, 'call': call_id, 'blame': blame_score})

    # 3. Select k calls probabilistically based on blame
    num_to_remove = min(k, len(calls_with_blame))
    if num_to_remove == 0:
        return vehicles, []

    blame_scores = np.array([c['blame'] for c in calls_with_blame])

    # Use exponential weighting or simple scores? Let's use scores directly + small constant
    weights = blame_scores + 1 # Add 1 to give non-zero probability to calls with 0 blame

    if np.sum(weights) <= 0: # Handle case where all weights are zero (e.g., early iterations)
        probabilities = np.ones(len(calls_with_blame)) / len(calls_with_blame)
    else:
        probabilities = weights / np.sum(weights)
        # Ensure probabilities sum to 1 due to potential floating point issues
        probabilities /= np.sum(probabilities)


    selected_indices = np.random.choice(
        len(calls_with_blame),
        size=num_to_remove,
        replace=False, # Don't remove the same call twice
        p=probabilities
    )

    calls_to_remove_info = [calls_with_blame[i] for i in selected_indices]

    # 4. Physically remove the selected calls
    removed_call_ids = []
    calls_to_remove_dict = {} # Group by vehicle index for efficient removal
    for item in calls_to_remove_info:
        v_idx = item['v_idx']
        call = item['call']
        if v_idx not in calls_to_remove_dict:
            calls_to_remove_dict[v_idx] = set()
        calls_to_remove_dict[v_idx].add(call)
        removed_call_ids.append(call)

    for v_idx, calls_set in calls_to_remove_dict.items():
        original_vehicle = vehicles[v_idx]
        vehicles[v_idx] = [c for c in original_vehicle if c not in calls_set]

    # 5. Build call info for the removed calls
    call_info = build_call_info(removed_call_ids, n_vehicles, problem)

    return vehicles, call_info

# Insertion functions
def random_insertion(vehicles, call_info, problem):
    n_vehicles = problem['n_vehicles']

    for call, compatible_vehicles in call_info:
        inserted = False
        
        # Choose a random compatible vehicle
        v_idx = random.choice(compatible_vehicles)
        
        # Special case for dummy vehicle
        if v_idx == n_vehicles:
            vehicles[n_vehicles].append(call)
            vehicles[n_vehicles].append(call)
            vehicles[n_vehicles].sort()
            continue
            
        # Get the actual vehicle route - NO SUBTRACTION NEEDED for v_idx
        vehicle = vehicles[v_idx]
        
        # Try positions until we find a feasible insertion
        for p_idx in range(len(vehicle) + 1):
            if inserted:
                break
                
            for d_idx in range(p_idx + 1, len(vehicle) + 2):
                temp_vehicle = vehicle.copy()
                temp_vehicle.insert(p_idx, call)
                temp_vehicle.insert(d_idx, call)
                
                # Use correct v_idx+1 for feasibility check (Utils2 expects 1-indexed)
                is_feas, _ = feasibility_helper(temp_vehicle, problem, v_idx+1)
                
                if is_feas:
                    vehicles[v_idx] = temp_vehicle
                    inserted = True
                    break

        # If we couldn't insert feasibly in any regular vehicle, use dummy
        if not inserted:
            vehicles[n_vehicles].append(call)
            vehicles[n_vehicles].append(call)
            vehicles[n_vehicles].sort()

    return vehicles

def greedy_insertion(vehicles, call_info, problem):
    n_vehicles = problem['n_vehicles']
    
    # Process each call
    for call, compatible_vehicles in call_info:
        # Get regular compatible vehicles (excluding dummy)
        regular_vehicles = [v for v in compatible_vehicles if v < n_vehicles]
        
        # Determine vehicles to try
        vehicles_to_try = regular_vehicles
            
        # Always include dummy vehicle as fallback
        if n_vehicles in compatible_vehicles:
            vehicles_to_try.append(n_vehicles)
        
        # Skip if no compatible vehicles
        if not vehicles_to_try:
            continue
            
        # Track pool of good insertion positions
        insertion_pool = []
        best_delta = float('inf')
        
        # Try each vehicle
        for v_idx in vehicles_to_try:
            vehicle = vehicles[v_idx]
            
            # Special case for dummy vehicle
            if v_idx == n_vehicles:
                # Calculate penalty cost for adding this call to dummy
                dummy_delta = problem['call_info'][call-1][4]  # Penalty is at index 4 of call_info
                insertion_pool.append({
                    'v_idx': n_vehicles,
                    'p_idx': len(vehicle),
                    'd_idx': len(vehicle) + 1,
                    'delta': dummy_delta
                })
                if dummy_delta < best_delta:
                    best_delta = dummy_delta
                continue
            
            # Skip if call too big for vehicle
            # call_size = problem['call_info'][call-1][3]  # Size is at index 3 of call_info
            # vehicle_capacity = problem['vehicle_info'][v_idx][3]  # Capacity is at index 3 of vehicle_info
            # if call_size > vehicle_capacity:
            #     continue
            
            # Calculate base cost for this vehicle using Utils2's helper function
            # Note: Utils2 expects 1-indexed vehicle numbers
            base_cost = cost_helper(vehicle, problem, v_idx+1)
            
            # Try ALL insertion positions
            vehicle_len = len(vehicle)
            
            for p_idx in range(vehicle_len + 1):
                for d_idx in range(p_idx, vehicle_len + 2):
                    # Create temporary vehicle
                    temp_vehicle = vehicle.copy()
                    temp_vehicle.insert(p_idx, call)
                    
                    # Adjust delivery index after pickup insertion
                    d_idx_adjusted = d_idx if d_idx <= p_idx else d_idx + 1
                    temp_vehicle.insert(d_idx_adjusted, call)
                    
                    # Check feasibility using Utils2's helper function
                    # Note: Utils2 expects 1-indexed vehicle numbers
                    is_feas, _ = feasibility_helper(temp_vehicle, problem, v_idx+1)
                    
                    if is_feas:
                        # Calculate cost delta using transport-only cost (more efficient)
                        # Note: Utils2 expects 1-indexed vehicle numbers
                        new_cost = cost_helper(temp_vehicle, problem, v_idx+1)
                        delta_cost = new_cost - base_cost
                        
                        # Add position to pool if good enough (within 200% of best)
                        if best_delta == float('inf') or delta_cost < 2.0 * best_delta:
                            insertion_pool.append({
                                'v_idx': v_idx,
                                'p_idx': p_idx,
                                'd_idx': d_idx,
                                'delta': delta_cost
                            })
                            
                        # Update best delta found
                        if delta_cost < best_delta:
                            best_delta = delta_cost
        
        # Select from pool using weighted probabilities
        if insertion_pool:
            # Sort by cost (lowest first)
            insertion_pool.sort(key=lambda x: x['delta'])
            
            # Calculate weights using exponential decay
            deltas = [pos['delta'] for pos in insertion_pool]
            
            # Handle normalization carefully
            if best_delta > 0:
                norm_deltas = [d/best_delta for d in deltas]
            else:
                # Handle negative or zero best delta
                min_positive = min([d for d in deltas if d > 0], default=0.1)
                norm_deltas = [(d - best_delta + min_positive) / min_positive for d in deltas]
            
            # Apply exponential weighting - adjustable parameter
            # -1.5 = more exploration, -3 = more greedy
            explore_factor = -2.0
            weights = [np.exp(explore_factor * nd) for nd in norm_deltas]
            
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
            vehicles[n_vehicles].append(call)
            vehicles[n_vehicles].append(call)

    return vehicles

def regret_insertion(vehicles, call_info, problem):
    """
    Insert calls using regret value with probabilistic selection.
    Higher regret calls have higher probability of being selected first,
    but lower regret calls still have a chance.
    """
    n_vehicles = problem['n_vehicles']
    
    # Process calls sorted by regret value (hardest to insert first)
    call_regrets = []
    
    for call, compatible_vehicles in call_info:
        # Calculate cost of best insertion position for each vehicle
        vehicle_best_costs = {}
        
        # Skip dummy vehicle for regret calculation
        for v_idx in [v for v in compatible_vehicles if v != n_vehicles]:
            vehicle = vehicles[v_idx]
            
            # Skip if call too big for vehicle
            call_size = problem['call_info'][call-1][3]
            vehicle_capacity = problem['vehicle_info'][v_idx][3]
            if call_size > vehicle_capacity:
                continue
                
            # Find best insertion for this vehicle
            best_cost = float('inf')
            for p_idx in range(len(vehicle) + 1):
                for d_idx in range(p_idx + 1, len(vehicle) + 2):
                    temp_vehicle = vehicle.copy()
                    temp_vehicle.insert(p_idx, call)
                    
                    d_idx_adjusted = d_idx if d_idx <= p_idx else d_idx + 1
                    temp_vehicle.insert(d_idx_adjusted, call)
                    
                    # Check feasibility
                    is_feas, _ = feasibility_helper(temp_vehicle, problem, v_idx+1)
                    if not is_feas:
                        continue
                        
                    # Calculate cost
                    cost_val = cost_helper(temp_vehicle, problem, v_idx+1)
                    if cost_val < best_cost:
                        best_cost = cost_val
            
            if best_cost < float('inf'):
                vehicle_best_costs[v_idx] = best_cost
                
        # Calculate regret value (difference between best and second best)
        if len(vehicle_best_costs) >= 2:
            sorted_costs = sorted(vehicle_best_costs.values())
            regret_value = sorted_costs[1] - sorted_costs[0]
        elif len(vehicle_best_costs) == 1:
            # If only one vehicle is feasible, high regret to prioritize
            regret_value = 10000
        else:
            # No feasible insertions, use dummy penalty
            regret_value = problem['call_info'][call-1][4]
            
        call_regrets.append((call, compatible_vehicles, regret_value))
    
    # Sort calls by regret value (highest first)
    call_regrets.sort(key=lambda x: x[2], reverse=True)
    
    # Create remaining calls to insert (we'll remove them as we go)
    remaining_calls = call_regrets.copy()
    
    # Insert calls probabilistically based on regret value
    while remaining_calls:
        # Calculate weights for each call's regret value
        weights = []
        for _, _, regret in remaining_calls:
            # Adjust this formula to control how much higher regret is favored
            # Higher exponent = more bias toward highest regret
            weight = regret ** 2  # Square the regret to give higher weights to higher regrets
            weights.append(weight)
        
        # Convert to probabilities
        if sum(weights) > 0:
            probs = [w / sum(weights) for w in weights]
        else:
            # If all weights are zero, use uniform distribution
            probs = [1.0 / len(remaining_calls)] * len(remaining_calls)
        
        # Select a call based on probabilities
        selected_idx = np.random.choice(len(remaining_calls), p=probs)
        call, compatible_vehicles, _ = remaining_calls.pop(selected_idx)
        
        # Insert this call using greedy insertion
        greedy_insertion(vehicles, [(call, compatible_vehicles)], problem)
    
    return vehicles

def try_insert_segment(vehicles, segment, problem):
    """
    Attempts to insert an entire segment into the best possible location.
    Returns (best_vehicles_state, best_cost_delta) or (None, float('inf')) if no feasible insertion.
    """
    n_vehicles = problem['n_vehicles']
    best_insertion_vehicles = None
    min_delta_cost = float('inf')

    # Get compatible vehicles for the *first* call in the segment (as a proxy)
    # A more robust check would ensure all calls are compatible, but this is simpler
    if not segment: return None, float('inf')
    first_call = segment[0]
    # Get 1-based compatible vehicle indices (including potential dummy n+1)
    compatible_vehicles_1_based = problem.get('vehicle_calls', {}).get(first_call+1, [])
    # Convert to 0-based indices AND filter out the dummy vehicle index (n_vehicles)
    compatible_vehicles_indices = [v-1 for v in compatible_vehicles_1_based if v <= n_vehicles] # Ensure v <= n_vehicles

    # Iterate through compatible *regular* vehicles only
    for v_idx in compatible_vehicles_indices:
        # Double check index validity before accessing (should be redundant now but safe)
        if v_idx < 0 or v_idx >= n_vehicles:
             print(f"Warning: Invalid v_idx {v_idx} encountered in try_insert_segment. Skipping.")
             continue

        vehicle = vehicles[v_idx]
        original_cost = cost_helper(vehicle, problem, v_idx + 1)

        # Iterate through all possible insertion points for the start of the segment
        for insert_pos in range(len(vehicle) + 1):
            temp_vehicle = vehicle[:insert_pos] + segment + vehicle[insert_pos:]

            # Check feasibility of the modified route
            is_feas, _ = feasibility_helper(temp_vehicle, problem, v_idx + 1)

            if is_feas:
                new_cost = cost_helper(temp_vehicle, problem, v_idx + 1)
                delta_cost = new_cost - original_cost

                if delta_cost < min_delta_cost:
                    min_delta_cost = delta_cost
                    # Store the state of *all* vehicles for this potential best insertion
                    temp_vehicles_state = [v.copy() for v in vehicles]
                    temp_vehicles_state[v_idx] = temp_vehicle
                    best_insertion_vehicles = temp_vehicles_state

    return best_insertion_vehicles, min_delta_cost

def segment_insertion(vehicles, call_info, removed_segment_ordered, problem):
    """
    Tries to insert the removed segment as a whole.
    If unsuccessful, falls back to greedy insertion for individual calls.
    """
    n_vehicles = problem['n_vehicles']

    # --- Try inserting the whole segment first ---
    best_segment_insertion_vehicles, segment_delta = try_insert_segment(vehicles, removed_segment_ordered, problem)

    if best_segment_insertion_vehicles is not None:
        # print(f"  Successfully inserted segment with delta {segment_delta:.2f}")
        return best_segment_insertion_vehicles # Return the state with the segment inserted
    else:
        # print("  Segment insertion failed, falling back to greedy.")
        # --- Fallback: Use greedy insertion for individual calls ---
        # The call_info should contain all calls from the segment
        return greedy_insertion(vehicles, call_info, problem)

# Operators
def worst_greedy(vehicles, problem, k):
    vehicles, removed_calls = worst_removal(vehicles, problem, k)
    vehicles = greedy_insertion(vehicles, removed_calls, problem)
    return vehicles

def random_greedy(vehicles, problem, k):
    vehicles, removed_calls = random_removal(vehicles, problem, k)
    vehicles = greedy_insertion(vehicles, removed_calls, problem)
    return vehicles

def related_greedy(vehicles, problem, k, relatedness_matrix):
    vehicles, removed_calls = related_removal(vehicles, problem, k, relatedness_matrix)
    vehicles = greedy_insertion(vehicles, removed_calls, problem)
    return vehicles

def segment_greedy(vehicles, problem, k):
    vehicles, removed_calls = segment_removal(vehicles, problem, k)
    vehicles = greedy_insertion(vehicles, removed_calls, problem)
    return vehicles

def worst_regret(vehicles, problem, k):
    vehicles, removed_calls = worst_removal(vehicles, problem, k)
    vehicles = regret_insertion(vehicles, removed_calls, problem)
    return vehicles

def random_regret(vehicles, problem, k):
    vehicles, removed_calls = random_removal(vehicles, problem, k)
    vehicles = regret_insertion(vehicles, removed_calls, problem)
    return vehicles

def segment_regret(vehicles, problem, k):
    vehicles, removed_calls = segment_removal(vehicles, problem, k)
    vehicles = regret_insertion(vehicles, removed_calls, problem)
    return vehicles

def related_regret(vehicles, problem, k, relatedness_matrix):
    vehicles, removed_calls = related_removal(vehicles, problem, k, relatedness_matrix)
    vehicles = regret_insertion(vehicles, removed_calls, problem)
    return vehicles

def shaw_greedy(vehicles, problem, k, relatedness_matrix, shaw_p=3):
    vehicles, removed_calls = shaw_removal(vehicles, problem, k, relatedness_matrix, shaw_p)
    vehicles = greedy_insertion(vehicles, removed_calls, problem)
    return vehicles

def shaw_regret(vehicles, problem, k, relatedness_matrix, shaw_p=3):
    vehicles, removed_calls = shaw_removal(vehicles, problem, k, relatedness_matrix, shaw_p)
    vehicles = regret_insertion(vehicles, removed_calls, problem)
    return vehicles

def segment_segment(vehicles, problem, k):
    # Make sure the removal function returns 3 values
    vehicles, call_info, removed_segment = segment_removal_for_segment_insertion(vehicles, problem, k)
    # Pass the correct removed_segment to the insertion function
    vehicles = segment_insertion(vehicles, call_info, removed_segment, problem)
    return vehicles

def historical_greedy(vehicles, problem, k, call_blame):
    vehicles, removed_calls = historical_removal(vehicles, problem, k, call_blame)
    vehicles = greedy_insertion(vehicles, removed_calls, problem)
    return vehicles

def historical_regret(vehicles, problem, k, call_blame):
    vehicles, removed_calls = historical_removal(vehicles, problem, k, call_blame)
    vehicles = regret_insertion(vehicles, removed_calls, problem)
    return vehicles

# Local search intensifier
def relocate_operator(vehicle, v_idx, problem):
    if len(vehicle) < 4: # Need at least 2 calls to relocate one
        return vehicle

    calls_in_route = sorted(list(set(c for c in vehicle if vehicle.count(c) == 2))) # Unique calls with P+D
    original_cost = cost_helper(vehicle, problem, v_idx + 1)
    best_vehicle = vehicle.copy()
    best_cost = original_cost
    found_improvement = False

    for call_to_move in calls_in_route:
        # Find current positions
        pickup_pos = -1
        delivery_pos = -1
        temp_indices = []
        for i, c in enumerate(vehicle):
            if c == call_to_move:
                temp_indices.append(i)
        if len(temp_indices) == 2:
             pickup_pos, delivery_pos = temp_indices[0], temp_indices[1]
        else:
             continue # Should not happen if call_counts was correct

        # Create route without the call
        temp_vehicle_base = [c for i, c in enumerate(vehicle) if i != pickup_pos and i != delivery_pos]

        # Try inserting pickup at all possible positions (ip)
        for ip in range(len(temp_vehicle_base) + 1):
            # Try inserting delivery at all possible positions after pickup (id)
            # Ensure delivery is after pickup
            for id_offset in range(len(temp_vehicle_base) - ip + 1):
                id_base = ip + id_offset # Index in temp_vehicle_base

                # Construct the potential new vehicle
                temp_vehicle = temp_vehicle_base[:ip] + [call_to_move] + temp_vehicle_base[ip:]
                # Insert delivery at the correct position relative to the inserted pickup
                temp_vehicle.insert(id_base + 1, call_to_move)


                # Check feasibility and cost
                is_feas, _ = feasibility_helper(temp_vehicle, problem, v_idx + 1)
                if is_feas:
                    new_cost = cost_helper(temp_vehicle, problem, v_idx + 1)
                    if new_cost < best_cost:
                        best_cost = new_cost
                        best_vehicle = temp_vehicle.copy()
                        found_improvement = True
                        # Keep searching for even better moves

    return best_vehicle

def exchange_operator(vehicle, v_idx, problem):
    if len(vehicle) < 8: # Need at least 4 calls (2 pairs) to swap
        return vehicle

    calls_in_route = sorted(list(set(c for c in vehicle if vehicle.count(c) == 2)))
    num_calls = len(calls_in_route)
    original_cost = cost_helper(vehicle, problem, v_idx + 1)
    best_vehicle = vehicle.copy()
    best_cost = original_cost
    found_improvement = False

    for i in range(num_calls):
        for j in range(i + 1, num_calls):
            call1 = calls_in_route[i]
            call2 = calls_in_route[j]

            # Find positions in the *original* vehicle for this iteration
            pos_p1, pos_d1 = -1, -1
            pos_p2, pos_d2 = -1, -1
            for k, c in enumerate(vehicle): # Use original vehicle to find indices
                if c == call1:
                    if pos_p1 == -1: pos_p1 = k
                    else: pos_d1 = k
                elif c == call2:
                    if pos_p2 == -1: pos_p2 = k
                    else: pos_d2 = k

            # Create a temporary vehicle based on the original for this swap attempt
            temp_vehicle = vehicle.copy()

            # Perform the swap in the copy
            # Ensure indices are valid before swapping
            if -1 in [pos_p1, pos_d1, pos_p2, pos_d2]: continue

            # Simple swap: Swap P1<->P2 and D1<->D2
            # Need to handle cases where indices might overlap after first swap if not careful
            # A safer way is to map values:
            val_p1, val_d1 = temp_vehicle[pos_p1], temp_vehicle[pos_d1]
            val_p2, val_d2 = temp_vehicle[pos_p2], temp_vehicle[pos_d2]

            temp_vehicle[pos_p1], temp_vehicle[pos_p2] = val_p2, val_p1
            temp_vehicle[pos_d1], temp_vehicle[pos_d2] = val_d2, val_d1


            # Check feasibility and cost
            is_feas, _ = feasibility_helper(temp_vehicle, problem, v_idx + 1)
            if is_feas:
                new_cost = cost_helper(temp_vehicle, problem, v_idx + 1)
                if new_cost < best_cost:
                    best_cost = new_cost
                    best_vehicle = temp_vehicle.copy()
                    found_improvement = True
                    # Keep searching for even better moves

    return best_vehicle

def two_opt_operator(vehicle, v_idx, problem):
    if len(vehicle) < 4: return vehicle # Need at least 2 calls for 2-opt

    original_cost = cost_helper(vehicle, problem, v_idx + 1)
    best_vehicle = vehicle.copy()
    improved = True

    while improved: # Keep applying 2-opt until no improvement
        improved = False
        current_best_cost = cost_helper(best_vehicle, problem, v_idx + 1)

        # Iterate through all valid segment start/end points
        # Ensure we only break between a delivery and the next pickup
        break_points = [i for i in range(1, len(best_vehicle) - 1) if i % 2 == 0] # Indices after deliveries

        for i_idx in range(len(break_points)):
            for j_idx in range(i_idx + 1, len(break_points)):
                start_node_idx = break_points[i_idx]
                end_node_idx = break_points[j_idx]

                # Create new route by reversing the segment
                temp_vehicle = best_vehicle[:start_node_idx] + best_vehicle[start_node_idx:end_node_idx][::-1] + best_vehicle[end_node_idx:]

                is_feas, _ = feasibility_helper(temp_vehicle, problem, v_idx + 1)
                if is_feas:
                    new_cost = cost_helper(temp_vehicle, problem, v_idx + 1)
                    if new_cost < current_best_cost:
                        best_vehicle = temp_vehicle.copy()
                        current_best_cost = new_cost
                        improved = True
                        # print(f"  2-opt improvement on V{v_idx+1}: {original_cost:.2f} -> {new_cost:.2f}")
                        break # Go back to outer loop to restart checks
            if improved: break
        if not improved: break # Exit while loop if no improvement in inner loops

    return best_vehicle

def inter_route_relocate_operator(vehicles, problem):
    """
    Tries to move a call from one vehicle to another.
    Uses a first improvement strategy across all possible moves.
    """
    n_vehicles = problem['n_vehicles']
    current_total_cost = cost(vehicles, problem) # Calculate initial total cost

    # Iterate through all source vehicles (excluding dummy)
    for v_idx_from in range(n_vehicles):
        vehicle_from = vehicles[v_idx_from]
        if not vehicle_from: continue # Skip empty vehicles

        calls_in_route = sorted(list(set(c for c in vehicle_from if vehicle_from.count(c) == 2)))

        for call_to_move in calls_in_route:
            # Find current positions in source vehicle
            pickup_pos = -1
            delivery_pos = -1
            temp_indices = []
            for i, c in enumerate(vehicle_from):
                if c == call_to_move:
                    temp_indices.append(i)
            if len(temp_indices) == 2:
                 pickup_pos, delivery_pos = temp_indices[0], temp_indices[1]
            else:
                 continue

            # Create source route without the call
            temp_vehicle_from = [c for i, c in enumerate(vehicle_from) if i != pickup_pos and i != delivery_pos]

            # Check feasibility and cost of the modified source route
            is_feas_from, _ = feasibility_helper(temp_vehicle_from, problem, v_idx_from + 1)
            if not is_feas_from: continue # If removing makes source infeasible, skip

            # Try inserting into all *other* compatible vehicles (including dummy)
            compatible_vehicles_for_call = problem.get('vehicle_calls', {}).get(call_to_move+1, []) # Get compatible vehicles for the call (1-based index)
            compatible_vehicles_indices = [v-1 for v in compatible_vehicles_for_call] # Convert to 0-based

            for v_idx_to in range(n_vehicles + 1): # Include dummy vehicle
                if v_idx_to == v_idx_from: continue # Don't move to the same vehicle
                if v_idx_to < n_vehicles and v_idx_to not in compatible_vehicles_indices: continue # Check compatibility for non-dummy

                vehicle_to = vehicles[v_idx_to]

                # Try inserting pickup at all possible positions (ip)
                for ip in range(len(vehicle_to) + 1):
                    # Try inserting delivery at all possible positions after pickup (id)
                    for id_offset in range(len(vehicle_to) - ip + 1):
                        id_base = ip + id_offset # Index in vehicle_to

                        # Construct the potential new destination vehicle
                        temp_vehicle_to = vehicle_to[:ip] + [call_to_move] + vehicle_to[ip:]
                        temp_vehicle_to.insert(id_base + 1, call_to_move)

                        # Check feasibility of the destination route
                        is_feas_to, _ = feasibility_helper(temp_vehicle_to, problem, v_idx_to + 1)

                        if is_feas_to:
                            # If both source (after removal) and destination (after insertion) are feasible,
                            # calculate the new *total* cost.
                            # Create a temporary full solution state
                            temp_vehicles = [v.copy() for v in vehicles]
                            temp_vehicles[v_idx_from] = temp_vehicle_from
                            temp_vehicles[v_idx_to] = temp_vehicle_to

                            new_total_cost = cost(temp_vehicles, problem)

                            if new_total_cost < current_total_cost:
                                # print(f"  Inter-Route Relocate improvement: {current_total_cost:.2f} -> {new_total_cost:.2f} (Call {call_to_move} from V{v_idx_from+1} to V{v_idx_to+1})")
                                return temp_vehicles # Return immediately on first improvement

    return vehicles # Return original if no improvement found

def apply_local_search(vehicles, problem):
    n_vehicles = problem['n_vehicles']
    intra_route_operators = [relocate_operator, exchange_operator, two_opt_operator]

    # --- Intra-Route VNS ---
    for v_idx in range(n_vehicles):
        vehicle_improved = True
        while vehicle_improved:
            vehicle_improved = False
            current_vehicle_cost = cost_helper(vehicles[v_idx], problem, v_idx + 1)
            
            # Shuffle operators for this iteration
            shuffled_operators = random.sample(intra_route_operators, len(intra_route_operators))

            for operator_func in shuffled_operators:
                new_vehicle = operator_func(vehicles[v_idx], v_idx, problem)
                new_vehicle_cost = cost_helper(new_vehicle, problem, v_idx + 1)

                if new_vehicle_cost < current_vehicle_cost:
                    vehicles[v_idx] = new_vehicle # Apply improvement
                    vehicle_improved = True
                    # print(f"    Intra LS Improvement on V{v_idx+1} using {operator_func.__name__}: {current_vehicle_cost:.2f} -> {new_vehicle_cost:.2f}")
                    break # Go back to the start of the while loop (re-shuffle and try again)
            # If no operator improved in this pass, the while loop terminates

    # --- Inter-Route LS (Applied after intra-route stabilizes) ---
    inter_route_improved = True
    while inter_route_improved:
         original_total_cost = cost(vehicles, problem)
         vehicles = inter_route_relocate_operator(vehicles, problem)
         new_total_cost = cost(vehicles, problem)
         if new_total_cost < original_total_cost:
             # print(f"   Inter LS Improvement: {original_total_cost:.2f} -> {new_total_cost:.2f}")
             inter_route_improved = True
         else:
             inter_route_improved = False

    return vehicles

# Algorithm
def adaptive_algorithm(initial_solution, problem, neighbours, relatedness_matrix, max_iter):
    # Initialize calls to vehicles as list of lists
    vehicles = initial_solution
    best_solution = [vehicle.copy() for vehicle in vehicles]  # Deep copy
    current_solution = [vehicle.copy() for vehicle in vehicles]  # Deep copy
    best_cost_value = cost(best_solution, problem)
    current_cost_value = best_cost_value
    feasibles = 0

    print(f'Initial cost: {current_cost_value:.2f}')
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
    # Ensure all operators in the current 'neighbours' list are included
    operator_weight_history = {op.__name__: [] for op in neighbours}
    operator_selection_counts = {op.__name__: 0 for op in neighbours}
    accepted_solutions = 0
    total_evaluations = 0

    # SA paramters
    T_initial = 0.1 * current_cost_value
    T_final = 0.1
    T = T_initial
    alpha = (T_final / T_initial) ** (1 / max_iter) if max_iter > 0 else 1.0 # Avoid division by zero

    # Initialize operator objects
    operators = [{"name": op.__name__, "func": op, "weight": 1.0,
                 "successes": 0, "attempts": 0, "best_improvements": 0, 'improvements': 0} for op in neighbours] # Start attempts at 0

    # --- Initialize call_blame ONLY ---
    call_blame = {} # Initialize blame dictionary
    # --- Ensure call_zones initialization is REMOVED ---

    # Main search loop
    iters_since_improvement = 0

    for iteration in range(max_iter):
        escape_threshold = 1000

        if iteration % 1000 == 0:
            print(f'Iteration: {iteration} | Current: {current_cost_value:.2f} | Best: {best_cost_value:.2f}')


        if iters_since_improvement >= escape_threshold:
            # print(f"Escape triggered at iteration {iteration}...")

            # Call the updated escape function
            new_current_solution, best_solution, best_cost_value, found_global = escape(
                current_solution, best_solution, best_cost_value, problem
            )

            # Update the current solution to the best one found during escape
            current_solution = [v.copy() for v in new_current_solution]
            current_cost_value = cost(current_solution, problem) # Recalculate cost

            if found_global:
                # Global best was already updated inside escape, just copy for safety
                best_solution = [v.copy() for v in best_solution]
                # print(f"Escape found new global best: {best_cost_value:.2f}")
                # Optional: Reset blame if escape found a new best
                # call_blame = {}
            # else:
                # print(f"Escape finished. New current cost: {current_cost_value:.2f}")

            iters_since_improvement = 0 # Reset counter after escape attempt

        # Update operator weights
        if iteration > 0 and iteration % 100 == 0: # Check iteration > 0
            update_operator_weights(operators)
            # Record operator weights every 100 iterations
            iteration_points.append(iteration)
            for op in operators:
                # Ensure the key exists before appending (should exist based on initialization)
                if op["name"] in operator_weight_history:
                     operator_weight_history[op["name"]].append(op["weight"])

       
        min_q = 2
        max_q = max(min_q, round(0.4 * problem['n_calls'])) 
        if problem['n_calls'] >= min_q:
            q = random.randint(min_q, max_q)
        else:
            q = problem['n_calls'] # If fewer than 2 calls, remove all

        # Select operator and apply
        total_weight = sum(op["weight"] for op in operators)
        # Ensure total_weight is positive to avoid division by zero
        if total_weight <= 0:
            print("Warning: Total operator weight is zero or negative. Resetting weights.")
            for op in operators: op['weight'] = 1.0
            total_weight = sum(op["weight"] for op in operators)

        # Calculate probabilities, handle potential division by zero if total_weight is still zero
        if total_weight > 0:
            probs = [op["weight"] / total_weight for op in operators]
            # Normalize probabilities due to potential floating point issues
            probs_sum = sum(probs)
            if abs(probs_sum - 1.0) > 1e-9:
                 probs = [p / probs_sum for p in probs]
        else:
            # Fallback to equal probability if total_weight is zero
            num_ops = len(operators)
            probs = [1.0 / num_ops] * num_ops if num_ops > 0 else []


        # Ensure probabilities list is not empty before choosing
        if not probs:
             print("Error: No operators available to select.")
             continue # Skip this iteration

        selected_op_idx = np.random.choice(len(operators), p=probs)
        selected_op = operators[selected_op_idx]
        operator_selection_counts[selected_op["name"]] += 1
        operators[selected_op_idx]["attempts"] += 1 # Increment attempt count HERE
        total_evaluations += 1


        new_solution = [vehicle.copy() for vehicle in current_solution]
        op_name = selected_op["name"]
        op_func = selected_op["func"]

        # --- Pass correct arguments based on operator name ---
        # Ensure q > 0 before calling operators that use it
        if q <= 0: continue

        try:
            if 'related' in op_name or 'shaw' in op_name:
                new_solution = op_func(new_solution, problem, q, relatedness_matrix)
            elif 'historical' in op_name:
                new_solution = op_func(new_solution, problem, q, call_blame) # Pass call_blame
            # --- Ensure 'zone' block is REMOVED ---
            else: # Other operators like random, worst, segment
                # Check if the operator expects q (most removal/insertion pairs do)
                # This might need adjustment based on specific operator signatures
                import inspect
                sig = inspect.signature(op_func)
                if 'k' in sig.parameters or 'q' in sig.parameters: # Check common names for number to remove
                     new_solution = op_func(new_solution, problem, q)
                else: # Operators that don't take q (e.g., pure local search if added here)
                     new_solution = op_func(new_solution, problem) # Adjust as needed
        except Exception as e:
            print(f"Error during operator execution ({op_name}): {e}")
            # Optionally skip this iteration or handle the error
            continue


        # --- Check feasibility and decide whether to accept ---
        is_feasible_lns = feasibility(new_solution, problem)
        # Get unique calls present in the solution *after* the operator ran
        calls_in_new_solution = set(call for veh in new_solution for call in veh if call != 0)

        if is_feasible_lns:
            feasibles += 1
            new_cost_value = cost(new_solution, problem) # Cost *after* LNS

            # Get cost of the solution *before* the LNS move
            cost_before_lns = current_cost_value # Use the stored current cost
            E = new_cost_value - cost_before_lns
            history["delta_value"].append(E)

            accepted_move = False
            # is_improving_move = False # Not strictly needed

            if E < 0: # Improving move
                operators[selected_op_idx]["improvements"] += 1
                accepted_move = True
                # is_improving_move = True

            # --- SA Acceptance Criterion ---
            elif E > 0: # Worsening move from LNS
                acceptance_prob = np.exp(-E / T) if T > 0 else 0 # Avoid division by zero
                if random.random() < acceptance_prob:
                    # operators[selected_op_idx]["successes"] += 1 # 'successes' might be better used for finding improving moves? Let's track accepted worsening separately if needed.
                    accepted_move = True
                    # print(f"Iter {iteration}: Accepted worse LNS {cost_before_lns:.2f} -> {new_cost_value:.2f} (Prob: {acceptance_prob:.3f}, T: {T:.2f})")

            # --- Process Accepted Move ---
            if accepted_move:
                accepted_solutions += 1
                current_solution = [vehicle.copy() for vehicle in new_solution] # Accept the LNS result
                current_cost_value = new_cost_value # Update current cost to reflect LNS result

                # --- Check if new best found ---
                if current_cost_value < best_cost_value:
                    # Optional: Apply Local Search ONLY if we found a potentially new best solution
                    # print(f"Iter {iteration}: Potential new best {current_cost_value:.2f} found by {selected_op['name']}. Applying LS...")
                    # current_solution_ls = apply_local_search(current_solution, problem)
                    # current_cost_value_ls = cost(current_solution_ls, problem)

                    # Now confirm if it's *still* the best (use original current_cost_value if LS not applied or made it worse)
                    # cost_to_compare = current_cost_value_ls if 'current_solution_ls' in locals() else current_cost_value
                    # solution_to_save = current_solution_ls if 'current_solution_ls' in locals() else current_solution

                    # Simpler: Check without LS first, apply LS later if desired
                    if current_cost_value < best_cost_value:
                        best_solution = [vehicle.copy() for vehicle in current_solution] # Deep copy
                        best_cost_value = current_cost_value
                        best_iteration = iteration
                        iters_since_improvement = 0 # Reset stagnation counter
                        operators[selected_op_idx]["best_improvements"] += 1 # Credit operator for finding new best
                        print(f"Iter {iteration}: NEW BEST solution: {best_cost_value:.2f} by {op_name}")
                        # Reset blame on finding new best? Optional.
                        # call_blame = {}
                    # else: # LS might have made it worse than the global best
                    #     iters_since_improvement += 1
                    #     # print(f"Iter {iteration}: LS applied, but cost {current_cost_value_ls:.2f} not better than global best {best_cost_value:.2f}")

                else:
                    # Accepted move, but not better than global best. Increment stagnation.
                    iters_since_improvement += 1

            else: # Move was not accepted (rejected worsening move)
                 iters_since_improvement += 1
                 # --- Update blame for rejected worsening move ---
                 # Only blame calls that were *part* of the proposed (but rejected) solution
                 for call_id in calls_in_new_solution:
                     call_blame[call_id] = call_blame.get(call_id, 0) + 1 # Simple increment

        else: # Infeasible solution from LNS
            iters_since_improvement += 1
            history["delta_value"].append(float('inf')) # Indicate infeasibility with infinity
            # --- Update blame for infeasible solution ---
            # Blame calls that were part of the infeasible solution attempt
            for call_id in calls_in_new_solution:
                call_blame[call_id] = call_blame.get(call_id, 0) + 1 # Simple increment

        # --- Cool Down Temperature ---
        T = T * alpha
        T = max(T, T_final) # Ensure T doesn't go below T_final

        # Record history
        history["iteration"].append(iteration)
        history["best_cost"].append(best_cost_value)
        history["current_cost"].append(current_cost_value)
        # Avoid division by zero if total_evaluations is 0
        history["acceptance_rate"].append(accepted_solutions / total_evaluations if total_evaluations > 0 else 0)
        history["operator_used"].append(selected_op["name"])

    # Final feasibility check of the returned best solution
    is_final_best_feas, final_best_details = feasibility(best_solution, problem)
    if not is_final_best_feas:
        print(f"ERROR: Final best_solution being returned is INFEASIBLE! Details: {final_best_details}")
    else:
        print(f"Final best solution cost: {best_cost_value:.2f}")


    # print(f'Feasible solutions generated: {feasibles} / {max_iter}')
    
    return (
        best_solution,
        best_cost_value,
        best_iteration,
        history,
        operator_weight_history,
        iteration_points
    )

neighbourhood = [ 
                # related_greedy,
                # segment_greedy,
                # random_greedy,
                # worst_greedy,
                random_regret,
                worst_regret,
                segment_regret,
                related_regret,
                segment_segment,
                historical_greedy, 
                historical_regret
                ]


def plot_operator_probabilities(operator_weight_history, iteration_points, max_iter):
    """
    Plots the selection probability of different operators over iterations using step plots.

    Args:
        operator_weight_history (dict): Dictionary where keys are operator names
                                        and values are lists of weights recorded at iteration_points.
        iteration_points (list): List of iteration numbers where weights were updated (e.g., [100, 200, ...]).
        max_iter (int): The total number of iterations the algorithm ran for.
    """
    plt.figure(figsize=(12, 7))
    operator_names = list(operator_weight_history.keys())
    num_operators = len(operator_names)

    # Check if data is available
    if not iteration_points or not all(len(w) > 0 for w in operator_weight_history.values()):
        print("Warning: No weight history data points to plot for operator probabilities.")
        return

    # --- Calculate Probabilities ---
    num_periods = len(iteration_points)
    probabilities = {op_name: [] for op_name in operator_names}

    for k in range(num_periods):
        # Calculate total weight at this update point k
        # Ensure we access the correct index k in the weight history list
        total_weight_k = sum(operator_weight_history[op_name][k] for op_name in operator_names if k < len(operator_weight_history[op_name]))

        if total_weight_k <= 0: # Avoid division by zero
            print(f"Warning: Total weight is zero or negative at iteration {iteration_points[k]}. Assigning equal probability.")
            prob_k = 1.0 / num_operators if num_operators > 0 else 0
            for op_name in operator_names:
                 probabilities[op_name].append(prob_k)
            continue

        # Calculate probability for each operator for the period starting at iteration_points[k]
        for op_name in operator_names:
            if k < len(operator_weight_history[op_name]):
                prob_k = operator_weight_history[op_name][k] / total_weight_k
                probabilities[op_name].append(prob_k)
            else:
                # Handle potential length mismatch (should not happen if recorded correctly)
                probabilities[op_name].append(0)
                print(f"Warning: Missing weight data for {op_name} at iteration {iteration_points[k]}.")

    # --- Prepare Data for Step Plot ---
    # Initial state (iteration 0 to first update point) assumes equal probability
    initial_prob = 1.0 / num_operators if num_operators > 0 else 0

    # Add the start and end points for plotting
    plot_iterations = [0] + iteration_points + [max_iter]

    for op_name in operator_names:
        # Probabilities corresponding to the start of each interval
        # Start with initial prob, then calculated probs, repeat last prob for final interval
        if probabilities[op_name]: # Check if list is not empty
             plot_probs = [initial_prob] + probabilities[op_name] + [probabilities[op_name][-1]]
        else: # Handle case where an operator somehow has no recorded probabilities
             plot_probs = [initial_prob] * (len(plot_iterations))
             print(f"Warning: No probabilities recorded for {op_name}. Plotting initial probability.")


        # Ensure lengths match for plotting
        if len(plot_iterations) != len(plot_probs):
             print(f"Error: Mismatch between iteration points ({len(plot_iterations)}) and probabilities ({len(plot_probs)}) for {op_name}. Skipping plot for this operator.")
             continue

        # Use step plot: probability plot_probs[i] is active from plot_iterations[i] to plot_iterations[i+1]
        plt.step(plot_iterations, plot_probs, where='post', label=op_name, linewidth=1.5)

    plt.xlabel("Iteration")
    plt.ylabel("Operator Selection Probability")
    plt.title("Operator Selection Probability Evolution")
    plt.ylim(0, 1.05) # Probabilities are between 0 and 1, add slight margin
    plt.xlim(0, max_iter)
    plt.legend(loc='center left', bbox_to_anchor=(1, 0.5)) # Place legend outside plot
    plt.grid(True, axis='y', linestyle='--', alpha=0.7)
    plt.tight_layout(rect=[0, 0, 0.85, 1]) # Adjust layout to prevent legend overlap
    plt.show()

def run():
    best_solutions = []
    best_costs = []
    run_times = [] # List to store runtime for each run

    file_index = 3
    file = filenames[file_index]


    problem = load(file)
    initial_solution = initial_solution_generator(problem)
    relatedness_matrix = calculate_relatedness(problem)

    print(f"--- Running for file: {file} ---")
    all_op_weights = {}
    all_iter_points = []
    max_iterations_run = 20000

    num_runs = 1
    for i in range(num_runs):
        print(f"\n--- Run {i+1}/{num_runs} ---") # Indicate which run is starting
        start_time = time.time()
        best_solution, best_cost, best_iteration, history, op_weight_history, iter_points = adaptive_algorithm(
        initial_solution, problem, neighbourhood, relatedness_matrix, max_iter=max_iterations_run)
        end_time = time.time()
        run_time = end_time - start_time
        run_times.append(run_time) # Store runtime

        # Print summary for this specific run
        print(f'Run {i+1} Best iteration: {best_iteration}')
        print(f"Run {i+1} Final cost: {best_cost:.2f}, Time: {run_time:.2f}s")
        feas, details = feasibility_check(best_solution, problem)
        print(f"Run {i+1} Feasibility: {feas}" + (f" Details: {details}" if not feas else ""))

        
        
        solution = reassemble_solution(best_solution)
        print(f"Run {i+1} Reassembled solution: {solution}")
        problem2 = load_problem(file)
        print(f"Run {i+1} Solution: {solution}")
        print(f'Run {i+1} Ahmad feasibility check:', feasibility_check2(solution, problem2))

        best_solutions.append(best_solution)
        best_costs.append(best_cost)

        # Store results from the first run for plotting (optional)
        if i == 0:
            all_op_weights = op_weight_history
            all_iter_points = iter_points


    print(f'\n--- Statistics for {file} ({num_runs} runs) ---')
    if best_costs: # Check if list is not empty
        print(f'Mean solution cost: {np.mean(best_costs):.2f}')
        print(f'Min solution cost: {min(best_costs):.2f}')
        print(f'Max solution cost: {max(best_costs):.2f}')
        print(f'Std Dev cost: {np.std(best_costs):.2f}')
    else:
        print("No successful runs to calculate cost statistics.")

    if run_times: # Check if list is not empty
        print(f'Mean runtime: {np.mean(run_times):.2f}s') # Print average runtime
        print(f'Min runtime: {min(run_times):.2f}s')
        print(f'Max runtime: {max(run_times):.2f}s')
        print(f'Std Dev runtime: {np.std(run_times):.2f}s')
    else:
        print("No successful runs to calculate runtime statistics.")


    # Plot operator probabilities from the first run (optional)
    # if all_op_weights and all_iter_points:
    #      plot_operator_probabilities(all_op_weights, all_iter_points, max_iterations_run)


if __name__ == "__main__":
    run()