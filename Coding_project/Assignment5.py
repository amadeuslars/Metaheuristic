import numpy as np
import random
from Utils import *
import time
import matplotlib.pyplot as plt

# Global variables for normalization factors
max_travel_cost = 0
biggest_time_window_diff = 0
biggest_cargo_size_diff = 0

# Load problem and precompute compatibility
problem = load_problem("Data/Call_300_Vehicle_90.txt")
compatibility_table = precompute_compatibility(problem)
route_cost_cache = {}

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
    # Time windows are in columns 4-7 of Cargo array
    # [early_pickup, late_pickup, early_delivery, late_delivery]
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

def get_cached_cost(solution, problem):
    solution_tuple = tuple(solution)
    if solution_tuple not in route_cost_cache:
        route_cost_cache[solution_tuple] = cost_function(solution, problem)
    return route_cost_cache[solution_tuple]

def update_operator_weights(operators):
    r = 0.1
    for op in operators:
        # Calculate base success rate
        successes = op['successes']
        success_bonus = 3 * successes
        
        # Calculate best solution bonus (if tracked)
        best_solutions_found = op.get('best_improvements', 0)
        best_solution_bonus = 20 * best_solutions_found 
        
        pi = (
            success_bonus +                         
            best_solution_bonus                     
        )
        
        op_old_weight = op['weight']
        new_weight = max(0.1, op_old_weight *(1-r) + r * pi/max(1, op['attempts']))
        op['weight'] =  new_weight
        
        # Reset statistics for next period
        op['successes'] = 0
        op['attempts'] = 1  # Start with 1 to avoid division by zero
        op['total_improvement'] = 0
        op['best_improvements'] = 0  # Reset best solution counter

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

def is_capacity_valid(vehicle, problem, v_idx):
    
    capacity = problem['VesselCapacity'][v_idx]
    current_load = 0
    call_count = {}
    
    for call in vehicle:
        call_count[call] = call_count.get(call, 0) + 1
        if call_count[call] == 1:  # First occurrence (pickup)
            current_load += problem['Cargo'][call-1][2]
        else:  # Second occurrence (delivery)
            current_load -= problem['Cargo'][call-1][2]
        
        if current_load > capacity:
            return False
    
    return True

def get_valid_calls(vehicles):
    """Extract valid calls (each appearing exactly twice) from vehicles"""
    valid_calls = []
    for v_idx, vehicle in enumerate(vehicles):
        call_counts = {}
        for call in vehicle:
            call_counts[call] = call_counts.get(call, 0) + 1
        valid_calls.extend([(v_idx, call) for call, count in call_counts.items() if count == 2])
    return valid_calls

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

# removal functions
def worst_removal(vehicles, problem, compatibility_table, k):
    n_vehicles = problem['n_vehicles']

    call_costs = {}
    for v_idx, vehicle in enumerate(vehicles):
        if not vehicle:
            continue

        calls_counts = {}
        for call in vehicle:
            calls_counts[call] = calls_counts.get(call, 0) + 1
        valid_calls = [call for call, count in calls_counts.items() if count == 2]

        if not valid_calls:
            continue

        base_cost = simplified_cost_function(vehicle, v_idx, problem)    
        for call in valid_calls:
            temp_vehicle = [c for c in vehicle if c != call]
            new_cost = simplified_cost_function(temp_vehicle, v_idx, problem)
            call_costs[(v_idx, call)] = base_cost - new_cost

    sorted_calls = sorted(call_costs.items(), key=lambda x: -x[1])[:k]

    candidate_count = min(len(sorted_calls), k * 3)  # Consider 3x more candidates than needed
    candidates = sorted_calls[:candidate_count]
    selected_calls = []

    if candidates:
        # Get candidate calls and their benefits
        candidate_items, benefits = zip(*candidates)
        
        # Convert benefits to probabilities
        total_benefit = sum(benefits)
        if total_benefit > 0:  # Avoid division by zero
            probs = [b/total_benefit for b in benefits]
            
            # Sample without replacement using probabilities
            num_to_select = min(k, len(candidates))
            selected_indices = np.random.choice(
                range(len(candidates)), 
                size=num_to_select, 
                replace=False, 
                p=probs
            )
            # Get the selected calls
            selected_calls = [candidate_items[i] for i in selected_indices]
        else:
            # If all benefits are 0, just choose randomly
            num_to_select = min(k, len(candidates))
            selected_calls = random.sample(candidate_items, num_to_select)

    removed_calls = remove_calls(vehicles, selected_calls)

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

    # Sorting by relatedness score            
    relatedness.sort(key=lambda x: x[2])

    k = min(k, len(relatedness) + 1) # not more than available calls
    calls_to_remove = [(seed_v_idx, seed_call)] + [(v, c) for v, c, _ in relatedness[:k - 1]]

    removed_calls = remove_calls(vehicles, calls_to_remove)

    # Get compatible vehicles for each call
    call_info = build_call_info(removed_calls, compatibility_table, n_vehicles)

    return vehicles, call_info

def random_removal(vehicles, problem, compatibility_table, k):
    n_vehicles = problem['n_vehicles']

    valid_calls = get_valid_calls(vehicles)

    if not valid_calls:
        return vehicles, []
    
    k = min(k, len(valid_calls))
    calls_to_remove = random.sample(valid_calls, k)

    removed_calls = remove_calls(vehicles, calls_to_remove)

    call_info = build_call_info(removed_calls, compatibility_table, n_vehicles)

    return vehicles, call_info

def dummy_removal(vehicles, problem, compatibility_table, k):
    n_vehicles = problem['n_vehicles']

    valid_calls = get_valid_calls(vehicles)

    if not valid_calls:
        return vehicles, []
    
    k = min(k, len(valid_calls))
    
    # Remove calls from the dummy vehicle
    dummy_calls = [(n_vehicles, call) for _, call in valid_calls if _ == n_vehicles]
    calls_to_remove = random.sample(dummy_calls, k)

    removed_calls = remove_calls(vehicles, calls_to_remove)

    call_info = build_call_info(removed_calls, compatibility_table, n_vehicles)

    return vehicles, call_info
# insertion functions
def random_insertion(vehicles, call_info, problem):
    n_vehicles = problem['n_vehicles']
    
    # Precompute call sizes
    call_sizes = {call: problem['Cargo'][call-1][2] for call, _ in call_info}
    
    for call, compatible_vehicles in call_info:
        inserted = False
        # Try vehicles in random order
        shuffled_vehicles = random.sample(compatible_vehicles, len(compatible_vehicles))
        
        for v_idx in shuffled_vehicles:
            vehicle = vehicles[v_idx]
            
            # Special case for dummy vehicle - always feasible
            if v_idx == n_vehicles:
                vehicle.append(call)
                vehicle.append(call)
                inserted = True
                break
                
            # For regular vehicle, try random positions but CHECK FEASIBILITY
            for _ in range(10):  # Try up to 10 random position combinations
                pickup_pos = random.randint(0, len(vehicle))
                delivery_pos = random.randint(pickup_pos, len(vehicle) + 1)
                
                # Create a temporary vehicle to test feasibility
                temp_vehicle = vehicle.copy()
                temp_vehicle.insert(pickup_pos, call)
                delivery_pos_adj = delivery_pos if delivery_pos <= pickup_pos else delivery_pos + 1
                temp_vehicle.insert(delivery_pos_adj, call)
                
                # Check if insertion is feasible
                if is_capacity_valid(temp_vehicle, problem, v_idx):
                    # If feasible, apply the insertion to the actual vehicle
                    vehicle.insert(pickup_pos, call)
                    vehicle.insert(delivery_pos_adj, call)
                    inserted = True
                    break
                    
            if inserted:
                break
                
        # If we couldn't insert feasibly in any regular vehicle, use dummy
        if not inserted:
            vehicles[n_vehicles].append(call)
            vehicles[n_vehicles].append(call)
    
    return vehicles

def regret_insertion(vehicles, call_info, problem):
    n_vehicles = problem['n_vehicles']
    
    # Shuffle call_info to avoid bias
    random.shuffle(call_info)
    
    # Precompute call sizes
    call_sizes = {call: problem['Cargo'][call-1][2] for call, _ in call_info}
    
    # While we have calls to insert
    remaining_calls = list(call_info)
    
    while remaining_calls:
        best_regret_call_idx = -1
        best_regret_score = -float('inf')
        best_vehicle = None
        best_pickup = None
        best_delivery = None
        best_cost = float('inf')
        
        # For each remaining call, calculate its regret value
        for idx, (call, compatible_vehicles) in enumerate(remaining_calls):
            # Find best and second best insertion positions
            insertion_costs = []
            best_positions = {}
            
            for v_idx in range(n_vehicles + 1):
                if v_idx not in compatible_vehicles:
                    continue
                
                vehicle = vehicles[v_idx]
                vehicle_len = len(vehicle)
                
                # Dummy vehicle - special case
                if v_idx == n_vehicles:
                    dummy_cost = float('inf')
                    for existing_call in set(vehicle):
                        dummy_cost += problem['Cargo'][existing_call-1][3]
                    dummy_cost += problem['Cargo'][call-1][3]  # Add the new call's penalty
                    
                    insertion_costs.append((v_idx, 0, 0, dummy_cost))
                    best_positions[v_idx] = (0, 0, dummy_cost)
                    continue
                
                # Try insertion positions
                min_cost = float('inf')
                best_p_idx = -1
                best_d_idx = -1
                
                # Check capacity first
                if call_sizes[call] > problem['VesselCapacity'][v_idx]:
                    continue
                
                # Try key positions
                if vehicle_len <= 4:
                    # For small routes, try all positions
                    pickup_indices = list(range(vehicle_len + 1))
                else:
                    # For larger routes, sample positions
                    pickup_indices = [0, vehicle_len//3, 2*vehicle_len//3, vehicle_len]
                
                for p_idx in pickup_indices:
                    for d_idx in range(p_idx + 1, vehicle_len + 2):
                        # Create a temporary vehicle
                        temp_vehicle = vehicle.copy()
                        temp_vehicle.insert(p_idx, call)
                        actual_d_idx = d_idx if d_idx <= p_idx else d_idx
                        temp_vehicle.insert(actual_d_idx, call)
                        
                        # Check capacity
                        if not is_capacity_valid(temp_vehicle, problem, v_idx):
                            continue
                        
                        # Evaluate cost
                        temp_vehicles = vehicles.copy()
                        temp_vehicles[v_idx] = temp_vehicle
                        temp_solution = reassemble_solution(temp_vehicles)
                        temp_cost = get_cached_cost(temp_solution, problem)
                        
                        if temp_cost < min_cost:
                            min_cost = temp_cost
                            best_p_idx = p_idx
                            best_d_idx = d_idx
                
                if best_p_idx != -1:
                    insertion_costs.append((v_idx, best_p_idx, best_d_idx, min_cost))
                    best_positions[v_idx] = (best_p_idx, best_d_idx, min_cost)
            
            # Sort insertions by cost (ascending)
            insertion_costs.sort(key=lambda x: x[3])
            
            # Calculate regret value (difference between best and 2nd best)
            if len(insertion_costs) >= 2:
                # Regret-2: difference between best and 2nd best insertion
                regret_value = insertion_costs[1][3] - insertion_costs[0][3]
                
                # If high regret and reasonably good insertion, prioritize this call
                if regret_value > best_regret_score:
                    best_regret_score = regret_value
                    best_regret_call_idx = idx
                    best_vehicle = insertion_costs[0][0]
                    best_pickup = insertion_costs[0][1]
                    best_delivery = insertion_costs[0][2]
                    best_cost = insertion_costs[0][3]
            elif len(insertion_costs) == 1:
                # Only one feasible insertion - give it high priority
                regret_value = 10000  # Arbitrary high value
                if regret_value > best_regret_score:
                    best_regret_score = regret_value
                    best_regret_call_idx = idx
                    best_vehicle = insertion_costs[0][0]
                    best_pickup = insertion_costs[0][1]
                    best_delivery = insertion_costs[0][2]
                    best_cost = insertion_costs[0][3]
        
        # Insert the call with highest regret
        if best_regret_call_idx != -1:
            call, _ = remaining_calls[best_regret_call_idx]
            
            vehicle = vehicles[best_vehicle]
            vehicle.insert(best_pickup, call)
            adjusted_d_idx = best_delivery
            if best_delivery > best_pickup:
                adjusted_d_idx = best_delivery
            vehicle.insert(adjusted_d_idx, call)
            
            # Remove from remaining calls
            remaining_calls.pop(best_regret_call_idx)
        else:
            # If no feasible insertions found, move remaining calls to dummy
            for call, _ in remaining_calls:
                vehicles[n_vehicles].append(call)
                vehicles[n_vehicles].append(call)
            break
    
    return vehicles

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

# Operators
def opt1_random_removal_random_insertion(solution, problem, compatibility_table, k):    
    vehicles = parse_solution_to_vehicles(solution)
    vehicles, call_info = random_removal(vehicles, problem, compatibility_table, k)
    vehicles = random_insertion(vehicles, call_info, problem)
    new_solution = reassemble_solution(vehicles)
    return new_solution

def opt2_related_removal_regret_insertion(solution, problem, compatibility_table, k):
    vehicles = parse_solution_to_vehicles(solution)
    vehicles, call_info = related_removal(vehicles, problem, compatibility_table, k)
    vehicles = regret_insertion(vehicles, call_info, problem)
    return reassemble_solution(vehicles)

def opt_3_worst_removal_regret_insertion(solution, problem, compatibility_table, k):
    vehicles = parse_solution_to_vehicles(solution)
    vehicles, call_info = worst_removal(vehicles, problem, compatibility_table, k)
    vehicles = regret_insertion(vehicles, call_info, problem)
    return reassemble_solution(vehicles)

def opt_4_dummy_removal_regret_insertion(solution, problem, compatibility_table, k):
    vehicles = parse_solution_to_vehicles(solution)
    vehicles, call_info = dummy_removal(vehicles, problem, compatibility_table, k)
    vehicles = regret_insertion(vehicles, call_info, problem)
    return reassemble_solution(vehicles)

# Algorithm
def Adaptive_algorithm(initial_solution, problem, max_iter):
    best_solution = initial_solution
    best_cost = get_cached_cost(initial_solution, problem)
    current_cost = best_cost
    current_solution = initial_solution
    iterations_since_best = 0
    T0 = 1000000
    T = T0
    cooling_rate = 0.9986
    infeasiblecount = 0
    # Operators
    operators = [
        {"name": "random_random", "func": opt1_random_removal_random_insertion, "params": {"k": 1}, "weight": 1.0, 
         "successes": 0, "attempts": 1,"best_improvements": 0},
        {"name": "random_random_3", "func": opt1_random_removal_random_insertion, "params": {"k": 3}, "weight": 1.0,
         "successes": 0, "attempts": 1,"best_improvements": 0},
        {"name": "related_regret", "func": opt2_related_removal_regret_insertion, "params": {"k": 2}, "weight": 1.0, 
         "successes": 0, "attempts": 1,"best_improvements": 0},
        {"name": "worst_regret", "func": opt_3_worst_removal_regret_insertion, "params": {"k": 2}, "weight": 1.0,
         "successes": 0, "attempts": 1,"best_improvements": 0},
        
        
    ]

    # Add to initialization:
    history = []
    
    # Track operator probabilities
    probability_history = {op["name"]: [] for op in operators}
    weight_iteration_points = []

    for i in range(max_iter):
        
        if i % 1000 == 0:
            print(f"Iteration {i}, Best cost: {best_cost}")
        
        # Update operator statistics
        if i % 100 == 0:
            update_operator_weights(operators)
            
            # Store weights for plotting
            weight_iteration_points.append(i)
            total_weight = sum([op['weight'] for op in operators])
            op_probs = [op['weight'] / total_weight for op in operators]
            
            for idx, op in enumerate(operators):
                probability_history[op["name"]].append(op_probs[idx])

        # Add escape mechanism
        if iterations_since_best > 500:
            print(f"Iteration {i}: Attempting escape after {iterations_since_best} iterations without improvement")
            
            for _ in range(5):
                perturbed_solution = opt1_random_removal_random_insertion(current_solution, problem, compatibility_table, 1)

                if feasibility_check(perturbed_solution, problem)[0]:
                    current_solution = perturbed_solution
                    current_cost = get_cached_cost(current_solution, problem)

                    if current_cost < best_cost:
                        best_solution = current_solution
                        best_cost = current_cost
                        iterations_since_best = 0

                    iterations_since_best = 0
                    break
            
            iterations_since_best = 0
            print('No escape found')
            current_solution = best_solution
            current_cost = best_cost
            
            continue

        # normalizing the weights
        total_weight = sum([op['weight'] for op in operators])
        for op in operators:
            op['weight'] /= total_weight
        
        # Select operator
        selected_op = random.choices(operators, [op['weight'] for op in operators])[0]
        
        selected_op_func = selected_op['func']
        selected_op_params = selected_op['params']
        selected_op['attempts'] += 1

        # Apply operator
        new_solution = selected_op_func(current_solution, problem, compatibility_table, **selected_op_params)
        new_cost = get_cached_cost(new_solution, problem)

        if not feasibility_check(new_solution, problem)[0]:
            infeasiblecount += 1
            iterations_since_best += 1
            continue

        if new_cost < best_cost:
            best_solution = new_solution
            current_solution = new_solution
            current_cost = new_cost
            best_cost = new_cost
            iterations_since_best = 0
            operators[operators.index(selected_op)]['best_improvements'] += 1
        
        else:
            p = np.exp(-(new_cost - current_cost) / T) 
            if random.random() < p:
                current_solution = new_solution
                current_cost = new_cost
                iterations_since_best += 1

        selected_op['successes'] += 1
        T = T * cooling_rate

        # Add to the main loop:
        history.append(current_cost)

    print(f"Best cost found: {best_cost}")
    print(f"Feasibility checks failed: {infeasiblecount}")
    # Return history with results:
    return best_solution, best_cost, history, probability_history, weight_iteration_points

# Function to plot operator probabilities
def plot_operator_weights(probability_history, iteration_points):
    plt.figure(figsize=(12, 6))
    
    for op_name, probs in probability_history.items():
        plt.plot(iteration_points, probs, '-o', label=op_name, linewidth=2, markersize=4)
    
    plt.axhline(0.25, color='black', linestyle='--', alpha=0.5)
    plt.xlabel('Iteration', fontsize=12)
    plt.ylabel('Selection Probability', fontsize=12)
    plt.title('Operator Selection Probabilities Over Time', fontsize=14)
    plt.grid(True, alpha=0.3)
    plt.legend(loc='center right')
    plt.tight_layout()
    plt.show()

# Main code
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
