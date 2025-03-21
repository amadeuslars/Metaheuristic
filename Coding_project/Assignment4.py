import numpy as np
import random
from Utils import *
import time

# Load problem and precompute compatibility
problem = load_problem("Data/Call_130_Vehicle_40.txt")
compatibility_table = precompute_compatibility(problem)
route_cost_cache = {}

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

def simplified_cost_function(vehicles, problem):
    n_vehicles = problem['n_vehicles']
    
    segment_cost = [0 for _ in range(n_vehicles + 1)]
    
    for vehicle_idx, vehicle in enumerate(vehicles):
        if vehicle_idx == n_vehicles:
            # Dummy vehicle - penalty costs for unserved calls
            current_call = None
            for call in sorted(set(vehicle)):  
                segment_cost[n_vehicles] += problem['Cargo'][call - 1][3]
        else:
            # Regular vehicle - calculating travel and port costs
            if len(vehicle) > 0:
                # unique calls and their count to ensure we're processing pickup/delivery pairs
                unique_calls = {}
                for call in vehicle:
                    unique_calls[call] = unique_calls.get(call, 0) + 1
                
                valid_vehicle_calls = [call for call, count in unique_calls.items() if count == 2]
                
                if valid_vehicle_calls:
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
                        FirstVisitCost = problem['FirstTravelCost'][vehicle_idx][int(problem['Cargo'][first_call, 0] - 1)]
                        segment_cost[vehicle_idx] += FirstVisitCost
                        
                        # Travel costs between consecutive nodes
                        if len(PortIndex) > 1:
                            for i in range(len(PortIndex) - 1):
                                segment_cost[vehicle_idx] += problem['TravelCost'][vehicle_idx][PortIndex[i]][PortIndex[i+1]]
                        
                        # Port costs (loading/unloading)
                        for call in valid_vehicle_calls:
                            segment_cost[vehicle_idx] += problem['PortCost'][vehicle_idx][call - 1]
    return segment_cost




    n_vehicles = problem['n_vehicles']
            
    return vehicles

def calculate_tardiness(vehicles, problem):
    print(vehicles)
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

def opt1_k_random_reinsert(solution, problem, compatibility_table, k):
    n_vehicles = problem['n_vehicles']
    vehicles = parse_solution_to_vehicles(solution)
   
    
    valid_calls = [] 
    for seg_idx, seg in enumerate(vehicles):
        counts = {}
        for call in seg:
            counts[call] = counts.get(call, 0) + 1
        for call, count in counts.items():
            if count == 2:
                valid_calls.append((seg_idx, call))
    
    k = min(k, len(valid_calls))
    if k == 0:
        return solution 
        
    # Selecting k calls to move
    calls_to_move = []
    call_sources = {} 
    
    for _ in range(k):
        if not valid_calls:
            break
            
        # Selecting a random call pair to move
        idx = random.choices(range(len(valid_calls)), k=1)[0]
        src_vehicle_index, call_to_move = valid_calls.pop(idx)
        
        calls_to_move.append(call_to_move)
        call_sources[call_to_move] = src_vehicle_index
        
        # Removing the call pair from the source vehicle
        src_vehicle = vehicles[src_vehicle_index]
        src_vehicle = [x for x in src_vehicle if x != call_to_move]
        vehicles[src_vehicle_index] = src_vehicle
    
    # Reinserting each call into a compatible vehicle
    for call in calls_to_move:
        src_vehicle_index = call_sources[call]
        
        # Finding allowed destinations for this call
        allowed_destinations = []
        for v in range(n_vehicles + 1):
            if v == src_vehicle_index:
                continue
            if compatibility_table[call - 1, v] == 1 and (call not in vehicles[v]):
                allowed_destinations.append(v)
        
        if not allowed_destinations:
            continue

        # Selecting destination vehicle
        destination_vehicle_idx = random.choices(
            allowed_destinations,
            weights=[2 if v != n_vehicles else 1 for v in allowed_destinations], 
            k=1)[0]
        destination_vehicle = vehicles[destination_vehicle_idx]
        
        # Inserting pickup and delivery
        pickup_idx = random.randint(0, len(destination_vehicle))
        delivery_idx = random.randint(pickup_idx, len(destination_vehicle) + 1)
        destination_vehicle.insert(pickup_idx, call)
        destination_vehicle.insert(delivery_idx, call)

    new_solution = reassemble_solution(vehicles)
    return new_solution

def opt3_tardiness_operator(solution, problem, compatibility_table):
  
    vehicles = parse_solution_to_vehicles(solution)
    n_vehicles = problem['n_vehicles']
    
    # tardiness for each call
    call_tardiness = calculate_tardiness(vehicles, problem)
    
    # If no tardiness found, random selection
    if not call_tardiness or all(t == 0 for t in call_tardiness.values()):
        
        return opt1_k_random_reinsert(solution, problem, compatibility_table, k=2)
    
   
    calls_sorted_by_tardiness = sorted(call_tardiness.keys(), 
                                       key=lambda c: call_tardiness[c],
                                       reverse=True)
    
    target_calls = calls_sorted_by_tardiness[:3]  # select n calls with highest tardiness
    
    valid_calls = []
    for v_idx, vehicle in enumerate(vehicles):
        call_counts = {}
        for call in vehicle:
            call_counts[call] = call_counts.get(call, 0) + 1
            
        for call, count in call_counts.items():
            if count == 2 and call in target_calls:
                valid_calls.append((v_idx, call))
    
    if not valid_calls:
        return solution  
    
    # Select calls to move
    calls_to_move = []
    call_sources = {}
    
    # up to n calls
    for v_idx, call in valid_calls[:5]:
        calls_to_move.append(call)
        call_sources[call] = v_idx
        
        # Removes the call pair from source vehicle
        src_vehicle = vehicles[v_idx]
        src_vehicle = [x for x in src_vehicle if x != call]
        vehicles[v_idx] = src_vehicle
    
    # For each call, finds best insertion position
    for call in calls_to_move:
        src_vehicle_idx = call_sources[call]
        best_tardiness = float('inf')
        best_vehicle = None
        best_pickup = None
        best_delivery = None
        
        # Trying all compatible vehicles
        dest_vehicles = []
        for v_idx in range(n_vehicles + 1):
            if v_idx == n_vehicles:  # Dummy vehicle last
                if compatibility_table[call-1, v_idx] == 1:
                    dest_vehicles.append(v_idx)
            elif v_idx != src_vehicle_idx and compatibility_table[call-1, v_idx] == 1:
                dest_vehicles.append(v_idx)
        
        # insertion in each compatible vehicle
        for v_idx in dest_vehicles:
            vehicle = vehicles[v_idx]
            
            for p_idx in range(len(vehicle) + 1):
                if problem['n_calls'] <= 20:
                    d_positions = range(p_idx + 1, len(vehicle) + 2)
                else:
                    max_samples = min(5, len(vehicle) - p_idx + 1)
                    d_positions = random.sample(range(p_idx + 1, len(vehicle) + 2), 
                                              max(1, max_samples))
                
                for d_idx in d_positions:
                    temp_vehicle = vehicle.copy()
                    temp_vehicle.insert(p_idx, call)
                    
                    actual_d_idx = d_idx if d_idx <= p_idx else d_idx
                    temp_vehicle.insert(actual_d_idx, call)
                    
                    temp_vehicles = vehicles.copy()
                    temp_vehicles[v_idx] = temp_vehicle
                    
                    # new tardiness for this vehicle only
                    new_tardiness = sum(calculate_tardiness({v_idx: temp_vehicle}, problem).values())
                    
                    if new_tardiness < best_tardiness:
                        best_tardiness = new_tardiness
                        best_vehicle = v_idx
                        best_pickup = p_idx
                        best_delivery = d_idx
        
        # Inserting at best position found
        if best_vehicle is not None:
            vehicle = vehicles[best_vehicle]
            vehicle.insert(best_pickup, call)
            
            adjusted_d_idx = best_delivery
            if best_delivery > best_pickup:
                adjusted_d_idx = best_delivery
            vehicle.insert(adjusted_d_idx, call)
        else:
            # If no better position found, adding to dummy
            vehicles[n_vehicles].append(call)
            vehicles[n_vehicles].append(call)
    
    new_solution = reassemble_solution(vehicles)
    _ = get_cached_cost(new_solution, problem)
    return new_solution

def opt2_greedy_reinsert(solution, problem, compatibility_table, k=2):
    n_vehicles = problem['n_vehicles']
    vehicles = parse_solution_to_vehicles(solution)
    
    # Calculate cost contribution of each call 
    segment_costs = simplified_cost_function(vehicles, problem)
    call_costs = {}
    
    for v_idx, vehicle in enumerate(vehicles):
        call_counts = {}
        for call in vehicle:
            call_counts[call] = call_counts.get(call, 0) + 1
            
        valid_calls = [call for call, count in call_counts.items() if count == 2]
        
        for call in valid_calls:
            if v_idx == n_vehicles:  # Dummy vehicle
                call_costs[call] = problem['Cargo'][call-1][3]  # Penalty cost
            else:
                n_calls = len(valid_calls)
                if n_calls > 0:
                    call_costs[call] = segment_costs[v_idx] / n_calls
    
    # Select k high-cost calls
    valid_calls = list(call_costs.keys())
    if not valid_calls:
        return solution
        
    weights = [call_costs[call] for call in valid_calls]
    total_weight = sum(weights)
    probs = [w / total_weight for w in weights] if total_weight > 0 else None
        
    k = min(k, len(valid_calls))
    selected_calls = np.random.choice(valid_calls, size=k, replace=False, p=probs)
    
    # Remove selected calls
    for call in selected_calls:
        for v_idx, vehicle in enumerate(vehicles):
            if call in vehicle:
                vehicles[v_idx] = [x for x in vehicles[v_idx] if x != call]
    
    # Precompute call size 
    call_sizes = {call: problem['Cargo'][call-1][2] for call in selected_calls}
    
    # Precompute vehicle remaining capacities
    vehicle_capacities = {}
    for v_idx in range(n_vehicles):
        capacity = problem['VesselCapacity'][v_idx]
        # Calculate current load profile
        load_profile = []
        current_load = 0
        for call in vehicles[v_idx]:
            call_count = vehicles[v_idx].count(call) 
            if call_count == 1:  # First occurrence (pickup)
                current_load += problem['Cargo'][call-1][2]
            load_profile.append(current_load)
        
        max_load = max(load_profile) if load_profile else 0
        vehicle_capacities[v_idx] = capacity - max_load
    
    # For each removed call, find best insertion 
    for call in selected_calls:
        best_cost = float('inf')
        best_vehicle = None
        best_pickup = None
        best_delivery = None
        
        # Sorting vehicles by estimated potential
        vehicle_potential = []
        for v_idx in range(n_vehicles + 1):
            if compatibility_table[call-1, v_idx] == 0:
                continue
                
            # Skipping vehicles with clearly insufficient capacity
            if v_idx < n_vehicles and vehicle_capacities[v_idx] < call_sizes[call]:
                continue
                
            # Potential score
            if v_idx == n_vehicles:
                # Dummy vehicle
                potential = float('inf')
            else:
                # Score based on spare capacity and current route cost
                potential = segment_costs[v_idx] / (vehicle_capacities[v_idx] + 1)
                
            vehicle_potential.append((v_idx, potential))
        
        # Sort by potential (best first)
        vehicle_potential.sort(key=lambda x: x[1])
        
        # Stopping after finding a good enough solution
        found_good_solution = False
        best_so_far = float('inf')
        
        # Vehicles in order of potential
        for v_idx, _ in vehicle_potential[:min(3, len(vehicle_potential))]:  # Try only top 3 vehicles
            vehicle = vehicles[v_idx]
            vehicle_len = len(vehicle)
            
            # If dummy vehicle and we found any feasible solution, skip
            if v_idx == n_vehicles and best_vehicle is not None:
                continue
                
            # Aggressive position sampling
            if vehicle_len <= 3:
                # For very small routes, trying all positions
                pickup_indices = list(range(vehicle_len + 1))
            else:
                # For larger routes, selective
                pickup_indices = [0, vehicle_len//2, vehicle_len]  # Start, middle, end only
            
            # Track if we've had any capacity failures in this vehicle
            capacity_failures = 0
            
            # Strategic positions only
            for p_idx in pickup_indices:
                
                if capacity_failures >= 2 and v_idx < n_vehicles:
                    break  # This vehicle is likely near capacity
                
                # Extremely limited delivery positions
                else:
                    d_positions = [p_idx + 1]  # Immediate delivery
                    if vehicle_len > 1:
                        d_positions.append(p_idx + random.randint(p_idx + 1, vehicle_len))  # one space after immediate

            
                for d_idx in d_positions:
                    # Quick capacity check before creating temp vehicle
                    if v_idx < n_vehicles:
                        if call_sizes[call] > vehicle_capacities[v_idx]:
                            capacity_failures += 1
                            continue
                    
                    # temp vehicle
                    temp_vehicle = vehicle.copy()
                    temp_vehicle.insert(p_idx, call)
                    actual_d_idx = d_idx if d_idx <= p_idx else d_idx
                    temp_vehicle.insert(actual_d_idx, call)
                    
                    # Quick capacity check
                    if v_idx < n_vehicles:
                        valid = is_capacity_valid(temp_vehicle, problem, v_idx)
                        if not valid:
                            capacity_failures += 1
                            continue
                    
                    # cached cost calculation
                    temp_vehicles = vehicles.copy()
                    temp_vehicles[v_idx] = temp_vehicle
                    temp_solution = reassemble_solution(temp_vehicles)
                    temp_cost = get_cached_cost(temp_solution, problem)
                    
                    if temp_cost < best_cost:
                        best_cost = temp_cost
                        best_vehicle = v_idx
                        best_pickup = p_idx
                        best_delivery = d_idx
                        
                        # If we found a solution significantly better than before, stop searching
                        if best_cost < best_so_far * 0.3:  # 20% improvement
                            found_good_solution = True
                            break
                
                if found_good_solution:
                    break
            
            if found_good_solution:
                break
        
        # Insert at best position found
        if best_vehicle is not None:
            vehicle = vehicles[best_vehicle]
            vehicle.insert(best_pickup, call)
            adjusted_d_idx = best_delivery
            if best_delivery > best_pickup:
                adjusted_d_idx = best_delivery
            vehicle.insert(adjusted_d_idx, call)
            
            # Updating capacity if not dummy vehicle
            if best_vehicle < n_vehicles:
                vehicle_capacities[best_vehicle] -= call_sizes[call]
        else:
            # No feasible position found, add to dummy
            vehicles[n_vehicles].append(call)
            vehicles[n_vehicles].append(call)
    
    new_solution = reassemble_solution(vehicles)
    return new_solution

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

def simulated_annealing2(problem, initial_solution, compatibility_table, max_iter):  
    Tf = 0.1
    incumbent = initial_solution
    best_solution = initial_solution
    seen_solutions = set()
    seen_solutions.add(tuple(initial_solution))
    skipcount = 0
    w = []
    operator_weights=[0, 0, 1]
    
    # Warming phase
    for _ in range(200):
        
        new_solution = opt1_k_random_reinsert(incumbent, problem, compatibility_table, k=1)
            
        E = get_cached_cost(new_solution, problem) - get_cached_cost(incumbent, problem)
        if feasibility_check(new_solution, problem)[0]:
            if E <= 0:
                incumbent = new_solution
                if get_cached_cost(incumbent, problem) < get_cached_cost(best_solution, problem):
                    best_solution = incumbent
            else:
                p = 0.8
                if np.random.rand() < p:
                    incumbent = new_solution
                w.append(E)
    
    if len(w) == 0:
        T_0 = 1_000_000
        alfa = (Tf / T_0) ** (1 / 9800)
        T = T_0
    else:
        delta_avg = np.mean(w)
        T_0 = -delta_avg / np.log(0.8)
        alfa = (Tf / T_0) ** (1 / 9800)
        T = T_0
  
    for iteration in range(100, max_iter):
        if iteration % 1000 == 0:
            print(f'Iteration: {iteration}')
        # Weighted selection for operators
        operator = np.random.choice([1, 2, 3], p=[w/sum(operator_weights) for w in operator_weights])
        
        if operator == 1:
            k = random.randint(1, 4)
            new_solution = opt1_k_random_reinsert(incumbent, problem, compatibility_table, k)
        elif operator == 2:
            new_solution = opt2_greedy_reinsert(incumbent, problem, compatibility_table)
        elif operator == 3:
            new_solution = opt3_tardiness_operator(incumbent, problem, compatibility_table)

        new_sol_tuple = tuple(new_solution)
        if new_sol_tuple in seen_solutions:
            skipcount += 1
            continue
        seen_solutions.add(new_sol_tuple)

        E = get_cached_cost(new_solution, problem) - get_cached_cost(incumbent, problem)
        if feasibility_check(new_solution, problem)[0]:
            if E < 0:
                incumbent = new_solution
                if get_cached_cost(incumbent, problem) < get_cached_cost(best_solution, problem):
                    best_solution = incumbent
            else:
                p = np.exp(-E / T)
                if np.random.rand() < p:
                    incumbent = new_solution
        T = alfa * T

    
    print(f'Skipcount: {skipcount}')
    final_solution = [int(x) for x in best_solution]
    
    return final_solution, cost_function(best_solution, problem)

# time_start = time.time()
# print(simulated_annealing2(problem, initial_solution_generator(problem), compatibility_table, 10_000))
# time_end = time.time()
# print(f'Execution time: {time_end - time_start:.2f} seconds')