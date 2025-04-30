from early_code.Utils import *
from functools import lru_cache
import random

problem = load_problem("Data/Call_7_Vehicle_3.txt")
compatibility_table = precompute_compatibility(problem)
route_cost_cache = {}



def cost_function_light(Solution, problem, vehicle_idx):
    num_vehicles = problem['n_vehicles']
    Cargo = problem['Cargo']
    TravelCost = problem['TravelCost']
    FirstTravelCost = problem['FirstTravelCost']
    PortCost = problem['PortCost']

    # Convert the route to 0-indexed form once.
    route_arr = np.array(Solution, dtype=int) - 1  
    N = route_arr.size

    if vehicle_idx == num_vehicles:
        # For dummy vehicle, use route_arr for cost calculation.
        NotTransportCost = np.sum(Cargo[route_arr, 3]) / 2
        return NotTransportCost
    elif N > 0:
        sortRout = np.sort(route_arr, kind='mergesort')
        I = np.argsort(route_arr, kind='mergesort')
        Indx = np.argsort(I, kind='mergesort')

        PortIndex = Cargo[sortRout, 1].astype(int)
        PortIndex[::2] = Cargo[sortRout[::2], 0]
        PortIndex = PortIndex[Indx] - 1

        Diag = TravelCost[vehicle_idx, PortIndex[:-1], PortIndex[1:]]
        FirstVisitCost = FirstTravelCost[vehicle_idx, int(Cargo[route_arr[0], 0] - 1)]
        RouteTravelCost = np.sum(np.hstack((FirstVisitCost, Diag.flatten())))
        CostInPorts = np.sum(PortCost[vehicle_idx, route_arr]) / 2

        NotTransportCost = 0

    TotalCost = NotTransportCost + RouteTravelCost + CostInPorts
    return TotalCost


# print(cost_function_light([2, 3, 3, 2], problem, 3))

def reinsert(solution, problem, compatibility_table):
    n_vehicles = problem['n_vehicles']
    global route_cost_cache

    # 1. Parse the solution into segments (car routes)
    vehicles = []
    current_vehicle = []
    for call in solution:
        if call == 0:
            vehicles.append(current_vehicle)
            current_vehicle = []
        else:
            current_vehicle.append(call)
    vehicles.append(current_vehicle)  # Append the last segment
    og_vehicles = vehicles.copy()
    # 2. Identify all call pairs by scanning each segment.
    #    We expect a call (nonzero number) to appear exactly twice in a segment.
    valid_calls = []  # each element is a tuple (segment_index, call)
    for seg_idx, seg in enumerate(vehicles):
        counts = {}
        for call in seg:
            counts[call] = counts.get(call, 0) + 1
        for call, count in counts.items():
            if count == 2:
                valid_calls.append((seg_idx, call))

    weight_parameter = 2
    # After valid_calls is computed:
    weights = []
    for seg_idx, call in valid_calls:
        
        # Increase the weight if seg_idx equals n_vehicles (dummy vehicle index)
        if seg_idx == n_vehicles:
            weights.append(weight_parameter)  # higher weight; adjust as needed
        else:
            weights.append(1)
    
    
    # Select a random call pair to move.
    src_vehicle_index, call_to_move = random.choices(valid_calls, weights=weights, k=1)[0]
    
    # Remove the call pair from the source vehicle.
    src_vehicle = vehicles[src_vehicle_index]
    src_vehicle = [x for x in src_vehicle if x != call_to_move]
    vehicles[src_vehicle_index] = src_vehicle

    # 3. Determine allowed destination vehicles for this call.
    # Only vehicles that are compatible (compatibility_table == 1) AND that do NOT already contain the call.
    allowed_destinations = []
    for v in range(n_vehicles + 1):
        if v == src_vehicle_index:
            continue  # Exclude the source vehicle.
        if compatibility_table[call_to_move - 1, v] == 1 and (call_to_move not in vehicles[v]):
            allowed_destinations.append(v)
            
    # If no allowed destination remains, return the original solution (or handle it as needed).
    if not allowed_destinations:
        print("No allowed destination vehicle available that doesn't already have call", call_to_move)
        return solution

    # Randomly choose one destination vehicle from the allowed ones.
    destination_vehicle_idx = random.choices(
    allowed_destinations,
    weights=[2 if v != n_vehicles else 1 for v in allowed_destinations],k=1)[0]
    destination_vehicle = vehicles[destination_vehicle_idx]
    pickup_idx = random.randint(0, len(destination_vehicle) + 1)
    delivery_idx = random.randint(0, len(destination_vehicle) + 2)
    destination_vehicle.insert(pickup_idx, call_to_move)
    destination_vehicle.insert(delivery_idx, call_to_move)
    
    # # 4. Generate all possible insertion permutations for call_to_move (as a pair) in destination_vehicle.
    # destination_vehicle_idx = random.choice(allowed_destinations)
    # destination_vehicle = vehicles[destination_vehicle_idx]
    # best_vehicle_perm = []
    # best_vehicle_perm_cost = np.inf
    # cache_skip = 0
    # threshold_value = 30000
    # found_threshold = False

    # if destination_vehicle_idx == src_vehicle_index:
    #     best_vehicle_perm = og_vehicles[destination_vehicle_idx]
    # # If destination_vehicle is empty, there's only one possibility.
    # elif not destination_vehicle:
    #     best_vehicle_perm = [call_to_move, call_to_move]
    # else:
    #     L = len(destination_vehicle)
    #     # The final route will have L+2 positions (indices 0 ... L+1).
    #     # Choose two indices (i, j) with i < j.
    #     for i in range(L + 2):
    #         for j in range(i, L + 1):
    #             new_route = destination_vehicle.copy()
    #             # Insert delivery first so that insertion at i doesn't shift the delivery index.
    #             new_route.insert(j, call_to_move)  # delivery insertion
    #             new_route.insert(i, call_to_move)  # pickup insertion
    #             candidate_key = tuple(new_route)
    #             if candidate_key in route_cost_cache:
    #                 cache_skip += 1
    #                 new_cost = route_cost_cache[candidate_key]
    #             else:
    #                 new_cost = cost_function_light(new_route, problem, destination_vehicle_idx)
    #                 route_cost_cache[candidate_key] = new_cost

    #             if new_cost < best_vehicle_perm_cost:
    #                 best_vehicle_perm_cost = new_cost
    #                 best_vehicle_perm = new_route
    #                 if best_vehicle_perm_cost < threshold_value:
    #                     found_threshold = True
    #                     break
    #         if found_threshold:
    #             break
                    
    
    # # Update the destination vehicle route with the best permutation found.
    # vehicles[destination_vehicle_idx] = best_vehicle_perm

    vehicles[destination_vehicle_idx] = destination_vehicle

    # 5. Reassemble the global solution list (insert 0 between segments).
    new_solution = []
    for i, seg in enumerate(vehicles):
        new_solution.extend(seg)
        if i < len(vehicles) - 1:
            new_solution.append(0)

    return new_solution



def local_search(problem, initial_solution, compatibility_table, max_iter):
    
    errors = 0
    cost = cost_function(initial_solution, problem)
    best_solution = initial_solution

    for i in range(max_iter):
        new_solution = reinsert(best_solution, problem, compatibility_table)
        if feasibility_check(new_solution, problem)[0]:
            new_cost = cost_function(new_solution, problem)
            if new_cost < cost:
                best_solution = new_solution
                cost = new_cost
                
        else:
            errors += 1

    return best_solution, cost, errors

# best_solution, cost, errors = local_search(problem, initial_solution_generator(problem), compatibility_table, 10000)
# print(best_solution, cost, errors)


# Simmulated annealing function
def simulated_annealing(problem, initial_solution, compatibility_table, max_iter):
    Tf = 0.1
    incumbent = initial_solution
    best_solution = initial_solution
    seen_solutions = set()
    # Save the initial solution as a tuple.
    seen_solutions.add(tuple(initial_solution))
    skipcount = 0
    w = []

    for _ in range(100):
        new_solution = reinsert(incumbent, problem, compatibility_table)
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
        new_solution = reinsert(incumbent, problem, compatibility_table)
        new_sol_tuple = tuple(new_solution)
        if new_sol_tuple in seen_solutions:
            skipcount += 1
            continue
        seen_solutions.add(new_sol_tuple)

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
    return best_solution, cost_function(best_solution, problem), np.inf

# best_solution, cost, errors = simulated_annealing(problem, initial_solution_generator(problem), compatibility_table, 10000)
# print(best_solution, cost, errors)

# Make a table of all the solutions we have already done computations on, 
# if the newly generated solution is in the table, skip to next one