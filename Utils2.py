import numpy as np
from enum import Enum

class ReasonNotFeasible(Enum):
    call_in_vehicle_not_allowed = 1
    vehicle_overloaded = 2
    time_window_wrong = 3
    pickup_delivery_order_wrong = 5

def load(filename_path):
    raw_vehicle_info = []
    raw_vehicle_call_list = []
    raw_call_info = []
    raw_travel_times = []
    raw_node_costs = []

    try:
        with open(filename_path) as input_file:
            comment_line = input_file.readline().strip()
            if not comment_line.startswith("%"): raise ValueError("Missing comment line 'number of nodes'")
            node_count = int(input_file.readline().strip())

            comment_line = input_file.readline().strip()
            if not comment_line.startswith("%"): raise ValueError("Missing comment line 'number of vehicles'")
            vehicle_count = int(input_file.readline().strip())

            comment_line = input_file.readline().strip()
            if not comment_line.startswith("%"): raise ValueError("Missing comment line 'for each vehicles (time, capacity)'")
            for _ in range(vehicle_count):
                raw_vehicle_info.append(input_file.readline().strip().split(","))

            comment_line = input_file.readline().strip()
            if not comment_line.startswith("%"): raise ValueError("Missing comment line 'number of calls'")
            call_count = int(input_file.readline().strip())

            comment_line = input_file.readline().strip()
            if not comment_line.startswith("%"): raise ValueError("Missing comment line 'for each vehicles (list of transportable calls)'")
            for _ in range(vehicle_count):
                raw_vehicle_call_list.append(input_file.readline().strip().split(","))

            comment_line = input_file.readline().strip()
            if not comment_line.startswith("%"): raise ValueError("Missing comment line 'for each call'")
            for _ in range(call_count):
                raw_call_info.append(input_file.readline().strip().split(","))

            comment_line = input_file.readline().strip()
            if not comment_line.startswith("%"): raise ValueError("Missing comment line 'travel times and costs'")
            data_line = input_file.readline()
            while data_line and not data_line.startswith("%"):
                raw_travel_times.append(data_line.strip().split(","))
                data_line = input_file.readline()

            if not data_line or not data_line.startswith("%"): raise ValueError("Missing comment line 'node times and costs'")
            data_line = input_file.readline()
            while data_line: # Read until EOF
                if data_line.strip(): 
                    raw_node_costs.append(data_line.strip().split(","))
                data_line = input_file.readline()

    except FileNotFoundError:
        print(f"Error: File not found at {filename_path}")
        raise
    except ValueError as val_err:
        print(f"Error parsing file {filename_path}: {val_err}")
        raise
    except Exception as exc:
        print(f"An unexpected error occurred while reading {filename_path}: {exc}")
        raise

    travel_data = {}
    for time_cost_row in raw_travel_times:
        try:
            # Key: (vehicle_idx, from_node, to_node), Value: (time, cost)
            travel_data[(int(time_cost_row[0]), int(time_cost_row[1]), int(time_cost_row[2]))] = (int(time_cost_row[3]), int(time_cost_row[4]))
        except (IndexError, ValueError):
            print(f"Warning: Skipping invalid travel time line: {time_cost_row}")

    node_data = {}
    for node_cost_row in raw_node_costs:
        try:
            # Key: (vehicle_idx, call_idx), Value: (pickup_time, pickup_cost, delivery_time, delivery_cost) - Assuming indices 2,3,4,5 based on original code
            node_data[(int(node_cost_row[0]), int(node_cost_row[1]))] = (int(node_cost_row[2]), int(node_cost_row[3]), int(node_cost_row[4]), int(node_cost_row[5]))
        except (IndexError, ValueError):
            print(f"Warning: Skipping invalid node cost line: {node_cost_row}")

    try:
        vehicle_details_array = np.array(raw_vehicle_info, dtype=int)
        call_details_array = np.array(raw_call_info, dtype=int)
    except ValueError:
        print("Error converting vehicle or call info to integer array.")
        raise

    vehicle_compatible_calls = {}
    for compat_row in raw_vehicle_call_list:
        try:
            vehicle_id = int(compat_row[0])
            compatible_set = set(map(int, compat_row[1:]))
            vehicle_compatible_calls[vehicle_id] = compatible_set
        except (IndexError, ValueError):
            print(f"Warning: Skipping invalid vehicle calls line: {compat_row}")

    problem_data = {
        "n_nodes": node_count,
        "n_vehicles": vehicle_count,
        "n_calls": call_count,
        "travel_time_cost": travel_data, 
        "node_time_cost": node_data,     
        "vehicle_info": vehicle_details_array, 
        "call_info": call_details_array,       
        "vehicle_calls": vehicle_compatible_calls, 
        "helper_feasibility_full": {},
        "helper_feasibility_single": {},
        "helper_cost_full": {},
        "helper_cost_partly": {},
    }

    return problem_data

def make_solution_hashable(sol):
    return tuple(tuple(route) for route in sol)

def make_route_hashable(route_list):
    return tuple(route_list)

def feasibility_check(current_solution, problem_spec):
    vehicle_count = problem_spec["n_vehicles"]
    feasibility_cache_full = problem_spec["helper_feasibility_full"]
    solution_key = make_solution_hashable(current_solution)

    if solution_key in feasibility_cache_full:
        return feasibility_cache_full[solution_key]

    # compatibility first across all vehicles
    vehicle_allowed_calls = problem_spec["vehicle_calls"]
    for v_idx, vehicle_route in enumerate(current_solution[:vehicle_count]): 
        compatible_calls_for_vehicle = vehicle_allowed_calls.get(v_idx + 1, set())
        calls_in_route = set(vehicle_route)
        if not calls_in_route.issubset(compatible_calls_for_vehicle):
            fail_reason = ReasonNotFeasible.call_in_vehicle_not_allowed
            feasibility_cache_full[solution_key] = (False, fail_reason)
            return False, fail_reason

    # Feasibility for each vehicle route individually using the helper
    for v_idx, vehicle_route in enumerate(current_solution):
        is_feasible, fail_reason = feasibility_helper(vehicle_route, problem_spec, v_idx + 1)
        if not is_feasible:
            feasibility_cache_full[solution_key] = (False, fail_reason)
            return False, fail_reason

    # If all vehicles are feasible
    feasibility_cache_full[solution_key] = (True, None)
    return True, None

def feasibility_helper(route_list, problem_spec, vehicle_index_1b):
    feasibility_cache_single = problem_spec["helper_feasibility_single"]
    route_key = make_route_hashable(route_list)
    memo_key = (vehicle_index_1b, route_key)

    if memo_key in feasibility_cache_single:
        return feasibility_cache_single[memo_key]

    vehicle_count = problem_spec["n_vehicles"]
    vehicle_details = problem_spec["vehicle_info"]
    call_details = problem_spec["call_info"]
    travel_data = problem_spec["travel_time_cost"]
    node_data = problem_spec["node_time_cost"]

    if vehicle_index_1b > vehicle_count:
        call_counts = {}
        for call_id in route_list:
            call_counts[call_id] = call_counts.get(call_id, 0) + 1
        for occurrence_count in call_counts.values():
            if occurrence_count != 2:
                feasibility_result = (False, ReasonNotFeasible.pickup_delivery_order_wrong)
                feasibility_cache_single[memo_key] = feasibility_result
                return feasibility_result
        feasibility_result = (True, None)
        feasibility_cache_single[memo_key] = feasibility_result
        return feasibility_result

    
    vehicle_index_0b = vehicle_index_1b - 1

    # (1) Check Pickup/Delivery Order and Count
    pickup_occurrences = {}
    delivery_occurrences = {}
    for call_id in route_list:
        if call_id not in pickup_occurrences:
            pickup_occurrences[call_id] = 1
        elif call_id not in delivery_occurrences:
            delivery_occurrences[call_id] = 1
        else: # More than one pickup or delivery
            feasibility_result = (False, ReasonNotFeasible.pickup_delivery_order_wrong)
            feasibility_cache_single[memo_key] = feasibility_result
            return feasibility_result
    # If all pickups have corresponding deliveries
    if pickup_occurrences.keys() != delivery_occurrences.keys():
        feasibility_result = (False, ReasonNotFeasible.pickup_delivery_order_wrong)
        feasibility_cache_single[memo_key] = feasibility_result
        return feasibility_result

    # (2) Capacity Check
    vehicle_capacity = vehicle_details[vehicle_index_0b][3]
    load_on_vehicle = 0
    picked_up_set = set()
    for call_id in route_list:
        size_of_call = call_details[call_id-1][3]
        if call_id not in picked_up_set: # Pickup
            load_on_vehicle += size_of_call
            picked_up_set.add(call_id)
            if load_on_vehicle > vehicle_capacity:
                feasibility_result = (False, ReasonNotFeasible.vehicle_overloaded)
                feasibility_cache_single[memo_key] = feasibility_result
                return feasibility_result
        else: # Delivery
            load_on_vehicle -= size_of_call

    # (3) Time Window Check
    current_time = vehicle_details[vehicle_index_0b][2] # Vehicle start time
    start_node = vehicle_details[vehicle_index_0b][1] # Vehicle home node
    previous_node = start_node
    time_window_visited_set = set() # Tracks calls already picked up for TW check

    for call_id in route_list:
        call_item_details = call_details[call_id-1]
        is_delivery_stop = call_id in time_window_visited_set

        if is_delivery_stop:
            destination_node = call_item_details[2] # Delivery node
            time_window_start = call_item_details[7] # Delivery TW lower
            time_window_end = call_item_details[8]   # Delivery TW upper
            # Delivery service time 
            stop_service_time = node_data.get((vehicle_index_1b, call_id), (0, 0, 0, 0))[2]
        else:
            destination_node = call_item_details[1] # Pickup node
            time_window_start = call_item_details[5] # Pickup TW lower
            time_window_end = call_item_details[6]   # Pickup TW upper
            # Pickup service time 
            stop_service_time = node_data.get((vehicle_index_1b, call_id), (0, 0, 0, 0))[0]

        # Travel time 
        leg_travel_time = travel_data.get((vehicle_index_1b, previous_node, destination_node), (0, 0))[0]
        current_time += leg_travel_time

        if current_time > time_window_end:
            feasibility_result = (False, ReasonNotFeasible.time_window_wrong)
            feasibility_cache_single[memo_key] = feasibility_result
            return feasibility_result

        current_time = max(current_time, time_window_start) 
        current_time += stop_service_time

        previous_node = destination_node
        if not is_delivery_stop:
            time_window_visited_set.add(call_id) 

    # If all checks passed
    feasibility_result = (True, None)
    feasibility_cache_single[memo_key] = feasibility_result
    return feasibility_result

def cost_function(current_solution, problem_spec):
    cost_cache_full = problem_spec["helper_cost_full"]
    solution_key = make_solution_hashable(current_solution)

    if solution_key in cost_cache_full:
        return cost_cache_full[solution_key]

    cumulative_cost = 0.0
    for v_idx, vehicle_route in enumerate(current_solution):
        cumulative_cost += cost_helper(vehicle_route, problem_spec, v_idx + 1)

    cost_cache_full[solution_key] = cumulative_cost
    return cumulative_cost

def cost_helper(route_list, problem_spec, vehicle_index_1b):
    cost_cache_partly = problem_spec["helper_cost_partly"]
    route_key = make_route_hashable(route_list)
    memo_key = (route_key, vehicle_index_1b)

    if memo_key in cost_cache_partly:
        return cost_cache_partly[memo_key]

    call_details = problem_spec["call_info"]
    travel_data = problem_spec["travel_time_cost"]
    node_data = problem_spec["node_time_cost"]
    vehicle_details = problem_spec["vehicle_info"]
    vehicle_count = problem_spec["n_vehicles"]

    if vehicle_index_1b > vehicle_count:
        penalty_cost = 0.0
        unserved_calls = set(route_list)
        for call_id in unserved_calls:
            penalty_cost += call_details[call_id-1][4]
        cost_cache_partly[memo_key] = penalty_cost
        return penalty_cost

  
    total_travel_cost = 0.0
    total_node_cost = 0.0
    vehicle_index_0b = vehicle_index_1b - 1

    served_calls_unique = set(route_list)
    for call_id in served_calls_unique:
        
        costs_for_node = node_data.get((vehicle_index_1b, call_id), (0, 0, 0, 0))
        total_node_cost += costs_for_node[1] + costs_for_node[3] 

    
    if route_list: 
        start_node = vehicle_details[vehicle_index_0b][1] 
        previous_node = start_node
        visited_pickup_set = set() 

        for call_id in route_list:
            call_item_details = call_details[call_id-1]
            is_delivery_stop = call_id in visited_pickup_set

            if is_delivery_stop:
                destination_node = call_item_details[2] # Delivery node
            else:
                destination_node = call_item_details[1] # Pickup node
                visited_pickup_set.add(call_id) # Mark as picked up

            
            leg_travel_cost = travel_data.get((vehicle_index_1b, previous_node, destination_node), (0, 0))[1]
            total_travel_cost += leg_travel_cost
            previous_node = destination_node

    route_total_cost = total_travel_cost + total_node_cost
    cost_cache_partly[memo_key] = route_total_cost
    return route_total_cost

def initial_solution_generator(problem_spec):
    vehicle_count = problem_spec["n_vehicles"]
    call_count = problem_spec["n_calls"]

    initial_sol = [[] for _ in range(vehicle_count)] # Empty lists for regular vehicles

    # Dummy vehicle contains pairs of all calls
    unserved_route = [call_id for call_id in range(1, call_count + 1) for _ in range(2)]
    initial_sol.append(unserved_route)

    return initial_sol

