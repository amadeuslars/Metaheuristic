import random
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

def parse_data(filename):
    data = {
        "num_nodes": 0,
        "num_vehicles": 0,
        "vehicles": [],
        "num_calls": 0,
        "vehicle_calls": {},
        "calls": [],
        "travel_times": [],
        "travel_costs": []
    }
    
    with open(filename, 'r') as file:
        lines = file.readlines()
    
    section = None
    for line in lines:
        line = line.strip()
        if not line or line.startswith('%'):
            if "number of nodes" in line:
                section = "num_nodes"
            elif "number of vehicles" in line:
                section = "num_vehicles"
            elif "for each vehicle:" in line:
                section = "vehicles"
            elif "number of calls" in line:
                section = "num_calls"
            elif "for each vehicle, vehicle index," in line:
                section = "vehicle_calls"
            elif "for each call:" in line:
                section = "calls"
            elif "travel times and costs:" in line:
                section = "travel_times"
            elif "node times and costs:" in line:
                section = 'travel_costs'
            continue
        
        # Parse sections
        if section == "num_nodes":
            data["num_nodes"] = int(line)
        elif section == "num_vehicles":
            data["num_vehicles"] = int(line)
        elif section == "vehicles":
            values = list(map(int, line.split(',')))
            data["vehicles"].append({
                "index": values[0],
                "home_node": values[1],
                "start_time": values[2],
                "capacity": values[3]
            })
        elif section == "num_calls":
            data["num_calls"] = int(line)
        elif section == "vehicle_calls":
            values = list(map(int, line.split(',')))
            data["vehicle_calls"][values[0]] = values[1:]
        elif section == "calls":
            values = list(map(int, line.split(',')))
            data["calls"].append({
                "index": values[0],
                "origin": values[1],
                "destination": values[2],
                "size": values[3],
                "penalty": values[4],
                "pickup_start": values[5],
                "pickup_end": values[6],
                "delivery_start": values[7],
                "delivery_end": values[8]
            })
        elif section == "travel_times":
            values = list(map(int, line.split(',')))
            data["travel_times"].append({
                "vehicle": values[0],
                "origin": values[1],
                "destination": values[2],
                "travel_time": values[3],
                "cost": values[4]
            })
        elif section == "travel_costs":
            values = list(map(int, line.split(',')))
            data["travel_costs"].append({
                "vehicle": values[0],
                "call": values[1],
                "origin_time": values[2],
                "origin_cost": values[3],
                "destination_time": values[4],
                "destination_cost": values[5]
            })
    
    return data


# Function for generation random solutions
def generate_random_solution(parsed_data, num_calls):
    # Extract vehicle and call information
    vehicles = [v["index"] for v in parsed_data["vehicles"]]  # List of real vehicle IDs
    calls = [c["index"] for c in parsed_data["calls"]]  # List of all call IDs
    solution = {v: [] for v in vehicles}  # Initialize solution dict
    solution[99] = []
    vehicles.append(99)
    final_solution = []

    # Step 1: Assign calls randomly to each vehicle twice   
    for i in range(num_calls):
        selected_vehicle = random.choice(vehicles)
        solution[selected_vehicle].append(calls[i])
        solution[selected_vehicle].append(calls[i])

    # Step 2: Shuffle the calls assigned to each vehicle and return list in the solution format
    
    for vehicle in solution:
        random.shuffle(solution[vehicle])
        final_solution.append(solution[vehicle])
        if vehicle == 99:
            continue
        else:
            final_solution.append(0)

    # Joins the elements of the final_solution list into a single list
    final_solution = [item for sublist in final_solution for item in (sublist if isinstance(sublist, list) else [sublist])] 
    return final_solution

# Function for checking the feasibility of a solution
from datetime import timedelta

def check_solution_feasibility(solution_list, parsed_data, num_vehicles):
    """
    Checks whether the solution (given in the format [x,x,0,x,x,...,0,x,x,...])
    is feasible for vehicles 1, 2, and 3 with respect to time windows, travel times, and capacity.
    The dummy vehicle (the last segment) is assumed always feasible.
    
    :param solution_list: Flat list in the format [x,x,0,x,x,...,0,x,x,...,0,...] where 0 is the separator.
    :param parsed_data: The dictionary returned by parse_data().
    :return: True if vehicles 1,2,3 satisfy the constraints; False otherwise.
    """
    c = "Feasible solution"
    # ---------------------
    # 1. Split the solution into segments.
    # ---------------------
    segments = []
    current_segment = []
    for item in solution_list:
        if item == 0:
            segments.append(current_segment)
            current_segment = []
        else:
            current_segment.append(item)
    # Append the last segment (dummy vehicle)
    segments.append(current_segment)
    
    # For clarity, assume:
    # segments[0] -> route for vehicle 1
    # segments[1] -> route for vehicle 2
    # segments[2] -> route for vehicle 3
    # segments[3] -> route for dummy vehicle (feasible by definition)
    if len(segments) < num_vehicles + 1:
        raise ValueError("The solution format must have four segments (three real vehicles and one dummy).")
    
    # ---------------------
    # 2. Prepare lookup dictionaries from parsed data.
    # ---------------------
    # Build a lookup for call details by call ID.
    calls_lookup = { call["index"]: call for call in parsed_data["calls"] }
    
    # Build a simple lookup for travel time between two nodes.
    # We ignore vehicle-specific travel times here for simplicity.
    travel_time_lookup = {}
    for record in parsed_data["travel_times"]:
        # record is a dict with keys: "origin", "destination", "travel_time", etc.
        key = (record["origin"], record["destination"])
        travel_time_lookup[key] = record["travel_time"]
    

    # ---------------------
    # 3. For each real vehicle (vehicles 1,2,3), simulate its route.
    # ---------------------
    vehicles = parsed_data["vehicles"]
    # We assume vehicles are stored in order (vehicle 1 is first, etc.)
    
    # Check feasibility for vehicles 
    for v_index in range(num_vehicles):
        route = segments[v_index]
        vehicle = vehicles[v_index]
        current_time = vehicle["start_time"]  # start time (assumed in same unit as time windows)
        current_node = vehicle["home_node"]
        current_load = 0
        capacity = vehicle["capacity"]
        
        # Initial check to see if a call is compatible with a vehicle
        # Get the list of allowed call IDs for this vehicle.
        allowed_calls = parsed_data["vehicle_calls"].get(vehicle["index"], [])
        for call_id in route:
            if call_id not in allowed_calls:
                c = "Vehicle {} is not allowed to serve call {}".format(vehicle["index"], call_id)
                return False, c
        # Dictionary to track which calls have been picked up.
        picked_up = {}
        
        # Process each call in the route sequentially.
        for call_id in route:
            if call_id not in calls_lookup:
                # If the call ID is not in the parsed calls, the solution is invalid.
                return False, "Call {} not found in problem data".format(call_id)
            
            call = calls_lookup[call_id]
            
            # Determine whether this is a pickup or a delivery.
            if call_id not in picked_up:
                # === PICKUP STEP ===
                # Travel from current_node to call origin.
                travel_key = (current_node, call["origin"])
                if travel_key not in travel_time_lookup:
                    # Missing travel info.
                    return False, "Missing travel info from {} to {}".format(current_node, call["origin"])
                travel_time = travel_time_lookup[travel_key]
                current_time += travel_time
                # If arriving earlier than the pickup window, wait until pickup_start.
                if current_time < call["pickup_start"]:
                    current_time = call["pickup_start"]
                # Check that we do not arrive after pickup_end.
                if current_time > call["pickup_end"]:
                    c = c = "Pickup window exceeded for call {}".format(call_id)
                    return False, c
                # Add loading time at node
                current_time += parsed_data["travel_costs"][call_id-1]["origin_time"]
                # Pickup: add call size to load.
                current_load += call["size"]
                if current_load > capacity:
                    c = "Capacity exceeded on vehicle {}".format(vehicle["index"])
                    return False, c
                # Mark this call as picked up.
                picked_up[call_id] = True
                # Update current location.
                current_node = call["origin"]
            else:
                # === DELIVERY STEP ===
                # Travel from current_node to call destination.
                travel_key = (current_node, call["destination"])
                if travel_key not in travel_time_lookup:
                    return False
                travel_time = travel_time_lookup[travel_key]
                current_time += travel_time
                if current_time < call["delivery_start"]:
                    current_time = call["delivery_start"]
                if current_time > call["delivery_end"]:
                    c = "Delivery window exceeded for call {}".format(call_id)
                    return False, c
                # Add unloading time at node
                current_time += parsed_data["travel_costs"][call_id-1]["destination_time"]
                # Delivery: subtract call size from load.
                current_load -= call["size"]
                # Update current location.
                current_node = call["destination"]
        
        # (Optional: you may require that the vehicle returns to its depot/home at the end.)
    
    # If we reached here without returning False, the routes for vehicles 1,2,3 are feasible.
    return True, c

# ---------------------

# Function to check the cost of a solution

def cost_function_real(solution_list, parsed_data, num_vehicles):
    """
    Computes the total cost of a solution.
    
    The solution is a flat list (e.g., [1,1,0,2,3,3,2,0,4,5,4,5,0,6,6,7,7])
    where 0 is the separator between routes. The first num_vehicles segments are
    for real vehicles and the final segment is for the dummy vehicle.
    
    :param solution_list: List of integers representing the solution.
    :param parsed_data: Dictionary with the problem data.
    :param num_vehicles: Number of real vehicles.
    :return: Total cost (a float or int).
    """
    
    # ---- Step 1. Split the solution into segments using 0 as separator ----
    segments = []
    current_segment = []
    for item in solution_list:
        if item == 0:
            segments.append(current_segment)
            current_segment = []
        else:
            current_segment.append(item)
    # Append the last segment (dummy vehicle)
    segments.append(current_segment)
    if len(segments) < num_vehicles + 1:
        raise ValueError("The solution must have {} segments ({} real vehicles and one dummy).".format(num_vehicles + 1, num_vehicles))
    
    total_travel_cost = 0.0
    total_port_cost = 0.0

    # ---- Step 2. Process each real vehicle's segment ----
    for i in range(num_vehicles):
        route = segments[i]
        if not route:
            continue  # Skip if no calls assigned to this vehicle.
        
        # Get the vehicle's details.
        vehicle = parsed_data["vehicles"][i]  # Assuming vehicles are in order.
        vehicle_id = vehicle["index"]
        current_node = vehicle["home_node"]
        route_cost = 0.0
        port_cost_vehicle = 0.0
        
        # For tracking whether a call is a pickup (first occurrence) or delivery (second).
        picked_up = {}
        
        # --- First, travel from the depot (home) to the first call's pickup ---
        first_call_id = route[0]
        # Find the call details.
        call_data = None
        for call in parsed_data["calls"]:
            if call["index"] == first_call_id:
                call_data = call
                break
        if call_data is None:
            raise ValueError("Call {} not found in data.".format(first_call_id))
        # For a pickup, we use the call's origin.
        next_node = call_data["origin"]
        # Look up travel cost from current_node (depot) to next_node.
        travel_cost = 0
        for rec in parsed_data["travel_times"]:
            if rec["vehicle"] == vehicle_id and rec["origin"] == current_node and rec["destination"] == next_node:
                travel_cost = rec["cost"]
                break
        route_cost += travel_cost
        current_node = next_node  # Update current position.
        
        # --- Process each call in the route ---
        for call_id in route:
            # Find call details.
            call_data = None
            for call in parsed_data["calls"]:
                if call["index"] == call_id:
                    call_data = call
                    break
            if call_data is None:
                raise ValueError("Call {} not found in data.".format(call_id))
            
            # Determine if this is a pickup or delivery.
            if call_id not in picked_up:
                # Pickup: use the call's origin.
                next_node = call_data["origin"]
                picked_up[call_id] = True
            else:
                # Delivery: use the call's destination.
                next_node = call_data["destination"]
            
            # Travel from the current node to next_node.
            travel_cost = 0
            for rec in parsed_data["travel_times"]:
                if rec["vehicle"] == vehicle_id and rec["origin"] == current_node and rec["destination"] == next_node:
                    travel_cost = rec["cost"]
                    break
            route_cost += travel_cost
            current_node = next_node
            
            # Add the port cost for this call.
            # We look up the record in travel_costs and take the average of origin_cost and destination_cost.
            port_cost = 0
            for rec in parsed_data["travel_costs"]:
                if rec["vehicle"] == vehicle_id and rec["call"] == call_id:
                    port_cost = (rec["origin_cost"] + rec["destination_cost"]) / 2.0
                    break
            port_cost_vehicle += port_cost
        
        # Since each call appears twice (pickup and delivery), we divide the port cost by 2.
        total_travel_cost += route_cost
        total_port_cost += port_cost_vehicle / 2.0

    # ---- Step 3. Process the dummy vehicle (last segment) ----
    dummy_route = segments[num_vehicles]
    dummy_penalty = 0.0
    for call_id in dummy_route:
        call_data = None
        for call in parsed_data["calls"]:
            if call["index"] == call_id:
                call_data = call
                break
        if call_data is None:
            raise ValueError("Call {} not found in data.".format(call_id))
        dummy_penalty += call_data["penalty"]
    dummy_penalty /= 2.0  # because each call appears twice
    
    # ---- Total Cost ----
    TotalCost = total_travel_cost + total_port_cost + dummy_penalty
    return TotalCost



