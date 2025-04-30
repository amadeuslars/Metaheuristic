import numpy as np
from collections import namedtuple
import random


np.set_printoptions(suppress=True, formatter={'float_kind':'{:0.1f}'.format})


def load_problem(filename):
    """

    :rtype: object
    :param filename: The address to the problem input file
    :return: named tuple object of the problem attributes
    """
    vehicle_info = []
    call_constraint = []
    call_info = []
    routes = []
    node_info = []
    with open(filename) as f:
        lines = f.readlines()
        num_nodes = int(lines[1])
        num_vehicles = int(lines[3])
        num_calls = int(lines[num_vehicles + 5 + 1])

        for i in range(num_vehicles):
            vehicle_info.append(lines[1 + 4 + i].split(','))

        for i in range(num_vehicles):
            call_constraint.append(lines[1 + 7 + num_vehicles + i].split(','))

        for i in range(num_calls):
            call_info.append(lines[1 + 8 + num_vehicles * 2 + i].split(','))

        for j in range(num_nodes * num_nodes * num_vehicles):
            routes.append(lines[1 + 2 * num_vehicles + num_calls + 9 + j].split(','))

        for i in range(num_vehicles * num_calls):
            node_info.append(lines[1 + 1 + 2 * num_vehicles + num_calls + 10 + j + i].split(','))
        f.close()

    # Cargo is the information about the calls, pickup and delivery locations, sizes, and time windows
    Cargo = np.array(call_info, dtype=int)[:, 1:]
    routes = np.array(routes, dtype=int)
    
    TravelTime = np.zeros((num_vehicles + 1, num_nodes + 1, num_nodes + 1))
    TravelCost = np.zeros((num_vehicles + 1, num_nodes + 1, num_nodes + 1))
    for j in range(len(routes)):
        TravelTime[routes[j, 0]][routes[j, 1], routes[j, 2]] = routes[j, 3]
        TravelCost[routes[j, 0]][routes[j, 1], routes[j, 2]] = routes[j, 4]
    
    VesselCapacity = np.zeros(num_vehicles)
    StartingTime = np.zeros(num_vehicles)
    FirstTravelTime = np.zeros((num_vehicles, num_nodes))
    FirstTravelCost = np.zeros((num_vehicles, num_nodes))
    vehicle_info = np.array(vehicle_info, dtype=int)
    for i in range(num_vehicles):
        VesselCapacity[i] = vehicle_info[i, 3]
        StartingTime[i] = vehicle_info[i, 2]
        for j in range(num_nodes):
            FirstTravelTime[i, j] = TravelTime[i + 1, vehicle_info[i, 1], j + 1] + vehicle_info[i, 2]
            FirstTravelCost[i, j] = TravelCost[i + 1, vehicle_info[i, 1], j + 1]
    TravelTime = TravelTime[1:, 1:, 1:]
    TravelCost = TravelCost[1:, 1:, 1:]
    VesselCargo = np.zeros((num_vehicles, num_calls + 1))
    call_constraint = np.array(call_constraint, dtype=object)
    for i in range(num_vehicles):
        VesselCargo[i, np.array(call_constraint[i][1:], dtype=int)] = 1
    VesselCargo = VesselCargo[:, 1:]

    LoadingTime = np.zeros((num_vehicles + 1, num_calls + 1))
    UnloadingTime = np.zeros((num_vehicles + 1, num_calls + 1))
    PortCost = np.zeros((num_vehicles + 1, num_calls + 1))
    node_info = np.array(node_info, dtype=int)
    for i in range(num_vehicles * num_calls):
        LoadingTime[node_info[i, 0], node_info[i, 1]] = node_info[i, 2]
        UnloadingTime[node_info[i, 0], node_info[i, 1]] = node_info[i, 4]
        PortCost[node_info[i, 0], node_info[i, 1]] = node_info[i, 5] + node_info[i, 3]

    LoadingTime = LoadingTime[1:, 1:]
    UnloadingTime = UnloadingTime[1:, 1:]
    PortCost = PortCost[1:, 1:]
    output = {
        'n_nodes': num_nodes,
        'n_vehicles': num_vehicles,
        'n_calls': num_calls,
        'Cargo': Cargo,
        'TravelTime': TravelTime,
        'FirstTravelTime': FirstTravelTime,
        'VesselCapacity': VesselCapacity,
        'LoadingTime': LoadingTime,
        'UnloadingTime': UnloadingTime,
        'VesselCargo': VesselCargo,
        'TravelCost': TravelCost,
        'FirstTravelCost': FirstTravelCost,
        'PortCost': PortCost
    }
    return output


def feasibility_check2(solution, problem):
    """

    :rtype: tuple
    :param solution: The input solution of order of calls for each vehicle to the problem
    :param problem: The pickup and delivery problem object
    :return: whether the problem is feasible and the reason for probable infeasibility
    """
    num_vehicles = problem['n_vehicles']
    Cargo = problem['Cargo']
    TravelTime = problem['TravelTime']
    FirstTravelTime = problem['FirstTravelTime']
    VesselCapacity = problem['VesselCapacity']
    LoadingTime = problem['LoadingTime']
    UnloadingTime = problem['UnloadingTime']
    VesselCargo = problem['VesselCargo']
    solution = np.append(solution, [0])
    ZeroIndex = np.array(np.where(solution == 0)[0], dtype=int)
    feasibility = True
    tempidx = 0
    c = 'Feasible'
    for i in range(num_vehicles):
        currentVPlan = solution[tempidx:ZeroIndex[i]]
        currentVPlan = currentVPlan - 1
        NoDoubleCallOnVehicle = len(currentVPlan)
        tempidx = ZeroIndex[i] + 1
        if NoDoubleCallOnVehicle > 0:

            if not np.all(VesselCargo[i, currentVPlan]):
                feasibility = False
                c = 'incompatible vessel and cargo'
                break
            else:
                LoadSize = 0
                currentTime = 0
                sortRout = np.sort(currentVPlan, kind='mergesort')
                I = np.argsort(currentVPlan, kind='mergesort')
                Indx = np.argsort(I, kind='mergesort')
                LoadSize -= Cargo[sortRout, 2]
                LoadSize[::2] = Cargo[sortRout[::2], 2]
                LoadSize = LoadSize[Indx]
                if np.any(VesselCapacity[i] - np.cumsum(LoadSize) < 0):
                    feasibility = False
                    c = 'Capacity exceeded'
                    break
                Timewindows = np.zeros((2, NoDoubleCallOnVehicle))
                Timewindows[0] = Cargo[sortRout, 6]
                Timewindows[0, ::2] = Cargo[sortRout[::2], 4]
                Timewindows[1] = Cargo[sortRout, 7]
                Timewindows[1, ::2] = Cargo[sortRout[::2], 5]

                Timewindows = Timewindows[:, Indx]

                PortIndex = Cargo[sortRout, 1].astype(int)
                PortIndex[::2] = Cargo[sortRout[::2], 0]
                PortIndex = PortIndex[Indx] - 1

                LU_Time = UnloadingTime[i, sortRout]
                LU_Time[::2] = LoadingTime[i, sortRout[::2]]
                LU_Time = LU_Time[Indx]
                Diag = TravelTime[i, PortIndex[:-1], PortIndex[1:]]
                FirstVisitTime = FirstTravelTime[i, int(Cargo[currentVPlan[0], 0] - 1)]

                RouteTravelTime = np.hstack((FirstVisitTime, Diag.flatten()))

                ArriveTime = np.zeros(NoDoubleCallOnVehicle)
                for j in range(NoDoubleCallOnVehicle):
                    ArriveTime[j] = np.max((currentTime + RouteTravelTime[j], Timewindows[0, j]))
                    if ArriveTime[j] > Timewindows[1, j]:
                        feasibility = False
                        c = 'Time window exceeded at call {}'.format(j)
                        break
                    currentTime = ArriveTime[j] + LU_Time[j]

    return feasibility, c


def cost_function(Solution, problem):
    """

    :param Solution: the proposed solution for the order of calls in each vehicle
    :param problem:
    :return:
    """

    num_vehicles = problem['n_vehicles']
    Cargo = problem['Cargo']
    TravelCost = problem['TravelCost']
    FirstTravelCost = problem['FirstTravelCost']
    PortCost = problem['PortCost']


    NotTransportCost = 0
    RouteTravelCost = np.zeros(num_vehicles)
    CostInPorts = np.zeros(num_vehicles)

    Solution = np.append(Solution, [0])
    ZeroIndex = np.array(np.where(Solution == 0)[0], dtype=int)
    tempidx = 0

    for i in range(num_vehicles + 1):
        currentVPlan = Solution[tempidx:ZeroIndex[i]]
        currentVPlan = currentVPlan - 1
        NoDoubleCallOnVehicle = len(currentVPlan)
        tempidx = ZeroIndex[i] + 1
        
        if i == num_vehicles:
            NotTransportCost = np.sum(Cargo[currentVPlan, 3]) / 2
        else:
            if NoDoubleCallOnVehicle > 0:
                sortRout = np.sort(currentVPlan, kind='mergesort')
                I = np.argsort(currentVPlan, kind='mergesort')
                Indx = np.argsort(I, kind='mergesort')

                PortIndex = Cargo[sortRout, 1].astype(int)
                PortIndex[::2] = Cargo[sortRout[::2], 0]
                PortIndex = PortIndex[Indx] - 1

                Diag = TravelCost[i, PortIndex[:-1], PortIndex[1:]]

                FirstVisitCost = FirstTravelCost[i, int(Cargo[currentVPlan[0], 0] - 1)]
                RouteTravelCost[i] = np.sum(np.hstack((FirstVisitCost, Diag.flatten())))
                CostInPorts[i] = np.sum(PortCost[i, currentVPlan]) / 2

    TotalCost = NotTransportCost + sum(RouteTravelCost) + sum(CostInPorts)
    return TotalCost

def precompute_compatibility(problem):
    vessel_cargo = problem['VesselCargo']  # shape: (n_vehicles, n_calls)
    # Transpose so rows are calls and columns are vehicles.
    compatibility_matrix = vessel_cargo.T
    # Create a dummy column with ones (meaning every call is compatible with the dummy vehicle)
    dummy_col = np.ones((compatibility_matrix.shape[0], 1), dtype=compatibility_matrix.dtype)
    # Append the dummy column to the compatibility matrix along the columns.
    compatibility_matrix = np.concatenate((compatibility_matrix, dummy_col), axis=1)
    
    return compatibility_matrix

# Random solution generator function
def random_solution(problem, compatibility_list):
   
    num_calls = problem['n_calls']
    num_vehicles = problem['n_vehicles']
    num_routes = num_vehicles + 1

    # Randomly assign each call (1-indexed) to a vehicle route (0-indexed here)
    # assignments[i] is the route for call i+1.
    assignments = np.empty(num_calls, dtype=int)
    
     # For each call, determine a compatible vehicle.
    for call in range(num_calls):
        # Get all vehicles (0-indexed) for which the call is compatible.
        comp_vehicles = np.where(compatibility_list[call] == 1)[0]
        if comp_vehicles.size > 0:
            # Randomly select one of the compatible vehicles.
            extended_list = np.concatenate([comp_vehicles, [num_vehicles]])
            chosen_vehicle = np.random.choice(extended_list)
        else:
            # Fallback: if no compatible vehicle is found, assign to extra route.
            chosen_vehicle = num_vehicles  
        assignments[call] = chosen_vehicle

    routes = []
    for vehicle in range(num_routes):
        # Find all call numbers assigned to this vehicle.
        # np.where returns indices (0-indexed) so we add 1 to convert to call numbers.
        calls_for_vehicle = np.where(assignments == vehicle)[0] + 1
        
        if calls_for_vehicle.size > 0:
            # Each call appears twice.
            route = np.repeat(calls_for_vehicle, 2)
            # Shuffle the order of calls in this route.
            np.random.shuffle(route)
        else:
            route = np.array([], dtype=int)

    
     # Append 0 as a separator after the route, except for the last route.
        if vehicle < num_routes - 1:
            route = np.concatenate([route, [0]])
        
        routes.append(route)
    
    # Flatten all routes into a single 1D NumPy array.
    solution = np.concatenate(routes)
    
    return solution

def feasibility_check_vehicle(vehicle, v_idx, problem):
    Cargo = problem['Cargo']
    TravelTime = problem['TravelTime']
    FirstTravelTime = problem['FirstTravelTime']
    VesselCapacity = problem['VesselCapacity']
    LoadingTime = problem['LoadingTime']
    UnloadingTime = problem['UnloadingTime']
    VesselCargo = problem['VesselCargo']
    n_vehicles = problem['n_vehicles']
    
    feasibility = True
    c = 'Feasible'
    
    # Check if route is empty
    if len(vehicle) == 0:
        return True, c
    
    # Special case for dummy vehicle - always feasible
    if v_idx == n_vehicles:
        # For dummy vehicle, just check that each call appears exactly twice
        call_counts = {}
        for call in vehicle:
            call_counts[call] = call_counts.get(call, 0) + 1
            
        for call, count in call_counts.items():
            if count != 2:
                return False, f"Call {call} appears {count} times in dummy vehicle"
        return True, c
    
    # Check compatibility for all calls in vehicle
    for call in vehicle:
        if not VesselCargo[v_idx, call-1]:  # Subtract 1 since calls are 1-indexed
            feasibility = False
            c = f'Call {call} incompatible with vessel {v_idx}'
            return feasibility, c
    
    # Process route for time windows and capacity
    NoDoubleCallOnVehicle = len(vehicle)
    if NoDoubleCallOnVehicle > 0:
        # Create call tracker to identify pickups and deliveries
        call_tracker = {}
        for i, call in enumerate(vehicle):
            if call in call_tracker:
                # This is a delivery (second occurrence)
                call_tracker[call] = (call_tracker[call], i)
            else:
                # This is a pickup (first occurrence)
                call_tracker[call] = i
        
        # Check if all calls appear exactly twice
        for call, positions in call_tracker.items():
            if not isinstance(positions, tuple):
                feasibility = False
                c = f'Call {call} is listed only once'
                return feasibility, c
            
            pickup_pos, delivery_pos = positions
            if pickup_pos > delivery_pos:
                feasibility = False
                c = f'Call {call} has delivery before pickup'
                return feasibility, c
        
        # Track capacity
        current_load = 0
        max_capacity = VesselCapacity[v_idx]
        
        # Track timing
        current_time = 0
        current_port = None
        
        # Process the route in sequence
        for i, call in enumerate(vehicle):
            is_pickup = call_tracker[call][0] == i  # True if pickup, False if delivery
            
            # Get port index (0-indexed)
            port_idx = int(Cargo[call-1, 0 if is_pickup else 1]) - 1
            
            # Calculate travel time to this port
            if i == 0:
                # First port in route, use FirstTravelTime
                travel_time = FirstTravelTime[v_idx, port_idx]
            else:
                # Use regular travel time from previous port
                travel_time = TravelTime[v_idx, current_port, port_idx]
            
            # Update current time with travel
            current_time += travel_time
            
            # Check time window
            if is_pickup:
                early = Cargo[call-1, 4]
                late = Cargo[call-1, 5]
                service_time = LoadingTime[v_idx, call-1]
            else:
                early = Cargo[call-1, 6]
                late = Cargo[call-1, 7]
                service_time = UnloadingTime[v_idx, call-1]
            
            # Wait if we arrive before the early time window
            current_time = max(current_time, early)
            
            # Check if we missed the time window
            if current_time > late:
                feasibility = False
                c = f'Time window exceeded for {"pickup" if is_pickup else "delivery"} of call {call}'
                return feasibility, c
            
            # Update capacity
            if is_pickup:
                current_load += Cargo[call-1, 2]
            else:
                current_load -= Cargo[call-1, 2]
            
            # Check capacity constraint
            if current_load > max_capacity:
                feasibility = False
                c = f'Capacity exceeded after {"pickup" if is_pickup else "delivery"} of call {call}'
                return feasibility, c
            
            # Add service time
            current_time += service_time
            
            # Update current port
            current_port = port_idx
    
    return feasibility, c

def initial_solution_generator(problem):

    num_calls = problem['n_calls']
    num_vehicles = problem['n_vehicles']
    solution = []

    for i in range(num_vehicles):
        solution.append(0)

    for i in range(1, num_calls + 1):
        solution.append(i)
        solution.append(i)

    return solution

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


problem = load_problem('Data/Call_35_Vehicle_7.txt')
print(problem["Cargo"][0,1])
# print(feasibility_check2([4, 4, 15, 1, 15, 11, 11, 16, 16, 0, 8, 14, 14, 8, 0, 6, 6, 5, 5, 18, 18, 13, 13, 0, 7, 7, 3, 3, 10, 10, 12, 12, 0, 9, 9, 17, 17, 0, 2, 2], problem))

# print(precompute_compatibility(problem))

# print(problem.keys())

# Cargo gives all the call information for a specific vehicle. Below gives
# Information on call one start node
# print(problem['Cargo'][0][2])

# Travel time gives a matrix for car i starting at node j
# Below we get the travel time of car 3 starting at node 2
# print(problem['TravelTime'][2][1])

# First travel time prints time it takes from vehicles start node to all the
# possible first moves. Below we get the list for vehicle 1 who has start node 8
# print(problem['FirstTravelTime'][0])

# The capacity of each car. 
# print(problem["VesselCapacity"])

# Gives the time at origin node of loading the cargo of a specific call.
# For example vehicle 1 uses 29 units to load call 2. -1 when call is incompatible
# print(problem['LoadingTime'])

# Gives the time at destination node for a specific call to unload the cargo.
# print(problem['UnloadingTime'])

# Vessel cargo gives a matrix of 0s and 1s where a 1 indicates that a vehicle
# is compatible with a specific call. Below prints comp.list for vehicle 1.
# print(problem['VesselCargo'][0])

# Travel cost gives a matrix where each number represents the cost of travelling
# from a node to another. Below gives for car 1 all the cost starting at node 1
# going to any of the 39 other nodes.
# print(problem['TravelCost'][0][0][16])

# Frist travel cost gives the cost of going from the vehicles start node to 
# any of the first possible moves. Below gives the matrix for vehicle 1.
# print(problem['FirstTravelCost'][0])

# Port cost gives the sum of origin node cost and destionation node cost for 
# each vehicle, so below are the port costs for vehicle 1 for all the calls
# print(problem['PortCost'][0])

