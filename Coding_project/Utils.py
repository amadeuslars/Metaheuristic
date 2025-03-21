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


def feasibility_check(solution, problem):
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
    for call in range(1, num_calls):
        # Get all vehicles (0-indexed) for which the call is compatible.
        comp_vehicles = compatibility_list[call]
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

problem = load_problem('Data/Call_7_Vehicle_3.txt')
# print(feasibility_check([4, 4, 18, 17, 18, 17, 0, 8, 14, 14, 8, 0, 6, 9, 6, 5, 9, 5, 12, 16, 16, 12, 0, 7, 7, 3, 3, 10, 10, 0, 1, 1, 0, 13, 13, 11, 11, 15, 15, 2, 2], problem))

# print(precompute_compatibility(problem))

# print(problem.keys())

# Cargo gives all the call information for a specific vehicle. Below gives
# Information on call one start node
# print(problem['Cargo'][0][0])

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
