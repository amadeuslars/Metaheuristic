import numpy as np
from collections import namedtuple
import random


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
    """
    Precompute a lookup table where for each call (indexed 0 to n_calls-1),
    we store a list of vehicles (0-indexed) that are compatible with that call.
    
    Parameters:
        problem (dict): Must contain 'n_calls', 'n_vehicles', and 'VesselCargo'.
        
    Returns:
        list of np.ndarray: A list where each element is an array of compatible vehicle indices for that call.
    """
    num_calls = problem['n_calls']
    vessel_cargo = problem['VesselCargo']  # shape: (n_vehicles, n_calls)
    compatibility_list = []
    for call in range(num_calls):
        # For call 'call+1', find vehicles where vessel_cargo is 1.
        compatible_vehicles = np.where(vessel_cargo[:, call] == 1)[0]
        compatibility_list.append(compatible_vehicles)
    return compatibility_list

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
