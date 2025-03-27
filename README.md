Pickup and Delivery Vehicle Routing Problem
Project Overview
This project implements advanced metaheuristic algorithms for solving the Maritime Pickup and Delivery Vehicle Routing Problem with Time Windows (PDPTW). The problem involves optimizing routes for a fleet of heterogeneous vessels to serve transportation requests (calls) between different ports, while respecting various operational constraints.

We are given:

A fleet of heterogeneous vessels (vehicles) with different capacities, speeds, and cost structures
A set of transportation requests (calls), each consisting of:
A pickup location and time window
A delivery location and time window
A cargo size
A penalty cost for not serving the call
Travel costs between ports, which may vary by vessel
Port operation costs for loading/unloading cargo
Time window constraints for pickup and delivery operations

The objective is to find a feasible assignment of calls to vessels and determine the sequence of port visits that minimizes the total cost. 

The total cost includes:
Travel costs between ports
Port operation costs
Penalty costs for unserved calls (assigned to the dummy vehicle)

Constraints:
Each call must be picked up before it can be delivered
Each call must be served by exactly one vessel or left unserved (assigned to the dummy vehicle)
Vessel capacity constraints must be respected at all times
Time window constraints for pickup and delivery operations must be adhered to
Some calls may only be compatible with certain vessels

This project implements several metaheuristic algorithms:

Adaptive Large Neighborhood Search (ALNS): The primary algorithm, which uses multiple removal and insertion operators with adaptive weights based on their performance.

Simulated Annealing: A temperature-based approach that allows accepting worse solutions with a decreasing probability over time.

Local Search: A hill-climbing approach that only accepts improvements but can get stuck in local optima.

Blind Random Search: A baseline method for comparison.

Key Components
Removal Operators:
Random Removal: Removes random calls from the solution
Related Removal: Removes calls that are related based on spatial proximity, time windows, and cargo size
Worst Removal: Removes calls that contribute most to the solution cost
Dummy Removal: Strategically removes calls from the dummy vehicle to try reinserts

Insertion Operators:
Random Insertion: Inserts calls at random feasible positions
Regret Insertion: Prioritizes calls with higher "regret" value if not inserted now
Greedy Insertion: Inserts calls at their best feasible positions
Limited Permutation Insertion: Considers permutations of nearby calls to find better arrangements

The algorithm dynamically adjusts operator weights based on their performance, favoring operators that:

Produce feasible solutions
Improve the current solution
Find new best solutions
Code Structure


Assignment5.py: Main implementation of the adaptive algorithm and operators
Utils.py: Utility functions for the problem
function_tester.py: Framework for testing different algorithms on multiple problem instances

To customize the run parameters, modify the run() function in function_tester.py.

The algorithm is tested on multiple problem instances with varying sizes:

Call_7_Vehicle_3 (small)
Call_18_Vehicle_5 (medium)
Call_35_Vehicle_7 (large)
Call_80_Vehicle_20 (very large)
Call_130_Vehicle_40 (extremely large)
Call_300_Vehicle_90 (massively large)

Results are recorded in results_history.csv including:

Best solution found
Average solution quality
Runtime performance
Improvement over initial solutions
