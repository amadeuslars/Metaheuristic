from Utils2 import *
import csv
import os
import datetime
import random
import time
import numpy as np
import concurrent.futures

filenames = [
    # "Data/Call_7_Vehicle_3.txt",
    # "Data/Call_18_Vehicle_5.txt",
    # "Data/Call_35_Vehicle_7.txt",
    # "Data/Call_80_Vehicle_20.txt",
    "Data/Call_130_Vehicle_40.txt",
    ]

# Helper functions
def parse_solution_to_vehicles(solution):
    vehicles = []
    current_vehicle = []

    for call in solution:
        if call == 0:
            vehicles.append(current_vehicle)
            current_vehicle = []
        else:
            current_vehicle.append(call)

    vehicles.append(current_vehicle)
    return vehicles

def reassemble_solution(vehicles):
    solution = []
    for i, vehicle in enumerate(vehicles):
        solution.extend(vehicle)
        if i < len(vehicles) - 1:
            solution.append(0)
    return solution

def cost(vehicles, problem):
    return cost_function(vehicles, problem)

def feasibility(vehicles, problem):
    return feasibility_check(vehicles, problem)

def escape(current, best_solution, best_cost, problem):
    updated_best_solution = [v.copy() for v in best_solution]
    updated_best_cost = best_cost
    found_new_global_best = False

    best_in_escape_solution = None
    best_in_escape_cost = float('inf')

    max_attempts = 20

    perturbation_operators = [
        (random_removal, random_insertion),
        
        
    ]

    for attempt in range(max_attempts):
        temp_escape_solution = [v.copy() for v in current]

        remove_op, insert_op = random.choice(perturbation_operators)

        min_k = max(2, round(0.20 * problem['n_calls']))
        max_k = max(min_k, round(0.70 * problem['n_calls']))
        if problem['n_calls'] >= min_k:
             k = random.randint(min_k, max_k)
        else:
             k = problem['n_calls']

        if k <= 0: continue

        try:
            if 'segment' in remove_op.__name__ and 'insertion' in remove_op.__name__:
                 temp_escape_solution, removed_calls, _ = remove_op(temp_escape_solution, problem, k)
            else:
                 temp_escape_solution, removed_calls = remove_op(temp_escape_solution, problem, k)

            temp_escape_solution = insert_op(temp_escape_solution, removed_calls, problem)

        except Exception as e:
            print(f"  Error during escape perturbation ({remove_op.__name__}/{insert_op.__name__}): {e}")
            continue

        is_feasible, details = feasibility(temp_escape_solution, problem)
        if is_feasible:
            perturbed_cost = cost(temp_escape_solution, problem)

            if perturbed_cost < best_in_escape_cost:
                best_in_escape_solution = [v.copy() for v in temp_escape_solution]
                best_in_escape_cost = perturbed_cost

            if perturbed_cost < updated_best_cost:
                updated_best_solution = [v.copy() for v in temp_escape_solution]
                updated_best_cost = perturbed_cost
                found_new_global_best = True


    if best_in_escape_solution is not None:
        solution_to_return = best_in_escape_solution
    else:
        solution_to_return = [v.copy() for v in current]


    return solution_to_return, updated_best_solution, updated_best_cost, found_new_global_best

def update_operator_weights(operators):
    r = 0.1

    for op in operators:
        attempts = max(1, op['attempts'])

        found_improving = op['improvements']
        found_new_best = op.get('best_improvements', 0)


        score_improving = 3 * found_improving
        score_best = 10.0 * found_new_best

        pi = score_improving + score_best

        period_score = pi / attempts

        op_old_weight = op['weight']
        new_weight = op_old_weight * (1 - r) + period_score * r

        op['weight'] = max(0.1, new_weight)

        op['successes'] = 0
        op['attempts'] = 0
        op['improvements'] = 0
        op['best_improvements'] = 0

def build_call_info(removed_calls, n_vehicles, problem):
    call_info = []
    for call in removed_calls:
        compatible_vehicles = []

        compatible_vehicles.append(n_vehicles)

        for v_idx in range(1, n_vehicles + 1):
            if call in problem['vehicle_calls'][v_idx]:
                compatible_vehicles.append(v_idx - 1)

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

    relatedness_matrix = np.zeros((n_calls, n_calls))

    max_travel_cost = 0
    for v_idx in range(1, n_vehicles + 1):
        for i in range(n_calls):
            origin_i = call_info[i][1]
            dest_i = call_info[i][2]

            for j in range(n_calls):
                origin_j = call_info[j][1]
                dest_j = call_info[j][2]

                travel_cost_oo = problem['travel_time_cost'].get((v_idx, origin_i, origin_j), [0, 0])[1]
                travel_cost_od = problem['travel_time_cost'].get((v_idx, origin_i, dest_j), [0, 0])[1]
                travel_cost_do = problem['travel_time_cost'].get((v_idx, dest_i, origin_j), [0, 0])[1]
                travel_cost_dd = problem['travel_time_cost'].get((v_idx, dest_i, dest_j), [0, 0])[1]

                max_cost = max(travel_cost_oo, travel_cost_od, travel_cost_do, travel_cost_dd)
                max_travel_cost = max(max_travel_cost, max_cost)

    max_pickup_tw_diff = 0
    max_delivery_tw_diff = 0

    for i in range(n_calls):
        pickup_early_i = call_info[i][5]
        pickup_late_i = call_info[i][6]
        delivery_early_i = call_info[i][7]
        delivery_late_i = call_info[i][8]

        for j in range(n_calls):
            pickup_early_j = call_info[j][5]
            pickup_late_j = call_info[j][6]
            delivery_early_j = call_info[j][7]
            delivery_late_j = call_info[j][8]

            pickup_tw_diff = abs(pickup_early_i - pickup_early_j) + abs(pickup_late_i - pickup_late_j)
            delivery_tw_diff = abs(delivery_early_i - delivery_early_j) + abs(delivery_late_i - delivery_late_j)

            max_pickup_tw_diff = max(max_pickup_tw_diff, pickup_tw_diff)
            max_delivery_tw_diff = max(max_delivery_tw_diff, delivery_tw_diff)

    max_tw_diff = max(max_pickup_tw_diff, max_delivery_tw_diff)

    max_cargo_size_diff = 0

    for i in range(n_calls):
        cargo_size_i = call_info[i][3]

        for j in range(n_calls):
            cargo_size_j = call_info[j][3]
            cargo_size_diff = abs(cargo_size_i - cargo_size_j)
            max_cargo_size_diff = max(max_cargo_size_diff, cargo_size_diff)

    for i in range(n_calls):
        origin_i = call_info[i][1]
        dest_i = call_info[i][2]
        cargo_size_i = call_info[i][3]
        pickup_early_i = call_info[i][5]
        pickup_late_i = call_info[i][6]
        delivery_early_i = call_info[i][7]
        delivery_late_i = call_info[i][8]

        for j in range(n_calls):
            if i == j:
                relatedness_matrix[i, j] = 1.0
                continue

            origin_j = call_info[j][1]
            dest_j = call_info[j][2]
            cargo_size_j = call_info[j][3]
            pickup_early_j = call_info[j][5]
            pickup_late_j = call_info[j][6]
            delivery_early_j = call_info[j][7]
            delivery_late_j = call_info[j][8]

            avg_travel_cost = 0
            count = 0

            for v_idx in range(1, n_vehicles + 1):
                if i+1 in problem['vehicle_calls'][v_idx] and j+1 in problem['vehicle_calls'][v_idx]:
                    oo_key = (v_idx, origin_i, origin_j)
                    travel_cost_oo = problem['travel_time_cost'].get(oo_key, [0, 0])[1]

                    od_key = (v_idx, origin_i, dest_j)
                    travel_cost_od = problem['travel_time_cost'].get(od_key, [0, 0])[1]

                    do_key = (v_idx, dest_i, origin_j)
                    travel_cost_do = problem['travel_time_cost'].get(do_key, [0, 0])[1]

                    dd_key = (v_idx, dest_i, dest_j)
                    travel_cost_dd = problem['travel_time_cost'].get(dd_key, [0, 0])[1]

                    vehicle_avg = (travel_cost_oo + travel_cost_od + travel_cost_do + travel_cost_dd) / 4
                    avg_travel_cost += vehicle_avg
                    count += 1

            if count == 0:
                avg_travel_cost = max_travel_cost
            else:
                avg_travel_cost /= count

            norm_travel_cost = 1 - (min(1.0, avg_travel_cost / max_travel_cost) if max_travel_cost > 0 else 0)

            pickup_tw_diff = abs(pickup_early_i - pickup_early_j) + abs(pickup_late_i - pickup_late_j)
            delivery_tw_diff = abs(delivery_early_i - delivery_early_j) + abs(delivery_late_i - delivery_late_j)

            norm_pickup_tw = 1 - (min(1.0, pickup_tw_diff / max_tw_diff) if max_tw_diff > 0 else 0)
            norm_delivery_tw = 1 - (min(1.0, delivery_tw_diff / max_tw_diff) if max_tw_diff > 0 else 0)

            cargo_size_diff = abs(cargo_size_i - cargo_size_j)
            norm_cargo_size = 1 - (min(1.0, cargo_size_diff / max_cargo_size_diff) if max_cargo_size_diff > 0 else 0)

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

    call_locations = []

    for v_idx, vehicle in enumerate(vehicles):
        call_counts = {}
        for pos, call in enumerate(vehicle):
            if call not in call_counts:
                call_counts[call] = [pos]
            else:
                call_counts[call].append(pos)

        for call, positions in call_counts.items():
            if len(positions) == 2:
                call_locations.append((v_idx, call, positions[0], positions[1]))

    call_costs = []

    for v_idx, call, pickup_pos, delivery_pos in call_locations:
        if v_idx == n_vehicles:
            penalty_cost = problem['call_info'][call-1][4]
            call_costs.append((v_idx, call, penalty_cost))
            continue

        vehicle = vehicles[v_idx]
        current_cost = cost_helper(vehicle, problem, v_idx+1)

        vehicle_without_call = [c for c in vehicle if c != call]
        cost_without_call = cost_helper(vehicle_without_call, problem, v_idx+1)

        cost_contribution = current_cost - cost_without_call

        if cost_contribution <= 0:
            cost_contribution = 0.1

        call_costs.append((v_idx, call, cost_contribution))

    if not call_costs:
        return vehicles, []

    call_costs.sort(key=lambda x: x[2], reverse=True)

    total_cost = sum(cost for _, _, cost in call_costs)
    if total_cost > 0:
        weights = [cost/total_cost for _, _, cost in call_costs]
        total_weight = sum(weights)
        probs = [w/total_weight for w in weights]
    else:
        probs = [1.0/len(call_costs)] * len(call_costs)

    k = min(k, len(call_costs))
    indices_to_remove = np.random.choice(
        len(call_costs),
        size=k,
        replace=False,
        p=probs
    )

    calls_to_remove = [call_costs[i][:2] for i in indices_to_remove]

    removed_calls = []

    for v_idx, call in calls_to_remove:
        pickup_pos = delivery_pos = None
        for pos, c in enumerate(vehicles[v_idx]):
            if c == call:
                if pickup_pos is None:
                    pickup_pos = pos
                else:
                    delivery_pos = pos
                    break

        if pickup_pos is not None and delivery_pos is not None:
            vehicles[v_idx].pop(delivery_pos)
            vehicles[v_idx].pop(pickup_pos)
            removed_calls.append(call)

    call_info = build_call_info(removed_calls, n_vehicles, problem)

    return vehicles, call_info

def random_removal(vehicles, problem, k):
    n_vehicles = problem['n_vehicles']

    call_locations = []

    for v_idx, vehicle in enumerate(vehicles):
        call_counts = {}
        for pos, call in enumerate(vehicle):
            if call not in call_counts:
                call_counts[call] = [pos]
            else:
                call_counts[call].append(pos)

        for call, positions in call_counts.items():
            if len(positions) == 2:
                call_locations.append((v_idx, call))

    dummy_calls = [(v_idx, call) for v_idx, call in call_locations if v_idx == n_vehicles]
    regular_calls = [(v_idx, call) for v_idx, call in call_locations if v_idx != n_vehicles]

    k = min(k, len(call_locations))
    calls_to_remove = []

    if dummy_calls:
        dummy_probability = 0.7

        for _ in range(k):
            if dummy_calls and (random.random() < dummy_probability or not regular_calls):
                selected_idx = random.randrange(len(dummy_calls))
                selected_location = dummy_calls.pop(selected_idx)
                calls_to_remove.append(selected_location)
            elif regular_calls:
                selected_idx = random.randrange(len(regular_calls))
                selected_location = regular_calls.pop(selected_idx)
                calls_to_remove.append(selected_location)
            else:
                break
    else:
        num_to_select = min(k, len(regular_calls))
        calls_to_remove = random.sample(regular_calls, num_to_select)

    removed_calls = []

    for v_idx, call in calls_to_remove:
        pickup_pos = delivery_pos = None
        for pos, c in enumerate(vehicles[v_idx]):
            if c == call:
                if pickup_pos is None:
                    pickup_pos = pos
                else:
                    delivery_pos = pos
                    break

        if pickup_pos is not None and delivery_pos is not None:
            vehicles[v_idx].pop(delivery_pos)
            vehicles[v_idx].pop(pickup_pos)
            removed_calls.append(call)

    call_info = build_call_info(removed_calls, n_vehicles, problem)

    return vehicles, call_info

def related_removal(vehicles, problem, k, relatedness_matrix):

    n_vehicles = problem['n_vehicles']

    call_locations = []
    for v_idx, vehicle in enumerate(vehicles):
        call_counts = {}
        for pos, call in enumerate(vehicle):
            if call not in call_counts:
                call_counts[call] = [pos]
            else:
                call_counts[call].append(pos)

        for call, positions in call_counts.items():
            if len(positions) == 2:
                call_locations.append((v_idx, call, positions[0], positions[1]))

    if not call_locations:
        return vehicles, []

    call_to_index = {call: call-1 for _, call, _, _ in call_locations}

    seed_location = random.choice(call_locations)
    seed_v_idx, seed_call, _, _ = seed_location

    selected_calls = [(seed_v_idx, seed_call)]
    remaining_calls = [(v_idx, call) for v_idx, call, _, _ in call_locations
                      if call != seed_call or v_idx != seed_v_idx]

    for _ in range(min(k-1, len(remaining_calls))):
        if not remaining_calls:
            break

        relatedness_scores = []

        for v_idx, call in remaining_calls:
            call_idx = call_to_index[call]

            max_relatedness = -1
            for selected_v_idx, selected_call in selected_calls:
                selected_idx = call_to_index[selected_call]
                relatedness = relatedness_matrix[call_idx, selected_idx]
                max_relatedness = max(max_relatedness, relatedness)

            relatedness_scores.append(max_relatedness)

        alpha = 2
        weights = [score ** alpha for score in relatedness_scores]

        if sum(weights) <= 0:
            weights = [1.0] * len(remaining_calls)

        total_weight = sum(weights)
        probabilities = [w / total_weight for w in weights]

        selected_idx = np.random.choice(len(remaining_calls), p=probabilities)
        selected_call = remaining_calls.pop(selected_idx)
        selected_calls.append(selected_call)

    removed_calls = []

    for v_idx, call in selected_calls:
        pickup_pos = delivery_pos = None
        for pos, c in enumerate(vehicles[v_idx]):
            if c == call:
                if pickup_pos is None:
                    pickup_pos = pos
                else:
                    delivery_pos = pos
                    break

        if pickup_pos is not None and delivery_pos is not None:
            vehicles[v_idx].pop(delivery_pos)
            vehicles[v_idx].pop(pickup_pos)
            removed_calls.append(call)

    call_info = build_call_info(removed_calls, n_vehicles, problem)

    return vehicles, call_info

def segment_removal(vehicles, problem, k):
    n_vehicles = problem['n_vehicles']
    potential_segments = []

    for v_idx, vehicle in enumerate(vehicles):
        if v_idx == n_vehicles or len(vehicle) < k:
            continue

        original_vehicle_cost = cost_helper(vehicle, problem, v_idx + 1)
        if original_vehicle_cost == float('inf'):
             continue

        for i in range(len(vehicle) - k + 1):
            segment_to_remove = vehicle[i : i + k]
            vehicle_without_segment = vehicle[:i] + vehicle[i+k:]

            is_valid_removal = True
            remaining_counts = {}
            for call in vehicle_without_segment:
                remaining_counts[call] = remaining_counts.get(call, 0) + 1
            for call, count in remaining_counts.items():
                if count == 1:
                    is_valid_removal = False
                    break
            if not is_valid_removal:
                continue

            is_feas_remaining, _ = feasibility_helper(vehicle_without_segment, problem, v_idx + 1)
            if not is_feas_remaining:
                continue

            cost_without_segment = cost_helper(vehicle_without_segment, problem, v_idx + 1)
            cost_contribution = original_vehicle_cost - cost_without_segment

            calls_removed_set = set(c for c in segment_to_remove)

            potential_segments.append({
                "v_idx": v_idx,
                "start_idx": i,
                "cost": cost_contribution,
                "calls": list(calls_removed_set)
            })

    if not potential_segments:
        return random_removal(vehicles, problem, k)

    costs = np.array([seg['cost'] for seg in potential_segments])
    min_cost = np.min(costs)
    if min_cost <= 0:
        costs += abs(min_cost) + 1

    total_cost = np.sum(costs)
    if total_cost > 0:
        probs = costs / total_cost
        probs /= np.sum(probs)
    else:
        probs = np.ones(len(potential_segments)) / len(potential_segments)

    chosen_idx = np.random.choice(len(potential_segments), p=probs)
    segment_to_remove_info = potential_segments[chosen_idx]

    v_idx_to_modify = segment_to_remove_info["v_idx"]
    start_idx = segment_to_remove_info["start_idx"]
    vehicles[v_idx_to_modify] = vehicles[v_idx_to_modify][:start_idx] + vehicles[v_idx_to_modify][start_idx+k:]

    removed_call_ids = segment_to_remove_info["calls"]

    call_info = build_call_info(removed_call_ids, n_vehicles, problem)

    return vehicles, call_info

def historical_removal(vehicles, problem, k, call_blame):
    n_vehicles = problem['n_vehicles']

    call_locations = get_calls(vehicles)

    if not call_locations:
        return vehicles, []

    calls_with_blame = []
    for v_idx, call_id in call_locations:
        blame_score = call_blame.get(call_id, 0)
        calls_with_blame.append({'v_idx': v_idx, 'call': call_id, 'blame': blame_score})

    num_to_remove = min(k, len(calls_with_blame))
    if num_to_remove == 0:
        return vehicles, []

    blame_scores = np.array([c['blame'] for c in calls_with_blame])

    weights = blame_scores + 1

    if np.sum(weights) <= 0:
        probabilities = np.ones(len(calls_with_blame)) / len(calls_with_blame)
    else:
        probabilities = weights / np.sum(weights)
        probabilities /= np.sum(probabilities)


    selected_indices = np.random.choice(
        len(calls_with_blame),
        size=num_to_remove,
        replace=False,
        p=probabilities
    )

    calls_to_remove_info = [calls_with_blame[i] for i in selected_indices]

    removed_call_ids = []
    calls_to_remove_dict = {}
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

    call_info = build_call_info(removed_call_ids, n_vehicles, problem)

    return vehicles, call_info

def random_insertion(vehicles, call_info, problem):
    n_vehicles = problem['n_vehicles']

    for call, compatible_vehicles in call_info:
        inserted = False

        v_idx = random.choice(compatible_vehicles)

        if v_idx == n_vehicles:
            vehicles[n_vehicles].append(call)
            vehicles[n_vehicles].append(call)
            vehicles[n_vehicles].sort()
            continue

        vehicle = vehicles[v_idx]

        for p_idx in range(len(vehicle) + 1):
            if inserted:
                break

            for d_idx in range(p_idx + 1, len(vehicle) + 2):
                temp_vehicle = vehicle.copy()
                temp_vehicle.insert(p_idx, call)
                temp_vehicle.insert(d_idx, call)

                is_feas, _ = feasibility_helper(temp_vehicle, problem, v_idx+1)

                if is_feas:
                    vehicles[v_idx] = temp_vehicle
                    inserted = True
                    break

        if not inserted:
            vehicles[n_vehicles].append(call)
            vehicles[n_vehicles].append(call)
            vehicles[n_vehicles].sort()

    return vehicles

# Insertion functions
def greedy_insertion(vehicles, call_info, problem):
    n_vehicles = problem['n_vehicles']

    for call, compatible_vehicles in call_info:
        regular_vehicles = [v for v in compatible_vehicles if v < n_vehicles]

        vehicles_to_try = regular_vehicles

        if n_vehicles in compatible_vehicles:
            vehicles_to_try.append(n_vehicles)

        if not vehicles_to_try:
            continue

        insertion_pool = []
        best_delta = float('inf')

        for v_idx in vehicles_to_try:
            vehicle = vehicles[v_idx]

            if v_idx == n_vehicles:
                dummy_delta = problem['call_info'][call-1][4]
                insertion_pool.append({
                    'v_idx': n_vehicles,
                    'p_idx': len(vehicle),
                    'd_idx': len(vehicle) + 1,
                    'delta': dummy_delta
                })
                if dummy_delta < best_delta:
                    best_delta = dummy_delta
                continue


            base_cost = cost_helper(vehicle, problem, v_idx+1)

            vehicle_len = len(vehicle)

            for p_idx in range(vehicle_len + 1):
                for d_idx in range(p_idx, vehicle_len + 2):
                    temp_vehicle = vehicle.copy()
                    temp_vehicle.insert(p_idx, call)

                    d_idx_adjusted = d_idx if d_idx <= p_idx else d_idx + 1
                    temp_vehicle.insert(d_idx_adjusted, call)

                    is_feas, _ = feasibility_helper(temp_vehicle, problem, v_idx+1)

                    if is_feas:
                        new_cost = cost_helper(temp_vehicle, problem, v_idx+1)
                        delta_cost = new_cost - base_cost

                        if best_delta == float('inf') or delta_cost < 2.0 * best_delta:
                            insertion_pool.append({
                                'v_idx': v_idx,
                                'p_idx': p_idx,
                                'd_idx': d_idx,
                                'delta': delta_cost
                            })

                        if delta_cost < best_delta:
                            best_delta = delta_cost

        if insertion_pool:
            insertion_pool.sort(key=lambda x: x['delta'])

            deltas = [pos['delta'] for pos in insertion_pool]

            if best_delta > 0:
                norm_deltas = [d/best_delta for d in deltas]
            else:
                min_positive = min([d for d in deltas if d > 0], default=0.1)
                norm_deltas = [(d - best_delta + min_positive) / min_positive for d in deltas]

            explore_factor = -2.0
            weights = [np.exp(explore_factor * nd) for nd in norm_deltas]

            total_weight = sum(weights)
            probs = [w/total_weight for w in weights]

            selected_idx = np.random.choice(len(insertion_pool), p=probs)
            selected = insertion_pool[selected_idx]

            v_idx = selected['v_idx']
            p_idx = selected['p_idx']
            d_idx = selected['d_idx']

            if v_idx == n_vehicles:
                vehicles[n_vehicles].append(call)
                vehicles[n_vehicles].append(call)
            else:
                vehicle = vehicles[v_idx]
                vehicle.insert(p_idx, call)

                d_idx_adjusted = d_idx
                if d_idx > p_idx:
                    d_idx_adjusted = d_idx + 1

                vehicle.insert(d_idx_adjusted, call)
        else:
            vehicles[n_vehicles].append(call)
            vehicles[n_vehicles].append(call)

    return vehicles

def regret_insertion(vehicles, call_info, problem):
    n_vehicles = problem['n_vehicles']

    call_regrets = []

    for call, compatible_vehicles in call_info:
        vehicle_best_costs = {}

        for v_idx in [v for v in compatible_vehicles if v != n_vehicles]:
            vehicle = vehicles[v_idx]

            call_size = problem['call_info'][call-1][3]
            vehicle_capacity = problem['vehicle_info'][v_idx][3]
            if call_size > vehicle_capacity:
                continue

            best_cost = float('inf')
            for p_idx in range(len(vehicle) + 1):
                for d_idx in range(p_idx + 1, len(vehicle) + 2):
                    temp_vehicle = vehicle.copy()
                    temp_vehicle.insert(p_idx, call)

                    d_idx_adjusted = d_idx if d_idx <= p_idx else d_idx + 1
                    temp_vehicle.insert(d_idx_adjusted, call)

                    is_feas, _ = feasibility_helper(temp_vehicle, problem, v_idx+1)
                    if not is_feas:
                        continue

                    cost_val = cost_helper(temp_vehicle, problem, v_idx+1)
                    if cost_val < best_cost:
                        best_cost = cost_val

            if best_cost < float('inf'):
                vehicle_best_costs[v_idx] = best_cost

        if len(vehicle_best_costs) >= 2:
            sorted_costs = sorted(vehicle_best_costs.values())
            regret_value = sorted_costs[1] - sorted_costs[0]
        elif len(vehicle_best_costs) == 1:
            regret_value = 10000
        else:
            regret_value = problem['call_info'][call-1][4]

        call_regrets.append((call, compatible_vehicles, regret_value))

    call_regrets.sort(key=lambda x: x[2], reverse=True)

    remaining_calls = call_regrets.copy()

    while remaining_calls:
        weights = []
        for _, _, regret in remaining_calls:
            weight = regret ** 2
            weights.append(weight)

        if sum(weights) > 0:
            probs = [w / sum(weights) for w in weights]
        else:
            probs = [1.0 / len(remaining_calls)] * len(remaining_calls)

        selected_idx = np.random.choice(len(remaining_calls), p=probs)
        call, compatible_vehicles, _ = remaining_calls.pop(selected_idx)

        greedy_insertion(vehicles, [(call, compatible_vehicles)], problem)

    return vehicles

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

def historical_greedy(vehicles, problem, k, call_blame):
    vehicles, removed_calls = historical_removal(vehicles, problem, k, call_blame)
    vehicles = greedy_insertion(vehicles, removed_calls, problem)
    return vehicles

def historical_regret(vehicles, problem, k, call_blame):
    vehicles, removed_calls = historical_removal(vehicles, problem, k, call_blame)
    vehicles = regret_insertion(vehicles, removed_calls, problem)
    return vehicles


# Algorithm
def adaptive_algorithm(initial_solution, problem, neighbours, relatedness_matrix, max_iter=None, max_time=None):
    """
    Implements the Adaptive Large Neighborhood Search algorithm with Simulated Annealing.
    """
    # --- Input Validation ---
    if max_iter is None and max_time is None:
        raise ValueError("Either max_iter or max_time must be specified.")
    if max_iter is not None and max_time is not None:
        print("Warning: Both max_iter and max_time specified. Algorithm will stop when the first limit is reached.")

    # --- Initialization ---
    vehicles = initial_solution # Current working solution (list of lists)
    best_solution = [vehicle.copy() for vehicle in vehicles] # Best solution found so far
    current_solution = [vehicle.copy() for vehicle in vehicles] # Solution at the start of the current iteration
    best_cost_value = cost(best_solution, problem) # Cost of the best solution
    current_cost_value = best_cost_value # Cost of the current solution
    feasibles = 0 # Counter for feasible solutions generated by LNS operators

    print(f'Initial cost: {current_cost_value:.2f}')
    best_iteration = 0 # Iteration where the best solution was found
    actual_iterations_run = 0 # Track actual iterations performed

    # --- History Tracking ---
    history = {
        "iteration": [], "best_cost": [], "current_cost": [],
        "acceptance_rate": [], "operator_used": [], "delta_value": []
    }
    iteration_points = [] # For plotting operator weights at intervals
    operator_weight_history = {op.__name__: [] for op in neighbours} # Track weights over time
    operator_selection_counts = {op.__name__: 0 for op in neighbours} # Track how often each operator is chosen
    accepted_solutions = 0 # Count accepted moves (improving or SA)
    total_evaluations = 0 # Count total LNS operator applications

    # --- Simulated Annealing Parameters ---
    T_initial = 0.2 * current_cost_value if current_cost_value > 0 else 1000 # Initial temperature (avoid T=0)
    T_final = 0.1 # Final temperature
    T = T_initial # Current temperature
    # Cooling rate (alpha) calculation based on stopping criterion

    alpha = (T_final / T_initial) ** (1 / 20000) # Geometric cooling for iteration limit
    

    # --- Operator Setup ---
    operators = [{"name": op.__name__, "func": op, "weight": 1.0,
                 "successes": 0, "attempts": 0, "best_improvements": 0, 'improvements': 0} for op in neighbours]

    # --- Historical Information ---
    call_blame = {} # Dictionary to track how often calls are in rejected/infeasible solutions

    # --- Escape Mechanism Parameters ---
    iters_since_improvement = 0 # Counter for iterations without finding a new best solution
    escape_threshold = 3000 # Iterations without improvement to trigger escape mechanism

    # --- Main Loop ---
    start_loop_time = time.time()
    iteration = 0
    while True:
        actual_iterations_run = iteration # Track actual iterations

        # --- Check Stopping Criteria ---
        elapsed_time = time.time() - start_loop_time
        stop_reason = None
        if max_iter is not None and iteration >= max_iter:
            stop_reason = f"Reached max iterations ({max_iter})"
        elif max_time is not None and elapsed_time >= max_time:
            stop_reason = f"Reached max time ({elapsed_time:.2f}s / {max_time}s)"

        if stop_reason:
            print(f"Stopping: {stop_reason}")
            break
        # --- End Stopping Criteria Check ---

        # --- Periodic Output ---
        if iteration % 1000 == 0 and iteration > 0: # Print status every 1000 iterations
            print(f'Iteration: {iteration} | Current: {current_cost_value:.2f} | Best: {best_cost_value:.2f} | Temp: {T:.2f} | Time: {elapsed_time:.2f}s')

        # --- Escape Mechanism ---
        if iters_since_improvement >= escape_threshold:
            # print(f"--- Iteration {iteration}: Triggering Escape Mechanism (Stuck for {iters_since_improvement} iters) ---")
            new_current_solution, best_solution_updated, best_cost_updated, found_global = escape(
                current_solution, best_solution, best_cost_value, problem
            )
            # Always update current solution from escape attempt
            current_solution = [v.copy() for v in new_current_solution]
            current_cost_value = cost(current_solution, problem) # Recalculate cost

            # Update global best only if escape found a better one
            if found_global:
                 print(f"Escape found new global best: {best_cost_updated:.2f}")
                 best_solution = [v.copy() for v in best_solution_updated]
                 best_cost_value = best_cost_updated
                 best_iteration = iteration # Mark iteration where escape found best

            iters_since_improvement = 0 # Reset counter after escape
            

        # --- Operator Weight Update ---
        if iteration > 0 and iteration % 100 == 0: # Update weights every 100 iterations
            update_operator_weights(operators)
            iteration_points.append(iteration) # Record iteration for plotting weights
            for op in operators:
                if op["name"] in operator_weight_history:
                     operator_weight_history[op["name"]].append(op["weight"])

        # --- Operator Selection ---
        total_weight = sum(op["weight"] for op in operators)
        if total_weight <= 0: # Reset weights if they all decay to zero or become negative
            print("Warning: Operator weights summed to zero or less. Resetting weights.")
            for op in operators: op['weight'] = 1.0
            total_weight = sum(op["weight"] for op in operators)

        if total_weight > 0:
            probs = [op["weight"] / total_weight for op in operators]
            # Normalize probabilities rigorously to handle potential floating point inaccuracies
            probs_sum = sum(probs)
            if abs(probs_sum - 1.0) > 1e-9:
                 probs = [p / probs_sum for p in probs]
        else: # Fallback if still no weight (e.g., no operators)
            num_ops = len(operators)
            probs = [1.0 / num_ops] * num_ops if num_ops > 0 else []

        if not probs: # Skip iteration if no operators/probabilities
             print("Warning: No operators available or probabilities invalid. Skipping iteration.")
             iteration += 1 # Ensure loop progresses
             continue

        # Select operator based on weights
        selected_op_idx = np.random.choice(len(operators), p=probs)
        selected_op = operators[selected_op_idx]
        op_name = selected_op["name"]
        op_func = selected_op["func"]

        # Update operator stats
        operator_selection_counts[op_name] += 1
        operators[selected_op_idx]["attempts"] += 1
        total_evaluations += 1
        solution_changed = False
        new_solution = None

        # --- Apply Selected Operator ---
        try:
            # Work on a copy of the current solution
            temp_solution = [v.copy() for v in current_solution]

            # Determine k/q value dynamically (number of calls to remove/modify)
            min_q = 2
            max_q = max(min_q, round(0.3 * problem['n_calls'])) # Example: up to 20% of calls
            # Ensure q is valid even for small problems
            q = random.randint(min_q, max_q) if problem['n_calls'] >= min_q else max(1, problem['n_calls'])

            if q <= 0: # Skip if q is invalid
                new_solution = current_solution
                solution_changed = False
            else:
                # Pass necessary arguments based on operator name convention or signature inspection
                import inspect
                sig = inspect.signature(op_func)
                params = sig.parameters

                if 'relatedness_matrix' in params:
                    new_solution = op_func(temp_solution, problem, q, relatedness_matrix)
                elif 'call_blame' in params:
                    new_solution = op_func(temp_solution, problem, q, call_blame)
                elif 'k' in params or 'q' in params: # Check if operator expects k/q
                    new_solution = op_func(temp_solution, problem, q)
                else:
                    # Handle operators that don't take k/q if necessary
                    print(f"Warning: Operator {op_name} does not accept parameter 'k' or 'q'. Calling without it.")
                    new_solution = op_func(temp_solution, problem) # Or adjust as needed

                solution_changed = True # Assume operator modifies the solution copy

        except Exception as e:
            print(f"Error during operator execution ({op_name}): {e}")
            # import traceback # Optional for detailed debugging
            # traceback.print_exc()
            new_solution = current_solution # Revert to current if operator fails
            solution_changed = False

        # --- Evaluate and Accept/Reject New Solution ---
        if solution_changed and new_solution is not None:
            is_feasible_lns, details = feasibility(new_solution, problem)
            # Get calls involved in the *new* solution for blame tracking
            calls_in_new_solution = set(call for veh in new_solution for call in veh if call != 0)

            if is_feasible_lns:
                feasibles += 1
                new_cost_value = cost(new_solution, problem)
                cost_before_lns = current_cost_value # Cost before applying the operator
                E = new_cost_value - cost_before_lns # Delta E (change in cost)
                history["delta_value"].append(E) # Record the change

                accepted_move = False
                if E < 0: # Improvement found (negative delta)
                    operators[selected_op_idx]["improvements"] += 1
                    accepted_move = True
                elif E >= 0: # Worse or equal solution, check Simulated Annealing criterion
                    acceptance_prob = np.exp(-E / T) if T > 0 else 0 # Avoid division by zero
                    if random.random() < acceptance_prob:
                        accepted_move = True

                if accepted_move:
                    accepted_solutions += 1
                    current_solution = new_solution # Keep the modified solution
                    current_cost_value = new_cost_value

                    # Check if it's a new global best
                    if current_cost_value < best_cost_value:
                        best_solution = [v.copy() for v in current_solution]
                        best_cost_value = current_cost_value
                        best_iteration = iteration
                        iters_since_improvement = 0 # Reset counter
                        operators[selected_op_idx]["best_improvements"] += 1 # Track new bests found
                        print(f"Iter {iteration}: NEW BEST solution: {best_cost_value:.2f} by {op_name}")
                    else:
                        iters_since_improvement += 1 # Accepted non-improving move
                else: # Move rejected
                     iters_since_improvement += 1
                     # Increment blame for calls *involved* in the rejected feasible solution
                     for call_id in calls_in_new_solution:
                         call_blame[call_id] = call_blame.get(call_id, 0) + 1
            else: # New solution was infeasible
                iters_since_improvement += 1
                history["delta_value"].append(float('inf')) # Record infeasible attempt
                # Increment blame for calls involved in the infeasible solution
                for call_id in calls_in_new_solution:
                    call_blame[call_id] = call_blame.get(call_id, 0) + 1
        else: # Operator failed or didn't change solution
            iters_since_improvement += 1
            history["delta_value"].append(0) # No change in cost


        # --- Update Temperature ---
        T = T * alpha
        T = max(T, T_final) # Ensure temperature doesn't go below final

        # --- Record History ---
        history["iteration"].append(iteration)
        history["best_cost"].append(best_cost_value)
        history["current_cost"].append(current_cost_value)
        history["acceptance_rate"].append(accepted_solutions / total_evaluations if total_evaluations > 0 else 0)
        history["operator_used"].append(selected_op["name"])

        iteration += 1 # Increment iteration counter

    # --- End of loop ---

    # --- Final Check and Output ---
    is_final_best_feas, final_best_details = feasibility(best_solution, problem)
    if not is_final_best_feas:
        print(f"ERROR: Final best_solution being returned is INFEASIBLE! Details: {final_best_details}")
        # Consider returning initial solution or raising an error if this happens
    else:
        print(f"Final best solution cost: {best_cost_value:.2f} found at iteration {best_iteration}")
        print(f"Algorithm ran for {actual_iterations_run} iterations in {elapsed_time:.2f} seconds.")

    # --- Return Results ---
    return (
        best_solution,
        best_cost_value,
        best_iteration,
        history,
        operator_weight_history,
        iteration_points,
        actual_iterations_run # Return the actual number of iterations performed
    )




neighbourhood = [
                related_greedy,
                random_greedy,
                worst_greedy,
                random_regret,
                segment_regret,
                historical_greedy,
                historical_regret,
                segment_greedy,
                worst_regret,
                ]



def process_file_logic(file_path, stop_criterion_config, max_iter_config, max_time_config, num_runs_config):
    """
    Processes a single data file: loads data, runs the adaptive algorithm multiple times,
    and gathers statistics. This function is designed to be run in a separate process.
    """
    best_solutions_for_file_runs = []
    best_costs_for_file_runs = []
    run_times_for_file_runs = []
    actual_iterations_all_runs = []

    try:
        problem = load(file_path)
        initial_solution_raw = initial_solution_generator(problem)
        initial_cost_val = cost(initial_solution_raw, problem)
        relatedness_matrix_val = calculate_relatedness(problem)
    except Exception as e:
        print(f"[Worker Error] Failed to load or initialize {file_path}: {e}")
        # Return structure indicating failure for this file
        return file_path, None, None, None, None, None, None, [], [], []

    print(f"\n--- [Worker] Starting processing for file: {file_path} (Initial Cost: {initial_cost_val:.2f}) ---")

    for i in range(num_runs_config):
        print(f"--- [Worker] Run {i+1}/{num_runs_config} for {file_path} ---")
        start_time_run = time.time()
        current_initial_solution_for_run = [v.copy() for v in initial_solution_raw]

        max_iter_for_run = max_iter_config if stop_criterion_config == 'iterations' else None
        max_time_for_run = max_time_config if stop_criterion_config == 'time' else None

        best_solution_run, best_cost_run, best_iter_run, _, _, _, actual_iters_run = adaptive_algorithm(
            current_initial_solution_for_run, problem, neighbourhood, relatedness_matrix_val,
            max_iter=max_iter_for_run, max_time=max_time_for_run
        )
        end_time_run = time.time()
        run_time_val = end_time_run - start_time_run
        
        run_times_for_file_runs.append(run_time_val)
        actual_iterations_all_runs.append(actual_iters_run)
        best_solutions_for_file_runs.append(best_solution_run)
        best_costs_for_file_runs.append(best_cost_run)

        print(f'[Worker] Run {i+1} Best iteration: {best_iter_run} (out of {actual_iters_run} actual) for {file_path}')
        print(f"[Worker] Run {i+1} Final cost: {best_cost_run:.2f}, Time: {run_time_val:.2f}s for {file_path}")
        feas_run, details_run = feasibility_check(best_solution_run, problem)
        print(f"[Worker] Run {i+1} Feasibility: {feas_run}" + (f" Details: {details_run}" if not feas_run else "") + f" for {file_path}")

    overall_best_cost_file = None
    overall_best_solution_file_obj = None
    avg_cost_file = None
    avg_time_file = None
    improvement_file = None

    if best_costs_for_file_runs:
        overall_best_cost_file = min(best_costs_for_file_runs)
        best_run_idx = best_costs_for_file_runs.index(overall_best_cost_file)
        overall_best_solution_file_obj = best_solutions_for_file_runs[best_run_idx]
        avg_cost_file = np.mean(best_costs_for_file_runs)

        if initial_cost_val is not None:
            if initial_cost_val > 0:
                improvement_file = ((initial_cost_val - overall_best_cost_file) / initial_cost_val) * 100
            elif overall_best_cost_file <= initial_cost_val: # Handles initial_cost_val == 0 or negative
                improvement_file = 0.0
    
    if run_times_for_file_runs:
        avg_time_file = np.mean(run_times_for_file_runs)

    return (
        file_path,
        overall_best_cost_file,
        overall_best_solution_file_obj,
        avg_cost_file,
        avg_time_file,
        improvement_file,
        initial_cost_val,
        best_costs_for_file_runs, # Pass raw list for stats in main
        run_times_for_file_runs,  # Pass raw list for stats in main
        actual_iterations_all_runs # Pass raw list for stats in main
    )


def run():

    # --- Configuration ---
    stop_criterion = 'time'  # Options: 'iterations' or 'time'
    max_iterations_config = 10000
    max_runtime_seconds_config = 100
    num_runs_per_file = 1 # Number of times to run the algorithm for each file
    # --- End Configuration ---
    
    # --- Configuration ---
    stop_criterion = 'time'  # Options: 'iterations' or 'time'
    max_iterations_config = 10000
    max_runtime_seconds_config = 600
    num_runs_per_file = 1 # Number of times to run the algorithm for each file
    # --- End Configuration ---

    # Timestamp for the entire batch of parallel processing
    overall_batch_timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    
    print(f"--- Starting parallel processing for {len(filenames)} files ---")
    print(f"Stop criterion: {stop_criterion}, Num runs per file: {num_runs_per_file}")
    if stop_criterion == 'iterations':
        print(f"Max Iterations per run: {max_iterations_config}")
    else:
        print(f"Max Runtime per run: {max_runtime_seconds_config}s")

    with concurrent.futures.ProcessPoolExecutor() as executor:
        # Submit all file processing tasks
        future_to_file = {
            executor.submit(process_file_logic, os.path.abspath(fname), stop_criterion, max_iterations_config, max_runtime_seconds_config, num_runs_per_file): fname
            for fname in filenames
        }

        for future in concurrent.futures.as_completed(future_to_file):
            original_fname = future_to_file[future]
            try:
                result_tuple = future.result()
                (
                    processed_fname,
                    overall_best_cost,
                    overall_best_solution_obj, # This is the [vehicles] structure
                    avg_cost,
                    avg_time,
                    improvement,
                    initial_cost_val,
                    best_costs_list,
                    run_times_list,
                    actual_iterations_list_file
                ) = result_tuple

                print(f"\n--- Processing completed for file: {processed_fname} ---")
                
                if initial_cost_val is None: # Indicates loading failure in worker
                    print(f"Failed to load or initialize {processed_fname}. Skipping statistics and CSV saving.")
                    continue

                avg_actual_iter_val = np.mean(actual_iterations_list_file) if actual_iterations_list_file else 0
                run_limit_desc_print = f"{max_iterations_config} iter/run" if stop_criterion == 'iterations' else f"{max_runtime_seconds_config}s/run"
                print(f'--- Statistics for {processed_fname} ({num_runs_per_file} runs, {run_limit_desc_print}, Avg Actual Iterations: {avg_actual_iter_val:.0f}) ---')
                print(f'Initial Cost: {initial_cost_val:.2f}')

                if best_costs_list:
                    print(f'Mean solution cost: {avg_cost:.2f}')
                    print(f'Best solution cost: {overall_best_cost:.2f}')
                    print(f'Improvement vs Initial: {improvement:.2f}%' if improvement is not None else "N/A")
                    print(f'Max solution cost: {max(best_costs_list):.2f}')
                    print(f'Std Dev cost: {np.std(best_costs_list):.2f}')
                else:
                    print("No successful runs to calculate cost statistics for this file.")

                if run_times_list:
                    print(f'Mean runtime: {avg_time:.2f}s')
                    print(f'Min runtime: {min(run_times_list):.2f}s')
                    print(f'Max runtime: {max(run_times_list):.2f}s')
                    print(f'Std Dev runtime: {np.std(run_times_list):.2f}s')
                else:
                    print("No successful runs to calculate runtime statistics for this file.")


            except Exception as exc:
                print(f'\n!!! An error occurred while processing result for {original_fname}: {exc}')
                import traceback
                traceback.print_exc()
            
            print(f"--- Finished post-processing for file: {original_fname} ---")

    print("\n--- All file processing tasks submitted and results handled. ---")

if __name__ == "__main__":
    run()