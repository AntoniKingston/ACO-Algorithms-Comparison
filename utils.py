import random
import re, math
from logging import exception

from collections import namedtuple
import time

import numpy as np
import copy

OptimizationInfo = namedtuple("OptimizationInfo", ["best_solutions_history", "elapsed_time", "pheromone_history"])


class CVRP:
    def __init__(self, problem_name):
        self.name = None
        self.n_vertices = None
        self.coords = []
        self.demands = []
        self.edge_weight_type = None
        self.n_trucks = None
        self.capacity = None
        self.depots = []
        self._parse_file(problem_name + ".vrp")
        self._parse_sol_file(problem_name + ".sol")
        self.demands = np.array(self.demands)
        self.heuristic_solution, self.heuristic_cost = self.optimize_brute_force()


    def _parse_file(self, filepath):
        with open(filepath, 'r') as file:
            lines = file.readlines()

        current_section = None
        metric = None
        dimension = None

        for line in lines:
            # Strip whitespace and ignore empty lines
            line = line.strip()
            if not line:
                continue

            # Stop parsing if EOF is reached
            if line.startswith("EOF"):
                break

            # Detect section headers
            if line.startswith("NODE_COORD_SECTION"):
                current_section = "NODE_COORD"
                continue
            elif line.startswith("DEMAND_SECTION"):
                current_section = "DEMAND"
                continue
            elif line.startswith("DEPOT_SECTION"):
                current_section = "DEPOT"
                continue

            # --- Parse Metadata (Header Information) ---
            if current_section is None:
                if ":" in line:
                    key, val = line.split(":", 1)
                    key = key.strip()
                    val = val.strip()

                    if key == "NAME":
                        self.name = val

                        # Fallback for n_trucks: Try to get it from the name (e.g. A-n33-k5 -> 5)
                        match = re.search(r'-k(\d+)$', self.name)
                        if match and self.n_trucks is None:
                            self.n_trucks = int(match.group(1))

                    elif key == "COMMENT":
                        # Primary for n_trucks: Extract from comment (e.g., No of trucks: 5)
                        match = re.search(r'No of trucks:\s*(\d+)', val, re.IGNORECASE)
                        if match:
                            self.n_trucks = int(match.group(1))

                    elif key == "CAPACITY":
                        self.capacity = int(val)
                    elif key == "DIMENSION":
                        dimension = int(val)
                        self.n = dimension
                    elif key == "EDGE_WEIGHT_TYPE":
                        metric = val

            # --- Parse Data Sections ---
            elif current_section == "NODE_COORD":
                parts = line.split()
                # Coords are often floats in general VRPs
                x, y = float(parts[1]), float(parts[2])
                self.coords.append((x, y))

            elif current_section == "DEMAND":
                parts = line.split()
                demand = int(parts[1])
                self.demands.append(demand)

            elif current_section == "DEPOT":
                parts = line.split()
                for p in parts:
                    depot_id = int(p)
                    if depot_id == -1:
                        # -1 signifies the end of the depot section
                        current_section = None
                        break
                    self.depots.append(depot_id-1)

        # Set the edge_weight_type tuple once metric and dimension are parsed
        if metric is not None:
            self.edge_weight_type = metric

        if self.coords and self.edge_weight_type:
            self._calculate_distance_matrix()

    def _parse_sol_file(self, filepath):
        optimal_solution = [self.depots[0]]
        with open(filepath, 'r') as file:
            lines = file.readlines()

        for line in lines:
            # Strip whitespace and ignore empty lines
            line = line.strip()
            if not line:
                continue

            # Stop parsing if EOF is reached
            if line.startswith("EOF"):
                break

            if line.startswith("Route"):
                segments = line.split(" ")
                for segment in segments[2:]:
                    optimal_solution.append(int(segment))
                optimal_solution.append(self.depots[0])
            else:
                _, cost = line.split(" ")
                self.minimal_cost = float(cost)
                self.optimal_solution = optimal_solution
                return

    def _calculate_distance_matrix(self):
        metric = self.edge_weight_type
        self.distance_matrix = np.zeros((len(self.coords), len(self.coords)))
        for i in range(len(self.coords)):
            for j in range(len(self.coords)):
                self.distance_matrix[i][j] = self._distance(self.coords[i], self.coords[j], metric)

    def __str__(self):
        """Returns a formatted summary of the CVRP instance."""
        total_demand = sum(self.demands)
        num_nodes = len(self.coords)

        return (
            f"--- CVRP Instance Summary ---\n"
            f"Name             : {self.name}\n"
            f"Dimension        : {self.n}\n"
            f"Trucks Available : {self.n_trucks}\n"
            f"Truck Capacity   : {self.capacity}\n"
            f"Edge Weight Type : {self.edge_weight_type}\n"
            f"Depot Node(s)    : {self.depots}\n"
            f"Total Nodes      : {num_nodes}\n"
            f"Total Demand     : {total_demand}\n"
            f"Distance Matrix  : {self.distance_matrix}\n"
            f"-----------------------------"
        )

    def optimize(self, hyperparameters, eval_info = False, save_pheromone = False):
        optimizer, alpha, beta, rho, n_ants, v, rho_loc, max_iterations, eval_info_interval = hyperparameters
        q = self.heuristic_cost
        maxx = 1/(rho*self.heuristic_cost)
        minn = maxx / 30
        pheromone_matrix = np.array([[maxx for _ in range(self.n)] for _ in range(self.n)])
        best_solutions_history = []
        pheromone_history = []
        best_cost = float('inf')
        local_update = CVRP._opt2locupd(optimizer)
        start_time = time.perf_counter()
        for iteration in range(1, max_iterations+1):
            solutions = []
            solution_costs = []
            for ant in range(n_ants):
                trucks_used = 0
                current_node = self.depots[0]
                unvisited_nodes = np.array([i for i in range(1, self.n)])
                available_nodes = copy.deepcopy(unvisited_nodes)
                capacity_left = self.capacity
                solution = [self.depots[0]]
                solution_cost = 0
                while (True):
                    previous_node = current_node
                    # Fallback to depot if no available nodes

                    current_node = self._sample_transition_node(current_node, available_nodes, self.distance_matrix,
                                                                pheromone_matrix, optimizer, (alpha, beta, v))
                    # Update solution
                    solution_cost += self.distance_matrix[previous_node][current_node]
                    if local_update:
                        pheromone_matrix[previous_node][current_node]*=rho_loc
                    solution.append(current_node)
                    # Update truck's load
                    capacity_left -= self.demands[current_node]
                    # Update available nodes
                    unvisited_nodes = unvisited_nodes[unvisited_nodes != current_node]
                    available_nodes = [self.depots[0]] + unvisited_nodes[self.demands[unvisited_nodes] <= capacity_left]
                    # End solution if no trucks left
                    if current_node == self.depots[0]:
                        trucks_used += 1
                        available_nodes = unvisited_nodes
                        capacity_left = self.capacity
                    if trucks_used == self.n_trucks:
                        break

                if len(solution) == self.n + self.n_trucks:
                    solutions.append(solution)
                    solution_costs.append(solution_cost)
                    if solution_cost < best_cost:
                            best_cost = solution_cost
                            best_solutions_history.append((solution_cost, solution, iteration))
            pheromone_matrix = self._update_pheromone_matrix(pheromone_matrix, solutions, solution_costs, optimizer, (rho, q, minn, maxx))
            if save_pheromone:
                pheromone_history.append(pheromone_matrix.copy())
            if eval_info and (iteration % eval_info_interval == 0 or iteration == 1):
                print(f"Iteration {iteration}: Best Cost = {best_cost}")
            elapsed_time = time.perf_counter() - start_time
        return OptimizationInfo(best_solutions_history, elapsed_time, pheromone_history)

    def optimize_brute_force(self):
        trucks_used = 0
        current_node = self.depots[0]
        unvisited_nodes = np.array([i for i in range(1, self.n)])
        available_nodes = copy.deepcopy(unvisited_nodes)
        capacity_left = self.capacity
        solution = [self.depots[0]]
        solution_cost = 0
        while (True):
            previous_node = current_node
            # Fallback to depot if no available nodes
            if len(available_nodes) == 0:
                current_node = self.depots[0]
                trucks_used += 1
                available_nodes = unvisited_nodes
                capacity_left = self.capacity
            else:
                current_node = available_nodes[np.argmin(self.distance_matrix[previous_node][available_nodes])]
            # Update solution
            solution_cost += self.distance_matrix[previous_node][current_node]
            solution.append(current_node)
            # End solution if no trucks left
            if trucks_used == self.n_trucks:
                break
            # Update truck's load
            capacity_left -= self.demands[current_node]
            # Update available nodes
            unvisited_nodes = unvisited_nodes[unvisited_nodes != current_node]
            available_nodes = unvisited_nodes[self.demands[unvisited_nodes] <= capacity_left]
        return (solution, solution_cost)

    def _sample_transition_node(self, current_node, available_nodes, distance_matrix, pheromone_matrix, optimizer, hyperparameters):
        transition_method = CVRP._opt2transmethod(optimizer)
        alpha, beta, v = hyperparameters
        if current_node != self.depots[0]:
            available_nodes = np.append(self.depots, available_nodes)
        # print(available_nodes)
        if transition_method == "PRPR":
            if v < np.random.rand():
                distances = distance_matrix[current_node][available_nodes]
                return available_nodes[np.argmax(1/distances)]
        probabilities = np.array([(pheromone_matrix[current_node][node]**alpha) / (distance_matrix[current_node][node]**beta) for node in available_nodes])
        probabilities += 1e-10 #for numerical stability
        probabilities /= np.sum(probabilities)
        if np.isnan(probabilities).any():
            print(current_node)
            print(available_nodes)
            print(distance_matrix[current_node][available_nodes])
            print(pheromone_matrix[current_node][available_nodes])
            print(probabilities)
            raise Exception(f"Nans in probabilities")

        return np.random.choice(available_nodes, p=probabilities)

    def _update_pheromone_matrix(self, pheromone_matrix, solutions, solution_costs, optimizer, hyperparameters):
        pheromone_bound_method = CVRP._opt2bndmethod(optimizer)
        pheromone_update_method = CVRP._opt2updmethod(optimizer)
        rho, q, min, max = hyperparameters
        pheromone_matrix*=rho
        if pheromone_update_method == "ELITIST" and solution_costs:
            best_idx = np.argmin(solution_costs)
            best_solution = solutions[best_idx]
            best_solution_cost = solution_costs[best_idx]
            for x, y in zip(best_solution[:-1], best_solution[1:]):
                pheromone_matrix[x][y] += q / best_solution_cost
        else:
            for solution, solution_cost in zip(solutions, solution_costs):
                for x, y in zip(solution[:-1], solution[1:]):
                    pheromone_matrix[x][y] += q / solution_cost
        if pheromone_bound_method == "MINMAX":
            pheromone_matrix[pheromone_matrix > max] = max
            pheromone_matrix[pheromone_matrix < min] = max
        return pheromone_matrix

    @staticmethod
    def _opt2transmethod(opt):
        if opt in ["AS", "MINMAX"]:
            return "STANDARD"
        elif opt == "ACS":
            return "PRPR"
        else:
            raise exception("invalid optimization method")
    @staticmethod
    def _opt2bndmethod(opt):
        if opt in ["AS", "ACS"]:
            return "STANDARD"
        return opt
    @staticmethod
    def _opt2updmethod(opt):
        if opt == "AS":
            return "STANDARD"
        else:
            return "ELITIST"
    @staticmethod
    def _opt2locupd(optimizer):
        return optimizer == "ACS"
    @staticmethod
    def _distance(coords1, coords2, metric):
        if metric == "EUC_2D":
            dist = math.sqrt((coords1[0] - coords2[0]) ** 2 + (coords1[1] - coords2[1]) ** 2)
        elif metric == "MAN_2D":
            dist = abs(coords1[0] - coords2[0]) + abs(coords1[1] - coords2[1])
        elif metric == "MAX_2D":
            dist = max(abs(coords1[0] - coords2[0]), abs(coords1[1] - coords2[1]))
        return dist

# def test_method(method, datasets, n_experiments, hyperparameters, random_state):
#     for dataset in datasets:
#         problem = CVRP(dataset)
#         best_solutions_history = method(problem, hyperparameters, random_state)
#         print(f"Dataset: {dataset}")
#         print(f"Best Cost: {best_solutions_history[-1][0]}")
#         print(f"Best Solution: {best_solutions_history[-1][1]}")
#         print(f"Best Solution Cost: {best_solutions_history[-1][0]}")
# ==========================================
# Example usage:
# ==========================================
if __name__ == "__main__":
    # Assuming the text block you provided is saved as 'A-n33-k5.vrp'
    # problem = CVRP('A-n33-k5.vrp')

    # print(f"Name: {problem.name}")
    # print(f"Trucks: {problem.n_trucks}")
    # print(f"Capacity: {problem.capacity}")
    # print(f"Edge Weight Type: {problem.edge_weight_type}")
    # print(f"Depots: {problem.depots}")
    # print(f"Node 3 Coords: {problem.coords[3]}")
    # print(f"Node 3 Demand: {problem.demands[3]}")
    pass