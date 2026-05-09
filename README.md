# ACO Algorithms Comparison for CVRP

This repository contains a Python implementation and comparative analysis of various **Ant Colony Optimization (ACO)** metaheuristics applied to the **Capacitated Vehicle Routing Problem (CVRP)**. 

The project evaluates the performance, stability, and convergence of three main variants:
1.  **Ant System (AS)** - The classic ACO approach.
2.  **Ant Colony System (ACS)** - Introduces pseudo-random proportional rules and local pheromone updates.
3.  **Max-Min Ant System (MMAS)** - Constrains pheromone values to a specific range to prevent premature convergence.

## 📋 Project Overview
The Capacitated Vehicle Routing Problem (CVRP) is an NP-hard optimization problem. The goal is to determine a set of optimal routes for a fleet of vehicles to deliver goods to a specific set of customers, starting and ending at a central depot, while respecting vehicle capacity constraints.

This project includes:
*   A custom CVRP engine capable of parsing `.vrp` and `.sol` files from the [CVRPLIB](http://vrp.atd-lab.inf.puc-rio.br/index.php/en/) repository.
*   Implementation of AS, ACS, and MMAS optimizers.
*   Automated hyperparameter grid search and parallelized testing.
*   Interactive visualizations for route construction and pheromone density.

## 📂 Repository Structure

### Core Logic
*   `utils.py`: Contains the `CVRP` class, responsible for file parsing, distance matrix calculation, and the main optimization loop.
*   `visualize.py`: Provides interactive visualization tools using `matplotlib` and `ipywidgets`.
*   `testing.py`: Framework for running multiple experiments in parallel, calculating statistics (Gap, Convergence Speed, Stagnation), and managing result persistence.

### Notebooks
*   `prod.ipynb`: The primary interactive entry point. Use this to run a specific problem instance, watch the ants build routes, and visualize pheromone heatmaps.
*   `plotting.ipynb`: Processes the `results.json` file to generate the comparative bar plots used in the final report.
*   `bf_sol_extract.ipynb`: Used for baseline comparisons, including greedy algorithm benchmarks.

### Documentation & Data
*   `requirements.txt`: List of Python dependencies.
*   `datasets/`: (Expected folder) Should contain `.vrp` and `.sol` files from sets A, B, and X.
*   `Final_Report.pdf`: Detailed documentation of the methodology and final findings.

## 🚀 Getting Started

### Installation
1. Clone the repository:
   ```bash
   git clone https://github.com/AntoniKingston/ACO-Algorithms-Comparison.git
   cd ACO-Algorithms-Comparison
   ```
2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

### Running the Optimizer
Open `prod.ipynb` in a Jupyter environment. You can load a problem instance and run the optimizer as follows:

```python
from utils import CVRP

# Load instance
cvrp = CVRP("datasets/A/A-n32-k5")

# Run ACS Optimizer
# Parameters: (method, alpha, beta, rho, n_ants, greediness, rho_loc, max_iter, interval)
params = ("ACS", 1.0, 3.0, 0.98, 25, 0.9, 0.99, 500, 50)
history = cvrp.optimize(params, eval_info=True)
```

## 📊 Visualization
One of the key features of this project is the interactive visualization found in `visualize.py`.
*   **Route Construction**: View how the best-found solution evolves over iterations. It includes an option to overlay the known optimal solution (BKS).
*   **Pheromone Heatmap**: An interactive heatmap showing which edges are being "reinforced" by the ant colony over time.

## 📈 Results Summary
Based on the benchmarks conducted on the A, B, and X datasets:
*   **ACS** showed the highest precision and reached solutions closest to the optimal (lowest Gap).
*   **AS** and **MMAS** performed well on smaller, random distributions (Set A) but struggled with clustered customers (Set B).
*   **Scalability**: For very large instances (Set X, $n > 150$), all variants require significant hyperparameter tuning to consistently find feasible solutions that meet the vehicle limit $k$.

## 🛠 Built With
*   **Python 3.14**
*   **NumPy**: Numerical calculations and matrix operations.
*   **Matplotlib**: Static and dynamic plotting.
*   **ipywidgets**: Interactive dashboard elements.
*   **Concurrent.futures**: Parallel processing for massive hyperparameter testing.

## ✍️ Author
*   **Antoni Kingston** - *Initial work & Research* - [AntoniKingston](https://github.com/AntoniKingston)
