# visualize.py

import io
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import ipywidgets as widgets
from IPython.display import display


def visualize_cvrp_solution(coords, best_solutions_history, depot_index=0,
                            interval_ms=500, minimal_cost=None,
                            optimal_solution=None):
    """
    Interactive visualisation of the CVRP best-solution history.
    Pre-renders all frames as PNG images for instant, flicker-free switching.

    Parameters
    ----------
    coords : list of (x, y)
        Node coordinates from CVRP.coords.
    best_solutions_history : list of (cost, solution, iteration)
        Output of CVRP.optimize().
    depot_index : int
        Index of the depot node (default 0).
    interval_ms : int
        Milliseconds between frames when playing (default 500).
    minimal_cost : float or None
        Known optimal cost (CVRP.minimal_cost). Shown in the subtitle.
    optimal_solution : list of int or None
        Known optimal route (CVRP.optimal_solution). Shown when checkbox is ticked.
    """
    if not best_solutions_history:
        print("No solutions to visualise.")
        return

    n_snapshots = len(best_solutions_history)
    palette = list(mcolors.TABLEAU_COLORS.values())
    # Distinct darker palette for optimal route overlay
    optimal_palette = list(mcolors.CSS4_COLORS.values())

    def _split_routes(solution):
        routes = []
        current = []
        for node in solution:
            current.append(node)
            if node == depot_index and len(current) > 1:
                routes.append(current)
                current = [depot_index]
        if len(current) > 1:
            routes.append(current)
        return routes

    xs = [c[0] for c in coords]
    ys = [c[1] for c in coords]

    # Pre-split optimal routes once (if provided)
    optimal_routes = _split_routes(optimal_solution) if optimal_solution else []

    def _render_frame(idx, cost, solution, iteration, show_optimal):
        fig, ax = plt.subplots(figsize=(9, 7))

        ax.scatter(xs, ys, c="steelblue", s=50, zorder=3, label="Customers")
        ax.scatter(
            [xs[depot_index]], [ys[depot_index]],
            c="red", s=120, marker="s", zorder=4, label="Depot",
        )
        for i, (x, y) in enumerate(coords):
            ax.annotate(str(i), (x, y), textcoords="offset points",
                        xytext=(4, 4), fontsize=7, color="gray")

        # Draw current best solution routes
        routes = _split_routes(solution)
        for r_idx, route in enumerate(routes):
            colour = palette[r_idx % len(palette)]
            route_xs = [coords[n][0] for n in route]
            route_ys = [coords[n][1] for n in route]
            ax.plot(route_xs, route_ys, marker="o", markersize=3,
                    linewidth=1.5, color=colour, label=f"Route {r_idx + 1}")

        # Draw optimal solution overlay
        if show_optimal and optimal_routes:
            for r_idx, route in enumerate(optimal_routes):
                colour = palette[r_idx % len(palette)]
                route_xs = [coords[n][0] for n in route]
                route_ys = [coords[n][1] for n in route]
                ax.plot(route_xs, route_ys, markersize=0,
                        linewidth=3.5, color=colour, alpha=0.35,
                        linestyle="--",
                        label=f"Optimal {r_idx + 1}" if r_idx == 0 else None)
            # Single legend entry for the optimal overlay
            if optimal_routes:
                ax.plot([], [], linewidth=3.5, color="gray", alpha=0.35,
                        linestyle="--", label="Optimal route")

        # Title with minimal cost info
        title_text = (
            f"Snapshot {idx + 1}/{n_snapshots}  —  "
            f"Iteration {iteration}  —  Cost {cost:.2f}"
        )
        if minimal_cost is not None:
            title_text += f"\nKnown optimal cost: {minimal_cost:.2f}  —  Gap: {((cost - minimal_cost) / minimal_cost) * 100:.1f}%"
        ax.set_title(title_text)

        ax.legend(loc="upper right", fontsize=7, ncol=2)
        ax.set_xlabel("X")
        ax.set_ylabel("Y")
        ax.set_aspect("equal", adjustable="datalim")
        plt.tight_layout()

        buf = io.BytesIO()
        fig.savefig(buf, format="png", dpi=100)
        plt.close(fig)
        buf.seek(0)
        return buf.read()

    # --- Pre-render frames (both variants) -------------------------------
    has_optimal = bool(optimal_routes)
    total = n_snapshots * (2 if has_optimal else 1)
    print(f"Pre-rendering {total} frames...")

    frames_normal = []
    frames_with_optimal = []

    for idx, (cost, solution, iteration) in enumerate(best_solutions_history):
        frames_normal.append(_render_frame(idx, cost, solution, iteration, False))
        if has_optimal:
            frames_with_optimal.append(_render_frame(idx, cost, solution, iteration, True))

    print("Done.")

    # --- Widgets ---------------------------------------------------------
    image_widget = widgets.Image(value=frames_normal[0], format="png")

    slider = widgets.IntSlider(
        value=0, min=0, max=n_snapshots - 1, step=1,
        description="Snapshot:",
        continuous_update=True,
        layout=widgets.Layout(width="60%"),
    )

    play = widgets.Play(
        value=0, min=0, max=n_snapshots - 1, step=1,
        interval=interval_ms,
        description="Play / Stop",
        repeat=False,
    )

    widgets.jslink((play, "value"), (slider, "value"))

    show_optimal_checkbox = widgets.Checkbox(
        value=False,
        description="Show optimal route",
        indent=False,
    )

    def _refresh(_=None):
        idx = slider.value
        if show_optimal_checkbox.value and has_optimal:
            image_widget.value = frames_with_optimal[idx]
        else:
            image_widget.value = frames_normal[idx]

    slider.observe(lambda change: _refresh(), names="value")
    show_optimal_checkbox.observe(lambda change: _refresh(), names="value")

    _refresh()

    controls = [play, slider]
    if has_optimal:
        controls.append(show_optimal_checkbox)

    display(widgets.VBox([
        widgets.HBox(controls),
        image_widget,
    ]))

def visualize_pheromone_history(pheromone_history, interval_ms=100):
    """
    Interactive heatmap visualisation of the pheromone matrix over iterations.
    Renders each frame on demand (no pre-rendering).

    Parameters
    ----------
    pheromone_history : list of np.ndarray
        List of pheromone matrices, one per iteration (from OptimizationInfo.pheromone_history).
    interval_ms : int
        Milliseconds between frames when playing (default 100).
    """
    if not pheromone_history:
        print("No pheromone history to visualise.")
        return

    import numpy as np

    n_iterations = len(pheromone_history)



    out = widgets.Output()

    def _draw(idx):
        matrix = pheromone_history[idx]
        global_min = matrix.min()
        global_max = matrix.max()
        with out:
            out.clear_output(wait=True)
            fig, ax = plt.subplots(figsize=(7, 6))
            im = ax.imshow(matrix, cmap="hot", aspect="equal",
                           vmin=global_min, vmax=global_max, origin="upper")
            fig.colorbar(im, ax=ax, label="Pheromone level")
            ax.set_title(f"Pheromone matrix  —  Iteration {idx + 1}/{n_iterations}")
            ax.set_xlabel("To node")
            ax.set_ylabel("From node")
            plt.tight_layout()
            plt.show()

    # --- Widgets ---------------------------------------------------------
    slider = widgets.IntSlider(
        value=0, min=0, max=n_iterations - 1, step=1,
        description="Iteration:",
        continuous_update=False,
        layout=widgets.Layout(width="60%"),
    )

    play = widgets.Play(
        value=0, min=0, max=n_iterations - 1, step=1,
        interval=interval_ms,
        description="Play / Stop",
        repeat=False,
    )

    widgets.jslink((play, "value"), (slider, "value"))
    slider.observe(lambda change: _draw(change["new"]), names="value")

    _draw(0)

    display(widgets.VBox([
        widgets.HBox([play, slider]),
        out,
    ]))
# # visualize.py
#
# import io
# import matplotlib.pyplot as plt
# import matplotlib.colors as mcolors
# import ipywidgets as widgets
# from IPython.display import display
#
#
# def visualize_cvrp_solution(coords, best_solutions_history, depot_index=0, interval_ms=500):
#     """
#     Interactive visualisation of the CVRP best-solution history.
#     Pre-renders all frames as PNG images for instant, flicker-free switching.
#
#     Parameters
#     ----------
#     coords : list of (x, y)
#         Node coordinates from CVRP.coords.
#     best_solutions_history : list of (cost, solution, iteration)
#         Output of CVRP.optimize().
#     depot_index : int
#         Index of the depot node (default 0).
#     interval_ms : int
#         Milliseconds between frames when playing (default 500).
#     """
#     if not best_solutions_history:
#         print("No solutions to visualise.")
#         return
#
#     n_snapshots = len(best_solutions_history)
#     palette = list(mcolors.TABLEAU_COLORS.values())
#
#     def _split_routes(solution):
#         routes = []
#         current = []
#         for node in solution:
#             current.append(node)
#             if node == depot_index and len(current) > 1:
#                 routes.append(current)
#                 current = [depot_index]
#         if len(current) > 1:
#             routes.append(current)
#         return routes
#
#     # --- Pre-render every snapshot to a PNG byte buffer ------------------
#     print(f"Pre-rendering {n_snapshots} frames...")
#     frames = []
#     xs = [c[0] for c in coords]
#     ys = [c[1] for c in coords]
#
#     for idx, (cost, solution, iteration) in enumerate(best_solutions_history):
#         fig, ax = plt.subplots(figsize=(9, 7))
#
#         ax.scatter(xs, ys, c="steelblue", s=50, zorder=3, label="Customers")
#         ax.scatter(
#             [xs[depot_index]], [ys[depot_index]],
#             c="red", s=120, marker="s", zorder=4, label="Depot",
#         )
#         for i, (x, y) in enumerate(coords):
#             ax.annotate(str(i), (x, y), textcoords="offset points",
#                         xytext=(4, 4), fontsize=7, color="gray")
#
#         routes = _split_routes(solution)
#         for r_idx, route in enumerate(routes):
#             colour = palette[r_idx % len(palette)]
#             route_xs = [coords[n][0] for n in route]
#             route_ys = [coords[n][1] for n in route]
#             ax.plot(route_xs, route_ys, marker="o", markersize=3,
#                     linewidth=1.5, color=colour, label=f"Route {r_idx + 1}")
#
#         ax.set_title(
#             f"Snapshot {idx + 1}/{n_snapshots}  —  "
#             f"Iteration {iteration}  —  Cost {cost:.2f}"
#         )
#         ax.legend(loc="upper right", fontsize=7, ncol=2)
#         ax.set_xlabel("X")
#         ax.set_ylabel("Y")
#         ax.set_aspect("equal", adjustable="datalim")
#         plt.tight_layout()
#
#         buf = io.BytesIO()
#         fig.savefig(buf, format="png", dpi=100)
#         plt.close(fig)
#         buf.seek(0)
#         frames.append(buf.read())
#
#     print("Done.")
#
#     # --- Widgets ---------------------------------------------------------
#     image_widget = widgets.Image(value=frames[0], format="png")
#
#     slider = widgets.IntSlider(
#         value=0, min=0, max=n_snapshots - 1, step=1,
#         description="Snapshot:",
#         continuous_update=True,
#         layout=widgets.Layout(width="60%"),
#     )
#
#     play = widgets.Play(
#         value=0, min=0, max=n_snapshots - 1, step=1,
#         interval=interval_ms,
#         description="Play / Stop",
#         repeat=False,
#     )
#
#     widgets.jslink((play, "value"), (slider, "value"))
#
#     def _on_slider_change(change):
#         image_widget.value = frames[change["new"]]
#
#     slider.observe(_on_slider_change, names="value")
#
#     display(widgets.VBox([
#         widgets.HBox([play, slider]),
#         image_widget,
#     ]))