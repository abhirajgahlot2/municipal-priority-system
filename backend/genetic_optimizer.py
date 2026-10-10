"""
genetic_optimizer.py – Member 3: Work-Order Optimizer
======================================================

Exposes a single public function:

    optimize_work_order(complaints, priority_scores, constraints=None) -> dict

The function uses a Genetic Algorithm (GA) to produce an optimised ordering
of municipal complaints that minimises a weighted urgency-cost objective.

Objectives that are DISABLED due to missing data
-------------------------------------------------
* travel_distance  – No latitude/longitude or distance matrix is available.
                     total_travel_distance is always returned as None.
* repair_time      – No per-complaint estimated repair times are in the
                     processed dataset.  total_estimated_repair_time is always
                     returned as None.

Both limitations are reported in the "notes" field of the return value.

Fitness function
----------------
Lower fitness = better schedule.

For each position i (0-indexed) in a schedule of n active complaints the
position weight is:

    position_weight(i) = (i + 1) / n       # 1/n for 1st slot, 1.0 for last

The urgency of complaint c is:

    urgency(c) = W_PRIORITY * priority_norm(c)
               + W_SAFETY   * safety_risk(c)
               + W_SEV      * severity(c)   / 10
               + W_TRAFFIC  * traffic_impact(c) / 10
               + W_PUBLIC   * public_impact(c)  / 10
               + W_PENDING  * min(days_pending(c) / 30, 1.0)
               + W_WEATHER  * weather_risk(c)   / 10
               + W_FREQ     * complaint_frequency(c) / 10

Positional cost for complaint c at position i:

    cost(c, i) = position_weight(i) * urgency(c)

Total cost (ordering penalty):

    ordering_cost = sum of cost(c, i)  for all active c at their positions i

Safety penalty – deters placing high-risk complaints far back:

    safety_penalty = SAFETY_PENALTY_FACTOR
                     * sum of safety_risk(c)
                       for each c in the LOWER half of the schedule
                       where safety_risk(c) > SAFETY_THRESHOLD

Overall fitness:

    fitness = ordering_cost + safety_penalty

Weights: Chosen to avoid excessive double-counting.
severity/traffic_impact/public_impact already influence safety_risk
(fuzzy engine) and priority_score (ANN), so they receive lower weights.

    W_PRIORITY = 0.35   primary ANN signal
    W_SAFETY   = 0.25   fuzzy engine output
    W_SEV      = 0.10   moderate, partially captured above
    W_TRAFFIC  = 0.08   moderate, partially captured above
    W_PUBLIC   = 0.08   moderate, partially captured above
    W_PENDING  = 0.07   independent age signal
    W_WEATHER  = 0.04   partly encoded in fuzzy persistence_risk
    W_FREQ     = 0.03   partly encoded in fuzzy persistence_risk (total = 1.00)

frequency_count is NOT used separately; it is already encoded in
complaint_frequency (the 0-10 fuzzy score), so using both would
double-count the same signal.
"""

from __future__ import annotations

import math
import random
from typing import Any

# ============================================================
# FITNESS WEIGHTS
# ============================================================

W_PRIORITY = 0.35
W_SAFETY   = 0.25
W_SEV      = 0.10
W_TRAFFIC  = 0.08
W_PUBLIC   = 0.08
W_PENDING  = 0.07
W_WEATHER  = 0.04
W_FREQ     = 0.03

# Safety-penalty parameters
SAFETY_PENALTY_FACTOR = 0.5
SAFETY_THRESHOLD      = 0.7   # safety_risk above this is "safety-critical"

# Status values treated as resolved/inactive (case-insensitive)
CLOSED_STATUSES = {"closed"}

# ============================================================
# DEFAULT GA PARAMETERS
# ============================================================

_DEFAULTS: dict[str, Any] = {
    "population_size": 100,
    "n_generations":   200,
    "mutation_rate":   0.02,
    "crossover_rate":  0.85,
    "elite_size":      2,
    "tournament_size": 5,
    "random_seed":     None,
    "include_closed":  False,
}


# ============================================================
# HELPERS – NORMALISATION
# ============================================================

def _safe_float(value: Any, default: float = 0.0) -> float:
    """Convert a value to float safely; return *default* on failure."""
    try:
        v = float(value)
        if not math.isfinite(v):
            return default
        return v
    except (TypeError, ValueError):
        return default


def _norm10(value: Any, default: float = 0.0) -> float:
    """Normalise a 0-10 field to [0, 1]."""
    return max(0.0, min(1.0, _safe_float(value, default) / 10.0))


def _norm_days(value: Any, cap: float = 30.0) -> float:
    """Normalise days_pending to [0, 1] capped at *cap* days."""
    return max(0.0, min(1.0, _safe_float(value, 0.0) / cap))


def _norm_safety(value: Any) -> float:
    """Clip safety_risk already in [0, 1]."""
    return max(0.0, min(1.0, _safe_float(value, 0.0)))


# ============================================================
# URGENCY
# ============================================================

def _urgency(complaint: dict, priority_norm: float) -> float:
    """
    Compute the urgency score for a single complaint, in [0, 1].

    Parameters
    ----------
    complaint     : complaint dict
    priority_norm : ANN priority score, already validated in [0, 1]
    """
    safety   = _norm_safety(complaint.get("safety_risk", 0.0))
    severity = _norm10(complaint.get("severity", 0.0))
    traffic  = _norm10(complaint.get("traffic_impact", 0.0))
    public   = _norm10(complaint.get("public_impact", 0.0))
    pending  = _norm_days(complaint.get("days_pending", 0.0))
    weather  = _norm10(complaint.get("weather_risk", 0.0))
    freq     = _norm10(complaint.get("complaint_frequency", 0.0))

    return (
        W_PRIORITY * priority_norm
        + W_SAFETY * safety
        + W_SEV    * severity
        + W_TRAFFIC * traffic
        + W_PUBLIC  * public
        + W_PENDING * pending
        + W_WEATHER * weather
        + W_FREQ    * freq
    )


# ============================================================
# FITNESS FUNCTION
# ============================================================

def _compute_fitness(
    order: list[int],
    urgencies: list[float],
    safety_scores: list[float],
) -> float:
    """
    Compute the fitness (cost) for a particular ordering.

    Parameters
    ----------
    order         : list of indices into urgencies/safety_scores
    urgencies     : pre-computed urgency for each active complaint
    safety_scores : normalised safety_risk for each active complaint

    Returns
    -------
    float – lower is better
    """
    n = len(order)
    if n == 0:
        return 0.0

    ordering_cost = 0.0
    for rank, idx in enumerate(order):
        position_weight = (rank + 1) / n
        ordering_cost  += position_weight * urgencies[idx]

    # Safety penalty: safety-critical complaints placed in the lower half
    half = n // 2
    penalty = 0.0
    for rank in range(half, n):          # lower half (rank >= half)
        idx = order[rank]
        if safety_scores[idx] > SAFETY_THRESHOLD:
            penalty += safety_scores[idx]

    return ordering_cost + SAFETY_PENALTY_FACTOR * penalty


# ============================================================
# BASELINE
# ============================================================

def _baseline_order(urgencies: list[float]) -> list[int]:
    """
    Return indices sorted by descending urgency (greedy baseline).
    """
    return sorted(range(len(urgencies)), key=lambda i: urgencies[i], reverse=True)


# ============================================================
# GA OPERATORS
# ============================================================

def _initial_population(
    n_individuals: int,
    n_complaints: int,
    rng: random.Random,
) -> list[list[int]]:
    """Return a list of random permutations of range(n_complaints)."""
    base = list(range(n_complaints))
    population = []
    for _ in range(n_individuals):
        ind = base[:]
        rng.shuffle(ind)
        population.append(ind)
    return population


def _tournament_select(
    population: list[list[int]],
    fitnesses: list[float],
    tournament_size: int,
    rng: random.Random,
) -> list[int]:
    """
    Tournament selection: sample *tournament_size* individuals and return
    the one with the lowest (best) fitness.
    """
    contestants = rng.sample(range(len(population)), k=min(tournament_size, len(population)))
    best = min(contestants, key=lambda i: fitnesses[i])
    return population[best][:]


def _order_crossover(parent1: list[int], parent2: list[int], rng: random.Random) -> list[int]:
    """
    Order Crossover (OX): produces one child that preserves relative order
    from both parents.

    1. Pick two random cut points.
    2. Copy the segment between the cut points from parent1 into the child.
    3. Fill remaining positions in order of appearance in parent2.
    """
    n = len(parent1)
    a, b = sorted(rng.sample(range(n), k=2))

    child = [None] * n
    # Copy segment from parent1
    child[a:b + 1] = parent1[a:b + 1]
    segment_set = set(child[a:b + 1])

    # Fill remaining positions with elements from parent2 in order
    fill_pos = [(b + 1 + i) % n for i in range(n - (b - a + 1))]
    fill_vals = [v for v in (parent2[(b + 1 + i) % n] for i in range(n)) if v not in segment_set]

    for pos, val in zip(fill_pos, fill_vals):
        child[pos] = val

    return child  # type: ignore[return-value]


def _swap_mutation(individual: list[int], mutation_rate: float, rng: random.Random) -> list[int]:
    """
    Swap mutation: each position is a candidate for a random swap with
    probability *mutation_rate*.
    """
    ind = individual[:]
    n   = len(ind)
    if n < 2:
        return ind
    for i in range(n):
        if rng.random() < mutation_rate:
            j = rng.randint(0, n - 1)
            ind[i], ind[j] = ind[j], ind[i]
    return ind


# ============================================================
# VALIDATION
# ============================================================

def _validate_inputs(
    complaints: list[dict],
    priority_scores: list[float],
    params: dict,
) -> None:
    """Raise ValueError for any invalid input."""

    if len(complaints) != len(priority_scores):
        raise ValueError(
            f"Length mismatch: {len(complaints)} complaints but "
            f"{len(priority_scores)} priority_scores."
        )

    # Validate priority_scores
    for i, s in enumerate(priority_scores):
        try:
            fs = float(s)
        except (TypeError, ValueError):
            raise ValueError(f"priority_scores[{i}] = {s!r} is not a number.")
        if not math.isfinite(fs):
            raise ValueError(
                f"priority_scores[{i}] = {s!r} is not finite (NaN or inf)."
            )
        if fs < 0.0 or fs > 1.0:
            raise ValueError(
                f"priority_scores[{i}] = {fs} is outside [0, 1]."
            )

    # Check for duplicate unique_key
    seen_keys: set = set()
    for i, c in enumerate(complaints):
        key = c.get("unique_key")
        if key is None:
            raise ValueError(f"complaints[{i}] missing 'unique_key'.")
        if key in seen_keys:
            raise ValueError(
                f"Duplicate unique_key '{key}' found in complaints."
            )
        seen_keys.add(key)

    # Validate GA parameters
    pop  = params["population_size"]
    ngen = params["n_generations"]
    mr   = params["mutation_rate"]
    cr   = params["crossover_rate"]
    es   = params["elite_size"]
    ts   = params["tournament_size"]

    if not isinstance(pop, int) or pop < 1:
        raise ValueError(f"population_size must be a positive integer, got {pop!r}.")
    if not isinstance(ngen, int) or ngen < 1:
        raise ValueError(f"n_generations must be a positive integer, got {ngen!r}.")
    if not (0.0 <= float(mr) <= 1.0):
        raise ValueError(f"mutation_rate must be in [0, 1], got {mr!r}.")
    if not (0.0 <= float(cr) <= 1.0):
        raise ValueError(f"crossover_rate must be in [0, 1], got {cr!r}.")
    if not isinstance(es, int) or es < 0:
        raise ValueError(f"elite_size must be a non-negative integer, got {es!r}.")
    if not isinstance(ts, int) or ts < 1:
        raise ValueError(f"tournament_size must be a positive integer, got {ts!r}.")


# ============================================================
# MAIN PUBLIC FUNCTION
# ============================================================

def optimize_work_order(
    complaints: list[dict],
    priority_scores: list[float],
    constraints: dict | None = None,
) -> dict:
    """
    Produce an optimised work-order for municipal complaints using a
    Genetic Algorithm.

    Parameters
    ----------
    complaints : list[dict]
        Each dict must contain 'unique_key' and may contain any subset of:
            severity, traffic_impact, public_impact, weather_risk,
            days_pending, complaint_frequency, frequency_count,
            safety_risk, status, complaint_type, descriptor, agency,
            created_date.

    priority_scores : list[float]
        ANN-generated priority score in [0, 1] for each complaint.
        Must have the same length as *complaints*.

    constraints : dict, optional
        Overrides for GA settings and scheduling options:

        GA settings (int/float):
            population_size   (default 100)
            n_generations     (default 200)
            mutation_rate     (default 0.02)
            crossover_rate    (default 0.85)
            elite_size        (default 2)
            tournament_size   (default 5)
            random_seed       (default None)

        Scheduling options:
            include_closed (bool, default False)
                If True, closed/resolved complaints are included in the
                work order. By default they are excluded.

        Disabled constraint keys (ignored, reported in notes):
            distance_matrix, repair_times, worker_count, coordinates

    Returns
    -------
    dict with keys:
        optimized_order            : list[unique_key]  – active complaints only
        total_travel_distance      : None              – disabled (no coordinates)
        total_estimated_repair_time: None              – disabled (no repair times)
        fitness                    : float             – best GA cost (lower = better)
        baseline_fitness           : float             – greedy descending-priority cost
        notes                      : list[str]         – disabled objectives + warnings
        excluded_complaints        : list[unique_key]  – closed/skipped complaints

    Raises
    ------
    ValueError
        On invalid input (mismatched lengths, duplicate IDs, out-of-range
        scores, non-finite scores, invalid GA settings).
    """

    # --------------------------------------------------------
    # 1. MERGE CONSTRAINTS WITH DEFAULTS
    # --------------------------------------------------------

    params: dict[str, Any] = dict(_DEFAULTS)
    if constraints:
        for key, value in constraints.items():
            if key in params:
                params[key] = value

    notes: list[str] = [
        "total_travel_distance is None: no latitude/longitude or distance "
        "matrix is available in the processed dataset.",

        "total_estimated_repair_time is None: no per-complaint repair time "
        "data is available in the processed dataset.",
    ]

    # Report unsupported/disabled constraint keys
    unsupported = {"distance_matrix", "repair_times", "worker_count", "coordinates"}
    if constraints:
        passed_unsupported = unsupported & set(constraints.keys())
        for key in sorted(passed_unsupported):
            notes.append(
                f"Constraint '{key}' was supplied but cannot be applied "
                f"(corresponding data is not available); it has been ignored."
            )

    # --------------------------------------------------------
    # 2. VALIDATE INPUTS
    # --------------------------------------------------------

    _validate_inputs(complaints, priority_scores, params)

    # --------------------------------------------------------
    # 3. EMPTY INPUT – EARLY EXIT
    # --------------------------------------------------------

    if len(complaints) == 0:
        return {
            "optimized_order":             [],
            "total_travel_distance":       None,
            "total_estimated_repair_time": None,
            "fitness":                     0.0,
            "baseline_fitness":            0.0,
            "notes":                       notes + ["No complaints provided; work order is empty."],
            "excluded_complaints":         [],
        }

    # --------------------------------------------------------
    # 4. FILTER ACTIVE COMPLAINTS
    # --------------------------------------------------------

    include_closed: bool = bool(params.get("include_closed", False))

    active_indices:   list[int] = []   # indices into original complaints/scores
    excluded_keys:    list     = []

    for i, c in enumerate(complaints):
        status = str(c.get("status", "")).strip().lower()
        if not include_closed and status in CLOSED_STATUSES:
            excluded_keys.append(c["unique_key"])
        else:
            active_indices.append(i)

    if excluded_keys:
        notes.append(
            f"{len(excluded_keys)} closed complaint(s) excluded from the "
            f"work order. Pass constraints={{'include_closed': True}} to include them."
        )

    # --------------------------------------------------------
    # 5. SINGLE ACTIVE COMPLAINT – TRIVIAL ORDER
    # --------------------------------------------------------

    if len(active_indices) == 0:
        notes.append("All complaints are closed; work order is empty.")
        return {
            "optimized_order":             [],
            "total_travel_distance":       None,
            "total_estimated_repair_time": None,
            "fitness":                     0.0,
            "baseline_fitness":            0.0,
            "notes":                       notes,
            "excluded_complaints":         excluded_keys,
        }

    if len(active_indices) == 1:
        only = complaints[active_indices[0]]["unique_key"]
        u    = _urgency(complaints[active_indices[0]], float(priority_scores[active_indices[0]]))
        trivial_fit = _compute_fitness([0], [u], [_norm_safety(complaints[active_indices[0]].get("safety_risk", 0.0))])
        return {
            "optimized_order":             [only],
            "total_travel_distance":       None,
            "total_estimated_repair_time": None,
            "fitness":                     trivial_fit,
            "baseline_fitness":            trivial_fit,
            "notes":                       notes,
            "excluded_complaints":         excluded_keys,
        }

    # --------------------------------------------------------
    # 6. PRE-COMPUTE URGENCIES AND SAFETY SCORES
    # --------------------------------------------------------

    urgencies:     list[float] = []
    safety_scores: list[float] = []

    for i in active_indices:
        urgencies.append(_urgency(complaints[i], float(priority_scores[i])))
        safety_scores.append(_norm_safety(complaints[i].get("safety_risk", 0.0)))

    # Mapping: local index → original unique_key
    active_keys = [complaints[i]["unique_key"] for i in active_indices]
    n_active    = len(active_indices)

    # --------------------------------------------------------
    # 7. BASELINE ORDER
    # --------------------------------------------------------

    baseline_local = _baseline_order(urgencies)
    baseline_fit   = _compute_fitness(baseline_local, urgencies, safety_scores)

    # --------------------------------------------------------
    # 8. GA PARAMETERS
    # --------------------------------------------------------

    population_size = int(params["population_size"])
    n_generations   = int(params["n_generations"])
    mutation_rate   = float(params["mutation_rate"])
    crossover_rate  = float(params["crossover_rate"])
    elite_size      = int(params["elite_size"])
    tournament_size = int(params["tournament_size"])
    seed            = params["random_seed"]

    rng = random.Random(seed)

    # --------------------------------------------------------
    # 9. INITIALISE POPULATION
    # --------------------------------------------------------

    population  = _initial_population(population_size, n_active, rng)
    fitnesses   = [_compute_fitness(ind, urgencies, safety_scores) for ind in population]

    best_individual = min(population, key=lambda ind: _compute_fitness(ind, urgencies, safety_scores))
    best_fitness    = _compute_fitness(best_individual, urgencies, safety_scores)

    # --------------------------------------------------------
    # 10. EVOLUTION LOOP
    # --------------------------------------------------------

    for _generation in range(n_generations):

        # Sort by fitness (ascending → best first)
        paired = sorted(zip(fitnesses, population), key=lambda x: x[0])
        fitnesses  = [p[0] for p in paired]
        population = [p[1] for p in paired]

        # Update global best
        if fitnesses[0] < best_fitness:
            best_fitness    = fitnesses[0]
            best_individual = population[0][:]

        # Elitism: carry forward the top individuals
        next_generation = [population[i][:] for i in range(min(elite_size, population_size))]

        # Fill the rest of the new generation
        while len(next_generation) < population_size:
            parent1 = _tournament_select(population, fitnesses, tournament_size, rng)
            parent2 = _tournament_select(population, fitnesses, tournament_size, rng)

            if rng.random() < crossover_rate:
                child = _order_crossover(parent1, parent2, rng)
            else:
                child = parent1[:]

            child = _swap_mutation(child, mutation_rate, rng)
            next_generation.append(child)

        population = next_generation
        fitnesses  = [_compute_fitness(ind, urgencies, safety_scores) for ind in population]

    # Final best
    final_best_fitness = min(fitnesses)
    final_best_idx     = fitnesses.index(final_best_fitness)
    final_order_local  = population[final_best_idx]

    if final_best_fitness < best_fitness:
        best_fitness    = final_best_fitness
        best_individual = final_order_local

    # --------------------------------------------------------
    # 11. ASSEMBLE RESULT
    # --------------------------------------------------------

    optimized_order = [active_keys[i] for i in best_individual]

    return {
        "optimized_order":             optimized_order,
        "total_travel_distance":       None,
        "total_estimated_repair_time": None,
        "fitness":                     round(best_fitness, 6),
        "baseline_fitness":            round(baseline_fit, 6),
        "notes":                       notes,
        "excluded_complaints":         excluded_keys,
    }
