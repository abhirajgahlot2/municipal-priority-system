import numpy as np
import skfuzzy as fuzz
from skfuzzy import control as ctrl

# ============================================================
# 1. INPUT VARIABLES
# ============================================================

severity = ctrl.Antecedent(np.arange(0, 10.1, 0.1), "severity")
traffic_impact = ctrl.Antecedent(np.arange(0, 10.1, 0.1), "traffic_impact")
public_impact = ctrl.Antecedent(np.arange(0, 10.1, 0.1), "public_impact")
weather_risk = ctrl.Antecedent(np.arange(0, 10.1, 0.1), "weather_risk")
days_pending = ctrl.Antecedent(np.arange(0, 30.1, 0.1), "days_pending")
complaint_frequency = ctrl.Antecedent(
    np.arange(0, 10.1, 0.1), "complaint_frequency"
)


# ============================================================
# 2. INPUT MEMBERSHIP FUNCTIONS
# ============================================================

def five_level_membership(variable, universe_max):

    x = variable.universe

    variable["very_low"] = fuzz.trapmf(x, [0, 0, 1.5, 3])
    variable["low"] = fuzz.trimf(x, [1.5, 3, 4.5])
    variable["medium"] = fuzz.trimf(x, [3.5, 5, 6.5])
    variable["high"] = fuzz.trimf(x, [5.5, 7, 8.5])
    variable["very_high"] = fuzz.trapmf(
        x, [7.5, 9, universe_max, universe_max]
    )


five_level_membership(severity, 10)
five_level_membership(traffic_impact, 10)
five_level_membership(public_impact, 10)
five_level_membership(weather_risk, 10)


# ============================================================
# 3. COMPLAINT FREQUENCY MEMBERSHIP FUNCTIONS
# ============================================================

complaint_frequency["none"] = fuzz.trapmf(
    complaint_frequency.universe, [0, 0, 0.5, 1.5]
)

complaint_frequency["low"] = fuzz.trimf(
    complaint_frequency.universe, [0.5, 2, 3.5]
)

complaint_frequency["medium"] = fuzz.trimf(
    complaint_frequency.universe, [2.5, 5, 7]
)

complaint_frequency["high"] = fuzz.trimf(
    complaint_frequency.universe, [6, 8, 9.5]
)

complaint_frequency["very_high"] = fuzz.trapmf(
    complaint_frequency.universe, [8.5, 9.5, 10, 10]
)


# ============================================================
# 4. DAYS PENDING MEMBERSHIP FUNCTIONS
# ============================================================

days_pending["recent"] = fuzz.trapmf(
    days_pending.universe, [0, 0, 1, 4]
)

days_pending["moderate"] = fuzz.trimf(
    days_pending.universe, [2, 7, 14]
)

days_pending["long"] = fuzz.trimf(
    days_pending.universe, [10, 17, 24]
)

days_pending["very_long"] = fuzz.trapmf(
    days_pending.universe, [20, 26, 30, 30]
)


# ============================================================
# 5. IMPACT RISK
# ============================================================

impact_risk = ctrl.Consequent(
    np.arange(0, 1.01, 0.01),
    "impact_risk"
)

impact_risk["low"] = fuzz.trapmf(
    impact_risk.universe, [0, 0, 0.15, 0.30]
)

impact_risk["moderate"] = fuzz.trimf(
    impact_risk.universe, [0.20, 0.35, 0.50]
)

impact_risk["high"] = fuzz.trimf(
    impact_risk.universe, [0.40, 0.55, 0.70]
)

impact_risk["very_high"] = fuzz.trimf(
    impact_risk.universe, [0.60, 0.75, 0.88]
)

impact_risk["critical"] = fuzz.trapmf(
    impact_risk.universe, [0.82, 0.92, 1, 1]
)


# ============================================================
# 6. IMPACT RULES
# ============================================================

impact_rules = [

    # Critical
    ctrl.Rule(
        severity["very_high"] &
        public_impact["very_high"],
        impact_risk["critical"]
    ),

    # Very high
    ctrl.Rule(
        severity["very_high"] &
        traffic_impact["very_high"],
        impact_risk["very_high"]
    ),

    ctrl.Rule(
        public_impact["very_high"] &
        traffic_impact["very_high"],
        impact_risk["very_high"]
    ),

    ctrl.Rule(
        severity["high"] &
        public_impact["high"],
        impact_risk["very_high"]
    ),

    # High
    ctrl.Rule(
        severity["high"] &
        traffic_impact["high"],
        impact_risk["high"]
    ),

    ctrl.Rule(
        severity["high"],
        impact_risk["high"]
    ),

    ctrl.Rule(
        severity["medium"] &
        public_impact["high"],
        impact_risk["high"]
    ),

    ctrl.Rule(
        severity["medium"] &
        traffic_impact["high"],
        impact_risk["high"]
    ),

    ctrl.Rule(
        public_impact["medium"] &
        traffic_impact["high"],
        impact_risk["high"]
    ),

    # Moderate
    ctrl.Rule(
        severity["medium"] &
        public_impact["medium"],
        impact_risk["moderate"]
    ),

    ctrl.Rule(
        severity["medium"] &
        traffic_impact["medium"],
        impact_risk["moderate"]
    ),

    ctrl.Rule(
        public_impact["medium"] &
        traffic_impact["medium"],
        impact_risk["moderate"]
    ),

    # Low
    ctrl.Rule(
        severity["very_low"] &
        traffic_impact["very_low"],
        impact_risk["low"]
    ),

    ctrl.Rule(
        severity["very_low"] &
        public_impact["very_low"],
        impact_risk["low"]
    ),

    ctrl.Rule(
        severity["low"] &
        traffic_impact["low"] &
        public_impact["low"],
        impact_risk["low"]
    ),

    ctrl.Rule(
        severity["low"],
        impact_risk["moderate"]
    ),

    ctrl.Rule(
        traffic_impact["low"] &
        public_impact["low"],
        impact_risk["low"]
    ),

    # --------------------------------------------------------
    # Coverage rules
    # --------------------------------------------------------

    ctrl.Rule(
        severity["very_high"],
        impact_risk["very_high"]
    ),

    ctrl.Rule(
        severity["high"],
        impact_risk["high"]
    ),

    ctrl.Rule(
        severity["medium"],
        impact_risk["moderate"]
    ),

    ctrl.Rule(
        severity["low"],
        impact_risk["low"]
    ),

    ctrl.Rule(
        severity["very_low"],
        impact_risk["low"]
    ),
]

impact_system = ctrl.ControlSystem(impact_rules)


# ============================================================
# 7. PERSISTENCE RISK
# ============================================================

persistence_risk = ctrl.Consequent(
    np.arange(0, 1.01, 0.01),
    "persistence_risk"
)

persistence_risk["low"] = fuzz.trapmf(
    persistence_risk.universe, [0, 0, 0.15, 0.30]
)

persistence_risk["moderate"] = fuzz.trimf(
    persistence_risk.universe, [0.20, 0.35, 0.50]
)

persistence_risk["high"] = fuzz.trimf(
    persistence_risk.universe, [0.40, 0.55, 0.70]
)

persistence_risk["very_high"] = fuzz.trimf(
    persistence_risk.universe, [0.60, 0.75, 0.90]
)

persistence_risk["critical"] = fuzz.trapmf(
    persistence_risk.universe, [0.82, 0.92, 1, 1]
)


# ============================================================
# 8. PERSISTENCE RULES
# ============================================================

persistence_rules = [

    ctrl.Rule(
        complaint_frequency["very_high"] &
        days_pending["very_long"],
        persistence_risk["critical"]
    ),

    ctrl.Rule(
        complaint_frequency["high"] &
        days_pending["long"],
        persistence_risk["very_high"]
    ),

    ctrl.Rule(
        complaint_frequency["very_high"],
        persistence_risk["very_high"]
    ),

    ctrl.Rule(
        days_pending["very_long"],
        persistence_risk["very_high"]
    ),

    ctrl.Rule(
        complaint_frequency["high"],
        persistence_risk["high"]
    ),

    ctrl.Rule(
        days_pending["long"],
        persistence_risk["high"]
    ),

    ctrl.Rule(
        complaint_frequency["medium"] &
        days_pending["moderate"],
        persistence_risk["high"]
    ),

    ctrl.Rule(
        complaint_frequency["medium"],
        persistence_risk["moderate"]
    ),

    ctrl.Rule(
        days_pending["moderate"],
        persistence_risk["moderate"]
    ),

    ctrl.Rule(
        complaint_frequency["none"] &
        days_pending["recent"],
        persistence_risk["low"]
    ),

    ctrl.Rule(
        complaint_frequency["low"] &
        days_pending["recent"],
        persistence_risk["low"]
    ),

    ctrl.Rule(
        complaint_frequency["none"],
        persistence_risk["low"]
    ),

    # --------------------------------------------------------
    # Coverage rules
    # --------------------------------------------------------

    ctrl.Rule(
        complaint_frequency["low"],
        persistence_risk["low"]
    ),

    ctrl.Rule(
        days_pending["recent"],
        persistence_risk["low"]
    ),
]

persistence_system = ctrl.ControlSystem(persistence_rules)


# ============================================================
# 9. FINAL RISK INPUTS
# ============================================================

final_impact_risk = ctrl.Antecedent(
    np.arange(0, 1.01, 0.01),
    "final_impact_risk"
)

final_persistence_risk = ctrl.Antecedent(
    np.arange(0, 1.01, 0.01),
    "final_persistence_risk"
)

final_safety_risk = ctrl.Consequent(
    np.arange(0, 1.01, 0.01),
    "final_safety_risk"
)


for variable in [final_impact_risk, final_persistence_risk]:

    variable["low"] = fuzz.trapmf(
        variable.universe, [0, 0, 0.20, 0.35]
    )

    variable["moderate"] = fuzz.trimf(
        variable.universe, [0.20, 0.40, 0.60]
    )

    variable["high"] = fuzz.trimf(
        variable.universe, [0.45, 0.65, 0.80]
    )

    variable["very_high"] = fuzz.trimf(
        variable.universe, [0.70, 0.82, 0.92]
    )

    variable["critical"] = fuzz.trapmf(
        variable.universe, [0.85, 0.93, 1, 1]
    )


# ============================================================
# 10. FINAL OUTPUT MEMBERSHIP FUNCTIONS
# ============================================================

final_safety_risk["low"] = fuzz.trapmf(
    final_safety_risk.universe, [0, 0, 0.15, 0.30]
)

final_safety_risk["moderate"] = fuzz.trimf(
    final_safety_risk.universe, [0.20, 0.35, 0.50]
)

final_safety_risk["high"] = fuzz.trimf(
    final_safety_risk.universe, [0.40, 0.55, 0.70]
)

final_safety_risk["very_high"] = fuzz.trimf(
    final_safety_risk.universe, [0.60, 0.75, 0.88]
)

final_safety_risk["critical"] = fuzz.trapmf(
    final_safety_risk.universe, [0.82, 0.92, 1, 1]
)


# ============================================================
# 11. FINAL SAFETY RULES
# ============================================================

final_rules = [

    # Critical
    ctrl.Rule(
        final_impact_risk["critical"],
        final_safety_risk["critical"]
    ),

    ctrl.Rule(
        final_impact_risk["very_high"] &
        final_persistence_risk["very_high"],
        final_safety_risk["critical"]
    ),

    # Very high
    ctrl.Rule(
        final_impact_risk["very_high"] &
        final_persistence_risk["high"],
        final_safety_risk["very_high"]
    ),

    ctrl.Rule(
        final_impact_risk["high"] &
        final_persistence_risk["high"],
        final_safety_risk["very_high"]
    ),

    # High
    ctrl.Rule(
        final_impact_risk["high"] &
        final_persistence_risk["moderate"],
        final_safety_risk["high"]
    ),

    ctrl.Rule(
        final_impact_risk["high"] &
        final_persistence_risk["low"],
        final_safety_risk["high"]
    ),

    ctrl.Rule(
        final_impact_risk["moderate"] &
        final_persistence_risk["high"],
        final_safety_risk["high"]
    ),

    # Moderate
    ctrl.Rule(
        final_impact_risk["moderate"] &
        final_persistence_risk["moderate"],
        final_safety_risk["moderate"]
    ),

    # Low
    ctrl.Rule(
        final_impact_risk["moderate"] &
        final_persistence_risk["low"],
        final_safety_risk["low"]
    ),

    ctrl.Rule(
        final_impact_risk["low"] &
        final_persistence_risk["high"],
        final_safety_risk["moderate"]
    ),

    ctrl.Rule(
        final_impact_risk["low"] &
        final_persistence_risk["moderate"],
        final_safety_risk["low"]
    ),

    ctrl.Rule(
        final_impact_risk["low"] &
        final_persistence_risk["low"],
        final_safety_risk["low"]
    ),

    # Very high persistence
    ctrl.Rule(
        final_impact_risk["moderate"] &
        final_persistence_risk["very_high"],
        final_safety_risk["very_high"]
    ),

    ctrl.Rule(
        final_impact_risk["low"] &
        final_persistence_risk["very_high"],
        final_safety_risk["high"]
    ),

    # --------------------------------------------------------
    # FINAL COVERAGE RULES
    # --------------------------------------------------------

    ctrl.Rule(
        final_impact_risk["critical"],
        final_safety_risk["critical"]
    ),

    ctrl.Rule(
        final_impact_risk["very_high"],
        final_safety_risk["very_high"]
    ),

    ctrl.Rule(
        final_impact_risk["high"],
        final_safety_risk["high"]
    ),

    ctrl.Rule(
        final_impact_risk["moderate"],
        final_safety_risk["moderate"]
    ),

    ctrl.Rule(
        final_impact_risk["low"],
        final_safety_risk["low"]
    ),
]

final_system = ctrl.ControlSystem(final_rules)


# ============================================================
# 12. MAIN FUNCTION
# ============================================================

def calculate_safety_risk(complaint):

    required = [
        "severity",
        "traffic_impact",
        "public_impact",
        "weather_risk",
        "days_pending",
        "complaint_frequency",
    ]

    missing = [
        key for key in required
        if key not in complaint
    ]

    if missing:
        raise ValueError(
            f"Missing complaint fields: {missing}"
        )

    values = {
        "severity": np.clip(
            float(complaint["severity"]), 0, 10
        ),
        "traffic_impact": np.clip(
            float(complaint["traffic_impact"]), 0, 10
        ),
        "public_impact": np.clip(
            float(complaint["public_impact"]), 0, 10
        ),
        "weather_risk": np.clip(
            float(complaint["weather_risk"]), 0, 10
        ),
        "days_pending": np.clip(
            float(complaint["days_pending"]), 0, 30
        ),
        "complaint_frequency": np.clip(
            float(complaint["complaint_frequency"]), 0, 10
        ),
    }

    # ========================================================
    # IMPACT STAGE
    # ========================================================

    impact_simulation = ctrl.ControlSystemSimulation(
        impact_system
    )

    impact_simulation.input["severity"] = values["severity"]
    impact_simulation.input["traffic_impact"] = values["traffic_impact"]
    impact_simulation.input["public_impact"] = values["public_impact"]

    impact_simulation.compute()

    calculated_impact = impact_simulation.output["impact_risk"]

    # ========================================================
    # PERSISTENCE STAGE
    # ========================================================

    persistence_simulation = ctrl.ControlSystemSimulation(
        persistence_system
    )

    effective_days_pending = min(
        30,
        values["days_pending"]
        + (values["weather_risk"] * 0.5)
    )

    persistence_simulation.input["days_pending"] = (
        effective_days_pending
    )

    persistence_simulation.input["complaint_frequency"] = (
        values["complaint_frequency"]
    )

    persistence_simulation.compute()

    calculated_persistence = (
        persistence_simulation.output["persistence_risk"]
    )

    # ========================================================
    # FINAL STAGE
    # ========================================================

    final_simulation = ctrl.ControlSystemSimulation(
        final_system
    )

    final_simulation.input["final_impact_risk"] = (
        calculated_impact
    )

    final_simulation.input["final_persistence_risk"] = (
        calculated_persistence
    )

    final_simulation.compute()

    calculated_safety = (
        final_simulation.output["final_safety_risk"]
    )

    return round(
        float(np.clip(calculated_safety, 0, 1)),
        4
    )


# ============================================================
# 13. QUICK TEST
# ============================================================

if __name__ == "__main__":

    tests = {

        "Low risk": {
            "severity": 3,
            "traffic_impact": 1,
            "public_impact": 2,
            "weather_risk": 1,
            "days_pending": 0,
            "complaint_frequency": 0,
        },

        "Medium risk": {
            "severity": 5,
            "traffic_impact": 5,
            "public_impact": 5,
            "weather_risk": 4,
            "days_pending": 7,
            "complaint_frequency": 3,
        },

        "High risk": {
            "severity": 7,
            "traffic_impact": 8,
            "public_impact": 8,
            "weather_risk": 7,
            "days_pending": 15,
            "complaint_frequency": 8,
        },

        "Critical risk": {
            "severity": 9,
            "traffic_impact": 10,
            "public_impact": 10,
            "weather_risk": 9,
            "days_pending": 25,
            "complaint_frequency": 10,
        },
    }

    for name, complaint in tests.items():

        risk = calculate_safety_risk(complaint)

        print(
            f"{name:15s}: "
            f"{risk:.4f} "
            f"({risk * 100:.2f}%)"
        )