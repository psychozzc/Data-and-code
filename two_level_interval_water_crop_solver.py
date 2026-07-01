from __future__ import annotations

import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Literal, Optional, Sequence, Tuple


PYTHON_TAG = f"py{sys.version_info.major}{sys.version_info.minor}"
LOCAL_SOLVER_DEPS = Path(__file__).with_name(f".solver_deps_{PYTHON_TAG}")
LEGACY_SOLVER_DEPS = Path(__file__).with_name(".solver_deps")
if LOCAL_SOLVER_DEPS.exists():
    sys.path.insert(0, str(LOCAL_SOLVER_DEPS))
elif sys.version_info[:2] == (3, 12) and LEGACY_SOLVER_DEPS.exists():
    sys.path.insert(0, str(LEGACY_SOLVER_DEPS))

import numpy as np
import pandas as pd


CaseName = Literal["best", "worst"]
Sense = Literal["<=", ">=", "="]


@dataclass(frozen=True)
class Interval:
    lower: np.ndarray
    upper: np.ndarray

    def __post_init__(self) -> None:
        lo = np.asarray(self.lower, dtype=float)
        hi = np.asarray(self.upper, dtype=float)
        if lo.shape != hi.shape:
            raise ValueError("Interval lower and upper arrays must have the same shape.")
        if np.any(lo > hi):
            raise ValueError("Interval lower values cannot exceed upper values.")
        object.__setattr__(self, "lower", lo)
        object.__setattr__(self, "upper", hi)

    def pick(self, case: CaseName, lower_is_better: bool = True) -> np.ndarray:
        if case == "best":
            return self.lower if lower_is_better else self.upper
        if case == "worst":
            return self.upper if lower_is_better else self.lower
        raise ValueError(f"Unknown case: {case}")


@dataclass
class ModelData:
    zones: List[str]
    crops: List[str]
    wd_blue: Interval
    wd_green: Interval
    wd_grey: Interval
    baseline_area: np.ndarray
    land_total: np.ndarray
    irrigation_efficiency: np.ndarray
    y_max: np.ndarray
    yield_response: np.ndarray
    price: np.ndarray
    cost: np.ndarray
    min_demand: np.ndarray
    available_water: float
    zone_demand_share: Optional[np.ndarray] = None

    def __post_init__(self) -> None:
        q = len(self.zones)
        p = len(self.crops)
        matrix_shape = (q, p)
        vector_zone = (q,)
        vector_crop = (p,)

        checks = {
            "wd_blue": self.wd_blue.lower.shape,
            "wd_green": self.wd_green.lower.shape,
            "wd_grey": self.wd_grey.lower.shape,
            "baseline_area": np.asarray(self.baseline_area).shape,
            "land_total": np.asarray(self.land_total).shape,
            "irrigation_efficiency": np.asarray(self.irrigation_efficiency).shape,
            "y_max": np.asarray(self.y_max).shape,
            "yield_response": np.asarray(self.yield_response).shape,
            "price": np.asarray(self.price).shape,
            "cost": np.asarray(self.cost).shape,
            "min_demand": np.asarray(self.min_demand).shape,
        }
        expected = {
            "wd_blue": matrix_shape,
            "wd_green": matrix_shape,
            "wd_grey": matrix_shape,
            "baseline_area": matrix_shape,
            "land_total": vector_zone,
            "irrigation_efficiency": vector_zone,
            "y_max": vector_crop,
            "yield_response": vector_crop,
            "price": vector_crop,
            "cost": vector_crop,
            "min_demand": vector_crop,
        }
        for name, actual in checks.items():
            if actual != expected[name]:
                raise ValueError(f"{name} shape {actual} does not match {expected[name]}.")

        self.baseline_area = np.asarray(self.baseline_area, dtype=float)
        self.land_total = np.asarray(self.land_total, dtype=float)
        self.irrigation_efficiency = np.asarray(self.irrigation_efficiency, dtype=float)
        self.y_max = np.asarray(self.y_max, dtype=float)
        self.yield_response = np.asarray(self.yield_response, dtype=float)
        self.price = np.asarray(self.price, dtype=float)
        self.cost = np.asarray(self.cost, dtype=float)
        self.min_demand = np.asarray(self.min_demand, dtype=float)

        if self.zone_demand_share is None:
            shares = np.full((q, p), 1.0 / q)
        else:
            shares = np.asarray(self.zone_demand_share, dtype=float)
            if shares.shape != matrix_shape:
                raise ValueError("zone_demand_share must have shape (zones, crops).")
        self.zone_demand_share = shares


@dataclass
class LPResult:
    status: str
    objective_value: float
    x: np.ndarray
    message: str = ""


def solve_lp(
    c: np.ndarray,
    constraints: Sequence[Tuple[np.ndarray, Sense, float]],
    bounds: Sequence[Tuple[float, Optional[float]]],
) -> LPResult:
    try:
        from scipy.optimize import linprog
    except ImportError as exc:
        raise RuntimeError(
            "SciPy is required. Install scipy or keep the local .solver_deps directory."
        ) from exc

    c = np.asarray(c, dtype=float)
    n = c.size
    if len(bounds) != n:
        raise ValueError("bounds length must equal number of variables.")

    a_ub: List[np.ndarray] = []
    b_ub: List[float] = []
    a_eq: List[np.ndarray] = []
    b_eq: List[float] = []
    for row, sense, rhs in constraints:
        row = np.asarray(row, dtype=float)
        if row.size != n:
            raise ValueError("Constraint row length must equal number of variables.")
        if sense == "<=":
            a_ub.append(row)
            b_ub.append(float(rhs))
        elif sense == ">=":
            a_ub.append(-row)
            b_ub.append(float(-rhs))
        elif sense == "=":
            a_eq.append(row)
            b_eq.append(float(rhs))
        else:
            raise ValueError(f"Unknown constraint sense: {sense}")

    scipy_bounds = [(float(lb), None if ub is None else float(ub)) for lb, ub in bounds]
    result = linprog(
        c,
        A_ub=np.vstack(a_ub) if a_ub else None,
        b_ub=np.array(b_ub, dtype=float) if b_ub else None,
        A_eq=np.vstack(a_eq) if a_eq else None,
        b_eq=np.array(b_eq, dtype=float) if b_eq else None,
        bounds=scipy_bounds,
        method="highs",
    )

    if result.success:
        return LPResult("optimal", float(result.fun), np.asarray(result.x, dtype=float), result.message)
    if result.status == 2:
        return LPResult("infeasible", np.nan, np.full(n, np.nan), result.message)
    if result.status == 3:
        return LPResult("unbounded", np.nan, np.full(n, np.nan), result.message)
    return LPResult("solver_error", np.nan, np.full(n, np.nan), result.message)


def _flat(i: int, j: int, p: int) -> int:
    return i * p + j


def _production_coefficients(
    data: ModelData,
    wd_blue: np.ndarray,
    wd_green: np.ndarray,
    area: np.ndarray,
    water_var: Optional[str],
) -> Tuple[np.ndarray, np.ndarray]:
    q, p = len(data.zones), len(data.crops)
    denom = wd_blue + wd_green
    denom = np.where(denom <= 0, 1e-12, denom)
    margin = data.price - data.cost
    k = data.yield_response.reshape(1, p)
    y_max = data.y_max.reshape(1, p)
    ni = data.irrigation_efficiency.reshape(q, 1)

    if water_var == "WE":
        base = y_max * (1.0 - k * wd_blue / denom) * area
        coeff = y_max * k * ni / denom
        return base, coeff
    if water_var == "xy":
        coeff_x = y_max * (1.0 - k * wd_blue / denom)
        coeff_y = y_max * k * ni / denom
        return coeff_x, coeff_y
    raise ValueError("water_var must be 'WE' or 'xy'.")


def solve_supply_level(
    data: ModelData,
    case: CaseName,
    weights: Tuple[float, float] = (0.5, 0.5),
    objective_scales: Tuple[float, float] = (1.0, 1.0),
) -> Dict[str, object]:
    q, p = len(data.zones), len(data.crops)
    n = q * p
    wd_blue = data.wd_blue.pick(case, lower_is_better=True)
    wd_green = data.wd_green.pick(case, lower_is_better=False)

    prod_const, prod_we = _production_coefficients(
        data, wd_blue, wd_green, data.baseline_area, water_var="WE"
    )
    margin = (data.price - data.cost).reshape(1, p)
    benefit_const = prod_const * margin
    benefit_we = prod_we * margin

    w1, w2 = weights
    s1, s2 = objective_scales
    c = (-w1 / s1) * benefit_we.reshape(-1) + (w2 / s2) * np.ones(n)

    constraints: List[Tuple[np.ndarray, Sense, float]] = []

    for j in range(p):
        row = np.zeros(n)
        for i in range(q):
            row[_flat(i, j, p)] = prod_we[i, j]
        rhs = float(data.min_demand[j] - np.sum(prod_const[:, j]))
        constraints.append((row, ">=", rhs))

    constraints.append((np.ones(n), "<=", float(data.available_water)))

    bounds: List[Tuple[float, Optional[float]]] = []
    for i in range(q):
        for j in range(p):
            demand_volume = wd_blue[i, j] * data.baseline_area[i, j] * data.y_max[j]
            lower = 0.8 * demand_volume / data.irrigation_efficiency[i]
            upper = demand_volume / data.irrigation_efficiency[i]
            bounds.append((max(0.0, lower), max(0.0, upper)))

    result = solve_lp(c, constraints, bounds)
    if result.status != "optimal":
        raise RuntimeError(f"Supply model failed in {case} case: {result.message}")

    we = result.x.reshape(q, p)
    benefit = float(np.sum(benefit_const + benefit_we * we))
    total_entitlement = float(np.sum(we))
    weighted_objective = float((-w1 / s1) * benefit + (w2 / s2) * total_entitlement)

    return {
        "case": case,
        "WE": we,
        "WE_zone": we.sum(axis=1),
        "economic_benefit": benefit,
        "total_entitlement": total_entitlement,
        "weighted_objective": weighted_objective,
    }


def solve_demand_level(
    data: ModelData,
    case: CaseName,
    supply_solution: Dict[str, object],
    weights: Tuple[float, float, float] = (1 / 3, 1 / 3, 1 / 3),
    objective_scales: Tuple[float, float, float] = (1.0, 1.0, 1.0),
) -> Dict[str, object]:
    q, p = len(data.zones), len(data.crops)
    n_x = q * p
    n_y = q * p
    n = n_x + n_y

    wd_blue = data.wd_blue.pick(case, lower_is_better=True)
    wd_green = data.wd_green.pick(case, lower_is_better=False)
    wd_grey = data.wd_grey.pick(case, lower_is_better=True)

    coeff_x, coeff_y = _production_coefficients(
        data, wd_blue, wd_green, data.baseline_area, water_var="xy"
    )
    margin = (data.price - data.cost).reshape(1, p)
    benefit_x = coeff_x * margin
    benefit_y = coeff_y * margin
    grey_x = coeff_x * wd_grey
    grey_y = coeff_y * wd_grey

    w1, w2, w3 = weights
    s1, s2, s3 = objective_scales
    c = np.zeros(n)
    c[:n_x] = (-w1 / s1) * benefit_x.reshape(-1) + (w3 / s3) * grey_x.reshape(-1)
    c[n_x:] = (-w1 / s1) * benefit_y.reshape(-1) + (w2 / s2) + (w3 / s3) * grey_y.reshape(-1)

    constraints: List[Tuple[np.ndarray, Sense, float]] = []

    for j in range(p):
        row = np.zeros(n)
        for i in range(q):
            row[_flat(i, j, p)] = coeff_x[i, j]
            row[n_x + _flat(i, j, p)] = coeff_y[i, j]
        constraints.append((row, ">=", float(data.min_demand[j])))

    we_zone = np.asarray(supply_solution["WE_zone"], dtype=float)
    for i in range(q):
        row = np.zeros(n)
        for j in range(p):
            row[n_x + _flat(i, j, p)] = 1.0
        constraints.append((row, "<=", float(we_zone[i])))

    for i in range(q):
        for j in range(p):
            max_irrigation_per_area = wd_blue[i, j] * data.y_max[j] / data.irrigation_efficiency[i]

            row_upper = np.zeros(n)
            row_upper[n_x + _flat(i, j, p)] = 1.0
            row_upper[_flat(i, j, p)] = -max_irrigation_per_area
            constraints.append((row_upper, "<=", 0.0))

            row_lower = np.zeros(n)
            row_lower[n_x + _flat(i, j, p)] = 1.0
            row_lower[_flat(i, j, p)] = -0.8 * max_irrigation_per_area
            constraints.append((row_lower, ">=", 0.0))

    rainy_names = {"rice", "maize", "peanut", "cotton", "姘寸ɑ", "鐜夌背", "鑺辩敓", "妫夎姳"}
    dry_names = {"wheat", "rapeseed", "灏忛害", "娌硅彍"}
    rainy_crop_idx = [j for j, crop in enumerate(data.crops) if crop.lower() in rainy_names]
    dry_crop_idx = [j for j, crop in enumerate(data.crops) if crop.lower() in dry_names]
    unknown_crops = [
        crop
        for crop in data.crops
        if crop.lower() not in rainy_names and crop.lower() not in dry_names
    ]
    if unknown_crops:
        raise ValueError(f"Unknown crop season for: {unknown_crops}")

    for i in range(q):
        rainy_row = np.zeros(n)
        for j in rainy_crop_idx:
            rainy_row[_flat(i, j, p)] = 1.0
        constraints.append((rainy_row, "<=", float(data.land_total[i])))

        dry_row = np.zeros(n)
        for j in dry_crop_idx:
            dry_row[_flat(i, j, p)] = 1.0
        constraints.append((dry_row, "<=", float(data.land_total[i])))

    bounds = [(0.0, None)] * n
    result = solve_lp(c, constraints, bounds)
    if result.status != "optimal":
        raise RuntimeError(f"Demand model failed in {case} case: {result.message}")

    x = result.x[:n_x].reshape(q, p)
    y = result.x[n_x:].reshape(q, p)
    benefit = float(np.sum(benefit_x * x + benefit_y * y))
    irrigation = float(np.sum(y))
    grey = float(np.sum(grey_x * x + grey_y * y))
    weighted_objective = float((-w1 / s1) * benefit + (w2 / s2) * irrigation + (w3 / s3) * grey)

    return {
        "case": case,
        "x": x,
        "y": y,
        "economic_benefit": benefit,
        "irrigation_water": irrigation,
        "grey_water": grey,
        "weighted_objective": weighted_objective,
    }


def solve_two_stage_extremum(
    data: ModelData,
    supply_weights: Tuple[float, float] = (0.5, 0.5),
    demand_weights: Tuple[float, float, float] = (1 / 3, 1 / 3, 1 / 3),
    supply_scales: Tuple[float, float] = (1.0, 1.0),
    demand_scales: Tuple[float, float, float] = (1.0, 1.0, 1.0),
) -> Dict[str, object]:
    case_results: Dict[str, Dict[str, object]] = {}
    for case in ("best", "worst"):
        supply = solve_supply_level(data, case, supply_weights, supply_scales)
        demand = solve_demand_level(data, case, supply, demand_weights, demand_scales)
        case_results[case] = {"supply": supply, "demand": demand}

    best_supply = case_results["best"]["supply"]
    worst_supply = case_results["worst"]["supply"]
    best_demand = case_results["best"]["demand"]
    worst_demand = case_results["worst"]["demand"]

    interval_solution = {
        "WE": _interval_from_cases(best_supply["WE"], worst_supply["WE"]),
        "x": _interval_from_cases(best_demand["x"], worst_demand["x"]),
        "y": _interval_from_cases(best_demand["y"], worst_demand["y"]),
        "economic_benefit": (
            min(best_demand["economic_benefit"], worst_demand["economic_benefit"]),
            max(best_demand["economic_benefit"], worst_demand["economic_benefit"]),
        ),
        "irrigation_water": (
            min(best_demand["irrigation_water"], worst_demand["irrigation_water"]),
            max(best_demand["irrigation_water"], worst_demand["irrigation_water"]),
        ),
        "grey_water": (
            min(best_demand["grey_water"], worst_demand["grey_water"]),
            max(best_demand["grey_water"], worst_demand["grey_water"]),
        ),
    }
    return {"cases": case_results, "interval_solution": interval_solution}


def _interval_from_cases(a: object, b: object) -> Interval:
    arr_a = np.asarray(a, dtype=float)
    arr_b = np.asarray(b, dtype=float)
    return Interval(np.minimum(arr_a, arr_b), np.maximum(arr_a, arr_b))


def to_long_table(matrix: np.ndarray, zones: Sequence[str], crops: Sequence[str], value: str) -> pd.DataFrame:
    rows = []
    for i, zone in enumerate(zones):
        for j, crop in enumerate(crops):
            rows.append({"zone": zone, "crop": crop, value: float(matrix[i, j])})
    return pd.DataFrame(rows)


def interval_to_long_table(interval: Interval, zones: Sequence[str], crops: Sequence[str], name: str) -> pd.DataFrame:
    rows = []
    for i, zone in enumerate(zones):
        for j, crop in enumerate(crops):
            rows.append(
                {
                    "zone": zone,
                    "crop": crop,
                    f"{name}_lower": float(interval.lower[i, j]),
                    f"{name}_upper": float(interval.upper[i, j]),
                }
            )
    return pd.DataFrame(rows)


def _case_variable_rows(
    matrix: np.ndarray,
    zones: Sequence[str],
    crops: Sequence[str],
    case: str,
    variable: str,
) -> pd.DataFrame:
    frame = to_long_table(matrix, zones, crops, "value")
    frame.insert(0, "variable", variable)
    frame.insert(0, "case", case)
    frame["lower"] = np.nan
    frame["upper"] = np.nan
    return frame[["case", "variable", "zone", "crop", "value", "lower", "upper"]]


def _interval_variable_rows(
    interval: Interval,
    zones: Sequence[str],
    crops: Sequence[str],
    variable: str,
) -> pd.DataFrame:
    frame = interval_to_long_table(interval, zones, crops, variable)
    frame = frame.rename(columns={f"{variable}_lower": "lower", f"{variable}_upper": "upper"})
    frame.insert(0, "variable", variable)
    frame.insert(0, "case", "interval")
    frame["value"] = np.nan
    return frame[["case", "variable", "zone", "crop", "value", "lower", "upper"]]


def export_results_to_csv(
    data: ModelData,
    result: Dict[str, object],
    output_dir: str | Path = "results_csv",
) -> Path:
    out_dir = Path(output_dir)
    if not out_dir.is_absolute():
        out_dir = Path(__file__).with_name(str(out_dir))
    out_dir.mkdir(parents=True, exist_ok=True)

    summary_rows = []
    for case in ("best", "worst"):
        supply = result["cases"][case]["supply"]
        demand = result["cases"][case]["demand"]
        summary_rows.extend(
            [
                {
                    "case": case,
                    "level": "supply",
                    "metric": "economic_benefit",
                    "value": supply["economic_benefit"],
                    "lower": np.nan,
                    "upper": np.nan,
                },
                {
                    "case": case,
                    "level": "supply",
                    "metric": "total_entitlement",
                    "value": supply["total_entitlement"],
                    "lower": np.nan,
                    "upper": np.nan,
                },
                {
                    "case": case,
                    "level": "supply",
                    "metric": "weighted_objective",
                    "value": supply["weighted_objective"],
                    "lower": np.nan,
                    "upper": np.nan,
                },
                {
                    "case": case,
                    "level": "demand",
                    "metric": "economic_benefit",
                    "value": demand["economic_benefit"],
                    "lower": np.nan,
                    "upper": np.nan,
                },
                {
                    "case": case,
                    "level": "demand",
                    "metric": "irrigation_water",
                    "value": demand["irrigation_water"],
                    "lower": np.nan,
                    "upper": np.nan,
                },
                {
                    "case": case,
                    "level": "demand",
                    "metric": "grey_water",
                    "value": demand["grey_water"],
                    "lower": np.nan,
                    "upper": np.nan,
                },
                {
                    "case": case,
                    "level": "demand",
                    "metric": "weighted_objective",
                    "value": demand["weighted_objective"],
                    "lower": np.nan,
                    "upper": np.nan,
                },
            ]
        )

    interval = result["interval_solution"]
    for metric in ("economic_benefit", "irrigation_water", "grey_water"):
        lower, upper = interval[metric]
        summary_rows.append(
            {
                "case": "interval",
                "level": "demand",
                "metric": metric,
                "value": np.nan,
                "lower": lower,
                "upper": upper,
            }
        )

    pd.DataFrame(summary_rows).to_csv(
        out_dir / "optimization_summary.csv",
        index=False,
        encoding="utf-8-sig",
    )

    variable_frames = []
    for case in ("best", "worst"):
        supply = result["cases"][case]["supply"]
        demand = result["cases"][case]["demand"]
        variable_frames.extend(
            [
                _case_variable_rows(supply["WE"], data.zones, data.crops, case, "WE"),
                _case_variable_rows(demand["x"], data.zones, data.crops, case, "x"),
                _case_variable_rows(demand["y"], data.zones, data.crops, case, "y"),
            ]
        )

    variable_frames.extend(
        [
            _interval_variable_rows(interval["WE"], data.zones, data.crops, "WE"),
            _interval_variable_rows(interval["x"], data.zones, data.crops, "x"),
            _interval_variable_rows(interval["y"], data.zones, data.crops, "y"),
        ]
    )
    pd.concat(variable_frames, ignore_index=True).to_csv(
        out_dir / "optimization_decision_variables.csv",
        index=False,
        encoding="utf-8-sig",
    )

    return out_dir


def scenario_definitions() -> List[Dict[str, object]]:
    return [
        {
            "scenario": "S0",
            "supply_weights": (0.5, 0.5),
            "demand_weights": (1 / 3, 1 / 3, 1 / 3),
            "interpretation": "Equal-weight baseline",
        },
        {
            "scenario": "S1",
            "supply_weights": (0.7, 0.3),
            "demand_weights": (0.5, 0.25, 0.25),
            "interpretation": "Economy-oriented",
        },
        {
            "scenario": "S2",
            "supply_weights": (0.3, 0.7),
            "demand_weights": (0.25, 0.5, 0.25),
            "interpretation": "Water-saving-oriented",
        },
        {
            "scenario": "S3",
            "supply_weights": (0.5, 0.5),
            "demand_weights": (0.25, 0.25, 0.5),
            "interpretation": "Pollution-mitigation-oriented",
        },
        {
            "scenario": "S4",
            "supply_weights": (0.4, 0.6),
            "demand_weights": (0.2, 0.4, 0.4),
            "interpretation": "Water-environment-oriented",
        },
        {
            "scenario": "S5",
            "supply_weights": (0.8, 0.2),
            "demand_weights": (0.6, 0.2, 0.2),
            "interpretation": "Strong economy preference",
        },
        {
            "scenario": "S6",
            "supply_weights": (0.2, 0.8),
            "demand_weights": (0.2, 0.5, 0.3),
            "interpretation": "Strong water-saving preference",
        },
    ]


def export_scenario_summary_to_csv(
    data: ModelData,
    output_dir: str | Path = "results_csv",
) -> pd.DataFrame:
    rows = []
    for scenario in scenario_definitions():
        result = solve_two_stage_extremum(
            data,
            supply_weights=scenario["supply_weights"],
            demand_weights=scenario["demand_weights"],
        )
        interval = result["interval_solution"]
        economic_lower, economic_upper = interval["economic_benefit"]
        irrigation_lower, irrigation_upper = interval["irrigation_water"]
        grey_lower, grey_upper = interval["grey_water"]

        best_supply = result["cases"]["best"]["supply"]
        worst_supply = result["cases"]["worst"]["supply"]
        best_demand = result["cases"]["best"]["demand"]
        worst_demand = result["cases"]["worst"]["demand"]
        rows.append(
            {
                "scenario": scenario["scenario"],
                "interpretation": scenario["interpretation"],
                "supply_weights": ", ".join(str(x) for x in scenario["supply_weights"]),
                "demand_weights": ", ".join(str(x) for x in scenario["demand_weights"]),
                "economic_lower_CNY": economic_lower,
                "economic_upper_CNY": economic_upper,
                "economic_lower_1e9_CNY": economic_lower / 1e9,
                "economic_upper_1e9_CNY": economic_upper / 1e9,
                "irrigation_lower_m3": irrigation_lower,
                "irrigation_upper_m3": irrigation_upper,
                "irrigation_lower_1e9_m3": irrigation_lower / 1e9,
                "irrigation_upper_1e9_m3": irrigation_upper / 1e9,
                "grey_lower_m3": grey_lower,
                "grey_upper_m3": grey_upper,
                "grey_lower_1e9_m3": grey_lower / 1e9,
                "grey_upper_1e9_m3": grey_upper / 1e9,
                "best_supply_weighted_objective": best_supply["weighted_objective"],
                "worst_supply_weighted_objective": worst_supply["weighted_objective"],
                "best_demand_weighted_objective": best_demand["weighted_objective"],
                "worst_demand_weighted_objective": worst_demand["weighted_objective"],
            }
        )

    frame = pd.DataFrame(rows)
    out_dir = Path(output_dir)
    if not out_dir.is_absolute():
        out_dir = Path(__file__).with_name(str(out_dir))
    out_dir.mkdir(parents=True, exist_ok=True)
    output_path = out_dir / "scenario_summary.csv"
    try:
        frame.to_csv(output_path, index=False, encoding="utf-8-sig")
    except PermissionError:
        output_path = out_dir / "scenario_summary_unscaled.csv"
        frame.to_csv(output_path, index=False, encoding="utf-8-sig")
    frame.attrs["output_path"] = output_path
    return frame


def build_toy_data() -> ModelData:
    raise RuntimeError("Case data has been removed. Provide a ModelData instance before running the solver.")

def main() -> None:
    data = build_toy_data()

    result = solve_two_stage_extremum(
        data,
        supply_weights=(0.5, 0.5),
        demand_weights=(1 / 3, 1 / 3, 1 / 3),
    )

    for case in ("best", "worst"):
        supply = result["cases"][case]["supply"]
        demand = result["cases"][case]["demand"]
        print(f"\n=== {case.upper()} CASE ===")
        print(
            "Supply: benefit={:.2f}, entitlement={:.2f}, weighted={:.2f}".format(
                supply["economic_benefit"],
                supply["total_entitlement"],
                supply["weighted_objective"],
            )
        )
        print(
            "Demand: benefit={:.2f}, irrigation={:.2f}, grey={:.2f}, weighted={:.2f}".format(
                demand["economic_benefit"],
                demand["irrigation_water"],
                demand["grey_water"],
                demand["weighted_objective"],
            )
        )
        print("\nWater entitlement WE:")
        print(to_long_table(supply["WE"], data.zones, data.crops, "WE").to_string(index=False))
        print("\nCropping area x:")
        print(to_long_table(demand["x"], data.zones, data.crops, "x").to_string(index=False))

    interval = result["interval_solution"]
    print("\n=== INTERVAL SOLUTION ===")
    print("Economic benefit interval:", interval["economic_benefit"])
    print("Irrigation water interval:", interval["irrigation_water"])
    print("Grey water interval:", interval["grey_water"])
    print("\nWE interval:")
    print(interval_to_long_table(interval["WE"], data.zones, data.crops, "WE").to_string(index=False))
    print("\nx interval:")
    print(interval_to_long_table(interval["x"], data.zones, data.crops, "x").to_string(index=False))
    print("\ny interval:")
    print(interval_to_long_table(interval["y"], data.zones, data.crops, "y").to_string(index=False))

    output_dir = export_results_to_csv(data, result)
    print(f"\nCSV results exported to: {output_dir}")

    scenario_summary = export_scenario_summary_to_csv(data)
    print("\n=== SCENARIO SUMMARY ===")
    print(
        scenario_summary[
            [
                "scenario",
                "economic_lower_1e9_CNY",
                "economic_upper_1e9_CNY",
                "irrigation_lower_1e9_m3",
                "irrigation_upper_1e9_m3",
                "grey_lower_1e9_m3",
                "grey_upper_1e9_m3",
            ]
        ].to_string(index=False)
    )
    scenario_output = scenario_summary.attrs.get("output_path", output_dir / "scenario_summary.csv")
    print(f"\nScenario summary exported to: {scenario_output}")


if __name__ == "__main__":
    main()

