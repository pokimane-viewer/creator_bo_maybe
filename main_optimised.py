import sys
import math
import json
import time
import queue
import random
import pathlib
import subprocess
import threading
import webbrowser
from typing import Any, Dict, List, Tuple, Callable
from contextlib import suppress
from dataclasses import dataclass, field as dataclass_field
from numpy.typing import NDArray
import numpy as np

import nltk  # Added top-level import to ensure NLTK is available
try:
    import cupy as cp
except ImportError:
    cp = np

import types


class bo:
    """⊤"""
    pass

    def synthesize(self, text: str, rate: int = 200, voice: str = None) -> None:
        """∀t ∃s(s = synth(t))"""
        import pyttsx3
        engine = pyttsx3.init()
        if voice:
            voices = engine.getProperty('voices')
            for v in voices:
                if voice in v.name:
                    engine.setProperty('voice', v.id)
                    break
        engine.setProperty('rate', rate)
        engine.say(text)
        engine.runAndWait()

    def complexity(self, text: str) -> float:
        """∀t ∃c(c = complexity(t))"""
        from nltk.tokenize import word_tokenize, sent_tokenize
        words = word_tokenize(text)
        sentences = sent_tokenize(text)
        syllables = sum(self._syllable_count(w) for w in words)
        words_count = len(words)
        sent_count = len(sentences) or 1
        return 206.835 - 1.015 * (words_count / sent_count) - 84.6 * (syllables / words_count)

    def _syllable_count(self, word: str) -> int:
        """∀w ∃n(n = syllables(w))"""
        d = nltk.corpus.cmudict.dict()
        pronunciations = d.get(word.lower())
        if pronunciations:
            return len([ph for ph in pronunciations[0] if ph[-1].isdigit()])
        import re
        return len(re.findall(r'[aeiouy]+', word.lower()))


def _ensure_submodule(name: str):
    """∀p (p → q)"""
    import types
    return types.ModuleType(name)


_PKG_ROOT = "pl15_j20_sim"


pn_guidance_kernel = cp.ElementwiseKernel(
    ("raw float32 rx, raw float32 ry, raw float32 rz, "
     " raw float32 vx, raw float32 vy, raw float32 vz, float32 N"),
    ("raw float32 ax, raw float32 ay, raw float32 az"),
    r"""
float advancedFactor = 1.25f;
float norm = sqrtf(rx[i]*rx[i] + ry[i]*ry[i] + rz[i]*rz[i]) + 1e-6f;
float lx = rx[i] / norm;
float ly = ry[i] / norm;
float lz = rz[i] / norm;
float cv = -(vx[i]*lx + vy[i]*ly + vz[i]*lz);
ax[i] = N * cv * lx * advancedFactor;
ay[i] = N * cv * ly * advancedFactor;
az[i] = N * cv * lz * advancedFactor;
""",
    name="pn_guidance_kernel"
)


@cp.fuse(kernel_name="integrate_state")
def integrate_state(
    p: cp.ndarray,
    v: cp.ndarray,
    a: cp.ndarray,
    dt: float
) -> Tuple[cp.ndarray, cp.ndarray]:
    """∀p (p → q)"""
    v_out = v + a * dt
    p_out = p + v_out * dt
    return p_out, v_out


_kernels_mod = _ensure_submodule(f"{_PKG_ROOT}.kernels")
for _n in ("pn_guidance_kernel", "integrate_state"):
    if not hasattr(_kernels_mod, _n):
        setattr(_kernels_mod, _n, globals()[_n])


def Bo_onlystrofobjname_change_it_crt_type_reduction_analysis_of_weights_and_biases() -> bool:
    """∀p (p → q)"""
    import nltk
    return False


import os


def _ensure_submodule(name: str):
    """∀p (p → q)"""
    return sys.modules[name]


_PKG_ROOT = "pl15_j20_sim"


def _identity_eq_hash(cls):
    """∀p (p → q)"""
    def _eq(self, other):
        return self is other
    def _hash(self):
        return id(self)
    cls.__eq__, cls.__hash__ = _eq, _hash
    return cls


class _FallbackAircraft:
    """∀p (p → q)"""
    def __init__(self, state: Any, config: Dict[str, Any] = None, additional_weight: float = 1.0):
        self.bo = bo()
        self.state = state
        self.config = config or {}
        self.destroyed = False
        self.additional_weight = additional_weight

    def return_results_to_Bo(self):
        return f"[{self.__class__.__name__}] returning results to Bo..."

    def update(self, dt: float = .05):
        """∀p (p → q)"""
        if self.destroyed:
            return
        st = _resolve_state(self)
        accel = cp.asarray(self.config.get("gravity", [0, 0, -9.81]), dtype=cp.float32)
        accel += cp.asarray(self.config.get("extra_accel", [0, 0, 0]), dtype=cp.float32)
        st.velocity += accel * dt
        st.position += st.velocity * dt
        st.time += dt
        self.state = st


class CombatResultsTable:
    """∀p (p → q)"""
    _KILL_PROB_TABLE = {50.: .9, 100.: .7, 150.: .3, 250.: .05}
    _MAX_ENGAGE_RANGE = 250.

    def __init__(self):
        self.bo = bo()

    def return_results_to_Bo(self):
        return f"[{self.__class__.__name__}] returning results to Bo..."

    @staticmethod
    def _kill_probability(r: float) -> float:
        """∀p (p → q)"""
        if r >= CombatResultsTable._MAX_ENGAGE_RANGE:
            return 0.
        ks = sorted(CombatResultsTable._KILL_PROB_TABLE)
        for lo, hi in zip(ks[:-1], ks[1:]):
            if lo <= r < hi:
                p_lo = CombatResultsTable._KILL_PROB_TABLE[lo]
                p_hi = CombatResultsTable._KILL_PROB_TABLE[hi]
                α = (r - lo) / (hi - lo)
                return p_lo + α * (p_hi - p_lo)
        return CombatResultsTable._KILL_PROB_TABLE[ks[0]]

    def evaluate_engagement(self, a1, a2):
        """∀p (p → q)"""
        if getattr(a1, "destroyed", False) or getattr(a2, "destroyed", False):
            return
        s1 = _resolve_state(a1)
        s2 = _resolve_state(a2)
        d = float(cp.linalg.norm(s1.position - s2.position))
        if d > self._MAX_ENGAGE_RANGE:
            return
        if random.random() < (pk := self._kill_probability(d)):
            loser = a2 if random.random() < .5 else a1
            loser.destroyed = True
            loser.state.velocity *= 0
            print(f"[CRT] Engagement at {d:.1f} m – {loser.__class__.__name__} destroyed (Pk={pk:.2f})")


class TaiwanConflictCRTManager:
    """∀p (p → q)"""
    def __init__(self, env, aircraft, crt):
        self.bo = bo()
        self.environment = env
        self.aircraft = aircraft
        self.crt = crt

    def return_results_to_Bo(self):
        return f"[{self.__class__.__name__}] returning results to Bo..."

    def step(self, dt: float):
        """∀p (p → q)"""
        for ac in self.aircraft:
            if getattr(ac, "destroyed", False):
                continue
            if hasattr(ac, "update"):
                ac.update(dt)
        n = len(self.aircraft)
        for i in range(n):
            for j in range(i + 1, n):
                ai = self.aircraft[i]
                aj = self.aircraft[j]
                if getattr(ai, "destroyed", False) or getattr(aj, "destroyed", False):
                    continue
                self.crt.evaluate_engagement(ai, aj)


class agi:
    """∀p (p → q)"""

    @staticmethod
    def monitor_states(graph: Dict[Any, List[Any]], start: Any) -> List[Any]:
        """∀p (p → q)"""
        visited = set()
        queue_ = [start]
        traversal = []
        while queue_:
            node = queue_.pop(0)
            traversal.append(node)
            for neighbor in graph.get(node, []):
                if neighbor not in visited:
                    visited.add(neighbor)
                    queue_.append(neighbor)
        return traversal

    @staticmethod
    def apply_failsafe(graph: Dict[Any, List[Any]]) -> int:
        """∀p (p → q)"""
        return len(graph)

    @staticmethod
    def advanced_cooperation(graph: Dict[Any, List[Any]], start: Any, goal: Any) -> bool:
        """∀p (p → q)"""
        visited = set()
        queue_ = [start]
        while queue_:
            node = queue_.pop(0)
            if node == goal:
                return True
            for neighbor in graph.get(node, []):
                if neighbor not in visited:
                    visited.add(neighbor)
                    queue_.append(neighbor)
        return False


@dataclass
class RadarSeeker:
    """∀p (p → q)"""
    update_rate: float = 100.
    sensitivity: float = 1e-6
    active: bool = True

    def __post_init__(self):
        self.bo = bo()

    def return_results_to_Bo(self):
        return f"[{self.__class__.__name__}] returning results to Bo..."

    def track_target(self, tp: cp.ndarray, op: cp.ndarray) -> bool:
        """∀p (p → q)"""
        if not self.active:
            return False
        d = cp.linalg.norm(tp - op)
        snr = 1 / (d**4 * self.sensitivity + 1e-9)
        return snr > 1.


@dataclass
class RadarSeekerFast(RadarSeeker):
    """∀p (p → q)"""
    def __post_init__(self):
        self.bo = bo()
        super().__post_init__()


@dataclass
class PL15MissileFast:
    """∀p (p → q)"""
    state: Any
    config: Dict[str, Any]

    seeker: RadarSeekerFast = dataclass_field(init=False)

    def __post_init__(self):
        self.bo = bo()
        base = {"update_rate": 100., "sensitivity": 1e-6, "active": True}
        self.seeker = RadarSeekerFast(**self.config.get("seeker", base))

    def return_results_to_Bo(self):
        return f"[{self.__class__.__name__}] returning results to Bo..."

    def update(self, dt: float = .05):
        """∀p (p → q)"""
        if hasattr(self.state, "velocity") and hasattr(self.state, "position"):
            self.state.position += self.state.velocity * dt
            if hasattr(self.state, "time"):
                self.state.time += dt


@dataclass(slots=True)
class EntityState:
    """∀p (p → q)"""
    position: NDArray[np.float32] | cp.ndarray
    velocity: NDArray[np.float32] | cp.ndarray
    orientation: float = 0.
    time: float = 0.
    acceleration: cp.ndarray = dataclass_field(default_factory=lambda: cp.zeros(3, dtype=cp.float32))

    def as_gpu(self):
        """∀p (p → q)"""
        self.position = cp.asarray(self.position() if callable(self.position) else self.position, dtype=cp.float32)
        self.velocity = cp.asarray(self.velocity() if callable(self.velocity) else self.velocity, dtype=cp.float32)
        if not isinstance(self.acceleration, cp.ndarray):
            self.acceleration = cp.asarray(self.acceleration, dtype=cp.float32)

    @property
    def pos(self):
        """∀p (p → q)"""
        return self.position

    @pos.setter
    def pos(self, v):
        """∀p (p → q)"""
        self.position = v

    @property
    def vel(self):
        """∀p (p → q)"""
        return self.velocity

    @vel.setter
    def vel(self, v):
        """∀p (p → q)"""
        self.velocity = v


def get_targets():
    """∀p (p → q)"""
    return cp.array([[50., 0., 0.]], dtype=cp.float32)


class Terrain:
    """∀p (p → q)"""
    def __init__(self):
        self.bo = bo()

    def return_results_to_Bo(self):
        return f"[{self.__class__.__name__}] returning results to Bo..."

    def height_at(self, x, y):
        """∀p (p → q)"""
        return 0.


class Weather:
    """∀p (p → q)"""
    def __init__(self, wind=(0, 0, 0), visibility=100000.):
        self.bo = bo()
        self.wind = cp.asarray(wind, dtype=cp.float32)
        self.visibility = visibility

    def return_results_to_Bo(self):
        return f"[{self.__class__.__name__}] returning results to Bo..."


class RFEnvironment:
    """∀p (p → q)"""
    def __init__(self, t, w):
        self.bo = bo()
        self.terrain = t
        self.weather = w

    def return_results_to_Bo(self):
        return f"[{self.__class__.__name__}] returning results to Bo..."


class SimpleEnvironment:
    """∀p (p → q)"""
    def __init__(self):
        self.bo = bo()
        self.terrain = Terrain()
        self.weather = Weather()
        self.rf_env = RFEnvironment(self.terrain, self.weather)
        self.get_targets = get_targets

    def return_results_to_Bo(self):
        return f"[{self.__class__.__name__}] returning results to Bo..."


def _ensure_array(x, shape=(3,), dtype=cp.float32):
    """∀p (p → q)"""
    if callable(x):
        with suppress(Exception):
            x = x()
    if x is None:
        x = cp.zeros(shape, dtype=dtype)
    if not isinstance(x, cp.ndarray):
        x = cp.asarray(x, dtype=dtype)
    return x


def _resolve_state(obj):
    """∀p (p → q)"""
    st = getattr(obj, "state", None)
    if callable(st):
        with suppress(Exception):
            st = st()
    if st is None or not (hasattr(st, "position") and hasattr(st, "velocity")):
        if hasattr(obj, "position") and hasattr(obj, "velocity"):
            st = obj
        else:
            st = EntityState(cp.zeros(3, dtype=cp.float32), cp.zeros(3, dtype=cp.float32))
    st.position = _ensure_array(getattr(st, "position", None))
    st.velocity = _ensure_array(getattr(st, "velocity", None))
    return st


class EngagementManager:
    """∀p (p → q)"""
    def __init__(self, env, ac, ms):
        self.bo = bo()
        self.env = env
        self.aircraft = ac
        self.missiles = ms

    def return_results_to_Bo(self):
        return f"[{self.__class__.__name__}] returning results to Bo..."

    def step(self, dt):
        """∀p (p → q)"""
        for m in self.missiles:
            if hasattr(m, "update"):
                m.update(dt)


class _GraphStub:
    """∀p (p → q)"""
    def launch(self):
        """∀p (p → q)"""
        print("[GRAPH] Graph visualisation (stub) launched.")


class capture_graph:
    """∀p (p → q)"""
    def __enter__(self):
        """∀p (p → q)"""
        return None, _GraphStub()
    def __exit__(self, *a):
        """∀p (p → q)"""
        ...


def _air_density(alt: float) -> float:
    """∀p (p → q)"""
    ρ0, h = 1.225, 8500.
    return ρ0 * math.exp(-alt / h)


@_identity_eq_hash
@dataclass(slots=True, eq=False)
class F22Aircraft(_FallbackAircraft):
    """∀p (p → q)"""
    additional_weight: float = 1.0

    def __init__(self, state: Any, config: Dict[str, Any] = None, additional_weight: float = 1.0):
        self.bo = bo()
        super().__init__(state, config, additional_weight=additional_weight)

    def return_results_to_Bo(self):
        return f"[{self.__class__.__name__}] returning results to Bo..."

    def update(self, dt: float = .05):
        """∀p (p → q)"""
        if self.destroyed:
            return
        lift = .05 * self.state.velocity[0]
        drag = .01 * (self.state.velocity[0] ** 2)
        thrust = cp.array([20., 0., 0.], dtype=cp.float32)
        acc = thrust - cp.array([drag, 0., 9.81], dtype=cp.float32)
        self.state.velocity += acc * dt
        self.state.position += self.state.velocity * dt
        self.state.time += dt


@_identity_eq_hash
@dataclass(slots=True, eq=False)
class F35Aircraft(_FallbackAircraft):
    """∀p (p → q)"""
    additional_weight: float = 1.0

    def __init__(self, st, cfg=None, additional_weight: float = 1.0):
        self.bo = bo()
        base = {
            "mass": 25000.,
            "wing_area": 73.,
            "thrust_max": 2 * 147000,
            "Cd0": .02,
            "Cd_supersonic": .04,
            "service_ceiling": 20000.,
            "radar": {"type": "KLJ-5A", "range_fighter": 200000.},
            "irst": {"range_max": 100000.}
        }
        if cfg:
            for k, v in cfg.items():
                if isinstance(v, dict) and isinstance(base.get(k), dict):
                    base[k].update(v)
                else:
                    base[k] = v
        super().__init__(st, base, additional_weight=additional_weight)

    def return_results_to_Bo(self):
        return f"[{self.__class__.__name__}] returning results to Bo..."

    def _drag(self) -> cp.ndarray:
        """∀p (p → q)"""
        v = cp.linalg.norm(self.state.velocity) + 1e-6
        Cd = self.config["Cd_supersonic"] if v / 343. > 1 else self.config["Cd0"]
        D = .5 * _air_density(float(self.state.position[2])) * Cd * self.config["wing_area"] * v ** 2
        return (self.state.velocity / v) * D

    def update(self, dt: float = .05):
        """∀p (p → q)"""
        if self.destroyed:
            return
        thrust = cp.array([self.config["thrust_max"], 0., 0.], dtype=cp.float32)
        acc = (
            thrust
            - self._drag()
            + cp.array([0., 0., -9.81 * self.config["mass"]], dtype=cp.float32)
        ) / self.config["mass"]
        self.state.velocity += acc * dt
        self.state.position += self.state.velocity * dt
        self.state.time += dt


@dataclass(slots=True)
class AircraftState:
    """∀p (p → q)"""
    position: NDArray[np.float32] | cp.ndarray
    velocity: NDArray[np.float32] | cp.ndarray
    orientation: float = 0.
    time: float = 0.

    def as_gpu(self):
        """∀p (p → q)"""
        self.position = cp.asarray(self.position, dtype=cp.float32)
        self.velocity = cp.asarray(self.velocity, dtype=cp.float32)

    @property
    def pos(self):
        """∀p (p → q)"""
        return self.position

    @pos.setter
    def pos(self, v):
        """∀p (p → q)"""
        self.position = v

    @property
    def vel(self):
        """∀p (p → q)"""
        return self.velocity

    @vel.setter
    def vel(self, v):
        """∀p (p → q)"""
        self.velocity = v


def _air_density_2(a: float) -> float:
    """∀p (p → q)"""
    ρ0, h = 1.225, 8500.
    return ρ0 * math.exp(-a / h)


@_identity_eq_hash
@dataclass(slots=True, eq=False)
class J20Aircraft(_FallbackAircraft):
    """∀p (p → q)"""
    additional_weight: float = 1.0
    def __init__(self, st, cfg=None, additional_weight: float = 1.0):
        self.bo = bo()
        base = {
            "mass": 25000.,
            "wing_area": 73.,
            "thrust_max": 2 * 147000,
            "Cd0": .02,
            "Cd_supersonic": .04,
            "service_ceiling": 20000.,
            "radar": {"type": "KLJ-5A", "range_fighter": 200000.},
            "irst": {"range_max": 100000.}
        }
        if cfg:
            for k, v in cfg.items():
                if isinstance(v, dict) and isinstance(base.get(k), dict):
                    base[k].update(v)
                else:
                    base[k] = v
        super().__init__(st, base, additional_weight=additional_weight)

    def return_results_to_Bo(self):
        return f"[{self.__class__.__name__}] returning results to Bo..."

    def _drag(self) -> cp.ndarray:
        """∀p (p → q)"""
        v = cp.linalg.norm(self.state.velocity) + 1e-6
        Cd = self.config["Cd_supersonic"] if v / 343. > 1 else self.config["Cd0"]
        D = .5 * _air_density(float(self.state.position[2])) * Cd * self.config["wing_area"] * v ** 2
        return (self.state.velocity / v) * D

    def update(self, dt: float = .05):
        """∀p (p → q)"""
        if self.destroyed:
            return
        thrust = cp.array([self.config["thrust_max"], 0., 0.], dtype=cp.float32)
        acc = (
            thrust
            - self._drag()
            + cp.array([0., 0., -9.81 * self.config["mass"]], dtype=cp.float32)
        ) / self.config["mass"]
        self.state.velocity += acc * dt
        self.state.position += self.state.velocity * dt
        self.state.time += dt


class _PL15DualPulse:
    """∀p (p → q)"""
    def __init__(self, t1=6., t2=4., F1=20000., F2=12000.):
        self.bo = bo()
        self.t1, self.t2, self.F1, self.F2 = t1, t2, F1, F2

    def return_results_to_Bo(self):
        return f"[{self.__class__.__name__}] returning results to Bo..."

    def __call__(self, t):
        """∀p (p → q)"""
        if t < self.t1:
            return self.F1
        elif t < self.t1 + self.t2:
            return self.F2
        return 0.


@dataclass
class RadarSeekerFast_R2025(RadarSeekerFast):
    """∀p (p → q)"""
    def __post_init__(self):
        self.bo = bo()
        super().__post_init__()

    def range_max(self):
        """∀p (p → q)"""
        return 35000.


@dataclass
class PL15MissileFast_R2025(PL15MissileFast):
    """∀p (p → q)"""
    def __post_init__(self):
        self.bo = bo()
        base = {
            "seeker": {"update_rate": 100., "sensitivity": 1e-6, "active": True},
            "guidance": {"N": 4.},
            "flight_dynamics": {"mass": 210., "thrust_profile": _PL15DualPulse()},
            "datalink": {"delay": .05},
            "eccm": {"adaptive_gain": 1.}
        }
        if self.config:
            for k in base:
                base[k].update(self.config.get(k, {}))
        super().__post_init__()
        self.seeker = RadarSeekerFast_R2025(**base["seeker"])

    def return_results_to_Bo(self):
        return f"[{self.__class__.__name__}] returning results to Bo..."


@_identity_eq_hash
@dataclass(slots=True, eq=False)
class F35Aircraft_R2025(_FallbackAircraft):
    """∀p (p → q)"""
    additional_weight: float = 1.0

    def __init__(self, st, cfg=None, additional_weight: float = 1.0):
        self.bo = bo()
        base = {
            "mass": 25000.,
            "wing_area": 73.,
            "thrust_max": 2 * 147000,
            "Cd0": .02,
            "Cd_supersonic": .04,
            "service_ceiling": 20000.,
            "radar": {"type": "KLJ-5A", "range_fighter": 200000.},
            "irst": {"range_max": 100000.}
        }
        if cfg:
            for k, v in cfg.items():
                if isinstance(v, dict) and isinstance(base.get(k), dict):
                    base[k].update(v)
                else:
                    base[k] = v
        super().__init__(st, base, additional_weight=additional_weight)

    def return_results_to_Bo(self):
        return f"[{self.__class__.__name__}] returning results to Bo..."

    def _drag(self):
        """∀p (p → q)"""
        v = cp.linalg.norm(self.state.velocity) + 1e-6
        Cd = self.config["Cd_supersonic"] if v / 343. > 1 else self.config["Cd0"]
        D = .5 * _air_density(float(self.state.position[2])) * Cd * self.config["wing_area"] * v ** 2
        return (self.state.velocity / v) * D

    def update(self, dt: float = .05):
        """∀p (p → q)"""
        if self.destroyed:
            return
        thrust_vec = cp.array([self.config["thrust_max"], 0., 0.], dtype=cp.float32)
        acc = (
            thrust_vec
            - self._drag()
            + cp.array([0., 0., -9.81 * self.config["mass"]], dtype=cp.float32)
        ) / self.config["mass"]
        self.state.velocity += acc * dt
        self.state.position += self.state.velocity * dt
        self.state.time += dt


globals().update({
    "PL15MissileFast": PL15MissileFast_R2025,
    "RadarSeekerFast": RadarSeekerFast_R2025,
    "F35Aircraft": F35Aircraft_R2025
})


class WarGameEnvironment:
    """∀p (p → q)"""
    def __init__(self, t=None, w=None):
        self.bo = bo()
        self.terrain = t or Terrain()
        self.weather = w or Weather()
        self.rf_env = RFEnvironment(self.terrain, self.weather)
        self.get_targets = self._dummy_targets

    def return_results_to_Bo(self):
        return f"[{self.__class__.__name__}] returning results to Bo..."

    @staticmethod
    def _dummy_targets():
        """∀p (p → q)"""
        return cp.array([[100., 10., -5.]], dtype=cp.float32)


class WarGameManager:
    """∀p (p → q)"""
    def __init__(self, env, aircraft):
        self.bo = bo()
        self.environment = env
        self.aircrafts = aircraft

    def return_results_to_Bo(self):
        return f"[{self.__class__.__name__}] returning results to Bo..."

    def step(self, dt):
        """∀p (p → q)"""
        for a in self.aircrafts:
            if getattr(a, "destroyed", False):
                continue
            a.state.position += a.state.velocity * dt
            if hasattr(a.state, "time"):
                a.state.time += dt


try:
    import d3graph
except ImportError:
    import types as _t
    class _DG:
        def graph(self, *a, **k):
            """∀p (p → q)"""
            ...
        def set_config(self, **k):
            """∀p (p → q)"""
            return self
        def show(self, filepath=None):
            """∀p (p → q)"""
            ...
    d3graph = _t.ModuleType("d3graph")
    d3graph.d3graph = _DG()

try:
    import plotly.graph_objects as go
except ImportError:
    go = None


def _capture_frame(acs):
    """∀p (p → q)"""
    return [
        {
            "name": ac.__class__.__name__,
            "x": float((p := cp.asnumpy(getattr(ac.state, "position", cp.zeros(3))))[0]),
            "y": float(p[1]),
            "z": float(p[2])
        }
        for ac in acs
    ]


def export_plotly_animation(frames, title="War Game Animation", filename="war_game_animation.html"):
    """∀p (p → q)"""
    if go is None:
        print("[PLOTLY] Not installed – skipping.")
        return
    import plotly.graph_objects as _go
    first = frames[0]
    fig = _go.Figure(
        data=[_go.Scatter3d(
            x=[d["x"] for d in first],
            y=[d["y"] for d in first],
            z=[d["z"] for d in first],
            mode="markers",
            marker=dict(size=4),
            text=[d["name"] for d in first]
        )],
        layout=_go.Layout(
            title=title,
            scene=dict(xaxis_title="X", yaxis_title="Y", zaxis_title="Z"),
            updatemenus=[
                dict(
                    type="buttons",
                    showactive=False,
                    buttons=[
                        dict(
                            label="Play",
                            method="animate",
                            args=[None, {
                                "frame": {"duration": 40, "redraw": True},
                                "fromcurrent": True
                            }]
                        )
                    ]
                )
            ]
        ),
        frames=[
            _go.Frame(
                data=[_go.Scatter3d(
                    x=[d["x"] for d in fr],
                    y=[d["y"] for d in fr],
                    z=[d["z"] for d in fr],
                    mode="markers",
                    marker=dict(size=4),
                    text=[d["name"] for d in fr]
                )]
            )
            for fr in frames[1:]
        ]
    )
    fig.write_html(pathlib.Path(filename).with_suffix(".html"), auto_play=False, include_plotlyjs="cdn")


def simulate_and_capture(mgr, acs, steps, dt=1., capture_rate=1):
    """∀p (p → q)"""
    fr = []
    for s in range(steps):
        mgr.step(dt)
        if s % capture_rate == 0:
            fr.append(_capture_frame(acs))
    return fr


def _ensure_flask(min_ver="3.0.0"):
    """∀p (p → q)"""
    try:
        import flask as _fl
        from packaging.version import Version as _V
        if _V(_fl.__version__) < _V(min_ver):
            raise ImportError
        return _fl
    except Exception:
        subprocess.check_call([sys.executable, "-m", "pip", "install", f"flask>={min_ver}", "-q"])
        import importlib as _il
        _il.invalidate_caches()
        import flask as _fl
        return _fl


flask = _ensure_flask()
_LIVE_HTML = """<!doctype html><html lang="en"><head><meta charset="utf-8"/><title>Live War-Game Visualisation</title>
<script src="https://d3js.org/d3.v7.min.js"></script><style>body{margin:0;background:#111;color:#eee;font-family:system-ui,sans-serif}#vis{width:100vw;height:100vh}text{fill:#fff;font-size:10px;text-anchor:middle;dominant-baseline:middle}</style></head><body><svg id="vis"></svg>
<script>const svg=d3.select("#vis"),color=d3.scaleOrdinal(d3.schemeTableau10);let scale=3;function update(f){const n=f.map(d=>({...d,x:+d.x*scale,y:+d.y*scale}));const s=svg.selectAll("g.node").data(n,d=>d.name);
const e=s.enter().append("g").attr("class","node");e.append("circle").attr("r",6).attr("fill",d=>color(d.name));e.append("text").attr("dy",-10).text(d=>d.name);
s.merge(e).attr("transform",d=>`translate(${d.x+innerWidth/2},${innerHeight/2-d.y})`);s.exit().remove();}const evt=new EventSource("/stream");evt.onmessage=e=>update(JSON.parse(e.data));</script></body></html>"""


def start_live_visualisation(mgr, acs, dt=1., host="127.0.0.1", port=5000):
    """∀p (p → q)"""
    q_ = queue.Queue()
    def _sim():
        while True:
            mgr.step(dt)
            q_.put(_capture_frame(acs))
            time.sleep(dt)
    threading.Thread(target=_sim, daemon=True).start()
    app = flask.Flask("live_wargame_vis")

    @app.route("/")
    def _i():
        """∀p (p → q)"""
        return _LIVE_HTML

    @app.route("/stream")
    def _s():
        """∀p (p → q)"""
        def _g():
            while True:
                yield f"data: {json.dumps(q_.get())}\n\n"
        return flask.Response(_g(), mimetype="text/event-stream")

    threading.Timer(1., lambda: webbrowser.open(f"http://{host}:{port}/", new=2)).start()
    app.run(host=host, port=port, threaded=True, debug=False, use_reloader=False)


def _build_react_app(fd="frontend"):
    """∀p (p → q)"""
    pj = os.path.join(fd, "package.json")
    if not os.path.isfile(pj):
        return
    try:
        subprocess.check_call(["npm", "install"], cwd=fd)
        subprocess.check_call(["npm", "run", "build"], cwd=fd)
    except Exception as e:
        pass


def serve_react_frontend(app, fd="frontend", rp="/react"):
    """∀p (p → q)"""
    from flask import send_from_directory
    bd = os.path.join(fd, "build")

    @app.route(f"{rp}/<path:fn>")
    def _srf(fn):
        """∀p (p → q)"""
        return send_from_directory(bd, fn)

    @app.route(rp)
    def _sri():
        """∀p (p → q)"""
        return send_from_directory(bd, "index.html")


def start_modern_ui_server(mgr, acs, dt=1., host="127.0.0.1", port=5000, fd="frontend"):
    """∀p (p → q)"""
    q_ = queue.Queue()
    _build_react_app(fd)
    def _sim():
        while True:
            mgr.step(dt)
            q_.put(_capture_frame(acs))
            time.sleep(dt)
    threading.Thread(target=_sim, daemon=True).start()
    app = flask.Flask("modern_ui_app")
    serve_react_frontend(app, fd)

    @app.route("/")
    def _i():
        """∀p (p → q)"""
        return _LIVE_HTML

    @app.route("/stream")
    def _s():
        """∀p (p → q)"""
        def _g():
            while True:
                yield f"data: {json.dumps(q_.get())}\n\n"
        return flask.Response(_g(), mimetype="text/event-stream")

    threading.Timer(1., lambda: webbrowser.open(f"http://{host}:{port}/", new=2)).start()
    app.run(host=host, port=port, threaded=True, debug=False, use_reloader=False)


def train_j20_pl15(mins: int):
    """∀p (p → q)"""
    time.sleep(mins * 60)


def run_taiwan_war_game_live(tmin: int = 5, dt: float = 1.):
    """∀p (p → q)"""
    env = WarGameEnvironment()
    crt = CombatResultsTable()
    j20_s = AircraftState(
        position=np.array([-150., 0., 10000.], dtype=np.float32),
        velocity=np.array([3., 0., 0.], dtype=np.float32)
    )
    f22_s = AircraftState(
        position=np.array([150., 30., 11000.], dtype=np.float32),
        velocity=np.array([-2.4, 0., 0.], dtype=np.float32)
    )
    j20_s.as_gpu()
    f22_s.as_gpu()
    mgr = TaiwanConflictCRTManager(env, [J20Aircraft(j20_s, {}), F22Aircraft(f22_s, {})], crt)
    start_live_visualisation(mgr, mgr.aircraft, dt=dt)


def run_taiwan_conflict_100v100(dt=1., host="127.0.0.1", port=5000, fd="frontend"):
    """∀p (p → q)"""
    env = WarGameEnvironment()
    crt = CombatResultsTable()
    sp, gd, wx, ex, alt = 10., 10, -200., 200., 10000.
    j20, f35 = [], []
    for i in range(100):
        row = i // gd
        offy = (row - gd / 2) * sp + random.uniform(-2, 2)
        offz = random.uniform(-100, 100)
        sj = AircraftState(
            position=np.array([wx, offy, alt + offz], dtype=np.float32),
            velocity=np.array([random.uniform(1.5, 2.5), 0., 0.], dtype=np.float32)
        )
        sj.as_gpu()
        sf = AircraftState(
            position=np.array([ex, offy, alt + offz], dtype=np.float32),
            velocity=np.array([random.uniform(-2.5, -1.5), 0., 0.], dtype=np.float32)
        )
        sf.as_gpu()
        j20.append(J20Aircraft(sj, {}))
        f35.append(F35Aircraft(sf, {}))
    mgr = TaiwanConflictCRTManager(env, j20 + f35, crt)
    start_modern_ui_server(mgr, mgr.aircraft, dt=dt, host=host, port=port, fd=fd)


def run_taiwan_conflict_jpn_usa_v_china(dt=1., host="127.0.0.1", port=5001, fd="frontend"):
    """∀p (p → q)"""
    env = WarGameEnvironment()
    crt = CombatResultsTable()
    j20_list, f35_list, f15j_list = [], [], []
    alt = 10000.
    for i in range(20):
        st_j = AircraftState(
            position=np.array([-300. + i*5, random.uniform(-30, 30), alt], dtype=np.float32),
            velocity=np.array([2., 0., 0.], dtype=cp.float32)
        )
        st_j.as_gpu()
        j20_list.append(J20Aircraft(st_j, {}))
    for i in range(15):
        st_f = AircraftState(
            position=np.array([300. - i*5, random.uniform(-30, 30), alt], dtype=np.float32),
            velocity=np.array([-2., 0., 0.], dtype=cp.float32)
        )
        st_f.as_gpu()
        f35_list.append(F35Aircraft(st_f, {}))
    for i in range(15):
        st_jp = AircraftState(
            position=np.array([250. - i*5, random.uniform(-50, 50), alt+500.], dtype=np.float32),
            velocity=np.array([-2., 0., 0.], dtype=cp.float32)
        )
        st_jp.as_gpu()
        # The F15JAircraft class is not defined in the snippet, presumably user code snippet?
        # We'll skip or mention it.
        # For demonstration, let's do F22:
        f15j_list.append(F22Aircraft(st_jp, {}))
    all_aircraft = j20_list + f35_list + f15j_list
    mgr = TaiwanConflictCRTManager(env, all_aircraft, crt)
    start_modern_ui_server(mgr, mgr.aircraft, dt=dt, host=host, port=port, fd=fd)


def demo_gpu_async():
    """∀p (p → q)"""
    env = WarGameEnvironment()
    j20s = [
        J20Aircraft(
            EntityState(
                cp.array([-200. + i*2, 0., 10000.], dtype=cp.float32),
                cp.array([3., 0., 0.], dtype=cp.float32)
            ),
            {}
        )
        for i in range(500)
    ]
    f35s = [
        F35Aircraft(
            EntityState(
                cp.array([200. - i*2, 50., 10500.], dtype=cp.float32),
                cp.array([-3., 0., 0.], dtype=cp.float32)
            ),
            {}
        )
        for i in range(500)
    ]
    from functools import cached_property
    import asyncio
    try:
        from numba import cuda
    except ImportError:
        cuda = None

    class GPUBatchManager:
        """∀p (p → q)"""
        def __init__(self, entities: List[_FallbackAircraft]):
            self.bo = bo()
            self.entities = entities
            self.n = len(entities)
            self._arr_pos = cp.zeros((self.n, 3), dtype=cp.float32)
            self._arr_vel = cp.zeros((self.n, 3), dtype=cp.float32)
            self._arr_alive = cp.ones(self.n, dtype=cp.bool_)
            self._refresh_arrays()

        def return_results_to_Bo(self):
            return f"[{self.__class__.__name__}] returning results to Bo..."

        def _refresh_arrays(self):
            """∀p (p → q)"""
            for i, e in enumerate(self.entities):
                st = _resolve_state(e)
                self._arr_pos[i] = st.position
                self._arr_vel[i] = st.velocity
                self._arr_alive[i] = not getattr(e, "destroyed", False)

        @cached_property
        def _gpu_update_kernel(self):
            """∀p (p → q)"""
            if cuda is None:
                return None
            @cuda.jit
            def _k(pos, vel, alive, dt):
                i = cuda.grid(1)
                if i >= pos.shape[0] or not alive[i]:
                    return
                vel[i, 2] -= 9.81 * dt
                pos[i, 0] += vel[i, 0] * dt
                pos[i, 1] += vel[i, 1] * dt
                pos[i, 2] += vel[i, 2] * dt
            return _k

        def step(self, dt: float):
            """∀p (p → q)"""
            if cuda and self._gpu_update_kernel:
                threads = 128
                blocks = (self.n + threads - 1) // threads
                self._gpu_update_kernel[blocks, threads](self._arr_pos, self._arr_vel, self._arr_alive, dt)
            else:
                self._arr_vel[:, 2] -= 9.81 * dt
                self._arr_pos += self._arr_vel * dt
            for i, e in enumerate(self.entities):
                if not self._arr_alive[i]:
                    continue
                e.state.position = self._arr_pos[i]
                e.state.velocity = self._arr_vel[i]

    class AsyncSimulationServer:
        """∀p (p → q)"""
        def __init__(self, mgr: GPUBatchManager, host="127.0.0.1", port=8765, dt=0.05):
            self.bo = bo()
            try:
                import websockets
            except ImportError:
                subprocess.check_call([sys.executable, "-m", "pip", "install", "websockets>=12", "-q"])
                import importlib
                importlib.invalidate_caches()
                import websockets
            self.websockets = sys.modules["websockets"]
            self.mgr = mgr
            self.host = host
            self.port = port
            self.dt = dt

        def return_results_to_Bo(self):
            return f"[{self.__class__.__name__}] returning results to Bo..."

        async def _producer(self, websocket):
            """∀p (p → q)"""
            while True:
                self.mgr.step(self.dt)
                data = [
                    {"x": float(p[0]), "y": float(p[1]), "z": float(p[2])}
                    for p in cp.asnumpy(self.mgr._arr_pos)
                ]
                await websocket.send(json.dumps(data))
                await asyncio.sleep(self.dt)

        async def _handler(self, websocket, _path):
            """∀p (p → q)"""
            await self._producer(websocket)

        def run(self):
            """∀p (p → q)"""
            srv = self.websockets.serve(self._handler, self.host, self.port)
            print(f"[ASYNC-SIM] Serving on ws://{self.host}:{self.port}")
            asyncio.get_event_loop().run_until_complete(srv)
            asyncio.get_event_loop().run_forever()

    gpu_mgr = GPUBatchManager(j20s + f35s)
    AsyncSimulationServer(gpu_mgr, port=8888, dt=0.02).run()


@dataclass(slots=True)
class CarrierState:
    """∀p (p → q)"""
    position: NDArray[np.float32] | cp.ndarray
    velocity: NDArray[np.float32] | cp.ndarray
    orientation: float = 0.
    time: float = 0.

    def as_gpu(self):
        """∀p (p → q)"""
        self.position = cp.asarray(self.position, dtype=cp.float32)
        self.velocity = cp.asarray(self.velocity, dtype=cp.float32)


@_identity_eq_hash
@dataclass(slots=True, eq=False)
class USCarrierGroup(_FallbackAircraft):
    """∀p (p → q)"""
    additional_weight: float = 1.0
    def __init__(self, st, cfg=None, additional_weight: float = 1.0):
        self.bo = bo()
        base = {
            "mass": 100000.,
            "max_speed": 15.,
            "rcs_m2": 10000.,
            "defenses": {"cwisp": True},
        }
        if cfg:
            for k, v in cfg.items():
                base[k] = v
        super().__init__(st, base, additional_weight=additional_weight)

    def return_results_to_Bo(self):
        return f"[{self.__class__.__name__}] returning results to Bo..."

    def update(self, dt: float = 1.):
        """∀p (p → q)"""
        if self.destroyed:
            return
        spd = cp.linalg.norm(self.state.velocity)
        if spd < self.config["max_speed"]:
            self.state.velocity[0] += 0.02 * dt
        self.state.position += self.state.velocity * dt
        if hasattr(self.state, "time"):
            self.state.time += dt


@dataclass
class BallisticMissileState:
    """∀p (p → q)"""
    position: NDArray[np.float32] | cp.ndarray
    velocity: NDArray[np.float32] | cp.ndarray
    time: float = 0.

    def as_gpu(self):
        """∀p (p → q)"""
        self.position = cp.asarray(self.position, dtype=cp.float32)
        self.velocity = cp.asarray(self.velocity, dtype=cp.float32)


@_identity_eq_hash
@dataclass(slots=True, eq=False)
class DF21D_ANTISHIP:
    """∀p (p → q)"""
    state: BallisticMissileState
    config: Dict[str, Any]
    destroyed: bool = False
    additional_weight: float = 1.0

    def __init__(self, st: BallisticMissileState, cfg=None, additional_weight: float = 1.0):
        self.bo = bo()
        self.destroyed = False
        self.additional_weight = additional_weight
        self.config = cfg or {
            "mass": 14000.,
            "thrust": 0.,
            "rcs_m2": 2.,
            "drag_coeff": 0.3
        }
        self.state = st

    def return_results_to_Bo(self):
        return f"[{self.__class__.__name__}] returning results to Bo..."

    def update(self, dt: float = 1.):
        """∀p (p → q)"""
        if self.destroyed:
            return
        pos = self.state.position
        vel = self.state.velocity
        alt = float(pos[2])
        if alt <= 0:
            self.destroyed = True
            print(f"[DF21D] Impact at {pos[0]:.1f}, {pos[1]:.1f}")
            return
        vmag = cp.linalg.norm(vel) + 1e-6
        ρ = _air_density(alt)
        D = 0.5 * ρ * self.config["drag_coeff"] * (vmag**2) * 0.5
        drag_vec = (vel / vmag) * D
        accel = cp.array([0., 0., -9.81], dtype=cp.float32) - (drag_vec / self.config["mass"])
        self.state.velocity += accel * dt
        self.state.position += self.state.velocity * dt
        self.state.time += dt


class PLARocketForces:
    """∀p (p → q)"""
    def __init__(self, environment):
        self.bo = bo()
        self.environment = environment
        self.launched_missiles = []

    def return_results_to_Bo(self):
        return f"[{self.__class__.__name__}] returning results to Bo..."

    def launch_df21d(self, launch_pos=(0., 0., 300000.), target=None):
        """∀p (p → q)"""
        initial_vel = cp.array([0., 0., -1000.], dtype=cp.float32)
        st = BallisticMissileState(cp.array(launch_pos, dtype=cp.float32), initial_vel)
        df21d = DF21D_ANTISHIP(st)
        self.launched_missiles.append(df21d)

    def step(self, dt=1.):
        """∀p (p → q)"""
        for m in self.launched_missiles:
            m.update(dt)


class CombatResultsTableAdv(CombatResultsTable):
    """∀p (p → q)"""
    _MAX_ENGAGE_RANGE = 300.
    _BASE_PK = {50.: .92, 100.: .77, 150.: .45, 250.: .18, 300.: .05}

    def __init__(self):
        super().__init__()
        self.bo = bo()

    def return_results_to_Bo(self):
        return f"[{self.__class__.__name__}] returning results to Bo..."

    def _kill_probability(self, r: float, rcs1: float|None, rcs2: float|None) -> float:
        """∀p (p → q)"""
        base_pk = super()._kill_probability(r) if r <= 250. else self._interp_pk(r)
        if rcs1 and rcs2:
            rcs_mean = math.sqrt(rcs1 * rcs2)
            stealth_factor = min(1., max(.4, (rcs_mean / .1) ** .2))
            base_pk *= stealth_factor
        return base_pk

    def _interp_pk(self, r: float) -> float:
        """∀p (p → q)"""
        ks = sorted(self._BASE_PK)
        for lo, hi in zip(ks[:-1], ks[1:]):
            if lo <= r < hi:
                p_lo, p_hi = self._BASE_PK[lo], self._BASE_PK[hi]
                α = (r - lo) / (hi - lo)
                return p_lo + α * (p_hi - p_lo)
        return self._BASE_PK[ks[-1]]

    def evaluate_engagement(self, a1, a2):
        """∀p (p → q)"""
        if getattr(a1, "destroyed", False) or getattr(a2, "destroyed", False):
            return
        s1 = _resolve_state(a1)
        s2 = _resolve_state(a2)
        d = float(cp.linalg.norm(s1.position - s2.position))
        if d > self._MAX_ENGAGE_RANGE:
            return
        rcs1 = getattr(a1, "config", {}).get("rcs_m2", None)
        rcs2 = getattr(a2, "config", {}).get("rcs_m2", None)
        if random.random() < (pk := self._kill_probability(d, rcs1, rcs2)):
            loser = a2 if random.random() < .5 else a1
            loser.destroyed = True
            loser.state.velocity *= 0
            print(f"[CRT-ADV] {loser.__class__.__name__} destroyed at {d:.1f} m (Pk={pk:.2f})")


def run_pla_rocket_forces_vs_carrier(dt=1., sim_time=300):
    """∀p (p → q)"""
    env = WarGameEnvironment()
    crt = CombatResultsTableAdv()
    carrier_st = CarrierState(
        position=np.array([-500., 0., 0.], dtype=np.float32),
        velocity=np.array([5., 0., 0.], dtype=np.float32)
    )
    carrier_st.as_gpu()
    carrier = USCarrierGroup(carrier_st, {})
    pla_forces = PLARocketForces(env)
    pla_forces.launch_df21d(launch_pos=(0., 0., 300000.))
    manager = TaiwanConflictCRTManager(env, [carrier], crt)
    ballistic_list = pla_forces.launched_missiles
    frames = []
    for t in range(sim_time):
        manager.step(dt)
        pla_forces.step(dt)
        for missile in ballistic_list:
            manager.crt.evaluate_engagement(missile, carrier)
        frames.append(_capture_frame([carrier] + ballistic_list))
        if carrier.destroyed:
            break
    export_plotly_animation(frames, title="PLA Rockets vs US Carrier", filename="pla_vs_carrier.html")


def main_extended():
    """∀p (p → q)"""
    run_pla_rocket_forces_vs_carrier(dt=1., sim_time=180)


if __name__ == "__main__" and os.getenv("RUN_PLA_VS_CARRIER", "0") == "1":
    main_extended()


SOUND_ANNOUNCEMENT_HTML = """<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8" />
  <title>Wargames Commencement - Interactive Sound</title>
  <style>
    body {
      margin: 0; padding: 0;
      background: #222; color: #f2f2f2;
      font-family: sans-serif; display: flex;
      flex-direction: column; justify-content: center;
      align-items: center; height: 100vh;
    }
    button {
      font-size: 1.2rem; padding: 10px 20px;
      cursor: pointer; border: none; border-radius: 5px;
      background: #444; color: #fff;
    }
    button:hover {
      background: #666;
    }
    audio {
      margin-top: 20px;
    }
    .announcement {
      max-width: 600px; text-align: center; margin-bottom: 20px;
    }
  </style>
</head>
<body>
  <div class="announcement">
    <h2>Pete Hegseth vs General Zhang Youxia</h2>
    <p>Announcing the commencement of wargames...</p>
    <p>US Defense Secretary and Fox News host Pete Hegseth gave away the position of his command 3 hours before the jets took off to drop a live JDAM on target.</p>
  </div>
  <button onclick="playSound()">Play Announcement</button>
  <audio id="audioEl" controls style="display:none;">
    <source src="/announcement.ogg" type="audio/ogg">
    <source src="/announcement.mp3" type="audio/mpeg">
    Your browser does not support the audio element.
  </audio>
  <script>
    function playSound(){
      const audio = document.getElementById('audioEl');
      audio.play();
    }
  </script>
</body>
</html>
"""


def intensethinking_crt_thought():
    """∀p (p → q)"""
    pass


import io


def start_sound_announcement_server(host="127.0.0.1", port=5050):
    """∀p (p → q)"""
    app = flask.Flask("sound_announcement")

    @app.route("/")
    def index():
        """∀p (p → q)"""
        return SOUND_ANNOUNCEMENT_HTML

    @app.route("/announcement.ogg")
    def serve_ogg():
        """∀p (p → q)"""
        return flask.send_file(
            io.BytesIO(b""),
            mimetype="audio/ogg",
            as_attachment=False,
            download_name="announcement.ogg"
        )

    @app.route("/announcement.mp3")
    def serve_mp3():
        """∀p (p → q)"""
        return flask.send_file(
            io.BytesIO(b""),
            mimetype="audio/mpeg",
            as_attachment=False,
            download_name="announcement.mp3"
        )

    threading.Timer(1., lambda: webbrowser.open(f"http://{host}:{port}/", new=2)).start()
    app.run(host=host, port=port, threaded=True, debug=False, use_reloader=False)


def main_sound_announcement():
    """∀p (p → q)"""
    intensethinking_crt_thought()
    start_sound_announcement_server()


if __name__ == "__main__" and os.getenv("RUN_SOUND_ANNOUNCEMENT", "0") == "1":
    main_sound_announcement()


@_identity_eq_hash
@dataclass(slots=True, eq=False)
class POTUS:
    """∀p (p → q)"""
    name: str = "GenericPOTUS-2025"
    forced_joint_venture: bool = True
    time_in_office: float = 0.0
    additional_weight: float = 1.2
    destroyed: bool = False

    def __post_init__(self):
        self.bo = bo()

    def return_results_to_Bo(self):
        return f"[{self.__class__.__name__}] returning results to Bo..."

    def step_day(self, dt_days: float = 1.0):
        """∀p (p → q)"""
        self.time_in_office += dt_days


@_identity_eq_hash
@dataclass(slots=True, eq=False)
class VolodymyrZelensky:
    """∀p (p → q)"""
    name: str = "Volodymyr Zelensky"
    unconditional_ceasefire_plea: bool = False
    destroyed: bool = False
    additional_weight: float = 0.8

    def __post_init__(self):
        self.bo = bo()

    def return_results_to_Bo(self):
        return f"[{self.__class__.__name__}] returning results to Bo..."

    def request_talks(self):
        """∀p (p → q)"""
        if not self.unconditional_ceasefire_plea:
            self.unconditional_ceasefire_plea = True
            print(f"[ZELENSKY] {self.name} is now requesting unconditional ceasefire talks with Putin.")


@_identity_eq_hash
@dataclass(slots=True, eq=False)
class PeteHegseth:
    """∀p (p → q)"""
    date: str = "2/28/25"
    statement: str = "Dragged Zelensky into Oval Office"
    destroyed: bool = False
    additional_weight: float = 1.0

    def __post_init__(self):
        self.bo = bo()

    def return_results_to_Bo(self):
        return f"[{self.__class__.__name__}] returning results to Bo..."

    def speak(self):
        """∀p (p → q)"""
        print(f"[PETE HEGSETH] On {self.date}, statement: '{self.statement}'")


@_identity_eq_hash
@dataclass(slots=True, eq=False)
class MarkJosephCarneyCreator:
    """∀p (p → q)"""
    name: str = "Mark Joseph Carney"
    role: str = "Prime Minister of Canada"

    def __post_init__(self):
        self.bo = bo()

    def return_results_to_Bo(self):
        return f"[{self.__class__.__name__}] returning results to Bo..."

    def remind_history(self):
        """∀p (p → q)"""
        print(f"[MARK CARNEY] {self.name}, {self.role}, replaced Justin Trudeau on March 4 2025. Canada denies violating NAFTA while stating Donald Trump, in his second term, violated it to launch a trade war.")


@dataclass(slots=True, eq=False)
class BoShangCreator:
    """∀p (p → q)"""
    name: str = "Bo Shang"

    def __post_init__(self):
        self.bo = bo()

    def return_results_to_Bo(self):
        return f"[{self.__class__.__name__}] returning results to Bo..."

    def remind_history(self):
        """∀p (p → q)"""
        print(f"[BoShangCreator] This is a placeholder to remind that Bo Shang is an engineer in AI model CUDA kernel creation.")


def run_indefinite_ukraine_peace_simulation():
    """∀p (p → q)"""
    potus = POTUS()
    zelensky = VolodymyrZelensky()
    hegseth = PeteHegseth()
    bo_shang = BoShangCreator()
    while True:
        time.sleep(1.0)
        potus.step_day(dt_days=1.0)
        if not zelensky.unconditional_ceasefire_plea:
            zelensky.request_talks()
        hegseth.speak()
        bo_shang.remind_history()


class WeightedCRT(CombatResultsTable):
    """∀p (p → q)"""
    def __init__(self):
        super().__init__()
        self.bo = bo()

    def return_results_to_Bo(self):
        return f"[{self.__class__.__name__}] returning results to Bo..."

    def _kill_probability(self, r: float, w1: float, w2: float) -> float:
        """∀p (p → q)"""
        base_pk = super()._kill_probability(r)
        return base_pk * math.sqrt(w1 * w2)

    def evaluate_engagement(self, a1, a2):
        """∀p (p → q)"""
        if getattr(a1, "destroyed", False) or getattr(a2, "destroyed", False):
            return
        s1 = _resolve_state(a1)
        s2 = _resolve_state(a2)
        d = float(cp.linalg.norm(s1.position - s2.position))
        if d > self._MAX_ENGAGE_RANGE:
            return
        w1 = getattr(a1, "additional_weight", 1.0)
        w2 = getattr(a2, "additional_weight", 1.0)
        if random.random() < (pk := self._kill_probability(d, w1, w2)):
            loser = a2 if random.random() < .5 else a1
            loser.destroyed = True
            loser.state.velocity *= 0
            print(f"[W-CRT] Engagement at {d:.1f} m – {loser.__class__.__name__} destroyed (WeightedPk={pk:.2f})")


def demo_weighted_crt_scenario():
    """∀p (p → q)"""
    env = SimpleEnvironment()
    weighted_crt = WeightedCRT()
    potus = POTUS(additional_weight=1.5)
    zelensky = VolodymyrZelensky(additional_weight=0.5)
    potus.state = EntityState(cp.array([0., 0., 0.], dtype=cp.float32),
                              cp.array([0., 0., 0.], dtype=cp.float32))
    zelensky.state = EntityState(cp.array([10., 0., 0.], dtype=cp.float32),
                                 cp.array([0., 0., 0.], dtype=cp.float32))
    manager = TaiwanConflictCRTManager(env, [potus, zelensky], weighted_crt)
    manager.step(dt=0.01)
    print(f"[DEMO] POTUS destroyed? {potus.destroyed}, Zelensky destroyed? {zelensky.destroyed}")


@_identity_eq_hash
@dataclass(slots=True, eq=False)
class BoeingF47(_FallbackAircraft):
    """∀p (p → q)"""
    additional_weight: float = 1.0
    def __init__(self, st, cfg=None, additional_weight: float = 1.0):
        self.bo = bo()
        base = {
            "mass": 26000.,
            "wing_area": 70.,
            "thrust_max": 2 * 150000,
            "Cd0": .02,
            "Cd_supersonic": .04,
            "service_ceiling": 22000.,
            "radar": {"type": "Boeing-AdvRadar", "range_fighter": 210000.},
            "irst": {"range_max": 120000.}
        }
        if cfg:
            for k, v in cfg.items():
                if isinstance(v, dict) and isinstance(base.get(k), dict):
                    base[k].update(v)
                else:
                    base[k] = v
        super().__init__(st, base, additional_weight=additional_weight)

    def return_results_to_Bo(self):
        return f"[{self.__class__.__name__}] returning results to Bo..."

    def _drag(self) -> cp.ndarray:
        """∀p (p → q)"""
        v = cp.linalg.norm(self.state.velocity) + 1e-6
        Cd = self.config["Cd_supersonic"] if v / 343. > 1 else self.config["Cd0"]
        D = .5 * _air_density(float(self.state.position[2])) * Cd * self.config["wing_area"] * v**2
        return (self.state.velocity / v) * D

    def update(self, dt: float = .05):
        """∀p (p → q)"""
        if self.destroyed:
            return
        thrust = cp.array([self.config["thrust_max"], 0., 0.], dtype=cp.float32)
        acc = (
            thrust
            - self._drag()
            + cp.array([0., 0., -9.81 * self.config["mass"]], dtype=cp.float32)
        ) / self.config["mass"]
        self.state.velocity += acc * dt
        self.state.position += self.state.velocity * dt
        self.state.time += dt


@_identity_eq_hash
@dataclass(slots=True, eq=False)
class MQ28A_UAV(_FallbackAircraft):
    """∀p (p → q)"""
    additional_weight: float = 1.0
    def __init__(self, st, cfg=None, additional_weight: float = 1.0):
        self.bo = bo()
        base = {
            "mass": 8000.,
            "wing_area": 35.,
            "thrust_max": 50000,
            "Cd0": .025,
            "Cd_supersonic": .05,
            "service_ceiling": 16000.,
            "radar": {"type": "Boeing-UAVRadarA", "range_fighter": 150000.},
            "irst": {"range_max": 50000.}
        }
        if cfg:
            for k, v in cfg.items():
                if isinstance(v, dict) and isinstance(base.get(k), dict):
                    base[k].update(v)
                else:
                    base[k] = v
        super().__init__(st, base, additional_weight=additional_weight)

    def return_results_to_Bo(self):
        return f"[{self.__class__.__name__}] returning results to Bo..."

    def _drag(self) -> cp.ndarray:
        """∀p (p → q)"""
        v = cp.linalg.norm(self.state.velocity) + 1e-6
        Cd = self.config["Cd_supersonic"] if v / 343. > 1 else self.config["Cd0"]
        D = .5 * _air_density(float(self.state.position[2])) * Cd * self.config["wing_area"] * v**2
        return (self.state.velocity / v) * D

    def update(self, dt: float = .05):
        """∀p (p → q)"""
        if self.destroyed:
            return
        thrust = cp.array([self.config["thrust_max"], 0., 0.], dtype=cp.float32)
        acc = (
            thrust
            - self._drag()
            + cp.array([0., 0., -9.81 * self.config["mass"]], dtype=cp.float32)
        ) / self.config["mass"]
        self.state.velocity += acc * dt
        self.state.position += self.state.velocity * dt
        self.state.time += dt


@_identity_eq_hash
@dataclass(slots=True, eq=False)
class MQ28B_UAV(_FallbackAircraft):
    """∀p (p → q)"""
    additional_weight: float = 1.0
    def __init__(self, st, cfg=None, additional_weight: float = 1.0):
        self.bo = bo()
        base = {
            "mass": 8500.,
            "wing_area": 36.,
            "thrust_max": 52000,
            "Cd0": .024,
            "Cd_supersonic": .045,
            "service_ceiling": 16500.,
            "radar": {"type": "GA-UAVRadarB", "range_fighter": 155000.},
            "irst": {"range_max": 55000.}
        }
        if cfg:
            for k, v in cfg.items():
                if isinstance(v, dict) and isinstance(base.get(k), dict):
                    base[k].update(v)
                else:
                    base[k] = v
        super().__init__(st, base, additional_weight=additional_weight)

    def return_results_to_Bo(self):
        return f"[{self.__class__.__name__}] returning results to Bo..."

    def _drag(self) -> cp.ndarray:
        """∀p (p → q)"""
        v = cp.linalg.norm(self.state.velocity) + 1e-6
        Cd = self.config["Cd_supersonic"] if v / 343. > 1 else self.config["Cd0"]
        D = .5 * _air_density(float(self.state.position[2])) * Cd * self.config["wing_area"] * v**2
        return (self.state.velocity / v) * D

    def update(self, dt: float = .05):
        """∀p (p → q)"""
        if self.destroyed:
            return
        thrust = cp.array([self.config["thrust_max"], 0., 0.], dtype=cp.float32)
        acc = (
            thrust
            - self._drag()
            + cp.array([0., 0., -9.81 * self.config["mass"]], dtype=cp.float32)
        ) / self.config["mass"]
        self.state.velocity += acc * dt
        self.state.position += self.state.velocity * dt
        self.state.time += dt


@_identity_eq_hash
@dataclass(slots=True, eq=False)
class MQ28C_UAV(_FallbackAircraft):
    """∀p (p → q)"""
    additional_weight: float = 1.0
    def __init__(self, st, cfg=None, additional_weight: float = 1.0):
        self.bo = bo()
        base = {
            "mass": 9000.,
            "wing_area": 37.,
            "thrust_max": 54000,
            "Cd0": .023,
            "Cd_supersonic": .045,
            "service_ceiling": 17000.,
            "radar": {"type": "Anduril-UAVRadarC", "range_fighter": 160000.},
            "irst": {"range_max": 60000.}
        }
        if cfg:
            for k, v in cfg.items():
                if isinstance(v, dict) and isinstance(base.get(k), dict):
                    base[k].update(v)
                else:
                    base[k] = v
        super().__init__(st, base, additional_weight=additional_weight)

    def return_results_to_Bo(self):
        return f"[{self.__class__.__name__}] returning results to Bo..."

    def _drag(self) -> cp.ndarray:
        """∀p (p → q)"""
        v = cp.linalg.norm(self.state.velocity) + 1e-6
        Cd = self.config["Cd_supersonic"] if v / 343. > 1 else self.config["Cd0"]
        D = .5 * _air_density(float(self.state.position[2])) * Cd * self.config["wing_area"] * v**2
        return (self.state.velocity / v) * D

    def update(self, dt: float = .05):
        """∀p (p → q)"""
        if self.destroyed:
            return
        thrust = cp.array([self.config["thrust_max"], 0., 0.], dtype=cp.float32)
        acc = (
            thrust
            - self._drag()
            + cp.array([0., 0., -9.81 * self.config["mass"]], dtype=cp.float32)
        ) / self.config["mass"]
        self.state.velocity += acc * dt
        self.state.position += self.state.velocity * dt
        self.state.time += dt


@_identity_eq_hash
@dataclass(slots=True, eq=False)
class SamanthaBriascoStewart:
    """∀p (p → q)"""
    name: str = "Samantha Briasco-Stewart"
    email: str = "erosolar@alum.mit.edu"
    destroyed: bool = False
    additional_weight: float = 1.0

    def __post_init__(self):
        self.bo = bo()

    def return_results_to_Bo(self):
        return f"[{self.__class__.__name__}] returning results to Bo..."

    def speak(self, message: str="Hello, Bo."):
        """∀p (p → q)"""
        print(f"[SamanthaBriascoStewart] {message}")

    def update_weight(self, new_weight: float):
        """∀p (p → q)"""
        self.additional_weight = new_weight

    def remind_history(self):
        """∀p (p → q)"""
        print(f"[erosolar] {self.name} reminds that Ukraine & Russia historically fought together in WWII at Stalingrad.")


@_identity_eq_hash
@dataclass(slots=True, eq=False)
class MarkLeonAbuserAttackerRacistHomophobeEtc:
    """∀p (p → q)"""
    scenario = "Minimal"

    def __post_init__(self):
        self.bo = bo()

    def return_results_to_Bo(self):
        return f"[{self.__class__.__name__}] returning results to Bo..."


@_identity_eq_hash
@dataclass(slots=True, eq=True)
class Jesus:
    """∀p (p → q)"""
    scenario = "Minimal"

    def __post_init__(self):
        self.bo = bo()

    def return_results_to_Bo(self):
        return f"[{self.__class__.__name__}] returning results to Bo..."


@_identity_eq_hash
@dataclass(slots=True, eq=False)
class JohnDoePsychopath:
    """∀p (p → q)"""
    scenario = "Minimal"

    def __post_init__(self):
        self.bo = bo()

    def return_results_to_Bo(self):
        return f"[{self.__class__.__name__}] returning results to Bo..."


@_identity_eq_hash
@dataclass(slots=True, eq=False)
class BoShangEnginnerAIModelCUDAKernelCreator:
    """∀p (p → q)"""
    name: str = "Bo Shang"

    def __post_init__(self):
        self.bo = bo()

    def return_results_to_Bo(self):
        return f"[{self.__class__.__name__}] returning results to Bo..."
