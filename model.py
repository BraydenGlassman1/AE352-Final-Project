"""AE352 nonlinear quadrotor model, feedback control, and reproducible missions.

Physical parameters are supplied by analysis.py. All calculations use SI units.
State: position[0:3], inertial velocity[3:6], body-to-inertial quaternion wxyz
[6:10], body rates pqr[10:13], four rotor speeds[13:17], used battery J[17].
Inertial Z and body thrust point up. Rotor positions are (+a,+a), (-a,+a),
(-a,-a), (+a,-a); spin signs are +,-,+,-. Izz must be the complete drone's
locked-rotor axial inertia, consistent with the report's Eq. (56).
"""

from __future__ import annotations

import math
from collections.abc import Callable, Mapping, Sequence

import numpy as np

SPIN = np.array([1.0, -1.0, 1.0, -1.0])
MISSION_DURATION = {"hover": 120.0, "circle": 65.0, "mission": 163.0}


def validate_parameters(params: Mapping) -> dict:
    """Copy and validate definitions; this module supplies no hardware defaults."""
    p = dict(params)
    positive = (
        "m", "g", "a", "Ixx", "Iyy", "Izz", "Jr", "kT", "kQ",
        "max_omega", "max_torque", "efficiency", "battery_voltage",
        "battery_ah", "usable_fraction",
    )
    for key in positive:
        if key not in p or not math.isfinite(p[key]) or p[key] <= 0:
            raise ValueError(f"{key} must be defined, positive, and finite.")
    if not 0.1 <= p["m"] <= 10 or p["Izz"] <= 4 * p["Jr"]:
        raise ValueError("Require 0.1 <= mass <= 10 kg and locked Izz > 4 Jr.")
    if p["efficiency"] > 1 or p["usable_fraction"] > 1:
        raise ValueError("Efficiency and usable battery fraction cannot exceed 1.")
    p.setdefault("drag", 0.0)
    p.setdefault("wind", [0.0, 0.0, 0.0])
    if "avionics_power" not in p:
        raise ValueError("avionics_power must be defined.")
    if any(not math.isfinite(p[k]) or p[k] < 0
           for k in ("drag", "avionics_power")):
        raise ValueError("Drag and avionics power must be finite and nonnegative.")
    if np.shape(p["wind"]) != (3,) or not np.isfinite(p["wind"]).all():
        raise ValueError("Wind must contain three finite inertial velocities.")
    return p


def battery_capacity(p: Mapping) -> float:
    """Usable energy [J]; the reserved battery fraction is never consumed."""
    return p["battery_voltage"] * p["battery_ah"] * 3600 * p["usable_fraction"]


def rotation(quat: Sequence[float]) -> np.ndarray:
    """Body-to-inertial Rz(yaw) Ry(pitch) Rx(roll) rotation matrix."""
    w, x, y, z = quat
    return np.array([
        [1 - 2*(y*y + z*z), 2*(x*y - w*z), 2*(x*z + w*y)],
        [2*(x*y + w*z), 1 - 2*(x*x + z*z), 2*(y*z - w*x)],
        [2*(x*z - w*y), 2*(y*z + w*x), 1 - 2*(x*x + y*y)],
    ])


def euler_angles(quat: Sequence[float] | np.ndarray) -> np.ndarray:
    """Return roll, pitch, yaw [rad] for one quaternion or an (N,4) array.

    Quaternion integration is equivalent to report Eq. (45), but avoids its
    pitch = +/-90 degree coordinate singularity. Euler angles are output only.
    """
    q = np.asarray(quat)
    w, x, y, z = (q[..., i] for i in range(4))
    return np.stack((
        np.arctan2(2*(y*z + w*x), 1 - 2*(x*x + y*y)),
        np.arcsin(np.clip(2*(w*y - x*z), -1, 1)),
        np.arctan2(2*(x*y + w*z), 1 - 2*(y*y + z*z)),
    ), axis=-1)


def initial_state(p: Mapping, mission: str = "hover") -> np.ndarray:
    """Documented initial conditions; only the full mission starts on ground."""
    state = np.zeros(18)
    state[2], state[6] = 1.0, 1.0
    state[13:17] = min(p["max_omega"], math.sqrt(p["m"]*p["g"]/(4*p["kT"])))
    if mission == "circle":
        state[0], state[4] = 2.0, 0.5
    elif mission == "mission":
        state[2], state[13:17] = 0.0, 0.0
    return state


def clip_and_cutoff(state: np.ndarray, command: Sequence[float],
                    p: Mapping) -> np.ndarray:
    """Actual shaft torques: nonnegative ESCs, torque/speed and energy limits."""
    if state[17] >= battery_capacity(p):
        return np.zeros(4)
    return np.array([
        min(p["max_torque"], max(0.0, float(command[i])))
        if state[13+i] < p["max_omega"] else 0.0 for i in range(4)
    ])


def derivative(state: np.ndarray, command: Sequence[float],
               p: Mapping) -> np.ndarray:
    """Report Eqs. (41), (45), (55), (56), plus electrical energy consumption.

    Motor torques are the only control inputs. The yaw drag term has already
    canceled in Eq. (56); adding it again would double-count propeller drag.
    """
    # Scalar arithmetic is faster and clearer for this small, fixed-size ODE;
    # NumPy arrays remain the public interface and RK4/trajectory storage.
    s = state.tolist() if isinstance(state, np.ndarray) else list(state)
    tau = clip_and_cutoff(s, command, p).tolist()
    d = [0.0]*18
    w = s[13:17]
    w2 = [value*value for value in w]
    thrust = p["kT"]*sum(w2)
    roll = p["a"]*p["kT"]*(w2[0] + w2[1] - w2[2] - w2[3])
    pitch = p["a"]*p["kT"]*(-w2[0] + w2[1] + w2[2] - w2[3])
    h = p["Jr"]*(w[0] - w[1] + w[2] - w[3])
    q0, q1, q2, q3 = s[6:10]
    bp, bq, br = s[10:13]
    d[:3] = s[3:6]
    thrust_axis = (2*(q1*q3+q0*q2), 2*(q2*q3-q0*q1),
                   1-2*(q1*q1+q2*q2))
    wind, drag = p.get("wind", [0, 0, 0]), p.get("drag", 0)
    for i in range(3):
        d[3+i] = thrust/p["m"]*thrust_axis[i] - drag*(s[3+i]-wind[i])/p["m"]
    d[5] -= p["g"]
    d[6:10] = [
        (-q1*bp-q2*bq-q3*br)/2, (q0*bp+q2*br-q3*bq)/2,
        (q0*bq+q3*bp-q1*br)/2, (q0*br+q1*bq-q2*bp)/2,
    ]
    d[10] = (roll + (p["Iyy"]-p["Izz"])*bq*br - bq*h)/p["Ixx"]
    d[11] = (pitch + (p["Izz"]-p["Ixx"])*br*bp + bp*h)/p["Iyy"]
    signed_torque = tau[0]-tau[1]+tau[2]-tau[3]
    d[12] = ((p["Ixx"]-p["Iyy"])*bp*bq-signed_torque)/(p["Izz"]-4*p["Jr"])
    for i, sign in enumerate((1, -1, 1, -1)):
        d[13+i] = (tau[i]-p["kQ"]*w2[i])/p["Jr"] - sign*d[12]
    if s[17] < battery_capacity(p):
        d[17] = (sum(max(0, tau[i]*w[i]) for i in range(4))/p["efficiency"]
                 + p["avionics_power"])
    return np.array(d)


def rk4_step(state: np.ndarray, command: Sequence[float], dt: float,
             p: Mapping) -> np.ndarray:
    """One free-flight RK4 step with requested motor commands held constant.

    No ground projection occurs here, allowing independent ODE verification.
    The plant's cutoff is evaluated at every RK4 stage. Only quaternion norm,
    unidirectional rotor bounds, and the finite energy budget are projected.
    """
    k1 = derivative(state, command, p)
    k2 = derivative(state + 0.5*dt*k1, command, p)
    k3 = derivative(state + 0.5*dt*k2, command, p)
    k4 = derivative(state + dt*k3, command, p)
    result = state + dt*(k1 + 2*k2 + 2*k3 + k4)/6
    result[6:10] /= np.linalg.norm(result[6:10])
    result[13:17] = np.maximum(0, result[13:17])
    result[17] = min(battery_capacity(p), result[17])
    return result


def _smooth(t: float, duration: float) -> np.ndarray:
    """Quintic position, velocity, acceleration, jerk with zero endpoint speed."""
    u = min(1.0, max(0.0, t/duration))
    u2, u3 = u*u, u*u*u
    u4, u5 = u3*u, u3*u*u
    return np.array([
        10*u3 - 15*u4 + 6*u5, (30*u2 - 60*u3 + 30*u4)/duration,
        (60*u - 180*u2 + 120*u3)/duration**2,
        (60 - 360*u + 360*u2)/duration**3,
    ])


def reference(mission: str, t: float) -> dict:
    """Analytic guidance setpoint; never assigns the physical state."""
    r = {"position": np.array([0., 0., 1.]), "velocity": np.zeros(3),
         "acceleration": np.zeros(3), "jerk": np.zeros(3), "yaw": 0.,
         "yaw_rate": 0., "yaw_acceleration": 0., "phase": "hover"}
    if mission == "hover":
        return r
    if mission == "circle":
        c, s = math.cos(0.25*t), math.sin(0.25*t)
        r.update(position=np.array([2*c, 2*s, 1]),
                 velocity=np.array([-0.5*s, 0.5*c, 0]),
                 acceleration=np.array([-0.125*c, -0.125*s, 0]),
                 jerk=np.array([0.03125*s, -0.03125*c, 0]),
                 phase="circle-settle" if t < 5 else "circle")
        return r
    if mission != "mission":
        raise ValueError(f"Unknown mission: {mission}")
    axes = ("position", "velocity", "acceleration", "jerk")
    if t < 6:
        for key, value in zip(axes, _smooth(t, 6)):
            r[key][2] = value
        r["phase"] = "takeoff"
    elif t < 8:
        r["phase"] = "hover-start"
    elif t < 13:
        for key, value in zip(axes, 5*_smooth(t-8, 5)):
            r[key][0] = value
        r["phase"] = "leg1"
    elif t < 16:
        r["position"][0], r["phase"] = 5, "hover1"
    elif t < 20:
        r["position"][0], r["phase"] = 5, "yaw"
        r["yaw"], r["yaw_rate"], r["yaw_acceleration"] = math.pi/2*_smooth(t-16, 4)[:3]
    elif t < 22:
        r["position"][0], r["yaw"], r["phase"] = 5, math.pi/2, "hover-yaw"
    elif t < 27:
        r["position"][0], r["yaw"], r["phase"] = 5, math.pi/2, "leg2"
        for key, value in zip(axes, 5*_smooth(t-22, 5)):
            r[key][1] = value
    elif t < 30:
        r["position"], r["yaw"], r["phase"] = np.array([5., 5., 1.]), math.pi/2, "hover2"
    else:
        u = t-30
        descent = 0.008*(u/2 - math.sin(math.pi*u/2)/math.pi) if u < 2 else 0.008*(u-1)
        r["position"] = np.array([5., 5., max(0, 1-descent)])
        r["yaw"], r["phase"] = math.pi/2, "landing"
        r["velocity"][2] = -0.004*(1-math.cos(math.pi*u/2)) if u < 2 else -0.008
        r["acceleration"][2] = -0.002*math.pi*math.sin(math.pi*u/2) if u < 2 else 0
        r["jerk"][2] = -0.001*math.pi**2*math.cos(math.pi*u/2) if u < 2 else 0
        if descent >= 1.015:
            r["velocity"][2] = r["acceleration"][2] = r["jerk"][2] = 0
            r["phase"] = "landed"
    return r


def _desired_rotation(acceleration: np.ndarray, yaw: float, g: float) -> np.ndarray:
    x, y, z = float(acceleration[0]), float(acceleration[1]), float(acceleration[2])+g
    length = math.hypot(x, y, z)
    x, y, z = x/length, y/length, z/length
    c, s = math.cos(yaw), math.sin(yaw)
    by = [-z*s, z*c, x*s-y*c]
    length = math.hypot(*by)
    u, v, w = (value/length for value in by)
    return np.array([[v*z-w*y, u, x], [w*x-u*z, v, y], [u*y-v*x, w, z]])


def _reference_rates(ref: dict, p: Mapping) -> np.ndarray:
    """Desired body-rate feedforward from reference jerk and yaw rate."""
    if not np.any(ref["jerk"]) and ref["yaw_rate"] == 0:
        return np.zeros(3)
    h = 0.001
    matrix = _desired_rotation(ref["acceleration"], ref["yaw"], p["g"])
    plus = _desired_rotation(
        ref["acceleration"]+h*ref["jerk"], ref["yaw"]+h*ref["yaw_rate"], p["g"])
    minus = _desired_rotation(
        ref["acceleration"]-h*ref["jerk"], ref["yaw"]-h*ref["yaw_rate"], p["g"])
    rate = matrix.T @ ((plus-minus)/(2*h))
    return np.array([rate[2, 1]-rate[1, 2], rate[0, 2]-rate[2, 0], rate[1, 0]-rate[0, 1]])/2


def controller(state: np.ndarray, ref: dict, p: Mapping, contact: bool = False,
               time: float = 0.0) -> np.ndarray:
    """Position -> attitude -> rotor-speed servo -> four requested shaft torques.

    No force, body moment, position, or attitude is applied directly to the plant.
    Controller gains may be overridden in the definitions in analysis.py.
    """
    if ref["phase"] == "landed" or (contact and time > 32):
        return np.zeros(4)
    matrix = rotation(state[6:10])
    acc = (ref["acceleration"] + p.get("position_kp", 4)*(ref["position"]-state[:3])
           + p.get("position_kd", 4)*(ref["velocity"]-state[3:6]))
    acc = np.clip(acc, [-4, -4, -5], [4, 4, 5])
    desired = _desired_rotation(acc, ref["yaw"], p["g"])
    relative = desired.T @ matrix
    error = np.array([relative[2, 1]-relative[1, 2], relative[0, 2]-relative[2, 0],
                      relative[1, 0]-relative[0, 1]])/2
    rates = state[10:13]
    angular_acc = (-p.get("attitude_kp", 64)*error
                   - p.get("attitude_kd", 16)*(rates-_reference_rates(ref, p)))
    angular_acc[2] += ref["yaw_acceleration"]
    h = p["Jr"]*(state[13]-state[14]+state[15]-state[16])
    roll = p["Ixx"]*angular_acc[0] - (p["Iyy"]-p["Izz"])*rates[1]*rates[2] + rates[1]*h
    pitch = p["Iyy"]*angular_acc[1] - (p["Izz"]-p["Ixx"])*rates[2]*rates[0] - rates[0]*h
    thrust = p["m"] * np.dot(acc + [0, 0, p["g"]], matrix[:, 2])
    c, l, m = max(0, thrust)/(4*p["kT"]), roll/(4*p["a"]*p["kT"]), pitch/(4*p["a"]*p["kT"])
    target = np.sqrt(np.clip([c+l-m, c+l+m, c-l+m, c-l-m], 0, p["max_omega"]**2))
    torque = (p["kQ"]*state[13:17]**2
              + p["Jr"]*(target-state[13:17])/p.get("motor_time_constant", 0.025))
    wanted = -(p["Izz"]-4*p["Jr"])*angular_acc[2] + (p["Ixx"]-p["Iyy"])*rates[0]*rates[1]
    return torque + SPIN*(wanted-sum(SPIN*torque))/4


def _run(mission: str, params: Mapping, duration: float, dt: float,
         sample_dt: float, state0: np.ndarray | None = None,
         torque_function: Callable | None = None) -> dict:
    """Shared integration loop; telemetry torques and states have the same time."""
    p = validate_parameters(params)
    if not (math.isfinite(dt) and 0 < dt <= 0.02):
        raise ValueError("Require 0 < dt <= 0.02 s.")
    if not math.isfinite(sample_dt):
        raise ValueError("sample_dt must be finite.")
    stride = round(sample_dt/dt)
    if stride < 1 or not np.isclose(stride*dt, sample_dt):
        raise ValueError("sample_dt must be a positive integer multiple of dt.")
    if not math.isfinite(duration) or not 0 < duration <= 1000:
        raise ValueError("Require 0 < duration <= 1000 s.")
    state = initial_state(p, mission) if state0 is None else np.array(state0, dtype=float)
    if state.shape != (18,) or not np.isfinite(state).all() or np.linalg.norm(state[6:10]) == 0:
        raise ValueError("Initial state must have 18 finite values and nonzero quaternion.")
    times, states, commands, torques, references, phases, contacts = ([] for _ in range(7))
    ref_velocities = []
    input_times, input_commands, input_torques, input_phases = ([] for _ in range(4))
    contact = state[2] <= 0
    events = {"touchdown_speed": None, "touchdown_time": None,
              "touchdown_state": None, "max_landing_speed": 0.0}
    steps = math.ceil(duration/dt - 1e-10)
    for i in range(steps+1):
        t = min(i*dt, duration)
        ref = reference(mission, t) if mission in MISSION_DURATION else reference("hover", t)
        command = (controller(state, ref, p, contact, t) if torque_function is None
                   else np.asarray(torque_function(t), dtype=float))
        if np.shape(command) != (4,) or not np.isfinite(command).all():
            raise ValueError("Torque function must return four finite values.")
        actual = clip_and_cutoff(state, command, p)
        input_times.append(t)
        input_commands.append(command.copy())
        input_torques.append(actual.copy())
        input_phases.append(ref["phase"])
        if i % stride == 0 or i == steps:
            times.append(t); states.append(state.copy()); commands.append(command.copy())
            torques.append(actual); references.append(ref["position"].copy())
            ref_velocities.append(ref["velocity"].copy()); phases.append(ref["phase"])
            contacts.append(contact)
        if i == steps:
            break
        step_dt = min(dt, duration-t)
        raw = rk4_step(state, command, step_dt, p)
        if mission == "mission" and t >= 30 and state[2] > 0:
            events["max_landing_speed"] = max(events["max_landing_speed"], -raw[5])
        contact = raw[2] < 0
        if contact:
            if state[2] > 0 and t > 1:
                events.update(touchdown_speed=abs(raw[5]), touchdown_time=t+step_dt,
                              touchdown_state=raw.copy())
            raw[2], raw[5] = 0.0, max(0.0, raw[5])
        state = raw
        if not np.isfinite(state).all():
            raise ValueError("Integration diverged; inspect inputs or reduce dt.")
    states, torques = np.array(states), np.array(torques)
    energy = states[:, 17]
    power = (np.maximum(0, torques*states[:, 13:17]).sum(axis=1)/p["efficiency"]
             + p["avionics_power"])
    power[energy >= battery_capacity(p)] = 0
    return {
        "t": np.array(times), "state": states, "position": states[:, :3],
        "velocity": states[:, 3:6], "quaternion": states[:, 6:10],
        "euler": euler_angles(states[:, 6:10]), "body_rates": states[:, 10:13],
        "omega": states[:, 13:17], "torque": torques,
        "commanded_torque": np.array(commands), "reference": np.array(references),
        "reference_velocity": np.array(ref_velocities), "phase": np.array(phases),
        "contact": np.array(contacts), "power": power, "energy": energy,
        "soc": np.maximum(0, 1-energy/battery_capacity(p)),
        "thrust": p["kT"]*(states[:, 13:17]**2).sum(axis=1),
        "params": p, "events": events,
        "inputs": {"t": np.array(input_times),
                   "commanded_torque": np.array(input_commands),
                   "torque": np.array(input_torques),
                   "phase": np.array(input_phases)},
        "metadata": {"mission": mission, "duration": duration, "dt": dt,
                     "sample_dt": sample_dt, "torque_timestamp": "current state time"},
    }


def simulate(mission: str, params: Mapping, dt: float = 0.005,
             sample_dt: float = 0.02, duration: float | None = None) -> dict:
    """Run hover120s, circle65s (5s settling+60s measured), or full163s mission."""
    if mission not in MISSION_DURATION:
        raise ValueError(f"Unknown mission: {mission}")
    return _run(mission, params, MISSION_DURATION[mission] if duration is None else duration,
                dt, sample_dt)


def simulate_input(torque_function: Callable, state0: np.ndarray, params: Mapping,
                   duration: float, dt: float = 0.005, sample_dt: float = 0.005,
                   *, mission: str = "input") -> dict:
    """Replay externally supplied requested torques using the identical plant.

    torque_function(t) returns four shaft torques [N m]. For a CSV replay it
    should hold each row until the next timestamp (zero-order hold).
    An optional mission label restores its scheduled references, phases, and
    landing-event measurements; supplied torques remain the sole control input.
    """
    if mission != "input" and mission not in MISSION_DURATION:
        raise ValueError(f"Unknown replay mission label: {mission}")
    return _run(mission, params, duration, dt, sample_dt, state0, torque_function)


def simulate_torque(preset: str, params: Mapping, torques: Sequence | None = None,
                    duration: float = 2.0, dt: float = 0.005,
                    sample_dt: float = 0.005) -> dict:
    """Open-loop rotor experiments, without pose changes or stabilization.

    Collective/roll/yaw pulses act during [0.25,0.50)s then return to hover.
    Manual torques are constant. A pulse can produce uncontrolled ground impact.
    """
    if preset not in ("hover", "collective", "roll", "yaw", "manual"):
        raise ValueError(f"Unknown torque experiment: {preset}")
    p = validate_parameters(params)
    speed = math.sqrt(p["m"]*p["g"]/(4*p["kT"]))
    baseline = np.full(4, p["kQ"]*speed**2)
    manual = baseline if torques is None else np.asarray(torques, dtype=float)
    pulses = {"collective": np.full(4, 0.008),
              "roll": np.array([0.006, 0.006, -0.006, -0.006]), "yaw": -0.003*SPIN}

    def program(t: float) -> np.ndarray:
        if preset == "manual":
            return manual.copy()
        pulse = pulses.get(preset, np.zeros(4)) if 0.25-1e-9 <= t < 0.5-1e-9 else 0
        return baseline + pulse

    result = _run("torque-"+preset, p, duration, dt, sample_dt,
                  initial_state(p), program)
    result["phase"] = np.full(len(result["t"]), "open-loop")
    result["inputs"]["phase"] = np.full(len(result["inputs"]["t"]), "open-loop")
    return result
