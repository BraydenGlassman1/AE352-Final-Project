"""Definitions, verification, CSV export, and paper figures (read sections in order).

Edit DEFAULT_PARAMS here; model.py contains the equations and run.py runs the work.
Positions are metres, time seconds, angles radians, motor torques N m, energy J.
"""
from __future__ import annotations

import csv
import json
from pathlib import Path
import numpy as np

# 1. PHYSICAL DEFINITIONS: report Table 3 (14 November draft).
J_PROP = 6.99e-5              # Report's propeller-only approximation [kg m²].
J_MOTOR_ASSUMED = 1.0e-5      # Unresolved motor term; provisional, not a measured value.
DEFAULT_PARAMS = dict(
    m=1.700, g=9.80665, a=0.17678,
    Ixx=0.017661, Iyy=0.018373, Izz=0.034584,
    kT=1.06775e-5, kQ=1.52451e-7,
    Jr=J_PROP + J_MOTOR_ASSUMED,
    max_omega=7000 * 2 * np.pi / 60, max_torque=0.12,  # Provisional conservative operating bounds.
    battery_voltage=14.8, battery_ah=5.0, usable_fraction=0.8,  # PROVISIONAL 4S pack.
    efficiency=0.75, avionics_power=5.0,  # Explicit loss/load assumptions.
    drag=0.0, wind=[0.0, 0.0, 0.0],  # Neglected as stated in the report.
    position_kp=4.0, position_kd=4.0,
    attitude_kp=64.0, attitude_kd=16.0, motor_time_constant=0.025,
)

NAMES = {"hover": "Hover at 1 m", "circle": "Circular flight", "mission": "Full flight sequence"}
COLORS = ("#007D8A", "#D77527", "#5265B4", "#A64978")
# Fixed reference schedules; colors denote stages, not measured pass/fail.
STAGES = {
    "hover": [(0, 120, "Hover", "#E7F2EF")],
    "circle": [(0, 5, "Settling", "#F5EBDD"), (5, 65, "Measured circle", "#E7F2EF")],
    "mission": [
        (0, 6, "Takeoff", "#DDECF7"), (6, 8, "Hover", "#E7F2EF"),
        (8, 13, "Leg 1", "#F7E8D9"), (13, 16, "Hover", "#E7F2EF"),
        (16, 20, "Left yaw", "#EBE2F4"), (20, 22, "Hover", "#E7F2EF"),
        (22, 27, "Leg 2", "#F7E8D9"), (27, 30, "Hover", "#E7F2EF"),
        (30, 156, "Landing approach", "#F5E0E4"), (156, 163, "Contact / coast", "#ECEFF1"),
    ],
}
SNAPSHOTS = {"hover": [0, 60, 120], "circle": [5, 13, 21], "mission": [3, 6, 18, 25]}


# 2. MEASURED RESULTS: criteria are explicit numerical interpretations of the brief.
def check_run(run):
    """Measure the solution; never infer goal success from a reference curve."""
    t, x, v, angles = run["t"], run["position"], run["velocity"], run["euler"]
    p, name = run["params"], run["metadata"]["mission"]
    checks = []

    def add(label, value, limit, passed):
        checks.append(dict(name=label, value=value, limit=limit, **{"pass": bool(passed)}))

    def at(time):
        return int(np.argmin(abs(t - time)))

    energy_wh = float(run["energy"][-1] / 3600)
    budget_wh = p["battery_voltage"] * p["battery_ah"] * p["usable_fraction"]
    speed = np.linalg.norm(v, axis=1)
    position_error = np.linalg.norm(x - run["reference"], axis=1)
    add("Finite dynamic states", bool(np.isfinite(run["state"]).all()), "all finite", np.isfinite(run["state"]).all())
    add("Usable battery remains", energy_wh, f"less than {budget_wh:.4f} Wh", energy_wh < budget_wh)
    add("Mass in project bounds", p["m"], "0.1 to 10 kg", 0.1 <= p["m"] <= 10)
    metrics = dict(duration_s=float(t[-1]), energy_Wh=energy_wh, usable_energy_Wh=budget_wh,
                   usable_remaining_percent=float(run["soc"][-1] * 100),
                   sampled_peak_power_W=float(np.max(run["power"])),
                   sampled_peak_pack_current_A=float(np.max(run["power"]) / p["battery_voltage"]),
                   sampled_peak_rotor_rpm=float(np.max(run["omega"]) * 60 / (2*np.pi)),
                   sampled_peak_motor_shaft_power_W=float(np.max(run["torque"] * run["omega"])),
                   maximum_position_error_m=float(np.max(position_error)))
    if name == "hover":
        error = float(np.max(np.linalg.norm(x - [0, 0, 1], axis=1)))
        add("Hover duration", float(t[-1]), "at least 120 s", t[-1] >= 120 - 1e-8)
        add("Hover position error", error, "at most 0.02 m", error <= 0.02)
        add("Hover speed", float(max(speed)), "at most 0.03 m/s", max(speed) <= 0.03)
        metrics["hover_position_error_m"] = error
    elif name == "circle":
        mask = t >= 5 - 1e-8
        radius_error = float(np.max(abs(np.linalg.norm(x[mask, :2], axis=1) - 2)))
        speed_error = float(np.max(abs(speed[mask] - 0.5)))
        altitude_error = float(np.max(abs(x[mask, 2] - 1)))
        add("Measured circle duration", float(t[-1] - 5), "at least 60 s", t[-1] - 5 >= 60 - 1e-8)
        add("Circle radial error", radius_error, "at most 0.02 m", radius_error <= 0.02)
        add("Circle speed error", speed_error, "at most 0.01 m/s", speed_error <= 0.01)
        add("Circle altitude error", altitude_error, "at most 0.02 m", altitude_error <= 0.02)
        metrics.update(radius_error_m=radius_error, speed_error_m_s=speed_error, altitude_error_m=altitude_error)
    elif name == "mission":
        add("Ground start", float(x[0, 2]), "0 m", abs(x[0, 2]) < 1e-10)
        for leg, start, finish, axis in [(1, 8, 13, 0), (2, 22, 27, 1)]:
            i, j = at(start), at(finish)
            distance = float(x[j, axis] - x[i, axis])
            path = float(np.sum(np.linalg.norm(np.diff(x[i:j + 1], axis=0), axis=1)))
            average = path / (t[j] - t[i])
            add(f"Leg {leg} distance", distance, "5 +/- 0.05 m", abs(distance - 5) <= 0.05)
            add(f"Leg {leg} average speed", float(average), "1 +/- 0.01 m/s", abs(average - 1) <= 0.01)
            add(f"Leg {leg} straightness", float(np.max(abs(x[i:j + 1, 1 - axis] - x[i, 1 - axis]))),
                "at most 0.02 m", np.max(abs(x[i:j + 1, 1 - axis] - x[i, 1 - axis])) <= 0.02)
            metrics[f"leg{leg}_distance_m"], metrics[f"leg{leg}_average_speed_m_s"] = distance, float(average)
        yaw = float(np.degrees(angles[at(22), 2] - angles[at(16), 2]))
        cruise = float(np.max(abs(x[(t >= 8) & (t <= 30), 2] - 1)))
        landing_drift = float(np.max(np.linalg.norm(x[t >= 30, :2] - [5, 5], axis=1)))
        add("Left yaw change", yaw, "90 +/- 1 degree", abs(yaw - 90) <= 1)
        add("Cruise altitude error", cruise, "at most 0.02 m", cruise <= 0.02)
        for time, label in [(16, "Stop before yaw"), (30, "Stop before landing")]:
            add(label, float(speed[at(time)]), "at most 0.02 m/s", speed[at(time)] <= 0.02)
        descent = float(run["events"]["max_landing_speed"])
        touchdown = run["events"]["touchdown_speed"]
        add("Maximum landing descent", descent, "at most 0.01 m/s", descent <= 0.01)
        add("Precontact touchdown speed", touchdown, "at most 0.01 m/s", touchdown is not None and touchdown <= 0.01)
        add("Ground contact achieved", bool(run["contact"][-1]), "landed at Z=0", run["contact"][-1] and x[-1, 2] == 0)
        add("Vertical landing drift", landing_drift, "at most 0.02 m", landing_drift <= 0.02)
        metrics.update(left_yaw_deg=yaw, cruise_altitude_error_m=cruise,
                       maximum_landing_speed_m_s=descent, touchdown_speed_m_s=touchdown,
                       touchdown_time_s=run["events"]["touchdown_time"], landing_drift_m=landing_drift)
    return {"pass": all(c["pass"] for c in checks), "checks": checks, "metrics": metrics}


def export_results(run, folder):
    """Save sampled telemetry and every 5 ms requested/applied motor input."""
    folder = Path(folder)
    folder.mkdir(parents=True, exist_ok=True)
    name = run["metadata"]["mission"]
    with (folder / f"{name}_telemetry.csv").open("w", newline="", encoding="utf-8") as file:
        writer = csv.writer(file)
        writer.writerow(["t_s", "phase", "x_m", "y_m", "z_m", "vx_m_s", "vy_m_s", "vz_m_s",
                         "roll_rad", "pitch_rad", "yaw_rad", *[f"omega_{i}_rad_s" for i in range(1, 5)],
                         *[f"torque_{i}_Nm" for i in range(1, 5)], "power_W", "energy_J", "usable_soc",
                         "qw", "qx", "qy", "qz", "p_rad_s", "q_rad_s", "r_rad_s",
                         "ref_x_m", "ref_y_m", "ref_z_m", "ref_vx_m_s", "ref_vy_m_s", "ref_vz_m_s",
                         "thrust_N", "contact", *[f"command_{i}_Nm" for i in range(1, 5)]])
        for i, time in enumerate(run["t"]):
            writer.writerow([time, run["phase"][i], *run["position"][i], *run["velocity"][i], *run["euler"][i],
                             *run["omega"][i], *run["torque"][i], run["power"][i], run["energy"][i], run["soc"][i],
                             *run["quaternion"][i], *run["body_rates"][i],
                             *run["reference"][i], *run["reference_velocity"][i],
                             run["thrust"][i], int(run["contact"][i]), *run["commanded_torque"][i]])
    inputs = run["inputs"]
    with (folder / f"{name}_motor_inputs.csv").open("w", newline="", encoding="utf-8") as file:
        writer = csv.writer(file)
        writer.writerow(["t_s", "phase", *[f"command_{i}_Nm" for i in range(1, 5)],
                         *[f"torque_{i}_Nm" for i in range(1, 5)]])
        for i, time in enumerate(inputs["t"]):
            writer.writerow([time, inputs["phase"][i], *inputs["commanded_torque"][i], *inputs["torque"][i]])
    save_json({key: run[key] for key in ("params", "events", "metadata")}, folder / f"{name}_metadata.json")


def load_results(folder, name):
    """Rebuild plotting data from the saved solution, without re-running physics."""
    folder = Path(folder)
    run = json.loads((folder / f"{name}_metadata.json").read_text(encoding="utf-8"))
    with (folder / f"{name}_telemetry.csv").open(newline="", encoding="utf-8") as file:
        rows = list(csv.DictReader(file))
    def columns(*names):
        return np.array([[float(row[key]) for key in names] for row in rows])
    run.update(t=columns("t_s")[:, 0], phase=np.array([r["phase"] for r in rows]),
               position=columns("x_m", "y_m", "z_m"), velocity=columns("vx_m_s", "vy_m_s", "vz_m_s"),
               euler=columns("roll_rad", "pitch_rad", "yaw_rad"), quaternion=columns("qw", "qx", "qy", "qz"),
               body_rates=columns("p_rad_s", "q_rad_s", "r_rad_s"),
               omega=columns(*[f"omega_{i}_rad_s" for i in range(1, 5)]),
               torque=columns(*[f"torque_{i}_Nm" for i in range(1, 5)]),
               commanded_torque=columns(*[f"command_{i}_Nm" for i in range(1, 5)]),
               reference=columns("ref_x_m", "ref_y_m", "ref_z_m"),
               reference_velocity=columns("ref_vx_m_s", "ref_vy_m_s", "ref_vz_m_s"),
               power=columns("power_W")[:, 0], energy=columns("energy_J")[:, 0],
               soc=columns("usable_soc")[:, 0], thrust=columns("thrust_N")[:, 0],
               contact=columns("contact")[:, 0].astype(bool))
    run["state"] = np.column_stack([run[k] for k in
                                   ("position", "velocity", "quaternion", "body_rates", "omega", "energy")])
    with (folder / f"{name}_motor_inputs.csv").open(newline="", encoding="utf-8") as file:
        inputs = list(csv.DictReader(file))
    run["inputs"] = dict(t=np.array([float(r["t_s"]) for r in inputs]), phase=np.array([r["phase"] for r in inputs]),
                         commanded_torque=np.array([[float(r[f"command_{i}_Nm"]) for i in range(1, 5)]
                                                    for r in inputs]),
                         torque=np.array([[float(r[f"torque_{i}_Nm"]) for i in range(1, 5)] for r in inputs]))
    return run


def save_json(data, path):
    """Convert NumPy scalar values without weakening JSON's finite-number rules."""
    def convert(value):
        if isinstance(value, np.ndarray):
            return value.tolist()
        if isinstance(value, np.generic):
            return value.item()
        raise TypeError(type(value).__name__)
    Path(path).write_text(json.dumps(data, indent=2, default=convert, allow_nan=False) + "\n", encoding="utf-8")


# 3. PAPER FIGURES: actual motor inputs and orientation snapshots from the solution.
def draw_drone(ax, quaternion, arm=0.17678, labels=False):
    """Render the measured pose; no exaggerated roll/pitch and no stock photograph."""
    from model import rotation
    R = np.asarray(rotation(quaternion)).reshape(3, 3)
    locations = np.array([[arm, arm, 0], [-arm, arm, 0], [-arm, -arm, 0], [arm, -arm, 0]])
    for i, location in enumerate(locations):
        points = np.array([[0, 0, 0], location]) @ R.T
        ax.plot(*points.T, color="#344C59", lw=3.0)
        theta = np.linspace(0, 2 * np.pi, 35)
        disk = location + np.column_stack([0.085 * np.cos(theta), 0.085 * np.sin(theta), np.zeros_like(theta)])
        ax.plot(*(disk @ R.T).T, color=COLORS[i], lw=1.0)
        if labels:
            q = (location * 1.2 + [0, 0, 0.04]) @ R.T
            ax.text(*q, str(i + 1), color=COLORS[i], fontsize=8)
    # Body +x identifies the heading, including during the 90-degree left yaw.
    nose = np.array([[0, 0, 0], [0.17, 0, 0]]) @ R.T
    ax.plot(*nose.T, color="#192F3C", lw=2)
    ax.scatter(*nose[-1], color="#192F3C", s=12)
    radius = 0.36
    ax.set(xlim=(-radius, radius), ylim=(-radius, radius), zlim=(-radius, radius))
    ax.set_box_aspect((1, 1, 0.72))
    ax.view_init(elev=24, azim=-55)
    ax.set_axis_off()


def _stage_shading(ax, run):
    from matplotlib.patches import Patch
    stages = STAGES[run["metadata"]["mission"]]
    seen, handles = set(), []
    for start, finish, label, color in stages:
        if label == "Contact / coast":
            start = run["events"].get("touchdown_time") or start
        if label == "Landing approach":
            finish = run["events"].get("touchdown_time") or finish
        ax.axvspan(start, finish, color=color, alpha=0.65, linewidth=0, zorder=0)
        if label not in seen:
            handles.append(Patch(facecolor=color, label=label))
            seen.add(label)
    return handles


def torque_figure(run):
    """One figure per goal: 4 torque histories, stage colors, zoom, actual poses."""
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D
    from matplotlib.ticker import MaxNLocator
    name, t, tau = run["metadata"]["mission"], run["inputs"]["t"], run["inputs"]["torque"]
    fig = plt.figure(figsize=(8.0, 6.4), layout="constrained")
    grid = fig.add_gridspec(3, 1, height_ratios=[0.12, 1.0, 0.52])
    title = fig.add_subplot(grid[0]); title.axis("off")
    title.text(0, 0.95, NAMES[name] + " | motor torque inputs", va="top",
               fontsize=12, fontweight="bold", color="#193440")
    title.text(0, 0.1, "Flight-stage shading and attitude snapshots from the simulated solution.",
               fontsize=8, color="#617781")
    ax = fig.add_subplot(grid[1])
    stages = _stage_shading(ax, run)
    styles = ["-", "--", "-.", ":"]
    for motor in range(4):
        ax.step(t, tau[:, motor], where="post", color=COLORS[motor], ls=styles[motor],
                lw=1.2, label=f"Motor {motor + 1}", zorder=3 + motor)
    ax.set(xlabel="Time [s]", ylabel="Applied shaft torque [N m]", xlim=(0, t[-1]))
    ax.grid(alpha=0.18)
    ax.ticklabel_format(axis="y", style="plain", useOffset=False)
    low, high = float(tau.min()), float(tau.max())
    pad = max((high - low) * 0.12, 0.001)
    ax.set_ylim(max(0, low - pad), high + pad)
    ax.yaxis.set_major_locator(MaxNLocator(nbins=5))
    motors = [Line2D([0], [0], color=COLORS[i], ls=styles[i], lw=1.5, label=f"Motor {i + 1}") for i in range(4)]
    ax.legend(handles=motors, loc="upper right", ncol=4, framealpha=0.95, fontsize=8)
    fig.legend(handles=stages, loc="lower center", ncol=min(len(stages), 3),
               fontsize=8, frameon=False, bbox_to_anchor=(0.5, -0.04))
    if name == "mission":
        # Slow landing makes the first 30 s look compressed; retain the full run
        # and add a clearly labeled zoom of the ascent-to-hover transition.
        zoom = ax.inset_axes([0.43, 0.60, 0.51, 0.25])
        for motor in range(4):
            zoom.step(t, tau[:, motor], where="post", color=COLORS[motor], ls=styles[motor], lw=1)
        window = (t >= 3.5) & (t <= 7.8)
        zoom.set(xlim=(3.5, 7.8), ylim=(float(tau[window].min()) - 0.0002,
                                     float(tau[window].max()) + 0.0002))
        zoom.axvline(6, color="#7B8790", lw=0.8, ls="--")
        zoom.set_title("Ascent → hover, 3.5–7.8 s", fontsize=8)
        zoom.ticklabel_format(axis="y", style="plain", useOffset=False)
        zoom.yaxis.set_major_locator(MaxNLocator(nbins=3))
        zoom.tick_params(labelsize=7); zoom.grid(alpha=0.15)
    if name == "circle":
        zoom = ax.inset_axes([0.43, 0.58, 0.51, 0.25])
        base = run["params"]["kQ"] * run["params"]["m"] * run["params"]["g"] / (4*run["params"]["kT"])
        measured = t >= 5
        for motor in range(4):
            zoom.step(t[measured], (tau[measured, motor] - base)*1e6, where="post",
                      color=COLORS[motor], ls=styles[motor], lw=0.9)
        zoom.set(xlim=(5, 65), title="Measured interval: torque minus hover [µN m]")
        zoom.title.set_fontsize(7)
        zoom.tick_params(labelsize=7); zoom.grid(alpha=0.15)
    if name == "hover":
        ax.text(0.02, 0.12, f"All four inputs ≈ {np.mean(tau):.5f} N m (traces coincide).",
                transform=ax.transAxes, fontsize=8, color="#52666F")
    snapshot_times = SNAPSHOTS[name]
    snapshots = grid[2].subgridspec(1, len(snapshot_times))
    for k, time in enumerate(snapshot_times):
        i = int(np.argmin(abs(run["t"] - time)))
        actual_time = run["t"][i]
        ax.axvline(actual_time, color="#657982", lw=0.6, ls=":", alpha=0.8)
        ax.text(actual_time, 0.02 + 0.06*(k % 2), chr(65 + k),
                transform=ax.get_xaxis_transform(), ha="center", fontsize=9, fontweight="bold")
        pose = fig.add_subplot(snapshots[k], projection="3d")
        draw_drone(pose, run["quaternion"][i], run["params"]["a"], labels=k == 0)
        angle = np.degrees(run["euler"][i])
        angle[abs(angle) < 0.005] = 0
        pose.set_title(f"{chr(65 + k)} · t = {actual_time:g} s\n"
                       f"roll {angle[0]:.2f}°, pitch {angle[1]:.2f}°\nyaw {angle[2]:.1f}°", fontsize=8, pad=0)
    return fig


def performance_figure(run):
    """Compact six-DOF histories plus measurable power/landing evidence."""
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(3, 2, figsize=(10.8, 8.2), layout="constrained")
    for k in range(3):
        axes[k, 0].plot(run["t"], run["position"][:, k], color=COLORS[0], lw=1.2, label="Actual")
        axes[k, 0].plot(run["t"], run["reference"][:, k], color=COLORS[1], lw=0.9, ls="--", label="Reference")
        axes[k, 0].set(ylabel=f"{'XYZ'[k]} [m]", xlabel="Time [s]")
        axes[k, 1].plot(run["t"], np.degrees(run["euler"][:, k]), color=COLORS[2], lw=1.2)
        axes[k, 1].set(ylabel=f"{['Roll','Pitch','Yaw'][k]} [deg]", xlabel="Time [s]")
        # Keep numerical roundoff from looking like a physical oscillation.
        for ax, values in [(axes[k, 0], np.r_[run["position"][:, k], run["reference"][:, k]]),
                           (axes[k, 1], np.degrees(run["euler"][:, k]))]:
            center = (values.max() + values.min()) / 2
            half_range = max(float(np.ptp(values)) * 0.6, 0.01)
            ax.set_ylim(center - half_range, center + half_range)
            ax.ticklabel_format(axis="y", style="plain", useOffset=False)
    axes[0, 0].legend(fontsize=8)
    for ax in axes.flat:
        ax.grid(alpha=0.2); ax.tick_params(labelsize=8)
    fig.suptitle(NAMES[run["metadata"]["mission"]] + " | integrated six-degree-of-freedom response", fontsize=13)
    return fig


def path_energy_figure(run):
    import matplotlib.pyplot as plt
    name = run["metadata"]["mission"]
    fig, axes = plt.subplots(2, 2, figsize=(10.8, 7.4), layout="constrained")
    axes[0, 0].plot(*run["reference"][:, :2].T, ls="--", color=COLORS[1], lw=1, label="Reference")
    axes[0, 0].plot(*run["position"][:, :2].T, color=COLORS[0], lw=1.3, label="Actual")
    axes[0, 0].set(xlabel="X [m]", ylabel="Y [m]", title="Ground track")
    axes[0, 0].axis("equal")
    axes[0, 0].legend(fontsize=8)
    axes[0, 1].plot(run["t"], np.linalg.norm(run["velocity"], axis=1), color=COLORS[0])
    axes[0, 1].set(xlabel="Time [s]", ylabel="Speed [m/s]", title="Center-of-mass speed")
    speed = np.linalg.norm(run["velocity"], axis=1)
    speed_pad = max(float(np.ptp(speed)) * 0.08, 0.005)
    axes[0, 1].set_ylim(max(0, float(speed.min()) - speed_pad), float(speed.max()) + speed_pad)
    axes[1, 0].plot(run["t"], run["power"], color=COLORS[2])
    axes[1, 0].set(xlabel="Time [s]", ylabel="Electrical power [W]", title="Modeled electrical load")
    power_pad = max(float(np.ptp(run["power"])) * 0.08, 1.0)
    axes[1, 0].set_ylim(max(0, float(run["power"].min()) - power_pad), float(run["power"].max()) + power_pad)
    if name == "mission":
        mask = run["t"] >= 30
        axes[1, 1].plot(run["t"][mask], -run["velocity"][mask, 2], color=COLORS[0], label="Downward speed")
        axes[1, 1].axhline(0.01, color="#AA4255", ls="--", lw=1, label="1 cm/s limit")
        axes[1, 1].set(xlabel="Time [s]", ylabel="Descent speed [m/s]", title="Vertical landing")
        axes[1, 1].legend(fontsize=8)
    else:
        axes[1, 1].plot(run["t"], run["energy"] / 3600, color=COLORS[0])
        axes[1, 1].set(xlabel="Time [s]", ylabel="Consumed energy [Wh]", title="Integrated energy")
    for ax in axes.flat:
        ax.grid(alpha=0.2); ax.tick_params(labelsize=8); ax.ticklabel_format(style="plain", useOffset=False, axis="y")
    fig.suptitle(NAMES[name] + " | trajectory and performance evidence", fontsize=13)
    return fig


def make_figures(run, folder, formats=("png", "svg")):
    """Write three readable paper panels for one maneuver, all from actual data."""
    import matplotlib.pyplot as plt
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 9, "axes.spines.top": False,
                         "axes.spines.right": False, "savefig.dpi": 300})
    folder = Path(folder); folder.mkdir(parents=True, exist_ok=True)
    name = run["metadata"]["mission"]
    for suffix, factory in [("motor_torques", torque_figure), ("six_dof", performance_figure),
                            ("performance", path_energy_figure)]:
        fig = factory(run)
        for extension in formats:
            fig.savefig(folder / f"{name}_{suffix}.{extension}", bbox_inches="tight")
        plt.close(fig)
