"""Run everything: python run.py; add --interactive for the 3D mission workbench.

Reading order: analysis.py definitions -> model.py equations -> main() below.
Only these three Python files are needed. Generated data/figures are subfolders.
"""
from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor
import csv
from pathlib import Path
import time
import numpy as np

from model import simulate, simulate_torque, simulate_input, initial_state, rotation
from analysis import (DEFAULT_PARAMS, J_PROP, NAMES, COLORS, check_run, export_results,
                      save_json, make_figures, load_results)


def basic_checks(p):
    """Small reproducible physics checks, separate from checking a plotted path."""
    from model import derivative, rk4_step
    state = initial_state(p, "hover")
    equilibrium = p["kQ"] * state[13:17] ** 2
    d = derivative(state, equilibrium, p)
    hover = bool(np.max(abs(d[:17])) < 1e-8)
    falling = state.copy(); falling[2] = 100; falling[13:17] = 0
    for _ in range(100):
        falling = rk4_step(falling, np.zeros(4), 0.01, p)
    freefall = bool(abs(falling[2] - (100 - 0.5 * p["g"])) < 1e-8)
    asym = state.copy(); asym[13:17] = [380, 410, 430, 370]
    cmd = np.array([0.02, 0.025, 0.019, 0.022])
    d = derivative(asym, cmd, p)
    signs = np.array([1, -1, 1, -1])
    total_rate = p["Izz"] * d[12] + p["Jr"] * signs @ d[13:17]
    external = -p["kQ"] * signs @ asym[13:17] ** 2
    momentum = bool(abs(total_rate - external) < 1e-10)
    return {"hover_equilibrium": hover, "analytic_freefall": freefall,
            "yaw_momentum_balance": momentum}


def replay_csv(path, mission, p, dt):
    """A saved requested-torque history is an explicit zero-order-held input."""
    with Path(path).open(newline="", encoding="utf-8") as file:
        rows = list(csv.DictReader(file))
    times = np.array([float(r["t_s"]) for r in rows])
    commands = np.array([[float(r[f"command_{i}_Nm"]) for i in range(1, 5)] for r in rows])
    if len(times) < 2 or np.any(np.diff(times) <= 0) or times[0] != 0:
        raise ValueError("Torque CSV must start at t=0 with strictly increasing times")

    def input_at(t):
        i = np.searchsorted(times, t + 1e-10, side="right") - 1
        return commands[max(i, 0)]

    return simulate_input(input_at, initial_state(p, mission), p, times[-1], dt=dt,
                          sample_dt=0.02, mission=mission)


# Optional display: Matplotlib replaces a large separate GUI package.
class Workbench:
    """Interactive poses, plots, parameters, and four manual motor-torque inputs."""

    def __init__(self, params, output):
        import matplotlib.pyplot as plt
        from matplotlib.widgets import Button, RadioButtons, Slider, TextBox
        self.plt, self.output, self.params = plt, Path(output), dict(params)
        self.pool = ThreadPoolExecutor(max_workers=1)
        self.pending = self.run = None
        self.playing, self.clock, self.last = False, 0.0, time.perf_counter()
        self.fig = plt.figure(figsize=(14, 8.5), facecolor="#F5F7F8")
        self.fig.canvas.manager.set_window_title("AE352 - three-file quadrotor workbench")
        self.fig.text(.025, .96, "FLIGHTLAB / AE352", fontsize=17, weight="bold", color="#193440")
        self.status = self.fig.text(.245, .955, "Select a mission; every state is numerically integrated.",
                                    fontsize=9, color="#526A76")
        self.ax3d = self.fig.add_axes([.245, .43, .45, .46], projection="3d")
        self.info = self.fig.add_axes([.75, .43, .235, .46]); self.info.axis("off")
        self.trace = self.fig.add_axes([.285, .18, .675, .19])
        self.radio = RadioButtons(self.fig.add_axes([.025, .70, .18, .18]),
                                  list(NAMES.values()), radio_props={"s": [30]*3})
        self.radio.on_clicked(self.choose_mission)
        self.chart = RadioButtons(self.fig.add_axes([.025, .53, .18, .15]),
                                  ["Position", "Attitude", "Rotors", "Energy"])
        self.chart.on_clicked(lambda _v: self.draw_traces())
        self.boxes = {}
        for k, (name, label, value) in enumerate([("m", "Mass / kg", params["m"]),
                                                 ("battery_ah", "Battery / Ah", params["battery_ah"]),
                                                 ("efficiency", "Efficiency", params["efficiency"])]):
            box = TextBox(self.fig.add_axes([.105, .465 - k*.052, .09, .032]),
                          label + " ", initial=str(value), label_pad=.04)
            self.boxes[name] = box
        self.recompute = Button(self.fig.add_axes([.025, .29, .18, .043]), "Recompute mission")
        self.recompute.on_clicked(lambda _event: self.choose_mission(self.radio.value_selected))
        self.torque_button = Button(self.fig.add_axes([.025, .225, .18, .043]), "Torque experiments…")
        self.torque_button.on_clicked(self.torque_window)
        self.export = Button(self.fig.add_axes([.025, .16, .18, .043]), "Export current run")
        self.export.on_clicked(lambda _event: self.export_current())
        self.play = Button(self.fig.add_axes([.025, .065, .08, .043]), "Play / pause")
        self.play.on_clicked(lambda _event: self.toggle())
        self.reset = Button(self.fig.add_axes([.115, .065, .08, .043]), "Reset")
        self.reset.on_clicked(lambda _event: self.seek(0))
        self.speed = Slider(self.fig.add_axes([.81, .065, .16, .025]), "Speed", 1, 32, valinit=4, valstep=1)
        self.slider_ax = self.fig.add_axes([.285, .07, .44, .025])
        self.slider = Slider(self.slider_ax, "Time / s", 0, 120, valinit=0)
        self.slider.on_changed(self.seek)
        self.timer = self.fig.canvas.new_timer(interval=100)
        self.timer.add_callback(self.tick); self.timer.start()
        self.fig.canvas.mpl_connect("close_event", lambda _event: self.pool.shutdown(wait=False, cancel_futures=True))
        self.choose_mission(NAMES["hover"])

    def choose_mission(self, label):
        if self.pending:
            return
        try:
            for key, box in self.boxes.items():
                self.params[key] = float(box.text)
            name = next(key for key, value in NAMES.items() if value == label)
            self.submit(simulate, name, self.params.copy())
        except (ValueError, StopIteration) as error:
            self.status.set_text(str(error))

    def submit(self, function, *args, **kwargs):
        if self.pending:
            return
        self.playing = False
        self.status.set_text("Computing the torque-driven solution…")
        self.fig.canvas.draw_idle()
        self.pending = self.pool.submit(function, *args, **kwargs)

    def toggle(self):
        self.playing = not self.playing
        if self.run is not None and self.clock >= self.run["t"][-1]:
            self.seek(0)

    def seek(self, time_value):
        self.clock = float(time_value)
        if self.slider.val != self.clock:
            self.slider.eventson = False
            self.slider.set_val(self.clock)
            self.slider.eventson = True
        if self.run is not None:
            self.draw_state()

    def load(self, run):
        from matplotlib.widgets import Slider
        self.run, self.clock = run, 0.0
        self.slider.disconnect_events(); self.slider_ax.clear()
        self.slider = Slider(self.slider_ax, "Time / s", 0, float(run["t"][-1]), valinit=0)
        self.slider.on_changed(self.seek)
        name = run["metadata"]["mission"]
        if name in NAMES:
            self.radio.eventson = False
            self.radio.set_active(list(NAMES).index(name))
            self.radio.eventson = True
            report = check_run(run)
            self.status.set_text(
                f'{NAMES[name]}: {"PASS" if report["pass"] else "REVIEW REQUIRED"} | numerical criteria')
        else:
            self.status.set_text("Open-loop torque experiment | no mission acceptance criteria")
        self.draw_traces(); self.draw_state()

    def draw_traces(self):
        if self.run is None:
            return
        r, chart = self.run, self.chart.value_selected
        data, labels, unit = {
            "Position": (r["position"], list("XYZ"), "Position [m]"),
            "Attitude": (np.degrees(r["euler"]), ["Roll", "Pitch", "Yaw"], "Attitude [deg]"),
            "Rotors": (r["omega"], [f"Rotor {i}" for i in range(1, 5)], "Speed [rad/s]"),
            "Energy": (r["energy"][:, None] / 3600, ["Consumed"], "Energy [Wh]"),
        }[chart]
        self.trace.clear()
        for k, label in enumerate(labels):
            self.trace.plot(r["t"], data[:, k], color=COLORS[k], lw=1.1, label=label)
        self.trace.set(xlabel="Time [s]", ylabel=unit)
        self.trace.legend(loc="upper right", ncol=len(labels), fontsize=8)
        self.trace.grid(alpha=.2)
        self.cursor = self.trace.axvline(self.clock, color="#6F8490", ls=":", lw=.8)
        self.fig.canvas.draw_idle()

    def draw_state(self):
        r = self.run
        i = int(np.clip(np.searchsorted(r["t"], self.clock, side="right") - 1, 0, len(r["t"]) - 1))
        ax = self.ax3d
        elev, azim = ax.elev, ax.azim
        ax.clear(); ax.view_init(elev=elev, azim=azim)
        ax.plot(*r["reference"].T, color="#A7B3BA", ls="--", lw=.8, label="Reference")
        ax.plot(*r["position"][:i + 1].T, color=COLORS[0], lw=1.3, label="Actual path")
        pos, R = r["position"][i], np.asarray(rotation(r["quaternion"][i])).reshape(3, 3)
        a, scale = r["params"]["a"], 1.6
        for k, motor in enumerate([[a,a,0],[-a,a,0],[-a,-a,0],[a,-a,0]]):
            motor = np.array(motor) * scale
            points = np.array([[0,0,0],motor]) @ R.T + pos
            ax.plot(*points.T, color="#304E5B", lw=2)
            theta = np.linspace(0, 2*np.pi, 20)
            disk = motor + np.column_stack([.07*np.cos(theta), .07*np.sin(theta), np.zeros_like(theta)])
            ax.plot(*(disk @ R.T + pos).T, color=COLORS[k], lw=1)
        points = np.vstack([r["position"], r["reference"]])
        center = (points.max(axis=0) + points.min(axis=0)) / 2
        span = max(2, np.ptp(points[:, :2], axis=0).max() + 1)
        ax.set(xlim=(center[0]-span/2, center[0]+span/2), ylim=(center[1]-span/2, center[1]+span/2),
               zlim=(0, max(2, float(points[:,2].max()) + .5)), xlabel="X [m]", ylabel="Y [m]", zlabel="Z [m]")
        ax.set_title(f'{r["phase"][i]} | t = {r["t"][i]:.2f} s | vehicle drawing ×1.6', fontsize=9)
        ax.set_box_aspect((1,1,.6)); ax.tick_params(labelsize=7)
        self.info.clear(); self.info.axis("off")
        text = (f'FLIGHT TELEMETRY\n\nAltitude       {pos[2]:8.3f} m\n'
                f'Ground speed   {np.linalg.norm(r["velocity"][i,:2]):8.3f} m/s\n'
                f'Heading        {np.degrees(r["euler"][i,2]):8.2f} deg\n'
                f'Power          {r["power"][i]:8.2f} W\n'
                f'Energy         {r["energy"][i]/3600:8.4f} Wh\n'
                f'Usable budget  {r["soc"][i]*100:8.2f} %\n\n'
                'ROTOR      rad/s       N m\n' + ''.join(
                    f'{k+1}       {r["omega"][i,k]:8.2f}   {r["torque"][i,k]:.5f}\n' for k in range(4)))
        self.info.text(0, .98, text, va="top", family="monospace", fontsize=9, color="#244654")
        if hasattr(self, "cursor"):
            self.cursor.set_xdata([self.clock, self.clock])
        self.fig.canvas.draw_idle()

    def tick(self):
        now = time.perf_counter()
        delta, self.last = min(.5, now - self.last), now
        if self.pending and self.pending.done():
            job, self.pending = self.pending, None
            try:
                self.load(job.result())
            except Exception as error:
                self.status.set_text("Simulation failed: " + str(error))
        if self.playing and self.run is not None and not self.pending:
            self.clock = min(float(self.run["t"][-1]), self.clock + delta * self.speed.val)
            self.slider.set_val(self.clock)
            if self.clock >= self.run["t"][-1]:
                self.playing = False

    def export_current(self):
        if self.run is not None:
            export_results(self.run, self.output / "interactive")
            self.status.set_text("Exported to " + str(self.output / "interactive"))

    def torque_window(self, _event=None):
        from matplotlib.widgets import Button, RadioButtons, Slider
        fig = self.plt.figure(figsize=(7, 6), facecolor="#F5F7F8")
        fig.canvas.manager.set_window_title("AE352 - four motor inputs")
        fig.text(.08, .94, "Individual motor shaft torques", fontsize=15, weight="bold")
        presets = RadioButtons(fig.add_axes([.08,.66,.84,.22]), ["hover","collective","roll","yaw","manual"])
        hover = self.params["kQ"] * self.params["m"] * self.params["g"] / (4*self.params["kT"])
        sliders = []
        for k in range(4):
            slider = Slider(fig.add_axes([.2,.55-k*.09,.67,.035]), f'Motor {k+1}', 0,
                            self.params["max_torque"], valinit=hover, valfmt="%.5f N m")
            slider.on_changed(lambda _value: presets.set_active(4))
            sliders.append(slider)
        button = Button(fig.add_axes([.25,.08,.50,.07]), "Run 2-second experiment")

        def launch(_event):
            self.submit(simulate_torque, presets.value_selected, self.params.copy(),
                        torques=[s.val for s in sliders] if presets.value_selected == "manual" else None)
            self.plt.close(fig)
        button.on_clicked(launch)
        fig._widgets = (presets, sliders, button)  # Keep Matplotlib widget callbacks alive.
        fig.text(.08,.015,"Open loop: unequal torques can destabilize the vehicle.",fontsize=9,color="#617781")
        fig.show()


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mission", choices=["all", *NAMES], default="all")
    parser.add_argument("--interactive", action="store_true",
                        help="Open the interactive 3D mission and torque workbench")
    parser.add_argument("--no-plots", action="store_true")
    parser.add_argument("--plots-only", action="store_true", help="Redraw saved CSV results without another simulation")
    parser.add_argument("--check", action="store_true", help="Add physical checks and a half-step convergence run")
    parser.add_argument("--dt", type=float, default=.005)
    parser.add_argument("--out", type=Path, default=Path(__file__).resolve().parent)
    parser.add_argument("--mass", type=float)
    parser.add_argument("--battery-ah", type=float)
    parser.add_argument("--motor-jr", type=float,
                        help="Motor contribution ONLY; added to the report propeller estimate")
    parser.add_argument("--torque", choices=["hover","collective","roll","yaw","manual"])
    parser.add_argument("--torques", type=float, nargs=4)
    parser.add_argument("--replay", type=Path,
                        help="Replay a full-resolution requested-torque CSV; choose its initial mission")
    args = parser.parse_args(argv)
    p = dict(DEFAULT_PARAMS)
    for key, value in [("m", args.mass), ("battery_ah", args.battery_ah)]:
        if value is not None: p[key] = value
    if args.motor_jr is not None:
        if args.motor_jr < 0: parser.error("Motor inertia cannot be negative")
        p["Jr"] = J_PROP + args.motor_jr
    if args.interactive:
        import matplotlib.pyplot as plt
        plt.rcParams.update({"font.size":9,"axes.spines.top":False,"axes.spines.right":False})
        app = Workbench(p, args.out / "results")
        plt.show()
        return 0
    import matplotlib
    matplotlib.use("Agg")
    results, figures = args.out / "results", args.out / "figures"
    results.mkdir(parents=True, exist_ok=True)
    names = list(NAMES) if args.mission == "all" else [args.mission]
    if args.plots_only:
        for name in names:
            make_figures(load_results(results, name), figures)
        print("Redrew saved results:", figures.resolve())
        return 0
    if args.torque: names = ["torque"]
    summaries = {}
    try:
        for name in names:
            print(f"Simulating {name} ...", flush=True)
            if args.torque:
                run = simulate_torque(args.torque, p, torques=args.torques, dt=args.dt)
            elif args.replay:
                if args.mission == "all": parser.error("Choose --mission when replaying a torque CSV")
                run = replay_csv(args.replay, name, p, args.dt)
            else:
                run = simulate(name, p, dt=args.dt)
            report = check_run(run)
            summaries[name] = report
            export_results(run, results)
            if not args.no_plots and not args.torque:
                make_figures(run, figures)
            print(f'  {"PASS" if report["pass"] else "REVIEW"} | '
                  f'energy {report["metrics"]["energy_Wh"]:.4f} Wh', flush=True)
            if args.check and not args.torque and not args.replay:
                fine = simulate(name, p, dt=args.dt/2)
                delta = float(np.max(np.linalg.norm(run["position"] - fine["position"], axis=1)))
                report["step_convergence"] = dict(
                    maximum_position_difference_m=delta, limit_m=.01, **{"pass":delta <= .01})
                report["pass"] = report["pass"] and delta <= .01
    except (ValueError, RuntimeError, KeyError) as error:
        parser.exit(2, f"Simulation failed: {error}\n")
    physical = basic_checks(p) if args.check else {}
    document = {"parameters":p,"missions":summaries,"physics_checks":physical,
                "pass":all(s["pass"] for s in summaries.values()) and all(physical.values())}
    save_json(document, results / "summary.json")
    print("Results:", results.resolve()); print("Figures:", figures.resolve())
    return 0 if document["pass"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
