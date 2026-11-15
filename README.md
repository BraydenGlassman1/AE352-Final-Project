# AE352 Team 25 - quadrotor simulation

This submission has **exactly three Python files**. The updated report's equations and physical parameters are preserved. The interactive simulation, individual motor experiments, saved torque inputs, performance checks, and paper figures are retained.

| Read in this order | Contents |
|---|---|
| `analysis.py` | Editable physical parameters at the top, acceptance criteria, data export/reload, and all figure code |
| `model.py` | Drone equations, controller, reference maneuvers, RK4 integration, and motor-input replay |
| `run.py` | Run the project, check results, regenerate figures, or open the interactive workbench |

`figures/` holds the paper-ready PNGs and vector SVGs. `results/` holds the numerical evidence and CSV files. There is no package hierarchy, notebook, bundled environment, historical code, or extra Python test file in the submission folder.

## Start in VS Code

Install Python 3.10 or newer. Open **this folder** in VS Code, select your Python interpreter, and use a terminal in the folder:

```console
python -m pip install -r requirements.txt
python run.py
```

This runs all three maneuvers and regenerates their data and figures. To include the basic physical checks and a half-step convergence comparison:

```console
python run.py --check
```

To explore the drone interactively:

```console
python run.py --interactive
```

Choose a mission, play/pause or scrub time, rotate the 3D view by dragging, inspect position/attitude/rotor/energy traces, or change mass, capacity, and efficiency. **Torque experiments** opens the four manual motor inputs and collective/roll/yaw pulse programs. These open-loop experiments can become unstable. Interactive exports go to `results/interactive/` and do not replace the paper's baseline data.

Matplotlib needs a desktop backend such as Tk for the interactive view. `python -m tkinter` checks Tk support; some Linux installations need their system's Tk package. Batch simulation and figure generation use a noninteractive backend. Use `python3` wherever that is your system's Python command.

## The requested motor-input figures

The three principal figures are:

- `figures/hover_motor_torques.png`
- `figures/circle_motor_torques.png`
- `figures/mission_motor_torques.png`

Each contains the four **applied motor shaft torques** versus time, with separate colors/line styles, shaded flight stages, marked timestamps, and drone snapshots calculated from the integrated orientation. The snapshots are simulated renderings, not photographs; their angles are not exaggerated. Hover's equal traces overlap. The circle includes a magnified torque-deviation inset. The full sequence includes an ascent-to-hover detail inset.

The matching SVGs are vector figures. Each maneuver also has a `*_six_dof` figure and a `*_performance` figure for the report's position/attitude, path, speed, power, and landing evidence. Figure functions and snapshot times are editable in `analysis.py`.

After editing figure labels or formatting, redraw the saved solution without another simulation:

```console
python run.py --plots-only
```

Use `--mission hover`, `--mission circle`, or `--mission mission` to work on one maneuver. Example: `python run.py --plots-only --mission mission`. By default, PNGs are saved at 300 dpi and SVGs preserve vector geometry.

## Where the updated values came from

The supplied **AE 352 Team Project: Quadcopter Digital Twin, Preliminary Design Review Report**, pages 3-6, gives these values. Its printed values are used directly:

| Parameter | Implemented report value |
|---|---:|
| Total mass | 1.700 kg |
| Center-to-rotor length | 0.250 m; X-arm coordinate `a = 0.17678 m` |
| `Ixx`, `Iyy`, `Izz` | 0.017661, 0.018373, 0.034584 kg m² |
| `kT`, `kQ` | 1.06775e-5 kg m, 1.52451e-7 kg m² |
| Propeller-only spin inertia | 6.99e-5 kg m² |

The unequal roll/pitch inertias are retained. The old 1.2 kg F450/APC configuration is not the nominal model in this folder.

**The report does not resolve every simulation input.** These additional values are provisional, visible together at the top of `analysis.py`, and must not be presented as report-provided measurements:

| Remaining input | Provisional numerical choice |
|---|---|
| Full rotating-assembly `Jr` | Report propeller estimate + assumed motor contribution 1e-5 = 7.99e-5 kg m² |
| Battery voltage/capacity | 4S, 14.8 V, 5.0 Ah; 74 Wh nameplate |
| Usable fraction | 80%, giving 59.2 Wh modeled usable energy |
| Motor/ESC efficiency, avionics | 75%, 5 W |
| Chosen operating bounds | 7000 RPM and 0.12 N m per motor; not certified flat motor limits |

Holybro recommends a 4S 3000-5000 mAh battery; the report itself gives only an assumed battery mass and dimensions. [Holybro X500 V2 specifications](https://holybro.com/products/x500-v2-kits). A roughly 0.5 kg [HRB 4S 5000 mAh example](https://hrb-power.com/products/hrb-4s-5000-50c-t-lipo-battery) supports a candidate energy budget, but its dimensions differ from the report and it is not asserted to be the team's selected pack. Rotor-inertia, efficiency, capacity, and yaw-inertia interpretation sensitivities were checked separately.

Two matters should be resolved with the report authors before claiming a calibrated physical twin. Equation (56) uses whole-aircraft **locked-rotor** inertia; the component calculation approximates the motors and propellers as point masses and omits their local spin inertias. We retain the reported `Izz` as an approximate locked-inertia input and flag that approximation instead of silently altering it. The report's exact motor-test rows are reproduced in Muhammad Usman's 2022 UiT thesis, *Automated multi-rotor draft survey of large vessels*, Table 6, printed p. 53 ([official repository](https://nva.sikt.no/registration/0198f43974ca-b8281b90-a534-4901-aa7a-2d2b3783a8f8)). That table cites a historical Holybro link; its upstream destination remains unverified. The [current Holybro-linked manufacturer table](https://cdn.shopify.com/s/files/1/0604/5905/7341/files/X500MotorSpec.png?v=1678791632) has different rows. Cite the reproduced thesis data accurately and confirm the exact motor revision, rather than attributing those rows to the current image.

## What the computation does

At each 0.005 s step, the reference and current state determine four requested motor torques. The plant bounds those requests and jointly integrates translation, quaternion attitude, body rates, rotor speeds, and electrical energy using RK4. The quaternion is an equivalent implementation of the report's Euler kinematics; exported angles are roll, pitch, and yaw. The controller never assigns an airborne position or orientation.

Motor input is held between controller updates. The plots use actual bounded shaft torque, while CSVs retain both the raw controller request and applied torque. Speed/energy cutoffs are part of the plant. At usable-energy exhaustion all electrical draw stops and the rotors coast. A simple ground constraint prevents penetration; touchdown speed is recorded **before** vertical velocity is reset.

Hover starts in equilibrium at 1 m for 120 s. Circle starts on its moving reference, settles for 5 s, then supplies 60 measured seconds. The ground-start sequence runs 163 s and deliberately descends at approximately 0.008 m/s. Each demonstration starts with a fresh modeled battery. All tracking tolerances are stated in `check_run`; they are engineering interpretations, not tolerances supplied by the course brief.

## Data and verification

`results/summary.json` gives the actual measured goal checks, energy, and half-step results. `results/verification.json` records 30 independent development verification groups and five targeted supplementary groups: equations/invariants, updated goals, true touchdown, convergence, old-parameter migration consistency, exact torque-input replay, figure data/pose consistency, and parameter sensitivities. This broader development harness is not an additional submission source file. `python run.py --check` reproduces the included basic mechanics checks, goal checks, and position convergence comparisons.

Each maneuver has three data files:

- `*_telemetry.csv`: state, quaternion/Euler attitude, reference, requested/applied inputs, power, and energy at 0.02 s intervals.
- `*_motor_inputs.csv`: all controller updates at 0.005 s, including requested and applied torques, with full floating-point precision.
- `*_metadata.json`: exact parameters, timing, and integration-step landing events.

The requested-input CSV is a reproducible motor-torque function under zero-order hold. For example, replay the full mission with the same initial conditions and parameters, keeping the replay outputs separate:

```console
python run.py --replay results/mission_motor_inputs.csv --mission mission --out replay_output
```

Other experiments can similarly use `--out experiment_output`, `--mass`, `--battery-ah`, or `--motor-jr` (motor contribution only). Example: `python run.py --mission mission --battery-ah 3 --out smaller_pack`. A changed capacity or mass alone is a numerical sensitivity, not a newly qualified aircraft design.

The verified nominal mission energies are approximately 6.776 Wh, 3.671 Wh, and 8.826 Wh. Full-sequence touchdown is 0.008 m/s, with maximum landing descent about 0.008054 m/s. These are numerical results under the declared assumptions; they do not validate hardware, an exact battery, or the unresolved rotor inertia.

Share this complete folder through the team's existing GitHub repository or Box. No file contains a machine-specific path. Keep generated evidence with its matching source version. Suggested captions and a short implementation paragraph are in `results/REPORT_NOTES.md`; revise those baseline notes if the parameters or experiments change.
