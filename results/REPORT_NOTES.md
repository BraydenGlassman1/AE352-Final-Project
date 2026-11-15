# Figure captions and implementation notes

These are software handoff notes for the team report. Revise wording to reflect the final selected components and acknowledge the provisional parameters listed in the README.

## Suggested figure captions

**Hover motor inputs.** Applied motor shaft torques for a 120 s equilibrium hover at 1 m using the updated report parameter set. All four inputs coincide at approximately 0.05951 N m. Snapshots show the simulated attitude at 0, 60, and 120 s. Hover is initialized airborne; this figure does not include takeoff.

**Circular-flight motor inputs.** Applied motor torques during a 65 s circular-flight simulation, with a 5 s settling period followed by 60 measured seconds at 2 m radius, 1 m altitude, and 0.5 m/s. The inset magnifies computed torque differences from the hover baseline; its micro-newton-metre scale describes the numerical solution, not measurement precision. Snapshot attitudes are taken from the integrated state at 5, 13, and 21 s.

**Full-sequence motor inputs.** Applied shaft torques for ground-start takeoff, two 5 m straight legs with a stop and a left 90-degree yaw between them, hovering, and a slow vertical landing. Shading identifies the reference flight stages, and the contact boundary is the simulated touchdown event. An inset shows the ascent-to-hover transition. Snapshots at 3, 6, 18, and 25 s use the actual simulated orientation, without angle exaggeration. Requested controller commands are bounded before being applied to the motors.

## Brief implementation paragraph

The six-degree-of-freedom rigid-body and four-rotor equations were integrated together using fixed-step fourth-order Runge-Kutta with a 0.005 s step. A unit quaternion implements attitude kinematics equivalent to the report's Euler-angle equations; reported angles are roll, pitch, and yaw. Position/velocity feedback, attitude feedback, a rotor-speed servo, and yaw torque allocation produce four requested motor torques. The plant enforces nonnegative torque, chosen torque/speed bounds, and finite usable energy; no airborne pose is assigned from the reference. Motor commands are held between 5 ms controller updates. Solutions were checked against analytical balances, the maneuver targets, and a halved timestep. Replaying the recorded full-resolution requested torques reproduces the nominal state history.

## Nominal measured results

| Maneuver | Energy used (Wh) | Usable energy remaining |
|---|---:|---:|
| hover | 6.7762 | 88.55% |
| circle | 3.6709 | 93.80% |
| mission | 8.8255 | 85.09% |

- Full-sequence touchdown: 0.008000 m/s before contact projection at 156.005 s.
- Maximum landing descent: 0.00805416 m/s.
- Measured leg averages: 0.9987056 and 0.9987084 m/s.
- Left yaw change: 90.000000 degrees.

Each goal uses a fresh modeled battery. These are conditional numerical performance results, not a continuous combined flight or measured hardware endurance. The report does not yet specify full rotor inertia or an exact battery; the nominal 14.8 V/5 Ah/80% usable budget is an explicit candidate assumption.

## Report items to reconcile

- Confirm the motor contribution to Jr and the exact battery; retain the current software values as provisional until then.
- Clarify that Eq. (56) needs whole-aircraft locked inertia; the point-mass component model omits local rotor/motor spin inertias. The code retains the printed Izz and flags the approximation.
- Cite the exact reproduced data: Muhammad Usman (2022), Automated multi-rotor draft survey of large vessels, UiT master's thesis, Table 6, printed p. 53. The thesis cites a historical Holybro motor-specification link. Its upstream destination remains unverified, and the current manufacturer table differs. Official repository: https://nva.sikt.no/registration/0198f43974ca-b8281b90-a534-4901-aa7a-2d2b3783a8f8.
- In Eq. (40), acceleration is intended where first derivatives of position are printed. The implemented final Eq. (41) is the acceleration equation.
- The angle vector [phi, theta, psi] represents attitude, not an inertial angular-velocity vector. The implemented kinematic mapping follows Eqs. (43)-(45).
