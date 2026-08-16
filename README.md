# INTRIX

**Simulated Bluetooth-Based Indoor Positioning Using Mathematical Trilateration**

A Class 12 STEM Exhibition project.

> INTRIX does not use physical Bluetooth hardware. It simulates Bluetooth-based
> distance measurements to demonstrate how mathematical trilateration can estimate
> an unknown indoor position.

---

## 1. What INTRIX Is

INTRIX is a software simulation that answers one question:

> **Given the known positions of several fixed Bluetooth beacons and their
> measured distances from an unknown device, can mathematical trilateration
> determine the device's indoor position?**

It simulates a building with a few fixed Bluetooth beacons at known locations,
simulates the imperfect distance measurements those beacons might produce, and
then uses **trilateration** — a purely mathematical technique — to work out
where an unknown device must be standing.

## 2. The Problem: Why Not Just Use GPS?

GPS works well outdoors because a receiver can get a clear line of sight to
satellites. Indoors, walls, floors, and roofs block or reflect those signals,
so GPS accuracy indoors is poor or unavailable entirely. Bluetooth-based
positioning is one common alternative: instead of satellites far above, it
uses small fixed beacons already inside the building.

## 3. How Bluetooth-Based Distance Estimation Fits In

In a real Bluetooth positioning system, a device measures the strength of the
signal (RSSI) coming from each nearby beacon. Signal strength tends to fall
off with distance, so it can be used to *estimate* — not measure exactly —
how far away each beacon is.

**INTRIX does not implement this physical process.** There is no real
Bluetooth hardware involved anywhere in this project. Instead, `modules/simulation.py`
generates distance values that behave the way real Bluetooth-based distance
estimates would: mostly accurate, with some random error mixed in.

## 4. Why the Measurements Are Simulated

Building and calibrating a real Bluetooth positioning system requires physical
beacons, hardware, and a controlled space to test in. Since the goal of this
project is to demonstrate the **mathematics of trilateration**, INTRIX
simulates realistic distance measurements instead — the mathematical engine
behaves identically whether the measured distance came from a real Bluetooth
radio or from this simulation.

## 5. The Critical Rule: The Device Position Is Never "Given Away"

It would be trivial (and mathematically meaningless) to type in a device's
X/Y position, calculate distances from it, and then feed the same numbers
back in to "recover" the same position. INTRIX deliberately avoids this:

```
Hidden simulated position (internal only)
        ↓
Calculate TRUE distance to each beacon
        ↓
Add measurement noise
        ↓
Simulated Bluetooth measurements  ← this is all the algorithm ever sees
        ↓
Trilateration  (beacon coordinates + measured distances only)
        ↓
Estimated position
        ↓
Compare with hidden ground truth  (evaluation only)
        ↓
Position error
```

The hidden ground-truth position lives only in `modules/simulation.py`
and in the app's session state, purely so the simulation has something
to test against. It is **never** passed into `trilaterate()`.

## 6. How Circles Are Formed

For a beacon at `(xi, yi)` with a measured distance `di`, the device could be
standing anywhere on the circle:

```
(x - xi)^2 + (y - yi)^2 = di^2
```

- **One beacon** → the device could be anywhere on one circle.
- **Two beacons** → the device must be at one of (at most) two points where
  the circles cross.
- **Three or more beacons** → the position becomes reliably determined, and
  extra beacons help average out measurement noise.

## 7. How Trilateration Works

With perfect (noise-free) measurements, all the circles would cross at
exactly one point. With real, noisy measurements they usually don't quite
agree, so INTRIX needs a way to find the position that fits **all** the
circles as well as possible.

`modules/trilateration.py` does this with **linearization + least squares**:

1. Pick one beacon as a reference.
2. Subtract its circle equation from every other beacon's circle equation.
   This cancels out the `x^2` and `y^2` terms, turning each pair of
   equations into a straight **linear** equation in `x` and `y`.
3. With 3+ beacons this produces an over-determined system of linear
   equations (more equations than unknowns).
4. `numpy.linalg.lstsq` solves this system for the `(x, y)` that best fits
   every equation at once — i.e. the best-fit intersection point.

Trilateration uses **distances**, not angles — this is what distinguishes it
from triangulation.

## 8. How Noise Affects Accuracy

`modules/simulation.py` can add configurable random noise (0%–30%) to the
true distances before they're handed to trilateration.

- **At 0% noise:** simulated measurements exactly equal the true distances,
  the circles agree almost perfectly, and the estimated position lands
  essentially on top of the hidden ground truth (error ≈ 0).
- **At higher noise:** measured distances drift further from the truth, the
  circles stop agreeing as well, and the best-fit position tends to drift
  further from the ground truth — so positioning error generally increases.
  (Any single random trial can still go either way — this is a general
  trend, not a guarantee every single time.)

## 9. How Position Error Is Calculated

```
Error = sqrt((Xactual - Xestimated)^2 + (Yactual - Yestimated)^2)
```

The ground-truth position is used **only here**, after the estimate has
already been produced, purely to grade how accurate it was.

## 10. Project Structure

```
INTRIX/
│
├── app.py                  Streamlit interface
├── requirements.txt
├── README.md
│
├── data/
│   ├── beacons.csv         Known, fixed beacon coordinates
│   └── locations.csv       Generic indoor destinations
│
├── modules/
│   ├── trilateration.py    Distance formula, trilateration, position error
│   ├── simulation.py       Hidden ground truth, simulated measurements, noise
│   ├── locator.py          Destination loading + distance calculation
│   └── visualization.py    Plotly map: beacons, circles, estimated position
│
└── tests/
    └── test_intrix.py      Simple test script (see "Testing" below)
```

## 11. Installation

Requires Python 3.9+.

```bash
cd INTRIX
pip install -r requirements.txt
```

## 12. Running the Application

```bash
streamlit run app.py
```

This opens INTRIX in your browser. Use the sidebar/slider to change the
measurement noise level, click **New Random Scenario** to test a different
hidden device position, and click **Locate Device** to run trilateration.

## 13. Replacing Beacon / Destination Data

INTRIX is general-purpose — it is not built specifically for any one kind of
building. To adapt it to a different space:

- Edit `data/beacons.csv` with your own beacon names and `(x, y)` coordinates
  (in meters, on any consistent 2D floor-plan grid). Use **3 or more**
  beacons, and make sure they are not all in a straight line.
- Edit `data/locations.csv` with your own destination names and coordinates.

No changes to the Python code are required — the mathematical engine in
`modules/trilateration.py` works the same regardless of the building.

## 14. Testing

`tests/test_intrix.py` is a small, dependency-free test script (no `pytest`
required) that checks:

1. **Distance formula** — distance between `(0,0)` and `(3,4)` is `5`.
2. **Zero-noise trilateration** — at 0% noise, the estimated position matches
   the hidden ground truth almost exactly (error ≈ 0).
3. **Noise effect** — average error at 25% noise is greater than average
   error at 5% noise, across many random trials.
4. **Different ground-truth positions** — trilateration stays accurate
   across 10 different random hidden positions.
5. **Beacon geometry** — the sample beacons are not collinear (a
   requirement for 2D trilateration to work at all).
6. **Destination locator** — distance from an estimated position to a
   destination is calculated correctly.

Run it with:

```bash
python tests/test_intrix.py
```

## 15. Limitations

- This is a mathematical/computational simulation, not a working Bluetooth
  positioning product. It does not use real Bluetooth hardware or real RSSI
  signals.
- The simulated noise model is a simplified stand-in for real-world Bluetooth
  signal variability (multipath, interference, obstacles, etc.), which is
  considerably more complex in practice.
- The simulation assumes a flat, single-floor 2D space.
- Trilateration's accuracy in real deployments also depends heavily on
  beacon placement/geometry, which this project only briefly touches on.

## 16. What INTRIX Is / Is Not

**INTRIX is NOT:**
- A real Bluetooth tracking device
- A GPS replacement
- A physical hardware system
- A building-specific (e.g. mall-only) navigation app
- An AI or machine-learning project
- A complete commercial indoor positioning solution

**INTRIX IS:**

> A mathematical and computational simulation demonstrating how distance
> measurements from fixed Bluetooth beacons can be used to estimate an
> unknown indoor position through trilateration.
