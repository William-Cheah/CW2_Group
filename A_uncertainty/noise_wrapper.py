"""Sensor-noise wrapper - plugs into env.unwrapped.sensor_hook (CONTRACT 6.2).

Signature required by the contract:  hook(readings, rng) -> (readings, sensor_fault)

Injects the four fault types from Section 1.1, each controlled by ONE number, fault_rate:
  1. logger drift    - cold-box logger reads too COLD (the dangerous direction:
                       warm blood looks safe)
  2. logger dropout  - logger freezes and repeats its last value
  3. false reports   - a field report is flipped (open <-> flooded)
  4. report blackout - telecom/API outage: no road reports for several steps
sensor_fault=True whenever any fault was injected this step (goes into the log).
"""
import copy
import numpy as np


class SensorNoise:
    def __init__(self, fault_rate=0.1, drift_c=1.5, blackout_steps=4):
        self.fault_rate = fault_rate          # 0.0 = clean sensors
        self.drift_c = drift_c                # how many degC too cold a drifting logger reads
        self.blackout_steps = blackout_steps  # 4 steps = 1 hour outage
        self.blackout_left = 0
        self.last_temp = None

    def __call__(self, readings, rng):
        r = copy.deepcopy(readings)
        fault = False
        if self.fault_rate <= 0:
            return r, False

        temps = np.array(r["box_temp"], dtype=float)
        if self.last_temp is None or len(self.last_temp) != len(temps):
            self.last_temp = temps.copy()

        # 1-2. cold-box logger faults (per van)
        for v in range(len(temps)):
            u = rng.random()
            if u < self.fault_rate / 2:                 # drift: reads too cold
                temps[v] -= self.drift_c
                fault = True
            elif u < self.fault_rate:                   # dropout: stuck at last value
                temps[v] = self.last_temp[v]
                fault = True
        self.last_temp = temps.copy()
        r["box_temp"] = temps

        # 4. report blackout (starts rarely, lasts blackout_steps)
        reports = np.array(r["road_reports"]).copy()
        if self.blackout_left == 0 and rng.random() < self.fault_rate / 5:
            self.blackout_left = self.blackout_steps
        if self.blackout_left > 0:
            reports[:] = -1
            self.blackout_left -= 1
            fault = True
        else:
            # 3. false reports: flip reports that exist
            has_report = reports != -1
            flip = has_report & (rng.random(len(reports)) < self.fault_rate)
            if flip.any():
                reports[flip] = 1 - reports[flip]
                fault = True
        r["road_reports"] = reports
        return r, fault


if __name__ == "__main__":
    rng = np.random.default_rng(0)
    for rate in [0.0, 0.1, 0.25, 0.5]:
        hook = SensorNoise(fault_rate=rate)
        faults, flips, blackouts = 0, 0, 0
        for t in range(1000):
            readings = {"rain_level": 2, "river_level": 0.5,
                        "road_reports": np.array([1, 0, 1, -1, 1, 0, -1]),
                        "box_temp": [5.0, 6.0], "van_seen_flooded": []}
            out, fault = hook(readings, rng)
            faults += fault
            if (out["road_reports"] == -1).all():
                blackouts += 1
            else:
                flips += int((out["road_reports"] != readings["road_reports"]).sum())
        print(f"fault_rate={rate:.2f}: steps with a fault={faults/10:.1f}%  "
              f"flipped reports={flips}  blackout steps={blackouts}")
