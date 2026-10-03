# Giant Fiber dependence of looming escape (malecns_v1.0, looming)

*Generated 2026-10-03T17:12:47Z | git 504549b6be (dirty)*

## Hypothesis
Looming-evoked takeoff in the model is mediated by the Giant Fiber (DNp01) and requires the real synaptic wiring from LPLC2/LC4 onto it.

## Intervention
Silence both DNp01 neurons (all their synapses removed, input clamped). Control: synaptic targets shuffled per presynaptic neuron (out-degree, weights and signs preserved).

## Runs
- `20261003-171139_looming_aa8130` - intact
- `20261003-171211_looming_b8aaad` - giant fiber lesion (DNp01)

## Controls
- `20261003-171139_looming_aa8130_shuffle` - intact [shuffle control]
- `20261003-171211_looming_b8aaad_shuffle` - giant fiber lesion (DNp01) [shuffle control]

## Findings (values verified against stored metrics)

| statement | run | metric | statistic | value |
|---|---|---|---|---|
| Intact connectome: mean takeoff | `20261003-171139_looming_aa8130` | takeoff | mean | 1 |
| Giant fiber lesion: mean takeoff | `20261003-171211_looming_b8aaad` | takeoff | mean | 0 |
| Shuffled connectome: mean takeoff | `20261003-171139_looming_aa8130_shuffle` | takeoff | mean | 0 |
| Intact: Giant Fiber spikes per trial | `20261003-171139_looming_aa8130` | gf_spikes | mean | 6.625 |
| Shuffled: Giant Fiber spikes per trial | `20261003-171139_looming_aa8130_shuffle` | gf_spikes | mean | 0 |

## Statistical comparisons (recomputed)

| run a | run b | metric | p | interpretation |
|---|---|---|---|---|
| `20261003-171139_looming_aa8130` | `20261003-171211_looming_b8aaad` | takeoff | 0.0002 | intact vs GF lesion: Fisher exact test on takeoff |
| `20261003-171139_looming_aa8130` | `20261003-171139_looming_aa8130_shuffle` | takeoff | 0.0002 | intact vs shuffle: Fisher exact test on takeoff |

## Spread across seeds
Std across seeds of takeoff: intact 0.000, lesion 0.000, shuffle 0.000. Std of gf_spikes intact 7.269.

## Controls assessment
Shuffling synaptic targets changed mean takeoff from 1.000 to 0.000 (Fisher p = 0.0002), and Giant Fiber spikes from 6.625 to 0.000.

## Limitations
Takeoff is read out from DNp01 firing by a hand-set threshold, so the lesion result partly follows from the readout design; the informative part is that LPLC2/LC4 drive reaches DNp01 through the real wiring but not through the shuffled one. The jump direction readout is hand-set and not validated.

### Standard model limitations (always included)
- Model = static connectome (synapse counts, predicted transmitter sign) + leaky integrate-and-fire dynamics
  (Shiu et al. 2024). No plasticity, no neuromodulation, no gap junctions, no cell-type-specific membrane properties.
- w_syn is a free parameter (calibrated per dataset: 0.275 mV FlyWire, 0.15 mV MaleCNS).
- Sensory encoders (looming, photoreceptors, odour, Johnston's organ) are engineered mappings onto real neuron
  groups, not models of the periphery.
- Motor side: descending-neuron rates -> hand-set readout -> motor primitives (walk/turn/jump/flight wrench).
  No joint-level control; wing aerodynamics are abstracted. Report as "connectome + engineered readout".
- Results are hypotheses about the wiring diagram, not evidence about real flies.

## Conclusion (hypothesis)
In this model, looming-evoked takeoff depends on the identified Giant Fiber pathway and on the specific connectome wiring (lost after target shuffling). This is a statement about the wiring diagram plus model assumptions, not about real flies.
