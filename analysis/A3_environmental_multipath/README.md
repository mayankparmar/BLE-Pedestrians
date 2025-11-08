# A3: Environmental Multipath Analysis

## Overview
This analysis compares controlled anechoic chamber measurements with real-world field data to characterize multipath fading, quantify environmental effects on BLE signals, and analyze angular signal dependency.

## Key Findings

### Anechoic Chamber (Controlled Environment)

**Angular Signal Pattern (0° to 180° rotation):**
- **0°**: -55.64 ± 0.57 dBm (facing forward)
- **45°**: -55.21 ± 1.46 dBm
- **90°**: -57.44 ± 0.97 dBm (perpendicular)
- **135°**: -55.66 ± 1.40 dBm
- **180°**: -61.76 ± 0.49 dBm (facing backward - **weakest signal**)

**Key Observations:**
- **6.12 dB attenuation** from 0° to 180° (body/device shadowing)
- Very stable signals: Average Std Dev = **0.98 dB**
- Minimal variability: Average Range = **3.20 dB**
- Clean angular pattern with predictable attenuation

### Field Environment (Real-World Conditions)

**BT Configuration (Back-Top):**
- 3m Clear: -66.35 ± 9.89 dBm (Range: 59 dB)
- 3m Occluded: -69.11 ± 9.92 dBm (Range: 50 dB)
- 5m Clear: -64.59 ± 11.19 dBm (Range: 50 dB)
- 5m Occluded: -67.54 ± 11.42 dBm (Range: 48 dB)

**Signal Characteristics:**
- Highly variable: Average Std Dev = **10.54 dB**
- Wide signal swings: Average Range = **54.50 dB**
- Occlusion adds ~3 dB additional attenuation

### Multipath Effect Quantification

**DRAMATIC MULTIPATH IMPACT:**

| Metric | Anechoic | Field (Clear) | Increase | Multipath Factor |
|--------|----------|---------------|----------|------------------|
| Std Dev | 0.98 dB | 10.54 dB | +9.56 dB | **10.74x** |
| Range | 3.20 dB | 54.50 dB | +51.30 dB | **17.03x** |

**Interpretation:**
The field environment shows **10.7x more signal variability** compared to the anechoic chamber. This massive increase indicates severe multipath propagation from:
- Ground reflections
- Wall/building reflections
- Scattering from objects
- Temporal fading as pedestrian moves
- Constructive/destructive interference

## Surprising Results

### 1. Extreme Multipath Effect
- Expected: 2-3x variability increase
- Actual: **10.7x to 17x** increase
- This suggests the outdoor walkway environment has significant multipath despite being "open"

### 2. Angular Dependency in Anechoic Chamber
- Clean 6 dB front-to-back ratio confirms device directionality
- 90° position shows slight attenuation (-57.44 dBm) vs. 0°/45°
- Nearly symmetric pattern (45° ≈ 135°)

### 3. Occlusion Has Modest Effect
- Occlusion adds only ~3 dB attenuation
- Much smaller than multipath-induced variability (10 dB std dev)
- Suggests multipath dominates over static obstruction

### 4. Distance Effect Is Non-Linear
- 5m sometimes shows **better** RSSI than 3m in field tests
- Confirms multipath creates standing wave patterns
- Cannot rely on simple distance-RSSI relationship in field

## Practical Implications

### For System Design

**Challenge:** Real-world RSSI has **10x more uncertainty** than lab conditions
- Lab calibration will underestimate field variability
- Need robust algorithms tolerant to ±10 dB fluctuations
- Averaging/filtering essential (but beware of lag)

**Recommendation:**
- Use **time-averaged RSSI** (moving window of 5-10 seconds)
- Implement **outlier rejection** (remove samples beyond 2σ)
- Consider **path loss + multipath** models, not just path loss
- Machine learning may outperform analytical models

### For Pedestrian Tracking

**Don't expect:**
- Stable RSSI for stationary pedestrian (will vary ±10 dB)
- Monotonic RSSI decrease with distance
- Symmetric LoS/nLoS behavior

**Do expect:**
- Rapid signal fluctuations (fading)
- Distance estimation errors of ±2-3 meters at 5m
- Need for probabilistic positioning (not deterministic)

### For Experimental Methodology

**Anechoic chamber is critical** for:
- Device characterization (antenna pattern, Tx power)
- Isolating hardware effects from environment
- Establishing baseline performance

**But remember:**
- Anechoic results are **10x more optimistic** than reality
- Field testing is mandatory for validation
- Multipath cannot be avoided, must be accommodated

## Fading Characteristics

### Temporal Fading
Field measurements show constant signal variation due to:
- Pedestrian movement changing path lengths
- Dynamic reflections as person walks
- Fresnel zone effects

### Spatial Fading
Standing wave patterns create:
- Constructive interference peaks (stronger than expected)
- Destructive interference nulls (weaker than expected)
- ~0.5λ periodicity at 2.4 GHz (λ = 12.5 cm, peaks every ~6 cm)

## Files Generated

### Data Files
- `anechoic_statistics.csv` - Angular pattern statistics (0-180°)
- `field_statistics.csv` - Field measurement statistics by configuration
- `fading_statistics.csv` - Temporal fading characteristics

### Visualizations
- `anechoic_angular_pattern.png/.pdf` - 4-panel angular analysis:
  - RSSI vs rotation angle
  - Variability vs angle
  - Polar radiation pattern
  - Min/max range by angle
- `environment_comparison.png/.pdf` - Anechoic vs field comparison:
  - RSSI distribution comparison
  - Signal stability comparison
  - Multipath factor quantification
  - Configuration variability
- `fading_analysis.png/.pdf` - Fading characteristics:
  - Fade depth by configuration
  - Temporal variation
  - Occlusion effect
  - Fade rate analysis

### Reports
- `multipath_analysis_report.txt` - Comprehensive findings

## Technical Details

### Anechoic Chamber Setup
- Fixed 3m separation
- Broadcaster on rotating platform (0° to 180°, 45° steps)
- Observer stationary
- No reflections (anechoic environment)

### Field Test Configurations
- **FB (Front-Back)**: Broadcaster orientation varies along path
- **BT (Back-Top)**: Specific orientation configuration
- **BD (Back-Down)**: Alternative orientation
- Both 3m and 5m distances tested
- Occlusion tests with obstacles

## Statistical Summary

```
Environment    | Mean RSSI | Std Dev | Range | Samples
---------------|-----------|---------|-------|--------
Anechoic 0°    | -55.64    | 0.57    | 2.0   | 105
Anechoic 180°  | -61.76    | 0.49    | 2.0   | 103
Anechoic Avg   | -57.14    | 0.98    | 3.2   | 529
Field 3m       | -66.35    | 9.89    | 59.0  | 770
Field 5m       | -64.59    | 11.19   | 50.0  | 347
Multipath Factor: 10.74x (std dev), 17.03x (range)
```

## Conclusions

1. **Multipath dominates** real-world BLE propagation (10x effect)
2. **Angular dependency** is clean and predictable in controlled settings
3. **Field measurements** require statistical/probabilistic treatment
4. **Distance estimation** from RSSI alone is unreliable without multipath mitigation
5. **Lab-to-field gap** is enormous - always validate in realistic conditions

## Next Steps

For follow-up analyses:
- **A4: Body Shadowing Effects** - Detailed LoS vs nLoS comparison
- **B5: Proximity Detection** - Robust algorithms accounting for multipath
- **E15: Regression Tasks** - ML-based distance estimation with multipath features

## Usage

Run the analysis:
```bash
python multipath_analysis.py
```

## Requirements
- pandas
- numpy
- matplotlib
- scipy
