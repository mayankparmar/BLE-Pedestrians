# A4: Body Shadowing Effects Analysis

## Overview
Detailed quantification of body shadowing effects by comparing Line-of-Sight (LoS) vs Non-Line-of-Sight (nLoS) RSSI measurements across multiple distances and pathway positions.

## Key Findings

### Overall Shadowing Effect
- **Mean Attenuation**: 10.43 ± 10.58 dB
- **Range**: -6.92 to 34.79 dB (highly variable)
- **Classification**: STRONG effect (> 6 dB)
- **Samples**: 27 matched configurations

### Shadowing by Distance
| Distance | LoS RSSI | nLoS RSSI | Shadowing | Std Dev |
|----------|----------|-----------|-----------|---------|
| 3m       | -58.96 dBm | -70.64 dBm | 11.68 dB | 13.06 dB |
| 7m       | -61.70 dBm | -71.92 dBm | 10.21 dB | 10.69 dB |
| 9m       | -62.75 dBm | -72.14 dBm | 9.39 dB  | 8.70 dB  |

**Pattern**: Shadowing effect decreases slightly with distance (11.68 → 9.39 dB)

### Shadowing by Position
| Position    | Shadowing | Std Dev | Interpretation |
|-------------|-----------|---------|----------------|
| Start       | 5.27 dB   | 6.25 dB | Weak effect    |
| Center      | 22.33 dB  | 7.87 dB | **Very strong**|
| End         | 3.68 dB   | 4.68 dB | Weak effect    |

**Surprising Finding**: Center position shows **4x stronger shadowing** than start/end!
- This contradicts expectations - center should be similar to other positions
- Likely due to specific geometry or reflection patterns at center point

### Statistical Significance

**Unexpected Result**: Despite 10+ dB mean difference, none are statistically significant!

| Distance | Effect Size | p-value | Significant? |
|----------|-------------|---------|--------------|
| 3m       | 5.09 dB     | 0.275   | NO           |
| 7m       | 5.19 dB     | 0.134   | NO           |
| 9m       | 2.69 dB     | 0.502   | NO           |

**Why?** High variability (σ = 10.58 dB) overwhelms the mean difference
- Overlapping distributions
- Small sample sizes (3 runs per configuration)
- Environmental multipath noise (as seen in A3)

## Interpretation

### Body Shadowing Is Real But Inconsistent

**Good news**: Average 10.43 dB attenuation confirms body shadowing exists

**Bad news**: 10.58 dB standard deviation means:
- Individual measurements unreliable
- Cannot confidently classify LoS vs nLoS from single RSSI reading
- Need statistical aggregation (averaging multiple samples)

### Position Matters More Than Expected

**Center position anomaly** (22.33 dB) suggests:
1. Specific geometric alignment at center creates maximum shadowing
2. Multipath reflections may be destructively interfering in nLoS
3. This position directly opposite observer → maximum body blockage

### Comparison with A1 Results

From A1 Path Loss Modeling:
- Reported average body shadowing: 4.32 dB

From A4 detailed analysis:
- Actual average body shadowing: 10.43 dB

**Why the difference?**
- A1 used distance-averaged aggregation
- A4 uses position-by-position calculation
- A4 reveals position-dependent variability hidden in A1

## Practical Implications

### For LoS/nLoS Classification

**Simple threshold won't work**:
- 10.58 dB std dev >> 10.43 dB mean difference
- Distributions heavily overlap
- Need probabilistic classification

**Recommendation**:
- Use machine learning (E14)
- Time-averaged RSSI (not instantaneous)
- Additional features beyond RSSI (e.g., RSSI variance, trend)

### For Distance Estimation

**Beware position effects**:
- Center position: 22 dB shadowing
- End position: 4 dB shadowing
- **5x difference** depending on where pedestrian is!

**Impact**:
- LoS/nLoS unknown → ±10 dB uncertainty
- Translates to ±2-3m distance error
- Must either: (1) know orientation, or (2) use probabilistic methods

### For System Design

**Don't rely on single RSSI reading**:
- Average over 5-10 seconds minimum
- Use median or trimmed mean (robust to outliers)
- Consider RSSI variance as additional feature

## Files Generated

### Data Files
- `shadowing_statistics.csv` - Per-configuration shadowing measurements
- `statistical_tests.csv` - t-test results for each distance

### Visualizations
- `body_shadowing_analysis.png/.pdf` - 4-panel analysis:
  - Shadowing vs distance
  - LoS vs nLoS comparison
  - Position-dependent shadowing
  - Shadowing consistency
- `statistical_significance.png/.pdf` - Statistical test results:
  - P-value visualization
  - Effect size comparison

### Reports
- `body_shadowing_report.txt` - Comprehensive findings

## Surprising Results

1. **Center position anomaly**: 22.33 dB (4x higher than start/end)
2. **Not statistically significant**: Despite 10 dB mean difference
3. **Higher variability than A1**: 10.58 dB vs 4.32 dB reported earlier
4. **Negative shadowing observed**: Min = -6.92 dB (nLoS stronger than LoS!)
   - Likely multipath constructive interference in nLoS cases

## Statistical Summary

```
Metric              | Value
--------------------|--------
Mean Shadowing      | 10.43 dB
Std Dev             | 10.58 dB
Coefficient of Var  | 101%
Min/Max             | -6.92 / 34.79 dB
Range               | 41.71 dB

Interpretation: Extremely high variability
```

## Conclusions

1. **Body shadowing is significant** (10+ dB) but **inconsistent** (10+ dB std dev)
2. **Position matters**: Center shows 4x stronger effect than edges
3. **Simple classification fails**: High overlap requires ML approaches
4. **Multipath dominates**: Variability from multipath > shadowing effect
5. **Statistical averaging required**: Single RSSI reading insufficient

## Next Steps

- **E14: Classification Tasks** - ML-based LoS/nLoS classifier
- **B5: Proximity Detection** - Robust distance estimation handling uncertainty
- **E16: Feature Engineering** - Extract features beyond raw RSSI

## Usage

```bash
python body_shadowing_analysis.py
```

## Requirements
- pandas
- numpy
- matplotlib
- scipy
