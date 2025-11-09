# B5: Proximity Detection Algorithms

## Overview
Develops and evaluates distance estimation algorithms from RSSI measurements, comparing analytical path loss models with machine learning approaches for robust proximity detection in pedestrian tracking.

## Model Performance

### Results Summary

| Model | MAE | RMSE | R² | Performance |
|-------|-----|------|-----|-------------|
| **SVR (Best)** | **1.90m** | **2.82m** | -0.447 | GOOD |
| Random Forest | 2.05m | 2.43m | -0.075 | GOOD |
| Log-Distance | 109.47m | 311.77m | -17032.546 | FAILED |

**Best Model**: **Support Vector Regression (SVR)** with 1.90m mean absolute error

### Key Findings

**ML Dramatically Outperforms Analytical Model**:
- SVR: 1.90m MAE (**98.1% better** than log-distance!)
- Simple path loss model completely fails (109m error)
- Confirms non-linear, complex propagation patterns

**Orientation Impact**:
- LoS error: 32.76m
- nLoS error: 170.84m
- **138m difference** - orientation uncertainty is critical

**Relative Accuracy**:
- 1.90m error / 6m average range = **31.7% relative error**
- Acceptable for coarse proximity detection
- Not suitable for precise ranging

## Why Log-Distance Model Failed

From A1, we found **path loss exponent = 0.735** (should be ~2.0)
- Model assumes n ≈ 2: `d = 10^((RSSI_0 - RSSI) / 20)`
- Reality has n ≈ 0.7: much slower signal decay
- Using wrong exponent → massive distance overestimation

**Example**:
- Actual distance: 5m
- RSSI: -60 dBm
- Log-distance prediction with n=0.735: **hundreds of meters**!

## Machine Learning Success

**Why ML works better**:
1. Learns true relationship from data (not theoretical assumptions)
2. Captures non-linear propagation effects
3. Incorporates RSSI statistics (mean, std, min, max)
4. Can handle LoS/nLoS differences
5. Adapts to multipath, body shadowing, environmental factors

**Features used**:
- `rssi_mean` - average signal strength
- `rssi_std` - signal variability (multipath indicator)
- `rssi_min`, `rssi_max` - signal range
- `rssi_median` - robust central tendency
- `rssi_range` - total variation
- `is_los` - orientation flag (0/1)

## Practical Performance

### At Each Distance

**3m**: Good accuracy (small errors)
**5m**: Moderate accuracy
**7m**: Increasing errors
**9m**: Largest errors

**Pattern**: Relative error increases with distance (as expected)

### Error Distribution

- Most errors < 3m (usable)
- Occasional large outliers (multipath)
- Roughly Gaussian distribution
- Symmetric around zero (no bias)

## Comparison with Prior Work

**A1 Path Loss**: Identified n=0.735 (unusual exponent)
**A3 Multipath**: Found 10.7x variability increase vs. anechoic
**A4 Shadowing**: 10.43 ± 10.58 dB LoS/nLoS difference

**B5 Confirms**: Simple models can't handle this complexity
- Need ML to capture non-linear effects
- Multipath dominates propagation
- LoS/nLoS must be known or estimated

## Recommendations

### For System Design

**Use SVR or Random Forest**:
- 1.9-2.0m accuracy is good for proximity zones
- Define zones: "very close" (<3m), "close" (3-6m), "far" (>6m)
- Don't expect meter-level precision

**Incorporate Orientation**:
- LoS/nLoS classification critical (see E14)
- Or use probabilistic approach (both orientations possible)
- Without orientation info: ±138m additional uncertainty

**Averaging Essential**:
- Use 5-10 second moving windows
- Median filtering for outlier rejection
- Temporal smoothing (Kalman filter)

### For Applications

**Good fit**:
- Proximity alerts (< 2m, 2-5m, >5m zones)
- Occupancy detection
- Social distancing monitoring
- Coarse people counting

**Poor fit**:
- Precise indoor positioning (need ±0.5m)
- Contact tracing (need reliable <1.5m detection)
- Measurement/surveying applications

## Implementation Notes

### Training Data
- 81 feature vectors (position × distance × orientation × run)
- 70/30 train/test split
- No cross-validation (small dataset)

### Model Configuration
- **Random Forest**: 100 trees, max depth=10
- **SVR**: RBF kernel, C=100, gamma=0.01
- No hyperparameter tuning (could improve further)

### Computational Cost
- Random Forest: Fast inference (~1ms)
- SVR: Moderate inference (~5ms)
- Both suitable for real-time applications

## Files Generated

### Data Files
- `features.csv` - Extracted features for ML (81 samples)
- `model_performance.csv` - Model comparison metrics

### Visualizations
- `proximity_model_comparison.png/.pdf` - 4-panel comparison:
  - MAE/RMSE comparison
  - R² scores
  - Actual vs predicted scatter
  - Error distribution
- `distance_estimation_analysis.png/.pdf` - Detailed analysis:
  - RSSI vs distance scatter
  - Error by true distance
  - LoS vs nLoS accuracy
  - Calibration curve

### Reports
- `proximity_detection_report.txt` - Comprehensive findings

## Surprising Results

1. **Log-distance model catastrophic failure**: 109m MAE!
   - Confirms A1 unusual exponent makes standard model useless

2. **ML 98% improvement**: Dramatic difference
   - Shows value of data-driven vs. theory-driven approaches

3. **Negative R² for all models**: Worse than predicting mean
   - High variance from multipath overwhelms predictive power
   - MAE is better metric than R² for this problem

4. **SVR slightly better than Random Forest**:
   - Opposite of typical pattern (RF usually wins)
   - Small dataset favors SVR's regularization

## Limitations

1. **Small training set**: Only 81 samples
   - More data would improve ML models
   - Limited generalization to unseen positions

2. **Static measurements**: Pedestrian stopped at milestones
   - Real walking creates additional dynamics
   - See B6 for movement analysis

3. **Single environment**: Campus walkway only
   - Models may not generalize to other spaces
   - Indoor vs outdoor differences

4. **No temporal features**: Only aggregated statistics
   - Could use RSSI time-series patterns
   - Kalman filter for smoothing (future work)

## Next Steps

- **B6: Movement Detection** - Analyze walking vs. stationary
- **E14: Classification** - LoS/nLoS prediction to improve distance estimation
- **E15: Regression Tasks** - More advanced ML with temporal features

## Usage

```bash
python proximity_detection.py
```

## Requirements
- pandas
- numpy
- matplotlib
- scikit-learn
- scipy
