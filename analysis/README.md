# BLE Pedestrian Dataset Analysis

## Quick Start

**For comprehensive findings**: See [`COMPREHENSIVE_FINDINGS.md`](COMPREHENSIVE_FINDINGS.md)

**For analysis summary**: See [`ANALYSIS_SUMMARY.md`](ANALYSIS_SUMMARY.md)

## Completed Analyses (5/17)

### 📊 A1: Path Loss Modeling
**Key Finding**: Path loss exponent = 0.735 (vs 2.0 theoretical)
- Unusual propagation suggests waveguide/reflection effects
- Body shadowing: 4.32 dB average (LoS vs nLoS)

**Location**: `A1_path_loss_modeling/`

### 📡 A2: Advertisement Interval Impact
**Key Finding**: 1000ms interval performs BEST (counterintuitive!)
- 1000ms: 53.41% packet reception
- 100ms: 12.36% packet reception (worst)
- 90% power savings with better performance

**Location**: `A2_advertisement_interval_impact/`

### 🌊 A3: Environmental Multipath Analysis
**Key Finding**: Real world shows 10.7x more variability than anechoic chamber
- Field: 10.54 dB std dev
- Anechoic: 0.98 dB std dev
- Multipath dominates signal behavior

**Location**: `A3_environmental_multipath/`

### 👤 A4: Body Shadowing Effects
**Key Finding**: 10.43 ± 10.58 dB shadowing (high mean, high variance)
- Center position: 22.33 dB (4x stronger than edges)
- Not statistically significant despite large difference
- Single RSSI unreliable for LoS/nLoS classification

**Location**: `A4_body_shadowing/`

### 📍 B5: Proximity Detection Algorithms
**Key Finding**: Machine learning achieves 1.90m accuracy; analytical model fails
- SVR: 1.90m MAE (best)
- Log-distance: 109m MAE (useless)
- 98% improvement with ML vs theory

**Location**: `B5_proximity_detection/`

## Major Insights

### 1. Analytical Models Fail
Traditional path loss models produce 109m errors. The unusual propagation environment (n=0.735) makes textbook equations useless. **Use machine learning instead.**

### 2. Multipath Dominates
Field measurements show 10x more variability than controlled conditions. Environmental reflections overwhelm line-of-sight propagation. **Expect ±10 dB uncertainty.**

### 3. Slower Advertisement is Better
1000ms interval outperforms 100ms by 4x while saving 90% power. **Recommendation: Use 1000ms.**

### 4. Distance Estimation Reality Check
Best achievable accuracy: ~2m at 3-9m range (31% relative error). **Define proximity zones, not precise distances.**

### 5. Orientation Matters But Is Unreliable
LoS/nLoS creates 10 dB difference on average, but with 10 dB variance. Cannot reliably classify from single RSSI. **Need statistical aggregation.**

## Practical Recommendations

### System Configuration
```
Advertisement Interval: 1000ms ✓
Deployment Distance: 3m (optimal)
Signal Processing: 5-10 second moving average
Outlier Rejection: Median filtering
```

### Distance Estimation
```python
# DON'T DO THIS (109m error!)
distance = 10**((rssi_ref - rssi) / (10 * path_loss_exponent))

# DO THIS (1.9m error)
from sklearn.svm import SVR
model = SVR(kernel='rbf', C=100, gamma=0.01)
model.fit(features, distances)
distance = model.predict(new_features)
```

### Proximity Zones (Recommended)
```
Very Close: < 3m
Close: 3-6m
Far: > 6m
```

## Directory Structure

```
analysis/
├── README.md (this file)
├── ANALYSIS_SUMMARY.md (progress overview)
├── COMPREHENSIVE_FINDINGS.md (detailed insights)
│
├── A1_path_loss_modeling/
│   ├── path_loss_analysis.py
│   ├── results/
│   └── README.md
│
├── A2_advertisement_interval_impact/
│   ├── adv_interval_analysis.py
│   ├── results/
│   └── README.md
│
├── A3_environmental_multipath/
│   ├── multipath_analysis.py
│   ├── results/
│   └── README.md
│
├── A4_body_shadowing/
│   ├── body_shadowing_analysis.py
│   ├── results/
│   └── README.md
│
└── B5_proximity_detection/
    ├── proximity_detection.py
    ├── results/
    └── README.md
```

## Running Analyses

Each analysis can be run independently:

```bash
# Example: Run path loss analysis
cd analysis/A1_path_loss_modeling
python path_loss_analysis.py

# Results saved to results/ subdirectory
# - CSV files with data
# - PNG/PDF visualizations
# - Text reports
```

## Results Overview

Each analysis generates:
- **CSV files**: Statistical summaries and raw results
- **Visualizations**: PNG (high-res) and PDF (vector) plots
- **Reports**: Comprehensive text summaries
- **README**: Analysis-specific documentation

## Key Metrics

| Metric | Value | Implication |
|--------|-------|-------------|
| Path Loss Exponent | 0.735 | Non-standard propagation |
| Multipath Factor | 10.7x | High environmental impact |
| Body Shadowing | 10.43 ± 10.58 dB | Large but variable |
| Advertisement Interval | 1000ms best | Counterintuitive result |
| ML Accuracy | 1.90m MAE | Good for zones |
| Analytical Model | 109m MAE | Unusable |

## Pending Analyses

**High Priority** (valuable additions):
- E14: LoS/nLoS Classification
- B6: Movement Detection
- F18: Contact Tracing Simulation
- D12: Cross-Dataset Validation

**Medium Priority** (good to have):
- E16: Feature Engineering
- C10: Pathway Analytics
- D13: Optimal Configuration

**Lower Priority** (marginal value):
- D11: Signal Variability
- C9: Occupancy Detection
- B7: Trajectory (GPS provides this)
- B8: Presence Detection (covered by B5)
- E15: Advanced Regression (covered by B5)

## Dependencies

```bash
pip install pandas numpy matplotlib scipy scikit-learn
```

All analyses use:
- Python 3.11+
- pandas (data manipulation)
- numpy (numerical computing)
- matplotlib (visualization)
- scipy (statistical analysis)
- scikit-learn (machine learning, B5 only)

## Citation

If using these analyses or insights, please acknowledge:
- Original dataset: BLE Pedestrian Tracking Dataset
- Analysis framework: Developed during dissertation work
- Key findings: See COMPREHENSIVE_FINDINGS.md

## Contact

For questions about the analyses or to request additional investigations, please refer to the individual analysis READMEs or the comprehensive findings document.

---

**Analysis Status**: 5/17 complete (29%)
**Last Updated**: Current session
**Repository**: BLE-Pedestrians
**Branch**: claude/pedestrian-dataset-collection-011CUvTBiKdfGYVgDaed2DWi
