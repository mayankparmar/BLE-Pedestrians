# B6: Movement Detection & Direction Analysis

## Overview
Distinguishes walking from stationary patterns and detects direction of movement using RSSI temporal characteristics. This analysis examines how RSSI changes over time to infer pedestrian movement and direction.

## Key Findings

### Movement Detection

**RSSI Temporal Patterns:**
- RSSI variability can indicate movement
- Rate of change in RSSI correlates with pedestrian activity
- Dynamic positions show different signal characteristics than static positions

**Signal Characteristics:**
- **Static positions (center)**: Lower temporal variability
- **Dynamic positions (start/end)**: Higher temporal variability
- RSSI range and standard deviation serve as movement indicators

### Direction Detection

**RSSI Trend Analysis:**
- **Positive slope**: Approaching (signal getting stronger)
- **Negative slope**: Departing (signal getting weaker)
- Trend strength measured by correlation coefficient

**Limitations:**
- Multipath fading can mask direction trends
- Body orientation changes affect signal strength
- Short measurement duration limits trend reliability

## Methodology

### 1. Data Loading
- Loaded data from Deployment Distance dataset
- 36 measurement sessions (5 positions × 3 runs × varying distances)
- 2,739 RSSI measurements analyzed

### 2. Movement Feature Extraction
- **RSSI variability**: Standard deviation of RSSI rate of change
- **RSSI trend**: Linear slope of RSSI over time
- **RSSI range**: Max - min RSSI during measurement
- **Mean RSSI rate**: Average absolute rate of change

### 3. Direction Detection
- Linear regression of RSSI vs time
- Slope sign indicates direction
- Correlation coefficient indicates trend strength

### 4. Static vs Dynamic Comparison
- Static positions: Center (milestone measurements)
- Dynamic positions: Start, mid_facing, mid_away, end

## Results Summary

### Movement Features
- **Average RSSI variability**: Varies by position and distance
- **RSSI std deviation**: Higher for dynamic positions
- **RSSI range**: Indicates signal fluctuation during measurement

### Direction Classification
- Approximately 50/50 split between approaching and departing
- Trend strength varies with distance and orientation
- LoS generally shows stronger trends than nLoS

### Static vs Dynamic
Comparison shows distinct signal characteristics:
- Dynamic positions: More variability
- Static positions: More stable signal
- Useful for occupancy vs. movement detection

## Visualizations Generated

### 1. Movement Detection Analysis (6-panel)
- RSSI variability vs distance
- Signal range variability
- Direction detection: RSSI slope distribution
- Direction classification by distance
- Movement trend reliability
- Signal change rate

### 2. Temporal Patterns
- Example RSSI time series for different positions
- Trend lines showing approach/departure
- Demonstrates temporal signal evolution

## Practical Applications

### Movement Detection
**Use Cases:**
- Detect when pedestrian is walking vs stationary
- Occupancy monitoring (present but static)
- Activity level assessment

**Method:**
- Monitor RSSI rate of change
- Apply threshold to variability metric
- Use rolling window (5-10 seconds)

### Direction Detection
**Use Cases:**
- Determine if pedestrian is approaching or leaving
- Traffic flow direction
- Queue progression

**Method:**
- Calculate RSSI slope over time window
- Positive slope = approaching
- Negative slope = departing
- Check trend strength for reliability

**Limitations:**
- Requires sufficient measurement duration
- Multipath creates noisy signals
- Body orientation changes can confound direction

## Comparison with Prior Work

**A3 Multipath**: Confirmed high variability (10.7x) makes direction detection challenging

**A4 Body Shadowing**: Orientation changes create ~10 dB swings that can mask movement trends

**B5 Proximity**: Movement detection complements distance estimation for trajectory tracking

## Recommendations

### For Real-Time Systems

**Movement Detection:**
```python
# Calculate RSSI rate of change
rssi_rate = np.diff(rssi) / np.diff(time)
rssi_variability = np.std(rssi_rate)

# Threshold for movement
is_moving = rssi_variability > threshold  # e.g., 0.5 dB/s
```

**Direction Detection:**
```python
# Linear trend over window
slope, _ = np.polyfit(time_window, rssi_window, 1)

if slope > 0.1:
    direction = "approaching"
elif slope < -0.1:
    direction = "departing"
else:
    direction = "stationary"
```

### System Design

**Configuration:**
- Measurement window: 5-10 seconds minimum
- Update rate: 1 Hz (1000ms advertisement interval)
- Smoothing: Moving average to reduce noise

**Integration:**
- Combine with distance estimation (B5) for full trajectory
- Use position classification (E14 planned) for context
- Apply to pathway analytics (C10) for flow analysis

## Limitations

1. **No GPS validation**: Dataset files don't include GPS data for ground truth
2. **Short durations**: Measurements are brief, limiting trend analysis
3. **Multipath interference**: High variability obscures true movement patterns
4. **Static measurements**: Data collected at milestones, not continuous walking
5. **Environment-specific**: Linear walkway may differ from complex spaces

## Future Enhancements

**Recommended Additions:**
- Kalman filtering for smoother trends
- Frequency domain analysis (FFT for gait detection)
- Machine learning classifier for movement states
- Integration with GPS for validation
- Longer continuous tracking sessions

**Related Analyses:**
- **B7: Trajectory Reconstruction** - Combine position and movement
- **E16: Feature Engineering** - Extract temporal patterns
- **C10: Pathway Analytics** - Apply to traffic flow

## Files Generated

### Data Files
- `movement_features.csv` - Extracted movement features (36 sessions)
- `direction_analysis.csv` - Direction detection results (36 sessions)

### Visualizations
- `movement_detection_analysis.png/.pdf` - 6-panel analysis
- `temporal_patterns.png/.pdf` - Example time series

### Reports
- `movement_detection_report.txt` - Comprehensive findings

## Key Metrics

| Metric | Value | Interpretation |
|--------|-------|----------------|
| Sessions analyzed | 36 | 5 positions × varying distances |
| RSSI measurements | 2,739 | Total data points |
| Avg RSSI variability | Varies | Depends on position/distance |
| Direction split | ~50/50 | Approaching vs departing |

## Insights

### 1. Movement is Detectable from RSSI
While noisy, RSSI temporal patterns can indicate movement. Variability metrics distinguish dynamic from static scenarios.

### 2. Direction Requires Careful Analysis
Simple slope analysis can detect direction, but multipath interference and short durations reduce reliability. Longer measurement windows needed for production systems.

### 3. Position Matters
Different pathway positions show distinct temporal characteristics. Center (static milestone) differs from start/end (approach/departure zones).

### 4. Complementary to Distance
Movement detection enhances distance estimation by providing context: is the signal changing because of movement or multipath?

## Usage

```bash
python movement_detection.py
```

## Requirements
- pandas
- numpy
- matplotlib
- scipy

## Next Steps
- **B7**: Use movement features for trajectory reconstruction
- **E16**: Engineer temporal features for ML models
- **C10**: Apply to pathway analytics and flow analysis
