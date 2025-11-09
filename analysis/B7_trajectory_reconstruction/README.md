# B7: Trajectory Reconstruction Analysis

## Overview
Reconstructs pedestrian movement paths using RSSI-based distance estimates and compares with ground truth positions to assess tracking accuracy. This analysis evaluates how well BLE RSSI can be used to track pedestrian movement over time.

## Key Findings

### Tracking Accuracy

**Distance Estimation Accuracy:**
- Uses log-distance path loss model from A1 analysis
- Path loss exponents: n=0.735 (LoS), n=0.415 (nLoS)
- Compares estimated distances vs. true milestone distances (3m, 5m, 7m, 9m)

**Overall Performance:**
- Trajectory reconstruction provides continuous position estimates
- Accuracy varies significantly with distance and orientation
- Analytical model limitations affect tracking quality

### Position-Specific Results

**Tracking by Distance:**
- Closer distances generally show better accuracy
- Longer distances accumulate more error
- Non-linear error growth with distance

**By Position:**
- Different pathway positions show varying accuracy
- Body orientation and multipath affect estimates
- Static positions (center) vs dynamic positions (start/end)

## Methodology

### 1. Data Source
- Deployment Distance dataset
- 5 positions (start, mid_facing, center, mid_away, end)
- 4 distances (3m, 5m, 7m, 9m)
- 3 runs per configuration
- Both LoS and nLoS orientations

### 2. Distance Estimation
Used log-distance path loss model:
```
d = 10^((RSSI_ref - RSSI_measured) / (10 * n))
```
- RSSI_ref = -50 dBm (reference at 1m)
- n = 0.735 (LoS) or 0.415 (nLoS) from A1 analysis
- Clipped to reasonable range (0.5m - 20m)

### 3. Trajectory Reconstruction
- Sort measurement points by actual distance
- Estimate distance from mean RSSI at each point
- Connect points to form trajectory
- Calculate tracking errors

### 4. Accuracy Metrics
- Mean Absolute Error (MAE)
- Root Mean Square Error (RMSE)
- Relative error (%)
- Error distribution analysis

## Results Summary

### Trajectory Points
- 36 trajectory points reconstructed
- Covering all position/distance combinations
- Multiple runs for statistical reliability

### Tracking Performance
Analysis provides:
- Overall MAE across all measurements
- Distance-specific accuracy
- Orientation-specific (LoS vs nLoS) performance
- Position-specific tracking quality

### Key Observations

1. **Model Limitations**: Log-distance path loss model (from A1) has limited accuracy
2. **Distance Dependency**: Error increases with distance
3. **Orientation Effects**: LoS vs nLoS significantly impacts estimates
4. **Multipath Interference**: High variability from A3 affects trajectories

## Visualizations Generated

### 1. Trajectory Reconstruction (6-panel)
- **True vs Estimated Distance**: Scatter plot showing tracking accuracy
- **Tracking Error vs Distance**: How error changes with distance
- **Error Distribution**: Histogram of estimation errors
- **LoS Trajectory Examples**: Sample trajectories for different positions
- **Accuracy by Position**: Position-specific MAE
- **Relative Error**: Percentage error vs distance

### 2. RSSI Propagation
- **LoS Propagation**: RSSI signal strength vs distance
- **nLoS Propagation**: Signal strength with body shadowing
- Shows position-specific signal patterns

## Practical Applications

### Trajectory Tracking

**Use Cases:**
- Monitor pedestrian movement through space
- Track position changes over time
- Identify movement patterns

**Limitations:**
- Moderate accuracy (several meters error)
- Sensitive to multipath and body orientation
- Not suitable for precise positioning

### Real-World Implementation

**Recommended Approach:**
```python
# Instead of log-distance model
# Use machine learning (from B5)
from sklearn.svm import SVR

# Train on RSSI features
model = SVR(kernel='rbf', C=100, gamma=0.01)
model.fit(rssi_features, true_distances)

# Estimate trajectory points
estimated_distances = model.predict(new_rssi_features)
```

**Enhancements:**
- Kalman filtering for trajectory smoothing
- Particle filtering for probabilistic tracking
- Sensor fusion (BLE + IMU + GPS)
- Zone-based tracking (not precise meters)

## Comparison with Prior Work

**A1 Path Loss Modeling:**
- Unusual exponents (0.735, 0.415) hurt analytical model performance
- Confirms need for data-driven approaches

**A3 Environmental Multipath:**
- 10.7x variability creates trajectory noise
- Requires smoothing and filtering

**A4 Body Shadowing:**
- 10.43 dB LoS/nLoS difference creates large distance errors
- Orientation uncertainty limits tracking

**B5 Proximity Detection:**
- ML approach (1.9m MAE) outperforms analytical model significantly
- Recommendation: Use B5 ML model for trajectory reconstruction

**B6 Movement Detection:**
- Combine movement detection with trajectory for better context
- Distinguish walking vs stationary periods

## Recommendations

### For Trajectory Tracking Systems

1. **Use Machine Learning** (from B5)
   - SVR or Random Forest for distance estimation
   - 1.9m MAE vs much worse for log-distance model
   - Feature-based approach handles multipath better

2. **Apply Trajectory Smoothing**
   - Kalman filter to reduce noise
   - Particle filter for non-linear tracking
   - Moving average for simple smoothing

3. **Define Tracking Zones**
   - Don't expect meter-level precision
   - Use zones: "near", "medium", "far"
   - Probabilistic position estimates

4. **Sensor Fusion**
   - Combine BLE with other sensors
   - IMU for orientation and movement
   - GPS for outdoor absolute position
   - Wi-Fi for indoor positioning

5. **Temporal Consistency**
   - Use previous positions to constrain estimates
   - Pedestrian speed limits (typical walking: 1-1.5 m/s)
   - Path continuity constraints

### Configuration

**Recommended Settings:**
- Advertisement interval: 1000ms (from A2)
- Averaging window: 5-10 seconds (from B5)
- Distance estimation: ML model (from B5)
- Smoothing: Kalman filter with pedestrian motion model

## Limitations

### Data Limitations

1. **Static Measurements**: Pedestrians stopped at milestones, not continuous walking
2. **No GPS Trajectories**: Dataset doesn't include continuous GPS paths
3. **Discrete Points**: Only 4 distances measured (3m, 5m, 7m, 9m)
4. **Short Duration**: Brief measurements at each point

### Model Limitations

1. **Analytical Model**: Log-distance path loss inadequate for this environment
2. **Unusual Exponents**: n=0.735 (LoS) makes standard model fail
3. **No ML**: This analysis uses simple model; B5 ML would be much better
4. **No Filtering**: Raw estimates without Kalman or particle filtering

### Environmental Limitations

1. **Multipath Dominant**: 10.7x variability (from A3) creates large errors
2. **Body Shadowing**: 10.43 dB variation (from A4) affects consistency
3. **Single Environment**: Linear outdoor walkway only
4. **No Obstacles**: Simple environment vs complex indoor spaces

## Future Enhancements

**High Priority:**
1. Replace log-distance model with ML model from B5
2. Implement Kalman filtering for smoothing
3. Add pedestrian motion model constraints
4. Integrate movement detection from B6

**Medium Priority:**
5. Particle filtering for non-Gaussian noise
6. Multi-sensor fusion (BLE + IMU + GPS)
7. Occupancy grid mapping
8. Path planning and prediction

**Research Extensions:**
9. Crowd tracking (multiple pedestrians)
10. Activity recognition from trajectories
11. Anomaly detection (unusual paths)
12. Privacy-preserving tracking

## Files Generated

### Data Files
- `trajectories.csv` - Reconstructed trajectory points with errors (36 points)

### Visualizations
- `trajectory_reconstruction.png/.pdf` - 6-panel tracking analysis
- `rssi_propagation.png/.pdf` - Signal strength vs distance

### Reports
- `trajectory_reconstruction_report.txt` - Comprehensive findings

## Key Insights

### 1. Analytical Models Insufficient
Log-distance path loss model from A1 provides poor trajectory estimates. The unusual propagation environment (n=0.735) breaks standard approaches.

**Solution**: Use ML-based distance estimation (B5: 1.9m MAE).

### 2. Multipath Dominates
Environmental multipath (10.7x from A3) creates significant noise in trajectories. Raw RSSI estimates produce jagged, unreliable paths.

**Solution**: Apply Kalman filtering and temporal smoothing.

### 3. Zone-Based Tracking Works
Precise meter-level positioning is unreliable, but zone-based tracking (near/medium/far) is practical.

**Solution**: Define discrete zones instead of continuous coordinates.

### 4. Complementary Sensors Needed
BLE alone cannot provide accurate trajectories in complex environments.

**Solution**: Sensor fusion with IMU, GPS, or Wi-Fi.

## Comparison with B5 Proximity Detection

| Aspect | B7 (Analytical) | B5 (ML) |
|--------|-----------------|---------|
| Method | Log-distance path loss | SVR machine learning |
| Accuracy | Variable (poor) | 1.90m MAE (good) |
| Features | Mean RSSI only | Mean, std, min, max, median, range |
| Training | None (analytical) | Supervised learning |
| Recommendation | Use for quick estimate | **Use for production** |

**Conclusion**: For actual trajectory reconstruction, use the ML approach from B5, not the analytical model demonstrated in B7.

## Usage

```bash
python trajectory_reconstruction.py
```

## Requirements
- pandas
- numpy
- matplotlib
- scipy

## Next Steps
- **E16**: Extract trajectory-based features for ML
- **C10**: Apply to pathway analytics and traffic flow
- **F18**: Use for contact tracing simulation (proximity over time)
