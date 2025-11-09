# B8: Presence Detection Reliability Analysis

## Overview
Assesses the reliability of detecting pedestrian presence using BLE signals. Analyzes detection probability at different distances, RSSI thresholds for reliable detection, packet reception rates, and temporal consistency.

## Key Findings

### Detection Probability
- Detection probability decreases with distance
- RSSI threshold selection trades off range vs false positives
- LoS provides more reliable detection than nLoS
- Different distances require different threshold strategies

### Reliable Detection Ranges
Different RSSI thresholds enable different detection ranges:
- **High threshold (-50 dBm)**: Very short range, very reliable
- **Medium threshold (-70 dBm)**: Good for <5m range
- **Low threshold (-90 dBm)**: Longer range but less reliable

### Packet Reception Rate
- Packet reception correlates with detection reliability
- Lower PRR at longer distances
- PRR serves as presence confidence indicator

### Temporal Consistency
- Inter-packet intervals vary with distance
- Consistency score quantifies detection stability
- Longer gaps reduce presence detection confidence

## Methodology

### 1. Detection Probability Analysis
For various RSSI thresholds (-90, -80, -70, -60, -50 dBm):
- Calculate probability of detecting at least one packet
- Analyze by distance and orientation
- Determine reliable detection ranges (>90% probability)

### 2. Packet Reception Rate
- Count packets received vs expected
- Calculate PRR for each measurement session
- Compare across distances and orientations

### 3. Temporal Consistency
- Calculate inter-packet intervals
- Identify maximum gaps in detection
- Compute consistency score = 1 / (std_interval + 0.001)

### 4. Detection Range Estimation
- Find maximum distance with ≥90% detection probability
- For each RSSI threshold and orientation
- Provides system design guidance

## Results Summary

### Data Analyzed
- 2,739 RSSI measurements
- 36 measurement sessions
- 5 RSSI thresholds tested
- Distances: 3m, 5m, 7m, 9m

### Detection Scenarios
- 20 detection scenarios analyzed
- LoS and nLoS orientations
- Multiple positions (start, mid_facing, center, mid_away, end)

## Visualizations Generated

### Presence Detection Analysis (6-panel)
1. **Detection Probability vs Distance**: Shows how detection probability changes with distance for different thresholds
2. **Packet Reception Rate**: PRR as function of distance
3. **Detection Range by Threshold**: Reliable range for each RSSI threshold
4. **Temporal Consistency**: Consistency score vs distance
5. **RSSI Distribution**: Signal strength with error bars
6. **Detection Success by Position**: Position-specific detection rates

## Practical Applications

### System Configuration

**Choose RSSI Threshold Based on Application:**

**Short-range presence (<3m):**
```python
RSSI_THRESHOLD = -60  # dBm
MIN_PACKETS = 3  # Require multiple detections
TIME_WINDOW = 5  # seconds
```

**Medium-range (3-7m):**
```python
RSSI_THRESHOLD = -75  # dBm
MIN_PACKETS = 2
TIME_WINDOW = 10  # seconds
```

**Long-range (>7m):**
```python
RSSI_THRESHOLD = -85  # dBm
MIN_PACKETS = 1
TIME_WINDOW = 15  # seconds, allow longer gaps
```

### Presence Detection Algorithm

```python
def detect_presence(rssi_measurements, threshold=-70, min_count=2, window_sec=5):
    """
    Detect pedestrian presence from RSSI measurements

    Args:
        rssi_measurements: List of (timestamp, rssi) tuples
        threshold: RSSI threshold in dBm
        min_count: Minimum packets required
        window_sec: Time window in seconds

    Returns:
        bool: True if presence detected reliably
    """
    # Filter by threshold
    detections = [ts for ts, rssi in rssi_measurements if rssi >= threshold]

    if len(detections) < min_count:
        return False

    # Check temporal consistency
    if len(detections) > 1:
        time_span = max(detections) - min(detections)
        if time_span < window_sec:
            return True

    return len(detections) >= min_count
```

### False Positive Mitigation

**Strategies:**
1. **Multiple packet requirement**: Don't trigger on single packet
2. **Temporal window**: Require detections within time span
3. **RSSI trend**: Check if signal is stable or fluctuating
4. **Packet rate**: Verify packets arrive at expected interval

### False Negative Mitigation

**Strategies:**
1. **Lower threshold**: Trade range for sensitivity
2. **Longer window**: Allow gaps in packet reception
3. **Probabilistic detection**: Use detection probability, not binary
4. **Orientation diversity**: Multiple observers at different angles

## Comparison with Prior Work

**A2 Advertisement Interval:**
- 1000ms interval recommended (53.41% PRR)
- Affects presence detection packet rate
- Longer intervals reduce detection frequency

**A4 Body Shadowing:**
- 10.43 dB LoS/nLoS difference affects detection range
- Orientation uncertainty impacts reliability
- Position matters (center 4x stronger shadowing)

**B5 Proximity Detection:**
- Distance estimation complements presence detection
- Can infer presence from distance estimates
- ML approach more robust than threshold

**B6 Movement Detection:**
- Movement affects packet reception patterns
- Walking vs stationary creates different temporal signatures
- Can combine with presence detection for activity state

## Recommendations

### For Occupancy Monitoring
- Use -70 dBm threshold for room-scale presence
- Require 3+ packets in 10-second window
- Update presence state every 5 seconds
- Handle temporary signal losses gracefully

### For Social Distancing
- Use -65 dBm for ~3m detection range
- Combine with distance estimation (B5)
- Require temporal consistency (2+ seconds)
- Account for body orientation effects

### For Access Control
- Use high threshold (-60 dBm) for near-range
- Require sustained detection (5+ seconds)
- Multiple packets mandatory
- Combine with other authentication factors

### For Traffic Counting
- Use lower threshold (-80 dBm) for wider coverage
- Allow brief detection (1-2 packets sufficient)
- Track transitions (enter/exit) not continuous presence
- Apply to pathway analytics (C10)

## Limitations

### Environmental
1. **Multipath interference**: 10.7x variability (A3) affects reliability
2. **Body shadowing**: LoS/nLoS creates 10 dB difference
3. **Position dependency**: Different positions show different detection rates
4. **Obstacle effects**: Not tested with environmental obstacles

### Methodological
1. **Static measurements**: Pedestrians stopped at milestones
2. **Limited distances**: Only 4 distances tested
3. **Known positions**: Real-world has unknown pedestrian locations
4. **Single broadcaster**: Multiple pedestrians not tested

### System
1. **Advertisement interval**: Fixed at ~1000ms (from A2)
2. **No diversity**: Single observer angle
3. **No tracking**: Binary presence, not continuous tracking
4. **No filtering**: Raw packet detection without smoothing

## Files Generated

### Data Files
- `detection_probability.csv` - Detection probability by threshold/distance/orientation
- `packet_reception_rate.csv` - PRR for each measurement session
- `temporal_consistency.csv` - Inter-packet interval statistics
- `detection_ranges.csv` - Reliable range for each threshold

### Visualizations
- `presence_detection_analysis.png/.pdf` - 6-panel comprehensive analysis

### Reports
- `presence_detection_report.txt` - Detailed findings

## Key Insights

### 1. Threshold Selection is Critical
RSSI threshold determines detection range and reliability. Lower threshold = longer range but more false positives. Must balance based on application requirements.

### 2. Packet Rate Matters
With 1000ms advertisement interval (A2), expect ~1 packet/second. Detection delay is inherent. For faster detection, would need faster advertising (but lower PRR per A2).

### 3. Multiple Packets Reduce False Positives
Requiring 2-3 packets in time window dramatically reduces false detections while maintaining good presence detection rate.

### 4. Temporal Consistency is Reliability Indicator
Consistent inter-packet intervals indicate stable detection. Large gaps suggest marginal detection zone or movement.

### 5. Orientation Matters
LoS provides better detection reliability than nLoS (per A4 body shadowing). System should account for orientation uncertainty.

## Usage

```bash
python presence_detection.py
```

## Requirements
- pandas
- numpy
- matplotlib
- scipy

## Next Steps
- **C9**: Apply to occupancy detection scenarios
- **F18**: Use for contact tracing (presence + proximity)
- **E14**: Classify presence confidence with ML
