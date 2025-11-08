# C9: Occupancy Detection Analysis

## Overview
Analyzes dwell time and occupancy patterns using BLE signals for space utilization monitoring.

## Key Findings
- **Dwell time analysis**: Mean measurement duration varies by position
- **Occupancy events**: 56 discrete presence events detected
- **Detection quality**: Decreases with distance (packets/sec metric)
- **Position dependency**: Different positions show different occupancy patterns
- **BLE suitability**: Good for binary occupancy (occupied/vacant), not precise counting

## Methodology
1. Dwell time calculation from measurement sessions
2. Occupancy event detection using RSSI threshold + temporal gaps
3. Position-specific occupancy pattern analysis
4. Distance effect on detection quality
5. Packet rate as occupancy reliability indicator

## Results
- 36 dwell periods analyzed
- 56 occupancy events detected
- Position and distance statistics generated
- Detection quality quantified (packets/second)

## Practical Applications
- **Room occupancy**: Vacant vs occupied detection
- **Dwell time monitoring**: How long people stay in areas
- **Space utilization**: Usage patterns by location
- **Capacity management**: Binary presence, not counting

## Recommendations
- Use packet count threshold for occupancy determination
- Apply 5-10 second temporal windows
- Account for distance-dependent quality
- Combine with B8 presence detection for robustness

## Files
- `dwell_times.csv` - Session duration and packet statistics
- `occupancy_events.csv` - Detected presence events
- `position_occupancy.csv` - Position-specific patterns
- `distance_occupancy.csv` - Distance effect analysis
- Visualizations: 6-panel occupancy analysis
- `occupancy_detection_report.txt` - Detailed findings
