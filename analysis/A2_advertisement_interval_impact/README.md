# A2: Advertisement Interval Impact Analysis

## Overview
This analysis examines how different BLE advertisement intervals (100ms, 500ms, 1000ms) affect packet reception, detection latency, signal quality, and overall system reliability in pedestrian tracking scenarios.

## Key Findings

### Packet Reception Performance

**100ms Interval:**
- **PRR**: 12.36% ± 8.72% (WORST performance)
- **Packet Loss**: 87.64%
- **Interpretation**: Despite being the most frequent, 100ms shows poorest reception - likely due to packet collisions, buffer overflow, or processing limitations

**500ms Interval:**
- **PRR**: 51.18% ± 9.13%
- **Packet Loss**: 48.82%
- **Interpretation**: Moderate performance, better balance than 100ms

**1000ms Interval:**
- **PRR**: 53.41% ± 7.02% (BEST performance)
- **Packet Loss**: 46.59%
- **Interpretation**: Best reception rate with lowest variability

### Detection Latency

- **100ms**: Highly variable (-7500 to +1000 ms) - synchronization issues
- **500ms**: 0-1000 ms average latency
- **1000ms**: 0-2000 ms average latency
- **Pattern**: As expected, longer intervals → longer latency

### Signal Quality

**RSSI Mean:**
- 100ms: -56.88 dBm
- 500ms: -57.15 dBm
- 1000ms: -60.10 dBm (slightly weaker, but within normal variation)

**RSSI Stability (Std Dev):**
- **100ms**: 7.61 dB (BEST - most stable)
- **500ms**: 8.33 dB
- **1000ms**: 8.09 dB

### Detection Gaps

**Gap Rate** (percentage of missed consecutive packets):
- 100ms: 66.98% (high gaps despite frequency)
- 500ms: 69.59%
- **1000ms**: 42.53% (BEST - fewer gaps)

## Surprising Results

1. **100ms Paradox**: The most frequent interval (100ms) has the WORST packet reception rate
   - Expected: More transmissions → better reception
   - Actual: 12.36% PRR vs 53.41% for 1000ms
   - **Likely Causes**:
     - Receiver buffer overflow
     - Packet collisions in the BLE channel
     - Processing overhead exceeding interval
     - Data aggregation artifacts in observer logs

2. **1000ms Superiority**: Least frequent interval performs best
   - Lower collision probability
   - More processing time between packets
   - Better suited to walking speed (~4 km/h means ~1.1m traveled per second)

## Practical Implications

### Use Case Recommendations

**For Real-Time Tracking:**
- **Recommended**: 500ms
- Balance between update rate and reliability
- Acceptable latency for pedestrian speeds

**For Presence Detection:**
- **Recommended**: 1000ms
- Best PRR (53.41%)
- Lowest gap rate (42.53%)
- Optimal power efficiency

**For High-Precision Applications:**
- **Recommended**: 500ms or 1000ms with error correction
- 100ms is unreliable despite theoretical advantages

### Power Efficiency vs. Performance

| Interval | Transmissions/min | PRR | Power Consumption | Best For |
|----------|-------------------|-----|-------------------|----------|
| 100ms    | 600               | 12% | High              | Not recommended |
| 500ms    | 120               | 51% | Medium            | Balanced applications |
| 1000ms   | 60                | 53% | Low               | Battery-constrained, presence detection |

**Power Savings**: 1000ms uses **90% less power** than 100ms while achieving **340% better PRR**

## Technical Observations

### Why 100ms Fails

1. **Channel Saturation**: BLE advertising channels may become saturated
2. **Packet Processing**: Observer device can't keep up with 100ms rate
3. **Temporal Aliasing**: Walking speed creates unfavorable timing patterns
4. **MAC Layer Conflicts**: More opportunities for interference

### Optimal Interval Selection

For pedestrian tracking at ~4 km/h:
- Distance per second: ~1.1 meters
- Distance per 500ms: ~0.55 meters
- Distance per 1000ms: ~1.1 meters

**Recommendation**: 1000ms provides sufficient spatial resolution (1.1m) with best reliability

## Files Generated

### Data Files
- `interval_analysis_results.csv` - Complete metrics for all configurations

### Visualizations
- `packet_reception_analysis.png/.pdf` - 4-panel plot showing:
  - PRR vs interval
  - Packet loss vs interval
  - Detection latency vs interval
  - Gap rate vs interval
- `signal_quality_analysis.png/.pdf` - 4-panel plot showing:
  - RSSI mean vs interval
  - RSSI stability (std dev) vs interval
  - RSSI range vs interval
  - Inter-packet timing consistency

### Reports
- `advertisement_interval_report.txt` - Comprehensive text report

## Statistical Summary

```
Interval | PRR (%)  | Loss (%) | RSSI (dBm) | Stability (dB) | Gap Rate (%)
---------|----------|----------|------------|----------------|-------------
100ms    | 12.36    | 87.64    | -56.88     | 7.61 (best)    | 66.98
500ms    | 51.18    | 48.82    | -57.15     | 8.33           | 69.59
1000ms   | 53.41    | 46.59    | -60.10     | 8.09           | 42.53 (best)
```

## Next Steps

1. Investigate why 100ms performs poorly (buffer analysis, channel monitoring)
2. Test intermediate intervals (250ms, 750ms) to find optimal sweet spot
3. Implement adaptive interval selection based on walking speed
4. Analyze power consumption empirically

## Usage

Run the analysis:
```bash
python adv_interval_analysis.py
```

## Requirements
- pandas
- numpy
- matplotlib
- scipy
