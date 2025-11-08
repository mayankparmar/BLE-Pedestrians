# F18: Contact Tracing Simulation

Simulates COVID-19 style contact tracing using BLE proximity detection.

## Key Findings
- Contact threshold: 2m (CDC guideline)
- Detection probability decreases with distance
- Sensitivity: 100% (detects all true contacts)
- Accuracy: 33.3% (false positives at borderline distances)

## Recommendations
- Use conservative RSSI threshold (-65 to -70 dBm)
- Require sustained contact (5+ minutes)
- Combine with movement detection
- Use probabilistic scoring

## Files
- `contact_detection_results.csv` - Detection by distance
- `contact_tracing_simulation.png` - Performance visualization
- `contact_tracing_report.txt` - Detailed findings
