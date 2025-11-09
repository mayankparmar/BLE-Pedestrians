# A1: Path Loss Modeling Analysis

## Overview
This analysis fits empirical path loss models to BLE RSSI measurements at different deployment distances (3m, 5m, 7m, 9m) to characterize signal propagation in both Line-of-Sight (LoS) and Non-Line-of-Sight (nLoS) conditions.

## Key Findings

### LoS Propagation
- **Path Loss Exponent**: 0.735 ± 0.230 (significantly lower than free-space value of 2.0)
- **RSSI at 1m**: -55.89 ± 1.76 dBm
- **Model Fit**: R² = 0.8368, RMSE = 0.580 dB
- **Interpretation**: The low path loss exponent (< 2.0) suggests waveguide or ground reflection effects enhancing signal propagation beyond free-space expectations

### nLoS Propagation
- **Path Loss Exponent**: 0.415 ± 0.390 (even lower, with higher uncertainty)
- **RSSI at 1m**: -62.31 ± 3.07 dBm
- **Model Fit**: R² = 0.5309, RMSE = 0.797 dB
- **Interpretation**: Limited body shadowing effect with high signal variability

### Body Shadowing Effect
- **Average Signal Attenuation**: 4.32 dB (nLoS compared to LoS)
- **Distance-specific effects**:
  - 3m: 5.09 dB attenuation
  - 7m: 5.19 dB attenuation
  - 9m: 2.69 dB attenuation (reduced effect at longer distances)

## Unexpected Results

The path loss exponents are significantly lower than the theoretical free-space value of 2.0:
- **Typical Indoor Values**: 2.0 - 4.0
- **Measured LoS**: 0.735
- **Measured nLoS**: 0.415

This suggests:
1. **Constructive Interference**: Ground reflections or multipath may be creating constructive interference
2. **Corridor/Walkway Effect**: The linear 24m walkway may act as a waveguide
3. **Limited Distance Range**: The 3-9m range may be too short to capture true path loss behavior
4. **Antenna Characteristics**: Directional effects or antenna positioning may influence results

## Files Generated

### Data Files
- `los_statistics.csv` - Statistical summary of LoS RSSI measurements
- `nlos_statistics.csv` - Statistical summary of nLoS RSSI measurements

### Visualizations
- `path_loss_analysis.png` / `.pdf` - Comprehensive 4-panel visualization showing:
  - LoS path loss with fitted model
  - nLoS path loss with fitted model
  - LoS vs nLoS comparison
  - Path loss exponent comparison
- `position_analysis.png` - RSSI variation across pathway positions (Start, Mid Facing, Center, Mid Away, End)

### Reports
- `path_loss_report.txt` - Detailed text report with all findings

## Usage

Run the analysis:
```bash
python path_loss_analysis.py
```

## Requirements
- pandas
- numpy
- matplotlib
- scipy

## Next Steps

For follow-up analyses:
1. **A2: Advertisement Interval Impact** - Analyze how different broadcast intervals affect detection
2. **A3: Environmental Multipath Analysis** - Compare controlled (anechoic) vs field measurements
3. **A4: Body Shadowing Effects** - Deeper investigation of orientation-dependent attenuation
