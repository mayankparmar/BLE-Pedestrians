# IEEE Access Version - Deployment Distance Paper

## Status: COMPLETED ✅

This directory contains the complete IEEE Access format version of the deployment distance paper, converted from the original MDPI template format with comprehensive revisions and enhancements.

### Document Information

**File:** `deployment_distance_ieee_access.tex` (633 lines)
**Target Journal:** IEEE Access
**Template:** IEEE Access LaTeX template (ieeeaccess.cls)
**Bibliography:** `ref.bib` (IEEE numbered citation style)

### Completed Revisions

1. ✅ **Directory structure created** with figures and bibliography
2. ✅ **Converted to IEEE Access LaTeX template** with proper document class
3. ✅ **Rewrote Abstract** with quantitative results (K-factors, p-values, specific findings)
4. ✅ **Wrote comprehensive Related Work section** (NEW - Section III) with 5 subsections reviewing 15+ papers
5. ✅ **Added detailed Limitations section** (NEW - Section 6.2) covering 5 limitation categories
6. ✅ **Added Practical Deployment Guidelines section** (NEW - Section 6.3) with 6 guideline categories
7. ✅ **Enhanced Conclusion** with 4 numbered key findings and future work directions
8. ✅ **Converted bibliography to IEEE numbered format** using IEEEtran style
9. ✅ **Added proper IEEE Access formatting** throughout

### Paper Structure

**Complete 8-section paper:**
1. **Introduction** - Research question, contributions, paper organization
2. **Background** - BLE technology, RSSI, path loss models, Rician distribution
3. **Related Work** (NEW) - BLE monitoring, RSSI positioning, body effects, wireless deployment, research gap
4. **Methodology** - System design, experimental location, data collection protocol, analysis methods
5. **Results** - Descriptive statistics, ANOVA/Tukey's HSD, Rician distribution fitting
6. **Discussion** - Interpretation of findings, Limitations (NEW), Practical Guidelines (NEW)
7. **Conclusion** - Key findings summary, future work, significance
8. **Bibliography** - IEEE numbered citation style

### Key Features

**Enhanced Content:**
- 8 figures (ANOVA plots, Tukey HSD, Rician distributions)
- 5 tables (median RSSI, statistical tests, Rician parameters, weather conditions)
- 40+ citations in IEEE format
- Comprehensive limitations addressing environmental scope, measurement constraints, hardware specificity
- Detailed deployment guidelines for practitioners covering 6 categories

**IEEE Access Requirements Met:**
- Two-column format with ieeeaccess document class
- Enhanced abstract with quantitative results
- Related Work section with gap analysis
- Limitations section explicitly addressing study constraints
- Practical implications and deployment guidelines
- Future work directions
- Proper IEEE citation style

### Key Findings Highlighted

1. **3m and 5m statistically equivalent** (p=0.5249 LoS, p=0.9992 nLoS)
2. **5m exhibits superior LoS dominance** (K-factor = 117.84)
3. **7m and 9m show Rayleigh fading** (K-factors approaching 0)
4. **Body shadowing significant** (27dB RSSI reduction at 3m Centre nLoS)

### Compilation

```bash
cd "Previous work/Deployment_Distance_IEEE_Access"
pdflatex deployment_distance_ieee_access.tex
bibtex deployment_distance_ieee_access
pdflatex deployment_distance_ieee_access.tex
pdflatex deployment_distance_ieee_access.tex
```

### Original Paper Location

`Previous work/Optimal_Horizontal_Deployment_Distance_of_a_BLE_Measurement_Device_for_Reliable_Pedestrian_Measurements_in_Outdoor_Linear_Walkway/`

---

**Conversion completed successfully. Paper ready for IEEE Access submission.**
