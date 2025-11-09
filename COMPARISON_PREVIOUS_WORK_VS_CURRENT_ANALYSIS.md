# Comparison: Previous Work vs Current Analysis & Publications Plan

## Executive Summary

After reviewing the "Previous work" directory (PhD thesis chapters) and comparing them with the current analysis summary and publications plan, I've identified significant **overlaps** and some important **gaps** that should inform modifications to the publications plan.

**Key Finding**: Many analyses in the current plan (A1-A4) already exist in comprehensive form in the previous work, but use different analytical approaches. The publications plan should be adjusted to either:
1. **Remove duplicates** that don't add new insights
2. **Merge analyses** to create stronger combined papers
3. **Add missing analyses** from the previous work that aren't yet in the current plan

---

## Detailed Comparison

### Category A: Signal Propagation & Environmental Influence

| Analysis ID | Current Plan | Previous Work Chapter | Overlap Assessment |
|-------------|-------------|---------------------|-------------------|
| **A1: Path Loss Modeling** | ✅ Complete<br>- Path loss exponent n=0.735<br>- Body shadowing: 4.32 dB | **Ch 4, Section 4.3**: LS and SS Fading<br>- Log-distance path loss model<br>- Residual analysis<br>- Rician fading | ⚠️ **HIGH OVERLAP**<br>Different analytical approach but same phenomenon |
| **A2: Advertisement Interval** | ✅ Complete<br>- 1000ms optimal (53% PRR)<br>- 100ms worst (12% PRR) | **Ch 4, Section 4.6**: Advertisement Interval<br>- 100ms, 500ms, 1000ms tested<br>- Packet loss analysis<br>- Bottleneck identified | ⚠️ **COMPLETE OVERLAP**<br>Identical experiment |
| **A3: Environmental Multipath** | ✅ Complete<br>- 10.74x variability vs anechoic<br>- Angular patterns | **Ch 4, Section 4.3**: SS Fading<br>- Rician K parameter analysis<br>- Scale parameter σ<br>**Ch 4, Section 4.2**: Antenna Characteristics<br>- Anechoic chamber vs outdoor | ⚠️ **HIGH OVERLAP**<br>Same data, different statistical methods |
| **A4: Body Shadowing** | ✅ Complete<br>- Mean: 10.43 ± 10.58 dB<br>- Position dependency | **Ch 4, Section 4.5**: Body Occlusion<br>- LoS vs nLoS analysis<br>- MAD technique<br>- Advertisement drop percentage<br>- Sub-experiments at key points | ⚠️ **COMPLETE OVERLAP**<br>Identical phenomenon, possibly same dataset |

**Recommendation for Category A**:
- ❌ **Remove A2** as standalone - it's fully covered in previous work
- 🔀 **Merge A1+A3+A4** with previous work analysis into single comprehensive propagation paper
- The "double hump" pattern from antenna characteristics (previous work) could strengthen A3
- Rician fading analysis from previous work adds depth to A3

---

### Category B: Pedestrian Detection & Tracking

| Analysis ID | Current Plan | Previous Work | Overlap Assessment |
|-------------|-------------|--------------|-------------------|
| **B5: Proximity Detection** | ✅ Complete<br>- SVR: 1.90m MAE<br>- 98% improvement over analytical | **Ch 5, Discussion**: Notes analytical models fail<br>- No ML-based proximity in previous work | ✅ **NEW CONTRIBUTION**<br>Strong standalone paper |
| **B6: Movement Detection** | ⏳ Pending | **Ch 5, Section 5.3**: Pause Detection<br>- Curve fitting<br>- Sliding window SD<br>- 15s significant pause threshold | ⚠️ **PARTIAL OVERLAP**<br>Previous work focuses on pauses, not general movement |
| **B7: Trajectory Reconstruction** | ⏳ Pending | **Ch 5, Section 5.5**: Direction of Travel<br>- Median RSSI comparison<br>- Antenna directional sensitivity | ⚠️ **PARTIAL OVERLAP**<br>Direction is covered, full trajectory is not |
| **B8: Presence Detection** | ⏳ Pending | **Ch 4, Section 4.5**: Body Occlusion experiments<br>- Detection probability implicit in analysis | ⚠️ **PARTIAL OVERLAP**<br>Implicit in occlusion work |

**Recommendation for Category B**:
- ✅ **Keep B5** - strong ML contribution not in previous work
- 🔀 **Merge B6+B7** with previous work on pause detection and direction
- 📊 **Enhance B8** with detection reliability metrics from previous work's occlusion experiments

---

### Category C: Space Utilization Analytics

| Analysis ID | Current Plan | Previous Work | Overlap Assessment |
|-------------|-------------|--------------|-------------------|
| **C9: Occupancy Detection** | ⏳ Pending | **Ch 5, Section 5.6**: Campus Study<br>- 17 locations, 28 participants<br>- BLE events per location<br>- Space utilization insights | ⚠️ **HIGH OVERLAP**<br>Campus study covers occupancy |
| **C10: Pathway Analytics** | ⏳ Pending | **Ch 5, Section 5.6**: Campus Study<br>- Journey analysis<br>- Walking pace calculation<br>- Table of BLE events by location | ⚠️ **HIGH OVERLAP**<br>Campus study covers pathway analytics |

**Recommendation for Category C**:
- 🔀 **Merge C9+C10** with campus study from previous work
- Add quantitative metrics to the qualitative insights from previous work

---

### Category D: Statistical & Comparative Studies

| Analysis ID | Current Plan | Previous Work | Overlap Assessment |
|-------------|-------------|--------------|-------------------|
| **D11: Signal Variability** | ⏳ Pending | **Ch 4**: Multiple experiments with 3 runs each<br>- Variability implicit in all analyses | ⚠️ **IMPLICIT OVERLAP**<br>Not explicitly analyzed in previous work |
| **D12: Cross-Dataset Validation** | ⏳ Pending | **Ch 4**: Multiple datasets (Antenna, Deployment, Advertisement Interval, etc.) | ⚠️ **IMPLICIT OVERLAP**<br>Datasets exist but cross-validation not done |
| **D13: Optimal Configuration** | ⏳ Pending | **Ch 5, Discussion Section 5.2**: Protocol for Using BLE<br>- Comprehensive recommendations<br>- Hardware, software, location guidelines | ⚠️ **HIGH OVERLAP**<br>Previous work has detailed protocol |

**Recommendation for Category D**:
- ✅ **Keep D11** - explicit variability analysis adds value
- ✅ **Keep D12** - cross-validation is new
- ❌ **Remove D13** as standalone - merge recommendations into other papers' Discussion sections

---

### Category E: Machine Learning Applications

| Analysis ID | Current Plan | Previous Work | Overlap Assessment |
|-------------|-------------|--------------|-------------------|
| **E14: Classification Tasks** | ⏳ Pending<br>- LoS/nLoS classification<br>- Position classification | **Ch 4**: LoS/nLoS experiments throughout<br>- No ML classification in previous work | ✅ **NEW CONTRIBUTION**<br>ML approach is novel |
| **E15: Regression Tasks** | ⏳ Pending<br>- Distance, speed estimation | **Ch 5, Section 5.6**: Walking pace calculation<br>- No ML regression in previous work | ✅ **PARTIAL NEW**<br>ML adds novelty to pace analysis |
| **E16: Feature Engineering** | ⏳ Pending<br>- 17 engineered features | Not in previous work | ✅ **NEW CONTRIBUTION**<br>Strong methodological contribution |

**Recommendation for Category E**:
- ✅ **Keep all E14/E15/E16** - ML approaches are novel
- Already combined with B5 in publications plan (good strategy)

---

### Category F: Practical Applications

| Analysis ID | Current Plan | Previous Work | Overlap Assessment |
|-------------|-------------|--------------|-------------------|
| **F18: Contact Tracing** | ⏳ Pending<br>- 2m threshold<br>- Sensitivity/specificity | **Ch 5, Section 5.4**: Interaction Detection<br>- 2 pedestrians pausing<br>- Interaction not conclusively detected | ⚠️ **RELATED BUT DIFFERENT**<br>Previous work shows difficulty, current plan may solve it |

**Recommendation for Category F**:
- ✅ **Keep F18** - builds on previous work's limitations
- Cite previous work's interaction detection challenges

---

## Missing from Current Analysis Plan (But Present in Previous Work)

### 🔴 Critical Missing Analyses:

1. **Antenna Directional Characteristics** (Ch 4, Section 4.2)
   - "Double hump" pattern discovery
   - Anechoic chamber baseline
   - Directional sensitivity: 0°-45° vs 135°-180°
   - **Publication Potential**: Medium (supporting analysis)
   - **Recommendation**: Add as supporting analysis for A3 or B7

2. **Pause Detection Methodology** (Ch 5, Section 5.3)
   - Novel protocol for pause detection
   - Curve fitting + sliding window SD + thresholding
   - 15-second "significant pause" threshold
   - **Publication Potential**: HIGH (novel methodology)
   - **Recommendation**: Create new analysis "B9: Pause Detection" or expand B6

3. **Deployment Distance Optimization** (Ch 4, Section 4.4)
   - 3m, 5m, 7m, 9m comparison
   - ANOVA + Tukey's HSD statistical validation
   - LoS vs nLoS performance
   - **Publication Potential**: Medium (system design contribution)
   - **Recommendation**: Add to D13 or A1+A2+A3+A4 combined paper

4. **Real-World Campus Deployment** (Ch 5, Section 5.6)
   - 28 participants, 17 locations, 24 days
   - 126,130 BLE advertisements
   - Journey analysis with walking pace
   - Space utilization metrics
   - **Publication Potential**: HIGH (real-world validation)
   - **Recommendation**: ⚠️ THIS IS CRITICAL - should be standalone paper or combined with C9+C10

---

## Revised Publications Plan

### Priority 1: Keep as Planned ⭐⭐⭐⭐⭐

**Paper 1: B5+E14+E16 - ML Proximity Detection**
- ✅ No overlap with previous work
- ✅ Novel ML contribution
- ✅ Strong results (98% improvement)
- **Action**: No changes needed

---

### Priority 2: Significant Revision Needed ⭐⭐⭐⭐

**Paper 2: A1+A2+A3+A4 - BLE Propagation (ORIGINAL PLAN)**

**Issues**:
- ❌ High overlap with previous work (Chapters 4.2, 4.3, 4.5, 4.6)
- ❌ A2 (Advertisement Interval) is identical to previous work
- ❌ Risk of self-plagiarism or redundant publication

**REVISED Paper 2: Comprehensive BLE Characterization**

**Recommendation**: Merge current analyses with MISSING previous work analyses

**New Structure**:
1. **Introduction**
   - Gap: Real-world BLE characterization missing in literature

2. **Experimental Setup**
   - Multiple datasets: Advertisement Interval, Deployment Distance, Antenna Characteristics
   - Both anechoic chamber and outdoor measurements

3. **Hardware Characterization** (⭐ ADD FROM PREVIOUS WORK)
   - Antenna directional sensitivity (Ch 4.2)
   - "Double hump" pattern
   - Anechoic chamber baseline

4. **Environmental Propagation**
   - Path Loss: n=0.735 (A1 + Ch 4.3)
   - Multipath: 10.7x variability (A3 + Ch 4.2-4.3)
   - Body Shadowing: 10.43 dB (A4 + Ch 4.5)

5. **System Configuration Impact**
   - ❌ Remove A2 as main section (already published in thesis?)
   - Add deployment distance optimization (Ch 4.4)
   - Reference advertisement interval findings (don't duplicate)

6. **Design Guidelines** (⭐ ADD FROM PREVIOUS WORK)
   - Protocol from Ch 5, Discussion 5.2
   - Optimal deployment distance: 3-5m
   - Advertisement interval: 500-1000ms

**Outcome**: Stronger paper without self-duplication

---

### Priority 3: Keep with Minor Adjustments ⭐⭐⭐⭐

**Paper 3: F18+B8 - Contact Tracing**

**Recommendation**:
- ✅ Keep as planned
- ✅ Add reference to previous work's interaction detection challenges (Ch 5.4)
- Position as solving the limitations identified in previous work

---

### Priority 4: Major Restructuring Needed ⭐⭐⭐

**Paper 4: B6+B7+C10 - Movement Analytics (ORIGINAL PLAN)**

**Issues**:
- Incomplete without pause detection from previous work
- Missing real-world campus validation

**REVISED Paper 4: Pedestrian Movement and Activity Detection**

**New Structure**:
1. **Introduction**
   - Pedestrian movement dynamics in outdoor spaces

2. **Movement Detection** (B6)
   - General movement patterns

3. **Pause Detection** (⭐ ADD FROM PREVIOUS WORK Ch 5.3)
   - Novel methodology: curve fitting + sliding window SD
   - "Significant pause" threshold (15s)
   - Validation against ground truth

4. **Direction Detection** (B7 + ⭐ Ch 5.5)
   - Antenna characteristics-based approach
   - Single-observer direction detection

5. **Pathway Analytics** (C10 + ⭐ Ch 5.6)
   - Real-world campus deployment
   - 28 participants, 17 locations, 126K advertisements
   - Journey reconstruction
   - Walking pace estimation

**Outcome**: Comprehensive movement analysis paper

---

### NEW Priority 5: Real-World Deployment ⭐⭐⭐⭐⭐

**NEW Paper 5: Campus Deployment Study**

**Why This Paper is Critical**:
- ⚠️ The campus study (Ch 5.6) is a major contribution NOT covered in current plan
- 28 participants over 24 days is significant real-world validation
- Space utilization insights have practical urban planning value

**Structure**:
1. **Introduction**
   - Need for real-world pedestrian behavior studies
   - Privacy-preserving approaches

2. **Methodology**
   - 17 locations, 28 participants, 24 days
   - Privacy-by-design approach
   - BLE beacon distribution

3. **Space Utilization Analysis** (C9 + ⭐ Ch 5.6)
   - BLE events per location
   - Temporal patterns
   - High-traffic vs low-traffic areas

4. **Individual Journey Analysis** (⭐ Ch 5.6)
   - Path reconstruction
   - Walking pace: 0.8 m/s example
   - Purpose inference (morning walk, commute)

5. **Privacy Considerations**
   - Hash encoding of UUIDs
   - Ephemeral data storage
   - Participant control mechanisms

6. **Urban Planning Implications**
   - Data-driven space (re)development
   - Pedestrian flow optimization

**Outcome**: Strong applied paper with real-world impact

---

## Summary of Recommendations

### ❌ Remove from Publications Plan

1. **A2: Advertisement Interval Impact** (standalone)
   - ⚠️ Complete duplication of Ch 4.6 from previous work
   - Use findings in Discussion sections only

2. **D13: Optimal Configuration** (standalone)
   - ⚠️ Already covered in Ch 5 Discussion 5.2
   - Merge into recommendations sections of other papers

### 🔀 Merge/Restructure

1. **A1+A3+A4 → Enhanced with previous work**
   - Add antenna characteristics (Ch 4.2)
   - Add deployment distance (Ch 4.4)
   - Strengthen with Rician fading analysis (Ch 4.3)

2. **B6+B7 → Add pause detection**
   - Include pause detection methodology from Ch 5.3
   - Add direction detection from Ch 5.5

3. **C9+C10 → Real-world deployment paper**
   - Build entirely around campus study (Ch 5.6)
   - This should be a MAJOR paper

### ➕ Add Missing Analyses

1. **Antenna Characteristics Analysis** (from Ch 4.2)
   - Not in current plan
   - Important for B7 (direction) and A3 (multipath)

2. **Pause Detection Methodology** (from Ch 5.3)
   - Novel contribution
   - Not explicitly in current plan

3. **Campus Deployment Study** (from Ch 5.6)
   - ⚠️ CRITICAL MISSING PAPER
   - Should be standalone publication

### ✅ Keep Unchanged

1. **B5+E14+E16**: ML Proximity Detection
2. **F18+B8**: Contact Tracing
3. **D11**: Signal Variability (new analysis)
4. **D12**: Cross-Dataset Validation (new analysis)

---

## Updated Publication Timeline

### Year 1

**Q1**: B5+E14+E16 (ML Proximity) → IEEE TMC ⭐⭐⭐⭐⭐
- Ready to draft immediately
- No overlap with previous work

**Q2**: REVISED A1+A3+A4 (Enhanced Propagation) → IEEE TAP/Access ⭐⭐⭐⭐
- Add antenna characteristics (Ch 4.2)
- Add deployment distance (Ch 4.4)
- Remove A2 duplication

**Q3**: Revisions on Papers 1-2

**Q4**: F18+B8 (Contact Tracing) → IEEE IoT ⭐⭐⭐⭐

### Year 2

**Q1**: Revisions on Paper 3

**Q2**: NEW PAPER - Campus Deployment → IEEE T-ITS or Pervasive Computing ⭐⭐⭐⭐⭐
- Built around Ch 5.6
- C9+C10+Real-world validation

**Q3**: REVISED B6+B7 (Movement + Pause) → IEEE T-ITS ⭐⭐⭐⭐
- Add pause detection (Ch 5.3)
- Add direction detection (Ch 5.5)

**Q4**: Short papers (D11, D12 if needed)

---

## Key Takeaways

### ⚠️ Critical Issues

1. **Self-Duplication Risk**: A2 is identical to previous work - must remove or heavily revise
2. **Missing Major Contribution**: Campus study (Ch 5.6) is not in publications plan but should be a major paper
3. **Overlap in Propagation**: A1, A3, A4 overlap significantly with Ch 4 - needs careful restructuring

### ✅ Strengths to Leverage

1. **ML Contribution**: B5+E14+E16 is entirely novel - proceed as planned
2. **Real-World Data**: Campus study provides unique validation opportunity
3. **Novel Methodologies**: Pause detection protocol, antenna-based direction detection

### 📊 Recommended Publication Count

**Original Plan**: 3-5 papers

**Revised Plan**: 5-6 papers
1. ML Proximity Detection (B5+E14+E16)
2. Enhanced BLE Propagation (REVISED A1+A3+A4 + Ch 4.2, 4.4)
3. Contact Tracing (F18+B8)
4. Campus Deployment Study (NEW - Ch 5.6 + C9+C10)
5. Movement & Activity Detection (B6+B7 + Ch 5.3, 5.5)
6. Optional: Variability & Cross-Validation (D11+D12)

---

## Immediate Actions Required

1. ✅ **Remove A2** from standalone publication consideration
2. ✅ **Add campus study** as Priority 5 paper
3. ✅ **Merge pause detection** into movement analytics paper
4. ✅ **Enhance A1+A3+A4** with antenna characteristics and deployment distance
5. ✅ **Cross-reference** previous work in all papers to avoid self-plagiarism concerns

---

**Date**: 2025-11-09
**Review Status**: Comprehensive comparison complete
**Next Steps**: Review this comparison and decide on publication strategy
