# Revised Publication Plan - Based on Thesis Review

**Date:** 2025-11-09
**Status:** Updated plan after comparing original analyses (A1-F18) with PhD thesis chapters

---

## Executive Summary

After reviewing the PhD thesis chapters in "Previous work" directory, critical issues were discovered with the original publication plan. This document outlines what changed and why.

---

## Key Differences: Earlier Plan vs. New Recommendations

### ✅ What Stays the Same

#### Paper 1: B5+E14+E16 (ML Proximity) - ⭐⭐⭐⭐⭐
- Still the strongest paper
- No changes needed
- IEEE TMC/Sensors Journal target maintained

#### Paper 3: F18+B8 (Contact Tracing) - ⭐⭐⭐⭐
- Still strong
- No major changes needed
- IEEE IoT target maintained

---

### 🔴 What CHANGED After Reviewing Thesis

#### Paper 2: A1+A2+A3+A4 (BLE Propagation) - ⚠️ CRITICAL ISSUE

**Earlier plan said:** "Foundational work - Establishes credibility"

**New finding:**
- **A2 is IDENTICAL** to thesis Chapter 4, Section 4.6 (same experiment: 100ms, 500ms, 1000ms intervals with packet loss analysis)
- **A4 has HIGH overlap** with thesis Chapter 4, Section 4.5 (body occlusion/shadowing)
- **A1+A3 overlap significantly** with thesis Chapter 4, Section 4.3 (path loss and fading)

**Risk:** Publishing A1+A2+A3+A4 as-is could be flagged as self-plagiarism or duplicate publication

**Solution:**
- ❌ Remove A2 from this paper (already published in thesis)
- 🔀 Add NEW content from thesis to strengthen it:
  - Antenna directional characteristics (Ch 4.2) - "double hump" pattern
  - Deployment distance optimization (Ch 4.4) - ANOVA analysis of 3m, 5m, 7m, 9m
- ✅ Result: Enhanced A1+A3+A4 with novel antenna/deployment content

---

### 🆕 Major Missing Paper (Not in Earlier Plan)

#### NEW Paper: Campus Deployment Study - ⭐⭐⭐⭐⭐

**Your thesis Chapter 5, Section 5.6 describes:**
- 28 participants, 17 locations, 24 days
- 126,130 BLE advertisements collected
- Journey reconstruction, walking pace analysis
- Space utilization metrics

**Issue:** This is NOT in the current analysis plan (A1-F18) but should be a MAJOR paper!

Earlier plan had C9+C10 for occupancy/pathway analytics, but the thesis shows you have a much bigger real-world deployment that deserves standalone publication.

---

## Revised Publications Plan (Incorporating Both)

| # | Earlier Plan | New Recommendation | Change Reason |
|---|--------------|-------------------|---------------|
| 1 | B5+E14+E16 (ML Proximity) | B5+E14+E16 (ML Proximity) | ✅ No change - still strongest |
| 2 | A1+A2+A3+A4 (Propagation) | A1+A3+A4+Ch4.2+Ch4.4 (Enhanced Propagation) | ⚠️ Remove A2 duplication, add antenna/deployment |
| 3 | F18+B8 (Contact Tracing) | F18+B8 (Contact Tracing) | ✅ No change |
| 4 | B6+B7+C10 (Movement) | B6+B7+Ch5.3+Ch5.5 (Movement + Pause Detection) | 🔀 Add pause detection methodology from thesis |
| 5 | C9+B8 (Occupancy) | Ch5.6+C9+C10 (Campus Deployment) | 🆕 Build around real-world study from thesis |
| 6 | D11+D12 (Reliability) | D11+D12 (Reliability) | ✅ Optional short paper |

---

## Why the Changes Are Necessary

### 1. Self-Plagiarism Risk
Publishing the exact same experiment (A2: advertisement interval) that's in your thesis could be rejected by journals or flagged during review. IEEE has strict policies on this.

### 2. Missing Major Contribution
The campus study (28 participants, 24 days, 17 locations) is publication-worthy on its own but wasn't captured in analyses A1-F18. This was hidden in your thesis but not in the current analysis plan.

### 3. Strengthening Papers
Adding the antenna characteristics and deployment distance optimization from your thesis makes Paper 2 stronger while avoiding duplication.

---

## Bottom Line

- **Earlier plan was correct** given the information available (analyses A1-F18)
- **New plan is updated** because reviewing your thesis revealed:
  1. Some analyses duplicate thesis content (A2, parts of A1/A3/A4)
  2. Major content exists in thesis but not in analysis plan (campus study, pause detection, antenna characteristics)
  3. Need to restructure to avoid self-plagiarism while capturing all novel contributions

---

## Recommended Action

Use this document to:
1. ✅ Keep the strong papers from earlier plan (B5+E14+E16, F18+B8)
2. ⚠️ Revise the propagation paper to avoid duplication
3. 🆕 Add the campus deployment as a new major paper
4. 🔀 Enhance movement paper with pause detection methodology

---

## Detailed Paper Breakdown

### Paper 1: ML-Based Proximity Detection (B5+E14+E16) ⭐⭐⭐⭐⭐
**Target:** IEEE Transactions on Mobile Computing / IEEE Sensors Journal
**Status:** Ready to proceed
**No changes from original plan**

### Paper 2: Enhanced BLE Propagation Analysis (A1+A3+A4+Ch4.2+Ch4.4) ⭐⭐⭐⭐
**Target:** IEEE Access / Sensors Journal
**Changes:**
- Remove A2 (duplicate of thesis Ch 4.6)
- Add antenna directional characteristics (Ch 4.2)
- Add deployment distance optimization (Ch 4.4)

### Paper 3: Contact Tracing Application (F18+B8) ⭐⭐⭐⭐
**Target:** IEEE Internet of Things Journal
**Status:** Ready to proceed
**No changes from original plan**

### Paper 4: Movement Pattern Analysis (B6+B7+Ch5.3+Ch5.5) ⭐⭐⭐⭐
**Target:** IEEE Sensors Journal
**Changes:**
- Add pause detection methodology from thesis Ch 5.3 and Ch 5.5

### Paper 5: Campus Deployment Study (Ch5.6+C9+C10) ⭐⭐⭐⭐⭐
**Target:** ACM/IEEE IoT Conference or Pervasive Computing
**Status:** NEW - Major paper identified from thesis
**Content:**
- 28 participants, 17 locations, 24 days
- 126,130 BLE advertisements
- Journey reconstruction
- Walking pace analysis
- Space utilization metrics

### Paper 6: Reliability Analysis (D11+D12) ⭐⭐⭐
**Target:** Short paper or technical note
**Status:** Optional - Lower priority
