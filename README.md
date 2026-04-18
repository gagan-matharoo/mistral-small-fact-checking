# Supplementary Materials: The F1 Mirage

**Paper Title:** The F1 Mirage: Unmasking Shortcut Learning in Mistral's Fact-Checking  
**Authors:** Gagandeep Kaur, Ismail Badache, Aznam Yacoub  
**Conference:** 30th International Conference on Knowledge-Based and Intelligent Information & Engineering (KES 2026)  

## Overview

This repository contains the supplementary materials, extended datasets, and detailed experimental metrics supporting the findings presented in our manuscript. Due to strict page limits in the conference proceedings, these materials were externalized to ensure full transparency and methodological reproducibility.

Our study investigates the probabilistic behavior of the Mistral-Small-24B architecture in automated fact-checking, demonstrating that the F1 score often masks shortcut learning and structural asymmetries rather than reflecting genuine factual verification.

## Repository Structure

The repository is organized into two main components:

### 1. `KES_2026_paper_563_Appendix.pdf`
This document consolidates all the supplementary data referenced in the main manuscript. It includes:
* **Prompt Instructions:** The exact, unaltered textual instructions provided to the architecture during our experiments. This covers the baseline prompt structure (from the original evaluation protocol) favoring the 'False' class as a fallback, and our modified prompt structure favoring the 'Conflicting' class.
* **Extended Metrics:** The complete performance tables across all evidence configurations (C1 to C7), including the per-class precision scores which corroborate the structural asymmetries discussed in Section 4.2.
* **Balanced Dataset Evaluation:** The resulting metrics from our control set. To ensure that our observations were not artifacts of the original class distribution, the experiments were rigorously replicated on a balanced dataset. These results confirm that the fundamental asymmetry in the model's probabilistic mapping remains identical even when the influence of the original dataset's prior distribution is neutralized.
* **Adversarial Testing:** The raw results and probability distributions for our cross-label inversion experiments (Configuration C8). This data demonstrates the architecture's behavior when trained on inverted labels and evaluated against the original ground truth. These tables definitively illustrate the collapse of performance and support our conclusion that the labels function as hollow syntactic variables devoid of epistemic meaning.

### 2. `src/`
This directory contains the source files utilized to conduct the experiments, perform the bootstrap resampling, and generate the reported metrics.

## Reproducibility

The data provided here corresponds precisely to the 1,000 bootstrap iterations and the repeated-measures ANOVA discussed in the manuscript. All metrics strictly reflect the isolated evaluation of the verdict prediction phase using the QuanTemp dataset.