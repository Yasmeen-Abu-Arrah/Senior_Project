---
title: "AI-Driven Arabic Speech Rehabilitation Platform"
---

# Introduction

## Overview
Communication is a critical vector for cognitive, social, and emotional development. Pediatric communication disorders present unique developmental hurdles that, if left unaddressed, can lead to permanent academic and social marginalization. This project explores the integration of computer vision, speech signal processing, and generative artificial intelligence to deliver an accessible, automated, and dialect-aware rehabilitation tool tailored for Palestinian children.

## Problem Statement
In Palestine, pediatric communication disorders pose a severe public health crisis. Children suffering from disorders such as Childhood Apraxia of Speech (CAS), verbal stuttering, phonological deletion, and environmental language delays face profound barriers to social integration. 

The core architectural issue lies in a compounding deficit of clinical resources, severe geographical and economic barriers, and a dialectal technological void. Modern digital therapy applications are built almost exclusively for Western languages or Modern Standard Arabic (MSA). They completely lack the phonetic modeling, acoustic tolerances, and vocabulary profiles required to process the distinct phonetics and localized variations of the Palestinian/Levantine dialect spoken by children.

## Local Statistics (Palestine Context)
This project anchors its justification in empirical data compiled by the Palestinian Central Bureau of Statistics (PCBS) and recent regional epidemiological research:

* **High Prevalence Rates:** Communication disabilities represent the **second most prevalent childhood disability** in Palestine, accounting for **37.3% of all pediatric disability cases**. 
* **Geographical Disparity:** The prevalence fluctuates based on geographical vulnerability, affecting **35.7% of disabled children in the West Bank** and rising to **40.1% within the Gaza Strip**.
* **Regional Vulnerability:** The Jenin Governorate—the home region of the Arab American University (AAUP)—historically records the highest overall disability rate in the West Bank at **4.1% of the total population**.
* **Socio-Economic Barriers:** Local case-control studies reveal that children residing in rural Palestinian communities face a drastically elevated risk of prolonged speech and language delays—with an **Odds Ratio (OR) of 4.4**—directly driven by low family income.

## Computational Engine: Lip Pucker Ratio (LPR) Logic
For the "Saif" Apraxia module, the system processes 3D landmarks on edge devices. Below is the implemented algorithmic validation block showing how the system normalizes distances to verify proper lip rounding metrics during production:

```{python}
#| label: fig-lpr-logic
#| fig-cap: "Algorithmic simulation of Lip Pucker Ratio (LPR) decision tree metrics"
import numpy as np

def calculate_lpr(width, height):
    return round(width / height, 2)

# Simulated values for Saif's speech patterns
# [Width (px), Height (px)]
test_cases = {
    "Normal Rest State":,
    "Incorrect Wide Drop (Error)":,
    "Correct Rounded Pucker (Success)": [32, 45]
}

for state, metrics in test_cases.items():
    ratio = calculate_lpr(metrics[0], metrics[1])
    print(f"{state:───────────────>35}: LPR = {ratio}")
```

## Project Objectives
The primary goal of this senior project is to develop a multi-dimensional, AI-driven, and low-latency software platform to screen, gamify, and rehabilitate Arabic communication disorders in children:

1. **Real-Time Vision for Articulatory Assessment:** Integrate *MediaPipe Face Mesh* to calculate the Lip Pucker Ratio (LPR) in real-time under the *Digital DTTC Protocol*.
2. **Acoustic Pipeline for Fluency Analysis:** Develop an audio signal processing engine using *Librosa* to analyze RMS energy envelopes and Zero Crossing Rates (ZCR) to detect disfluencies down to 200 milliseconds of latency.
3. **Language Models for Phonological Mapping:** Fine-tune an Arabic *Wav2Vec 2.0* or *Hubert* model to perform automated *Forced Alignment* and compute *Levenshtein Distance* matrices at the phoneme level.
4. **Adaptive Generative AI Storytelling Engine:** Architect a dynamic backend utilizing conversational LLM APIs (*GPT-4o*) paired with localized TTS systems to dynamically scale vocabulary complexity based on real-time verbal performance metrics.
5. **Distributed Tele-Therapy Architecture:** Construct a real-time web-streaming communication platform via *WebRTC Protocols* that embeds gaze-tracking logic to transmit continuous focus matrices directly to clinician monitoring dashboards.
