---
title: "AI-Driven Rehabilitation System for Arabic Pediatric Communication Disorders"
subtitle: "Senior Graduation Project - Chapter 1"
author: "Computer Systems Engineering Student"
institution: "Arab American University (AAUP)"
date: today
format:
  pdf:
    toc: true
    number-sections: true
    colorlinks: true
    geometry:
      - top=30mm
      - left=30mm
      - right=30mm
      - bottom=30mm
---

# Introduction

## Overview
Communication is a fundamental human right and a critical vector for cognitive, social, and emotional development. Pediatric communication disorders present unique developmental hurdles that, if left unaddressed, can lead to permanent academic and social marginalization. This project explores the integration of computer vision, speech signal processing, and generative artificial intelligence to deliver an accessible, automated, and dialect-aware rehabilitation tool tailored for Palestinian children.

## Problem Statement
In the Arab world, and specifically within Palestine, pediatric communication disorders pose a severe and expanding public health and educational crisis. Children suffering from disorders such as Childhood Apraxia of Speech (CAS), verbal stuttering, phonological deletion, and environmental language delays face profound barriers to social integration. Left untreated, these early childhood speech impediments transform into permanent cognitive and social liabilities.

The core of the problem lies in a compounding deficit of clinical, geographical, and technological resources:

* **Severe Clinical Shortages:** There is an acute shortage of certified Speech-Language Pathologists (SLPs) relative to the vulnerable pediatric population.
* **Geographical and Economic Barriers:** Existing speech therapy infrastructure in Palestine is heavily centralized within major urban centers. Families residing in rural or marginalized areas face prohibitive transportation costs and long clinic waiting lists, leaving thousands of rural children entirely unserved.
* **Dialectal Technological Void:** Modern digital therapy applications and AI speech tools are built almost exclusively for Western languages or Modern Standard Arabic (MSA). They lack the phonetic modeling, acoustic tolerances, and vocabulary profiles required to process the distinct phonetics and localized variations of the Palestinian/Levantine dialect spoken by children.

Consequently, there is an urgent, unmet engineering need to design and implement an automated, accessible, intelligent, and dialect-aware system capable of screening, tracking, and assisting in the remote rehabilitation of Arabic pediatric communication disorders.

## Local Statistics (Palestine Context)
To establish the critical socio-economic necessity of this engineering intervention, this project anchors its problem statement in empirical data compiled by the Palestinian Central Bureau of Statistics (PCBS) and recent regional epidemiological research:

* **High Prevalence Rates:** According to census data from the PCBS, communication disabilities represent the **second most prevalent childhood disability** in Palestine, accounting for **37.3% of all pediatric disability cases**. 
* **Geographical Disparity:** The prevalence fluctuates based on geographical vulnerability, affecting **35.7% of disabled children in the West Bank** and rising to **40.1% within the Gaza Strip**.
* **Regional Vulnerability (Jenin Governorate):** The Jenin Governorate—the home region of the Arab American University (AAUP)—historically records the highest overall disability rate in the West Bank at **4.1% of the total population**, creating an immediate, hyper-local demand for targeted technological intervention.
* **Socio-Economic Barriers:** Local case-control studies reveal that children residing in rural Palestinian communities face a drastically elevated risk of prolonged speech and language delays—with an **Odds Ratio (OR) of 4.4**—directly driven by low family income and restricted physical access to urban speech centers.
* **Consanguinity and Environmental Trauma:** Congenital factors, heavily correlated with local consanguinity rates, account for **33.6% of communication disabilities** in Palestine, while acute environmental stressors and systemic trauma heavily exacerbate fluency disorders (stuttering) and sudden speech regression in children.

## Project Objectives
The primary goal of this senior project is to develop a multi-dimensional, AI-driven, and low-latency software platform to screen, gamify, and rehabilitate Arabic communication disorders in children. The specific engineering and clinical objectives are detailed below:

### Objective 1: Implement Real-Time Computer Vision for Articulatory Assessment (The "Saif" Module)
To integrate the *MediaPipe Face Mesh* framework to extract 3D facial landmarks on edge mobile devices. The system will mathematically calculate the Lip Pucker Ratio (LPR) in real time to visually monitor and correct jaw drops and lip-rounding mechanics for children with Childhood Apraxia of Speech (CAS) under the *Digital DTTC Protocol*.

### Objective 2: Build an Acoustic Pipeline for Fluency and Stuttering Analysis (The "Layla" Module)
To develop an audio signal processing engine using the *Librosa Python Library* to analyze Root Mean Square (RMS) energy envelopes and Zero Crossing Rates (ZCR). The application will automatically detect speech disfluencies (repetitions and blocks) down to 200 milliseconds of latency, while validating vocal chord relaxation via the *Easy Onsets* clinical technique.

### Objective 3: Deploy Language Models for Phonological Deletion Mapping (The "Khaled" Module)
To fine-tune a pre-trained Arabic *Wav2Vec 2.0* or *Hubert* speech-to-text model to perform automated *Forced Alignment* on children's speech. The system will utilize *Levenshtein Distance* matrices at the phoneme level to detect syllable deletions and trigger a cognitive *Minimal Pairs* visual feedback mechanism.

### Objective 4: Engineer an Adaptive Generative AI Storytelling Engine (The "Sara" Module)
To architect a dynamic backend utilizing conversational LLM APIs (*GPT-4o*) paired with localized Text-to-Speech (TTS) systems. The framework will generate personalized, interest-driven interactive stories that adapt their prompt-difficulty level dynamically based on the child's recorded word counts to combat environmental language delays.

### Objective 5: Design a Distributed Low-Latency Remote Tele-Therapy Architecture (The "Adam" Module)
To construct a robust real-time communication platform utilizing the *WebRTC Protocol*. The system will deploy real-time *Gaze Tracking Algorithms* to evaluate joint attention metrics, transmitting instant biometric alerts regarding child distraction to the clinician’s dashboard over an optimized *WebRTC DataChannel* in under 100 milliseconds.
