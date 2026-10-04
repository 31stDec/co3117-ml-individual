# Data

Raw data is NOT committed (the self-check fails on data/raw/**, archives and binary arrays).
Describe here how to obtain the exact dataset version frozen in protocol.yaml.

## Dataset: UCI Human Activity Recognition Using Smartphones

### Source & Acquisition
- **URL:** https://archive.ics.uci.edu/dataset/240/human+activity+recognition+using+smartphones
- **Version:** 1.0
- **Download and unpack command:** `python src/data.py --download` (unpacks into `data/raw/`)
- **License & Citation:** Davide Anguita, Alessandro Ghio, Luca Oneto, Xavier Parra and Jorge L. Reyes-Ortiz. *A Public Domain Dataset for Human Activity Recognition Using Smartphones*. 21th European Symposium on Artificial Neural Networks, Computational Intelligence and Machine Learning, ESANN 2013.

### Description
The dataset was built from recordings of 30 subjects performing activities of daily living (ADL) while carrying a waist-mounted smartphone with embedded inertial sensors.

### Activity Labels (6 classes)
1. WALKING
2. WALKING_UPSTAIRS
3. WALKING_DOWNSTAIRS
4. SITTING
5. STANDING
6. LAYING

### Features
- 561 features derived from accelerometer and gyroscope 3-axial raw signals (time and frequency domain).
- Pre-computed by authors using sliding windows of 2.56 sec with 50% overlap.

### Split Policy
- **Method:** Subject-aware split (no entity leakage — same person never in both train and test).
- **Original split:** 70% train (21 subjects), 30% test (9 subjects).
- **Validation:** Cross-validation within training population only.
- **Test set:** Sealed — used only for final comparison.

### Data Protocol (Frozen at R0)
| Item | Value |
|------|-------|
| Primary metric | Macro-F1 |
| Secondary metrics | Accuracy, Confusion Matrix |
| Random seed | 42 |
| Preprocessing | Standard z-score fit on training data only; transform val/test |
| Baseline | Majority-class classifier |

### Leakage Controls
- Scalers/encoders fit on training data only.
- Subject IDs never shared across train/test.
- Raw inertial signals preserved for HMM sequential modeling.
