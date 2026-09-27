# CO3117 - Machine Learning | Individual Longitudinal Assignment

## One Dataset, One Use Case, Many Models

**Student:** [Phạm Minh Trí - 2353235]  
**Semester:** HK261 (2026–2027)  
**Faculty:** Computer Science and Engineering, HCMUT, VNU-HCM

---

## 🎯 Use Case

**Dataset:** UCI Human Activity Recognition Using Smartphones  
**Source:** https://archive.ics.uci.edu/dataset/240/human+activity+recognition+using+smartphones  
**Prediction target:** Predict a person's current physical activity from smartphone inertial measurements  
**Primary metric:** Macro-F1  
**Secondary metrics:** Accuracy, Confusion Matrix

## 📁 Repository Structure

```
co3117-ml-individual/
├── README.md                  # This file
├── PROGRESS.md                # Instructor dashboard — one row per course week
├── MODEL_LOG.md               # Per-model record for all implemented models
├── AI_USE.md                  # AI usage disclosure log
├── REFERENCES.md              # All references and citations
├── SUBMISSION_PART1.md        # Part I submission checklist
├── SUBMISSION_PART2.md        # Part II submission checklist
├── requirements.txt           # Python dependencies
├── data/
│   └── README.md              # Dataset description, source, split policy
├── docs/
│   ├── index.md               # Wiki/blog index page
│   ├── pre-release/
│   │   └── PRE_RELEASE_CATCHUP.md
│   └── weekly/                # Weekly wiki/blog posts
├── exercises/                 # Written drills (handwritten scans + corrections)
├── exam/                      # Exam preparation materials
├── src/
│   ├── data.py                # Data loading, splitting, preprocessing
│   ├── metrics.py             # Evaluation metrics
│   ├── from_scratch/          # Depth-A implementations
│   └── reference_adapters/    # Depth-B/C wrappers
├── experiments/               # Experiment scripts and configs
│   ├── part1_pre_midterm/
│   └── part2_post_midterm/
├── tests/                     # Unit and sanity tests
├── results/
│   ├── metrics.csv            # Master comparison table
│   └── figures/               # Plots and visualizations
└── report/                    # Final reports (PDF)
```

## 🚀 Quick Start

```bash
# Clone and setup
git clone <repo-url>
cd co3117-ml-individual

# Create virtual environment
python -m venv venv
venv\Scripts\activate        # Windows
# source venv/bin/activate   # Linux/Mac

# Install dependencies
pip install -r requirements.txt

# Download dataset
python src/data.py --download

# Run experiments
python experiments/part1_pre_midterm/w05_perceptron.py
```

## 📊 Data Protocol (Frozen at R0)

| Item | Value |
|------|-------|
| Dataset | UCI HAR Using Smartphones v1.0 |
| Target | Activity label (6 classes) |
| Split | Subject-aware split (no entity leakage) |
| Primary metric | Macro-F1 |
| Random seed | 42 |
| Test set | Sealed — used only for final comparison |

## 📅 Timeline

- **R0 (Release checkpoint):** Before W06 class
- **Part I deadline:** 14 October 2026
- **Midterm:** 16 October 2026
- **Part II deadline:** 2 calendar days before official final exam

## 📚 Key Resources

- [ML-From-Scratch](https://github.com/eriklindernoren/ML-From-Scratch) — Primary code microscope
- [numpy-ml](https://github.com/ddbourgin/numpy-ml) — HMM/sequence reference
- [pyprobml](https://github.com/probml/pyprobml) — Probabilistic ML bridge
- [NPTEL ML Course](https://nptel.ac.in/courses/106106139) — Video lectures
- [scikit-learn](https://scikit-learn.org/stable/) — Benchmark implementations
