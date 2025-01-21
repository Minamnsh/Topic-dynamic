# Topic Dynamics in the DFG Research Grants Program: Code Repository

This repository contains the code and scripts developed during my Master's thesis, **"Topic Dynamics in the DFG Research Grants Program."** The analysis focuses on identifying and exploring research themes funded by the Deutsche Forschungsgemeinschaft (DFG) using advanced topic modeling and visualization techniques.

## Key Features
- **Methods Implemented:**
  - Non-Negative Matrix Factorization (NMF)
  - Term Frequency-Inverse Document Frequency (TF-IDF)
  - Coherence Score Calculation for Topic Optimization
  - Topic Visualization (Histograms and Coherence Score Plots)
- **Data:** Extracted from the **GEPRIS** platform, containing research project descriptions.
- **Languages & Tools:** Python, Pandas, NumPy, Matplotlib, Gensim, Scikit-learn, and NLTK.

## File Structure
- `data/`: Contains sample datasets (e.g., GEPRIS exports).
- `scripts/`: Python scripts for preprocessing, modeling, and evaluation.
- `notebooks/`: Jupyter notebooks for interactive data analysis and visualization.
- `models/`: Serialized models (e.g., NMF, TF-IDF vectorizer) for reuse.

## Getting Started
1. Clone the repository:
   ```bash
   git clone <repository-url>
