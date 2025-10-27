# Epitech - Data Science Projects

This repository contains various data science and machine learning projects completed as part of the Epitech curriculum.

## 📋 Table of Contents

- [Projects Overview](#projects-overview)
- [Rush 1: YouTube Channel Analysis](#rush-1-youtube-channel-analysis)
- [Rush 2: Pharmaceutical Sales Analysis](#rush-2-pharmaceutical-sales-analysis)
- [Rush 3: Credit Risk Analysis](#rush-3-credit-risk-analysis)
- [Rush 4: Marketing Campaign Clustering](#rush-4-marketing-campaign-clustering)
- [Utilities](#utilities)
- [Requirements](#requirements)
- [Getting Started](#getting-started)

## Projects Overview

This repository contains four main "Rush" projects focusing on different aspects of data science:

1. **Rush 1**: Analysis of top 300 YouTube channels
2. **Rush 2**: Pharmaceutical sales data analysis and visualization
3. **Rush 3**: Credit risk prediction using KNN and Regression models
4. **Rush 4**: Customer segmentation using clustering techniques

## Rush 1: YouTube Channel Analysis

Analysis of the top 300 YouTube channels.

**Key Files:**
- `Rush 1/Analyse_top300_chaine_youtube_Stefan_Beaulieu.xlsb.xlsx` - Analysis spreadsheet

## Rush 2: Pharmaceutical Sales Analysis

Comprehensive analysis of pharmaceutical sales data across different time periods (hourly, daily, weekly, monthly).

**Directory Structure:**
- `Data/` - Raw sales data files
- `Clean/` - Python scripts for data cleaning
- `Cleaned_data/` - Processed and cleaned datasets
- `Analyse/` - Analysis scripts with visualizations

**Key Features:**
- Data cleaning and preprocessing
- Time-series analysis
- Sales trend visualization
- Molecule-specific sales analysis

**Molecules Analyzed:**
- M01AB, M01AE (Anti-inflammatory drugs)
- N02BA, N02BE (Pain relievers)
- N05B, N05C (Psychiatric medications)
- R03 (Respiratory system medications)
- R06 (Antihistamines)

## Rush 3: Credit Risk Analysis

Machine learning project for predicting credit risk using classification models.

**Directory Structure:**
- `Data/` - Raw credit data
- `Clean/` - Data cleaning scripts
- `Cleaned_data/` - Preprocessed datasets
- `Analyse/` - Exploratory data analysis
- `KNN/` - K-Nearest Neighbors classification
- `Regression/` - Regression model implementation
- `Risque_data/` - Risk-scored datasets

**Key Features:**
- Credit risk scoring
- KNN classification model
- Regression analysis
- Feature engineering (age, income, credit amount, family status, etc.)
- Model comparison and evaluation

**Models Used:**
- K-Nearest Neighbors (KNN)
- Regression models

## Rush 4: Marketing Campaign Clustering

Customer segmentation project using clustering algorithms to identify customer groups.

**Directory Structure:**
- `Data/` - Raw campaign data
- `Clean/` - Data cleaning scripts
- `Cleaned_data/` - Preprocessed datasets
- `Clusturing/` - Clustering analysis and models

**Key Features:**
- Customer segmentation
- K-Means clustering
- Hierarchical clustering
- PCA (Principal Component Analysis) for dimensionality reduction
- Customer behavior analysis based on:
  - Demographics (Year of Birth, Education, Marital Status)
  - Purchase behavior (Wines, Fruits, Meat, Fish, Sweets, Gold products)
  - Campaign responses
  - Channel preferences (Web, Catalog, Store)

## Utilities

### Random.py

A utility script for creating balanced groups of people based on skill levels.

**Features:**
- Takes a list of people with skill ratings
- Creates balanced groups with similar average skill levels
- Interactive input for customizing group size and number of groups
- Uses random sampling to find optimal group distribution

**Usage:**
```bash
python Random.py
```

The script will prompt you for:
- Number of groups (default: 2)
- Number of people per group (default: 4)

## Requirements

This project uses Python 3.x with the following main libraries:

- `pandas` - Data manipulation and analysis
- `numpy` - Numerical computing
- `matplotlib` - Data visualization
- `seaborn` - Statistical data visualization
- `scikit-learn` - Machine learning algorithms
- `scipy` - Scientific computing

To install dependencies:

```bash
pip install pandas numpy matplotlib seaborn scikit-learn scipy
```

## Getting Started

1. **Clone the repository:**
   ```bash
   git clone https://github.com/StefanB35/Epitech.git
   cd Epitech
   ```

2. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```
   *(Note: Create a requirements.txt file with the libraries listed above if needed)*

3. **Navigate to the project you want to explore:**
   ```bash
   cd "Rush 2"
   ```

4. **Run the analysis scripts:**
   ```bash
   python Analyse/Analyse_Pharma_Ventes_Daily.py
   ```

## Project Structure

```
Epitech/
├── Random.py                    # Group balancing utility
├── Rush 1/                      # YouTube analysis
├── Rush 2/                      # Pharmaceutical sales
│   ├── Data/
│   ├── Clean/
│   ├── Cleaned_data/
│   └── Analyse/
├── Rush 3/                      # Credit risk analysis
│   ├── Data/
│   ├── Clean/
│   ├── Cleaned_data/
│   ├── Analyse/
│   ├── KNN/
│   ├── Regression/
│   └── Risque_data/
└── Rush 4/                      # Marketing clustering
    ├── Data/
    ├── Clean/
    ├── Cleaned_data/
    └── Clusturing/
```

## Author

**Stéfan Beaulieu**

## License

This project is part of the Epitech curriculum.
