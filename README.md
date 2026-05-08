# N26 User Income & Expense Prediction

A machine learning project that predicts future income and expenses for N26 bank users based on their historical transaction behaviour.

## Problem Statement

Given 4 months of a user's transaction history (Feb–May 2016), predict their total income and total expenses for the following 2 months (Jun–Jul 2016).

## Dataset

| File | Description |
|------|-------------|
| `transactions_data_training.csv` | ~408K transactions across 10,000 users |
| `user_profile.csv` | Demographics for 9,470 users (age, country, employment, income band) |
| `transaction_types.csv` | Maps transaction type codes to direction (In/Out) |
| `mcc_group_definition.csv` | Maps MCC group numbers to spending categories |

## Approach

### 1. Data Preparation
- Map transaction types to `In` / `Out` direction
- Convert transaction dates to datetime
- Split data into two time windows:
  - **Feature window** (Feb–May 2016) — historical behaviour used to build features
  - **Target window** (Jun–Jul 2016) — actual income/expenses to predict

### 2. Feature Engineering (55 features per user)
- **Counts** — number of transactions, by direction, by type
- **Amounts** — sum, mean, median, std, max for In and Out transactions
- **MCC categories** — spending per category, number of categories, dominant category
- **Temporal** — activity span, transactions per day, active months, weekend ratio
- **Behavioural** — in/out ratio, card usage %, spending trend (slope), transaction size variability

### 3. Modelling
Two separate regression models are trained — one for income, one for expenses.

Models compared:
- Baseline (Mean / Median)
- Linear Regression, Ridge, Lasso, ElasticNet
- Decision Tree
- Random Forest
- Gradient Boosting

Target variable is log-transformed (`log1p`) before training to handle skewed distributions.

### 4. Results

| Target | Best Model | MAE | R² | Lift over Baseline |
|--------|-----------|-----|----|--------------------|
| Income | Random Forest | 228 | 0.34 | 33% |
| Expenses | Random Forest | 236 | 0.42 | 43% |

## Project Structure

```
N26/
├── n26.ipynb                        # Main notebook
├── transactions_data_training.csv   # Transaction data
├── user_profile.csv                 # User demographics
├── transaction_types.csv            # Transaction type reference
├── mcc_group_definition.csv         # MCC category reference
├── requirements.txt                 # Python dependencies
└── README.md
```

## Setup

```bash
pip install -r requirements.txt
jupyter notebook n26.ipynb
```

## Requirements

- Python 3.8+
- pandas, numpy, scikit-learn, matplotlib, seaborn
