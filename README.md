# Open Banking Credit Risk Model

**Can 90 days of transaction data replace years of credit history?**

MSc FinTech Dissertation Project | University of Glasgow | Chetan Prakash

---

## The Problem

Traditional credit scoring fails for thin-file customers — young adults, gig workers, and first-time borrowers with no formal credit history. In India alone, 190M+ people cannot access credit because they have no CIBIL score.

These customers do have bank accounts. They transact daily. They pay rent, buy groceries, receive salaries.

This project tests whether **90 days of open banking transaction data** can predict credit default as accurately as traditional bureau data — and whether a model trained on existing customers can score new customers with zero credit history.

---

## Results

| Model | AUC | Brier Score | Accuracy | Defaulters Caught |
|-------|-----|-------------|----------|-------------------|
| Baseline — Logistic Regression | 0.7103 | 0.1942 | 69.9% | 43% |
| Baseline — XGBoost | 0.6804 | 0.2042 | 67.4% | 33% |
| Open Banking — Logistic Regression | 0.9112 | 0.1166 | 84.4% | 84% |
| Open Banking — XGBoost | 0.8992 | 0.1218 | 83.0% | 74% |
| **Hybrid — Logistic Regression** | **0.9108** | **0.1407** | **80.2%** | **89%** |
| Hybrid — XGBoost | 0.8997 | 0.1214 | 83.3% | 78% |

**+28% AUC improvement** over the traditional bureau baseline. The hybrid model catches **89% of defaulters** (289/324) using only transaction data at inference — no credit history needed.

---

## Project Structure

```
open-banking-credit-risk/
│
├── notebooks/
│   ├── 01_Baseline_Traditional_Credit_Model.ipynb
│   ├── 02_OpenBanking_Transaction_Model.ipynb
│   └── 03_Hybrid_Model.ipynb
│
├── data/
│   ├── old_cust_credit.csv           # 9,000 existing customers — training
│   ├── old_cust_transactions.csv     # 1.4M transactions (existing customers)
│   ├── new_cust_credit.csv           # 1,000 new customers — external features only
│   ├── new_cust_transactions.csv     # 154K transactions (new customers)
│   ├── old_txn_features.csv          # Pre-engineered features (existing customers)
│   └── new_txn_features.csv          # Pre-engineered features (new customers)
│
├── outputs/                          # Generated charts (auto-created on run)
├── requirements.txt
└── README.md
```

---

## How It Works

### Notebook 1 — Baseline
Trains on 9,000 existing customers with full credit features. Tests on 1,000 new customers with **external features only** — no credit history, no savings data. This simulates what a bank actually knows about a new applicant. Result: AUC 0.71, catches only 43% of defaulters.

### Notebook 2 — Open Banking
Ignores credit data entirely. Engineers 56 behavioural features from raw transactions across 24 spending categories (Groceries, Gambling, Salary Credit, Utilities, Rent, etc.) and 5 payment methods. Tests on new customer transactions. Result: AUC 0.91, catches 84% of defaulters.

### Notebook 3 — Hybrid (Flagship)
Trains on existing customers using both credit features and transaction features (77 total). At inference, scores new customers using **only their transaction data** — credit columns are zeroed. Result: AUC 0.91, catches 89% of defaulters.

---

## Data Pipeline — How the Transaction Data Was Built

The transaction data in this project was built through a deliberate three-stage pipeline designed to ensure the spending patterns are realistic and grounded in real open banking behaviour.

### Stage 1 — Real Data Collection via Plaid API
Real transaction data was first collected from **5 live bank accounts using the Plaid API** (open banking infrastructure). This gave access to genuine transaction records including transaction amounts, categories, merchant names, payment methods, and timestamps over a 90-day window. This real data was not large enough to train models on directly, but it served a critical purpose — it revealed the actual behavioural patterns that distinguish financially stable customers from high-risk ones. Key observations from the Plaid data included the regularity of utility payments among stable customers, the presence of salary credits as a stability signal, and the correlation between high ATM withdrawal frequency and financial instability.

### Stage 2 — Pattern-Informed Synthetic Data Generation
Using the behavioural insights from the Plaid API data as the foundation, transaction data was then generated programmatically for each customer in the UCI German Credit Dataset. Rather than generating random transactions, the generation logic was directly informed by what the real Plaid data showed:

- **High-risk customers** (those marked as defaulters in the German Credit data): higher food and bar spending, erratic transaction amounts, infrequent utility payments, higher ATM withdrawal frequency — all patterns observed in the Plaid data for financially stressed accounts
- **Low-risk customers** (non-defaulters): regular utility and insurance payments, consistent salary credits, stable retail spending, lower spending volatility — patterns observed in the Plaid data for stable accounts

This approach ensured the synthetic transaction data was not arbitrary — it was anchored to real open banking behaviour.

### Stage 3 — Scaling with SDV (Synthetic Data Vault)
The pattern-informed transaction data was then scaled using the **SDV GaussianCopulaSynthesizer** to produce 9,000 existing customer profiles and 1,000 new customer profiles, preserving the statistical distributions and behavioural correlations learned from the original Plaid-sourced data. All data is fully GDPR and PSD2 compliant — no real customer information is present in the final datasets.

This pipeline reflects exactly how production open banking credit scoring systems are built in practice: real API data informs the feature logic, and synthetic data is used to scale to production volumes while protecting privacy.

---

## Feature Engineering

56 behavioural features engineered from raw transactions:

| Group | Examples | Signal |
|-------|---------|--------|
| **Volume** | avg_txn_amount, total_spend, std_txn_amount | Spending level and stability |
| **Category Ratios** | ratio_gambling, ratio_salary_credit, ratio_utilities | Spending behaviour patterns |
| **Category Totals** | total_gambling, total_fees_penalties, total_savings_transfer | Risk and stability amounts |
| **Payment Methods** | ratio_credit_card, ratio_cash_deposit | Financial habits |
| **Composite Scores** | high_risk_ratio, stability_ratio, essential_ratio | Aggregated risk signals |
| **Timing** | late_night_ratio, weekend_spend_ratio | Behavioural patterns |
| **Income Signals** | has_salary, cash_ratio | Income stability |

---

## Key Finding on Model Selection

Logistic Regression outperforms XGBoost in the hybrid setting. When credit features are zeroed at inference time for new customers, XGBoost's tree-splitting mechanism is disrupted by the artificially zero values. Logistic Regression naturally ignores zero-valued features and relies entirely on the transaction signals.

This supports the **regulatory case for interpretable models** in credit scoring: the transparent model is not only explainable to regulators (aligned with IFRS 9 and SR 11-7 expectations), it also performs better in the real-world thin-file deployment scenario.

---

## Real-World Relevance (India)

This model design maps directly onto India's **Account Aggregator (AA) framework**:

- Existing customers provide the training signal (credit history + transactions)
- New customers share 90 days of transaction data via AA consent
- The model scores them instantly — no CIBIL lookup needed

Companies including Perfios, Finbox, CreditVidya, and Setu are building exactly this type of pipeline for NBFC-AA credit decisioning.

---

## Setup

```bash
git clone https://github.com/Chetanp-g/open_banking_credit_risk.git
cd open_banking_credit_risk
pip install -r requirements.txt
jupyter notebook
```

Run notebooks in order: 01 → 02 → 03. Notebook 03 depends on the engineered features saved by Notebook 02.

---

## Tech Stack

| Tool | Purpose |
|------|---------|
| Python 3.9+ | Core language |
| Pandas / NumPy | Data manipulation and feature engineering |
| Scikit-learn | Logistic Regression, preprocessing, evaluation |
| XGBoost | Gradient-boosted tree model |
| Matplotlib / Seaborn | Visualisation |
| Plaid API | Real open banking transaction data collection |
| SDV (Synthetic Data Vault) | Privacy-preserving synthetic data scaling |

---

## About

**Chetan Prakash** | MSc Financial Technology (Merit) — University of Glasgow
Risk Analyst | Credit Risk Modelling | Open Banking | Python | SQL

chetanp.g21@gmail.com

*This project is the empirical foundation of my MSc dissertation: "Effectiveness of Open Banking in Credit Scoring and Risk Assessment" (2024)*
