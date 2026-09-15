# Ames Housing Price Predictor

An end-to-end machine learning project that predicts house sale prices in Ames, Iowa — from raw, messy data all the way to a live, deployed app anyone can use.

**[Try it live](https://ames-housing-price-prediction-production.up.railway.app)** ↗

---

## What this is

This isn't a "load a clean dataset and call `.fit()`" project. It's the full pipeline, built and debugged by hand:

- Cleaning a real, messy dataset (19 columns with missing values, each requiring a different judgment call)
- Exploratory data analysis to understand what actually drives house prices
- Feature engineering (derived features, encoding, scaling)
- A **linear regression model implemented from scratch in NumPy** — gradient descent, cost function, and L2 regularization, all hand-coded, not imported
- Benchmarking that from-scratch model against scikit-learn's Ridge, Lasso, Random Forest, and XGBoost
- A simple, non-technical UI (Streamlit) so anyone — not just developers — can get a prediction
- Live deployment on Railway

## The pipeline

| Phase | What happened |
|---|---|
| 1. Data Cleaning | Identified 19 columns with missing values, and — critically — figured out which "missing" values actually meant "this feature doesn't exist" (no pool, no garage) vs. genuinely missing data (LotFrontage, Electrical) |
| 2. EDA | Found `SalePrice` was right-skewed and log-transformed it; identified `OverallQual`, `GrLivArea`, and `Neighborhood` as the strongest predictors; caught and removed two structural outliers |
| 3. Feature Engineering | Built `HouseAge`, `RemodAge`, and `TotalSF` from raw date/area columns; one-hot encoded categoricals; scaled features with `StandardScaler` |
| 4. Model From Scratch | Implemented `predict()`, `compute_cost()`, and `compute_gradient()` by hand in NumPy; diagnosed overfitting (14x train/validation gap) and fixed it through feature trimming and L2 regularization |
| 5. Benchmarking | Compared the from-scratch model against sklearn's Ridge and Lasso (tuned via `GridSearchCV`) |
| 6. Stretch Models | Added Random Forest and XGBoost for comparison — useful negative result: none of them beat the linear models here, telling us the relationships in this data are mostly linear |
| 7. Deployment | Wrapped the final model in a Streamlit app and deployed it live on Railway |

## Final model

**Lasso Regression (alpha = 0.001)** — chosen over the alternatives because it matched the best validation performance while automatically zeroing out ~150 low-signal features, making it simpler and faster to serve than the tree-based alternatives.

| Model | Validation MSE (log scale) |
|---|---|
| **Lasso (tuned)** | **0.01538** |
| XGBoost | 0.01617 |
| From-scratch (manual) | 0.01654 |
| Ridge (tuned) | 0.01661 |
| Plain Linear Regression | 0.01887 |
| Random Forest | 0.02177 |

## A real bug worth mentioning

Midway through this project, a positional-indexing bug (`.iloc[:, -1]` silently pointing to the wrong column after the dataframe's structure changed) meant the model was accidentally trained on the wrong target for almost a full day. Every result was re-validated and retrained once this was caught. It's mentioned here because catching and fixing it — rather than shipping numbers that looked fine but weren't — was as much a part of this project as the modeling itself.

## Tech stack

Python · NumPy · pandas · scikit-learn · XGBoost · Streamlit · Railway

## Running it locally

```bash
git clone https://github.com/hasnainhafeezqureshi-prog/ames-housing-price-prediction
cd ames-housing-price-prediction
pip install -r requirements.txt
streamlit run app.py
```

## What's next

- A second, smaller-scoped project applying this same pipeline to real estate listings in DHA, Karachi
- Possibly exposing this model behind a proper API again (it briefly was, via FastAPI) for use by other services, alongside the Streamlit UI

## Author

**Hasnain Qureshi**
[GitHub](https://github.com/hasnainhafeezqureshi-prog) · [Upwork](https://www.upwork.com/freelancers/~01751f28c6877bb1dc?mp_source=share) · [Fiverr](https://www.fiverr.com/s/wbAgZpq)
