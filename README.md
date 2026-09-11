# Hotel Booking Cancellations: An Explanatory and Predictive Approach
Hotel booking cancellations generate significant operational and financial challenges for hotels. 
The objective of this project is to identify the main drivers of reservation cancellations and develop predictive models capable of
estimating the probability that a booking will be canceled before arrival.

## Objective
This project approaches hotel cancellation prediction from two complementary perspectives:

- **Explanatory Model (statsmodels):** The goal is to *understand* which factors influence cancellations and in what direction.
  Statistical interpretability is prioritized, featuring variable selection based on p-values and VIF, alongside Odds Ratio analysis
  to quantify the effect of each predictor.
- **Predictive Model (scikit-learn):** The goal is to *maximize performance* on unseen data. This pipeline applies transformations,
- regularization, and classification threshold adjustment, evaluating the model using AUC and the confusion matrix.

This distinction drives different methodological decisions at every stage: the variables included, the transformations applied, and 
the metrics used to evaluate the results.

## Project Structure
Main notebook:
```
hotel_booking_cancellations_analysis.ipynb  
```

## Dataset
Data loaded directly from:
```
https://raw.githubusercontent.com/eduardofc/data/refs/heads/main/reservas_hotel.csv
```

(~119k records).


> Note: The original dataset is in Spanish. All column names are renamed to English at the beginning of the notebook.


**Target**: `reserva_cancelada` — 1 if cancelled, 0 if not (~37% cancellation rate).

**Key features**: lead time, arrival date, stay duration, meal plan, country of origin, market segment, distribution channel, 
deposit type, customer type, previous cancellations, average daily rate, and more.

## Methodology

### 1. EDA & Feature Engineering
- Removed data-leakage variables (`reservation_status`, `reservation_status_date`)
- Temporal feature extraction from `arrival_date`; dropped year, quarter, and weekday due to low variability
- Missing value imputation: median for `children`, constant `'Unknown'` for `country`
- Removed invalid records: zero-occupancy bookings, stays with 0 nights, prices ≤ 0 or < 25
- Binarized zero-concentrated variables; capped `adults` at 10; created `room_change` flag
- `room_change` later excluded from final models due to potential data leakage

### 2. Train/Test Split
- 80/20 split with stratification on the target variable (`random_state=99`)

### 3. Explanatory Model (statsmodels)
- Logistic Regression estimated with `statsmodels.Logit` for statistical inference
- Iterative variable selection based on p-values (> 0.05) and VIF (> 5)
- Dropped `required_parking_spaces` (near-perfect separation) and `is_repeated_guest` (collinearity with `previous_bookings_not_canceled`)
- **Final model:** McFadden's Pseudo R² ≈ 0.246 (with `room_change` excluded)
- Odds Ratios computed for interpretation

**Key findings:**
- Previous cancellations: strongest risk factor (~128× higher odds)
- Transient customer type: ~7× higher cancellation odds
- Previous successful bookings: ~99% reduction in cancellation odds
- German customers (DEU): ~72% lower cancellation odds than reference group

### 4. Predictive Model (scikit-learn)
- Applied Yeo-Johnson transform to `lead_time` and Box-Cox to `average_daily_rate`
- Clipped `stays_nights` at 14 nights; standardized remaining numerical features
- Compared Ridge vs. Lasso Logistic Regression via 5-fold CV (ROC-AUC)
- Hyperparameter tuning with `LogisticRegressionCV`
- **Final model:** Lasso Logistic Regression (L1, C=1, `liblinear` solver)

**Performance on test set:**
| Metric | Value |
|---|---|
| AUC | 0.897 |
| Accuracy | 82.1% |
| Threshold | 0.467 |

> The model shows strong discriminative capacity with no evidence of overfitting. Its main limitation is recall — if the priority is
detecting cancellations for overbooking strategies, lowering the threshold is recommended.

## Requirements

```
pandas
numpy
matplotlib
seaborn
scikit-learn
statsmodels
```

Install with:

```bash
pip install pandas numpy matplotlib seaborn scikit-learn statsmodels
```

Usage:

Open and run the notebook sequentially. The dataset is loaded directly from a public URL — no local file download is needed.

```bash
jupyter notebook hotel_booking_cancellations_analysis.ipynb
```
