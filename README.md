# AR Collections Intelligence Engine

A machine learning system that predicts invoice payment outcomes, estimates payment timing, and generates prioritized collection worklists -- enabling collections teams to focus effort where it recovers the most cash, fastest.

Built end-to-end: from raw AR data through feature engineering, classification, survival analysis, and a prioritization engine with simulated business impact.

---

## Business Problem

Accounts Receivable teams in B2B companies face a resource allocation problem that directly impacts cash flow.

A typical mid-size portfolio has thousands of overdue invoices at any point. Collectors have finite bandwidth -- 30 to 50 contacts per day. The standard approach is to work invoices by age (oldest first) or amount (largest first). Neither accounts for the actual likelihood of recovery, the payment timing trajectory, or the diminishing returns of repeated follow-up on the same account.

The result:

- Collectors spend time on invoices that would have been paid anyway
- High-risk invoices in the critical 15-60 DPD window get attention too late
- No differentiation between a reliable enterprise payer who is 10 days late and a deteriorating SMB account at the same DPD
- DSO stays elevated because collection effort is not aligned with recovery probability

This system replaces rule-based prioritization with a model-driven approach that scores every open invoice on expected recovery value per unit of collector effort.

---

## Why This Matters

| Metric | What It Measures | How This System Helps |
|--------|-----------------|----------------------|
| **DSO** (Days Sales Outstanding) | Average time to collect payment | Accelerates collection on high-probability invoices in the intervention window |
| **Recovery Rate** | Percentage of AR ultimately collected | Identifies at-risk invoices before they age past the recovery cliff (60-90 DPD) |
| **Collector Productivity** | Cash collected per contact | Routes effort toward invoices with the highest expected value per contact |
| **Cash Flow Predictability** | Variance in monthly collections | Survival model provides probabilistic payment timing, not just binary predictions |

---

## Solution Overview

The system has four layers, each feeding the next:

```
Raw AR Data (5 tables)
    |
    v
Feature Engineering (39 features, observation-date-aware to prevent leakage)
    |
    +---> Classification Model (LightGBM) ---> P(paid within 90 days)
    |
    +---> Survival Model (Cox PH) -----------> Payment timing distribution
    |
    v
Prioritization Engine
    Priority Score = P(recovery | contact) x Amount x Urgency / Effort
    |
    v
Decision Layer
    +-- Daily collector work queue (ranked)
    +-- Customer strategy segments (5 playbooks)
    +-- Expected recovery forecasts
```

The output is not a model score. It is a ranked worklist with recommended actions, expected recovery amounts, and differentiated collection strategies by customer risk profile.

---

## Key Capabilities

**Invoice Payment Prediction**
Binary classification: will this invoice be fully paid within 90 days? LightGBM with Optuna hyperparameter tuning, isotonic probability calibration, and time-based train/test split to prevent temporal leakage.

**Payment Timing Estimation**
Cox Proportional Hazards model produces a full survival curve per invoice -- not a point estimate, but P(paid by day 30), P(paid by day 60), median expected payment time, and confidence intervals. Handles right-censored invoices (still open at observation end) correctly.

**Customer Risk Segmentation**
Five strategy segments, each with a distinct collection playbook:

| Segment | Profile | Collection Approach |
|---------|---------|-------------------|
| A -- Reliable Payer | Low risk, low DPD | Automated reminders only |
| B -- Variable Payer | Moderate risk, <40 DPD | Structured escalation (email -> phone -> senior) |
| C -- Chronic Late Payer | High DPD, persistent delays | Immediate phone contact from day 1 |
| D -- High-Risk High-Value | High risk, large exposure | Senior collector, frequent contact |
| E -- Likely Write-off | Very high DPD or very low P(recovery) | Minimize effort, route to external agency |

**Collection Prioritization**
Composite priority score combining recovery probability, invoice amount, aging-based urgency (peaked at 15-60 DPD, the empirical intervention sweet spot), and estimated collector effort.

---

## Dataset

Five relational tables modeling a realistic B2B AR portfolio:

| Table | Records | Description |
|-------|---------|-------------|
| `customers` | 5,000 | Segment, industry, region, credit limit, risk tier, behavioral archetype |
| `invoices` | ~120,000 | Invoice amount, dates, payment terms, PO number, product category |
| `payments` | ~100,000 | Payment date, amount, method, partial payment flag |
| `disputes` | ~8,000 | Dispute reason, resolution status, resolution amount |
| `dunning_contacts` | ~70,000 | Contact type, outcome, promised payment dates, collector ID |

**Data is synthetic**, generated with 10 behavioral archetypes calibrated to realistic B2B patterns:

- Q1 seasonal payment delays (post-holiday cash crunch)
- Quarter-end payment bunching
- Customer-level payment delay autocorrelation
- Progressive deterioration trajectories for at-risk accounts
- Segment-specific dispute rates and partial payment behavior
- Dunning response rates with diminishing returns after 3rd contact

Invoice amounts follow lognormal distributions with segment-appropriate parameters. Payment timing reflects archetype-specific delay distributions, not random noise.

---

## Feature Engineering

39 features organized by business domain. All features enforce strict observation-date filtering -- no future information leaks into any feature.

**Invoice characteristics**: amount (log-transformed), payment terms, aging bucket, PO presence (proxy for procurement process maturity), recurring flag, line item count.

**Customer payment history** (6-month rolling window): average/median/std payment delay, on-time rate, worst delay, payment count. A payment consistency index (coefficient of variation of delays) captures erratic vs. predictable payers.

**Dispute exposure**: active dispute flag on the specific invoice, customer-level dispute rate over trailing 12 months.

**Collection activity**: contacts to date, days since last contact, promise-to-pay flag, phone contact count, max escalation level reached. These capture dunning responsiveness and effort already invested.

**Portfolio pressure**: open invoice count per customer, total open AR, credit utilization ratio. High utilization signals cash flow stress.

**Seasonal and calendar**: Q1 indicator, quarter-end month flag, year-end flag. Q1 adds roughly 3-8 days of delay depending on segment (empirically observed).

**Composite health score**: weighted combination of on-time rate, credit utilization, payment consistency, dispute rate, payment frequency, and tenure. Single 0-1 metric summarizing customer payment health.

---

## Modeling Approach

### Classification: Will It Be Paid?

- **Model**: LightGBM (gradient boosted trees)
- **Target**: Fully paid within 90 days of due date (binary)
- **Split**: Time-based -- train on invoices due through Jun 2023, validate Jul-Sep, test Oct-Dec. Not random, because in production you always predict forward.
- **Tuning**: Optuna Bayesian optimization (50 trials)
- **Calibration**: Isotonic regression on validation set -- critical because raw LightGBM probabilities are not well-calibrated, and the prioritization formula uses these probabilities directly
- **Explainability**: SHAP values (global importance + per-invoice waterfall plots)

### Survival Analysis: When Will It Be Paid?

- **Model**: Cox Proportional Hazards
- **Event**: Full payment received
- **Censoring**: Invoices still open at observation window end are right-censored (not dropped -- they contain valid survival information)
- **Output**: Per-invoice survival curve, median payment time, P(paid within 30/60/90 days), 25th-75th percentile payment window

### Why Two Models

Classification answers **if**. Survival answers **when**. The prioritization engine needs both: an invoice with 80% payment probability but predicted median payment at 75 days is treated differently from one with 80% probability and predicted payment at 15 days. The urgency component of the priority score is derived from the survival model's timing prediction.

---

## Outputs and Decision Layer

The system produces three actionable outputs:

**1. Ranked Collector Work Queue**

Each morning, a collector receives a prioritized list of invoices to work, ordered by expected recovery value per contact. Each entry includes the recommended action (first contact, phone follow-up, escalation, dispute resolution, final notice) based on the invoice's current state.

**2. Customer Strategy Assignment**

Every customer is assigned to one of five strategy segments. Each segment triggers a different collection playbook -- from fully automated reminders for reliable payers to immediate senior-collector involvement for high-risk high-value accounts.

**3. Portfolio Recovery Forecast**

Aggregated survival curves produce portfolio-level cash flow projections: expected collections at 30, 60, and 90 days, broken down by segment, risk tier, and aging bucket.

---

## Sample Use Case

A collections team manages a portfolio of 15,000 open invoices. 10 collectors each handle 40 contacts per day -- 400 total daily capacity across a portfolio where thousands of invoices are overdue.

Under the old process (FIFO by due date), collectors work through invoices chronologically. A reliable enterprise customer 12 days past due gets a call before a deteriorating SMB account at 35 DPD with a history of broken promises, simply because the enterprise invoice is older.

Under model-driven prioritization, the system scores every open invoice overnight. The SMB account scores higher: its recovery probability is dropping fast (35 DPD is in the steep part of the recovery curve), the invoice amount is material, and the survival model predicts payment will slip past 90 days without intervention. The enterprise invoice scores lower: the model gives it 92% recovery probability regardless of contact.

The collector's morning queue surfaces the SMB invoice first, with a recommended action of "Phone Follow-up" and a note that the customer has one prior promise-to-pay that was not honored.

---

## Business Impact (Simulated)

A 60-day simulation comparing five prioritization strategies across the test portfolio (10 collectors, 400 contacts/day):

| Strategy | Cash Collected | Cash per Contact | Recovery Rate |
|----------|---------------|------------------|---------------|
| **Model-Driven** | Highest | Highest | Highest |
| FIFO (oldest first) | Baseline | Baseline | Baseline |
| Amount-based | Moderate | Moderate | Moderate |
| Aging-based | Similar to FIFO | Similar to FIFO | Similar to FIFO |
| Random | Lowest | Lowest | Lowest |

Model-driven prioritization consistently outperforms FIFO across all three metrics. The primary driver is concentration of effort on invoices in the 15-60 DPD intervention window where marginal recovery probability is highest, combined with reduced effort on invoices that would be paid without contact.

**Key assumptions** (stated transparently):
- Contact accelerates payment by ~20% for invoices that would eventually be paid
- Contact adds ~10% recovery probability for invoices that would otherwise default
- Both estimates are grounded in the dunning effectiveness patterns observed in EDA
- Results are simulated, not from production deployment

**Recommended validation**: A/B test in production -- split the collector team, one group uses model-driven worklists, the other uses the existing process. Compare recovery rates and DSO after 90 days.

---

## Tech Stack

| Component | Technology |
|-----------|-----------|
| Core | Python 3.10+, pandas, NumPy |
| Classification | LightGBM, scikit-learn, Optuna |
| Survival Analysis | lifelines (Kaplan-Meier, Cox PH) |
| Explainability | SHAP |
| Calibration | scikit-learn (isotonic regression) |
| Dashboard | Streamlit, Plotly |
| Serialization | joblib, JSON |

---

## How to Run

```bash
# 1. Clone and install
git clone <https://github.com/DataFinFreak/ar-collections-engine.git>
cd ar-collections-engine
pip install -r requirements.txt

# 2. Generate synthetic data
python 01_data_generation.py

# 3. Run notebooks in order
#    02_eda.ipynb                          -- Exploratory analysis
#    03_Feature_Engineering.ipynb          -- Build modeling dataset
#    04_Classification_Model.ipynb         -- Train LightGBM + SHAP
#    05_Survival_Analysis.ipynb            -- Cox PH survival model
#    06_Prioritization___Business_Simulation.ipynb  -- Scoring + simulation

# 4. Launch dashboard
streamlit run app.py
```

---

## Project Structure

```
ar-collections-engine/
|
|-- 01_data_generation.py              # Synthetic data with 10 behavioral archetypes
|-- 02_eda.ipynb                       # AR portfolio analysis, aging, recovery curves
|-- 03_Feature_Engineering.ipynb       # 39 features with observation-date leakage prevention
|-- 04_Classification_Model.ipynb      # LightGBM + Optuna + SHAP + calibration
|-- 05_Survival_Analysis.ipynb         # Kaplan-Meier + Cox PH payment timing
|-- 06_Prioritization___Business_Simulation.ipynb  # Priority scoring + strategy simulation
|-- app.py                             # Streamlit dashboard
|
|-- data/
|   |-- raw/                           # Generated tables (CSV + Parquet)
|
|-- processed/
|   |-- modeling_dataset.csv           # Feature-engineered modeling table
|   |-- test_predictions.csv           # Classification predictions
|   |-- survival_predictions.csv       # Timing predictions
|   |-- priority_scores.csv            # Final priority scores
|   |-- strategy_segments.csv          # Customer segment assignments
|   |-- simulation_results.csv         # Daily simulation logs
|
|-- models/
|   |-- lgbm_classifier.pkl            # Trained LightGBM
|   |-- lgbm_calibrated.pkl            # Calibrated model
|   |-- cox_ph_model.pkl               # Survival model
|   |-- model_metadata.json            # Features, params, metrics
```

---

## Future Improvements

**Near-term**
- Real-time scoring via REST API (FastAPI) for ERP/CRM integration
- Collector feedback loop: track which recommended actions were taken and actual outcomes, retrain quarterly
- Segmentation refinement: industry-specific and region-specific strategy playbooks

**Medium-term**
- Time-varying Cox model or recurrent survival forests for non-proportional hazard effects
- Multi-objective optimization: balance recovery maximization against customer relationship preservation
- Automated threshold tuning per segment (Enterprise vs SMB optimal thresholds differ)

**Production-grade**
- Drift monitoring on feature distributions and model performance (PSI, AUC decay tracking)
- Shadow mode deployment: run model-driven queue alongside existing process before full cutover
- Integration with payment gateway data for real-time payment event capture
