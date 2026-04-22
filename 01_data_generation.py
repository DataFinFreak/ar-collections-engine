"""
AR Collections Intelligence Engine — Synthetic Data Generator (Stable Version)
=============================================================================
Generates 5 tables with realistic B2B AR behavioral patterns.

Fixes included:
1. Removed stray character causing NameError
2. Added safe clipping to prevent timestamp overflow
3. Stabilized deterioration growth so delays do not explode
4. Added small guards for empty tables
"""

import numpy as np
import pandas as pd
from datetime import timedelta
import warnings
import os
import time

warnings.filterwarnings("ignore")
np.random.seed(42)

# ============================================================
# CONFIG
# ============================================================

OUTPUT_DIR = "./data/raw/"
os.makedirs(OUTPUT_DIR, exist_ok=True)

START_DATE = pd.Timestamp("2022-01-01")
END_DATE = pd.Timestamp("2023-12-31")
OBSERVATION_END = pd.Timestamp("2023-12-31")
NUM_CUSTOMERS = 5000

t0 = time.time()

# ============================================================
# REFERENCE DATA
# ============================================================

ARCHETYPES = {
    "reliable_enterprise": {
        "weight": 0.15,
        "segments": ["Enterprise"],
        "delay_mean": 3,
        "delay_std": 5,
        "dispute_rate": 0.03,
        "partial_rate": 0.02,
        "writeoff_rate": 0.005,
        "qe_bunch": 0.10,
        "q1_add": 2,
        "q1_std": 2,
        "terms": [45, 60],
        "cl_range": (3e6, 20e6),
        "amt_log_mean": 12.5,
        "amt_log_std": 0.7,
        "inv_range": (1, 4),
        "risk": "Low",
        "responsiveness": 0.85,
        "autocorr": 0.30,
        "deter_prob": 0.02,
    },
    "slow_but_steady_enterprise": {
        "weight": 0.10,
        "segments": ["Enterprise"],
        "delay_mean": 15,
        "delay_std": 5,
        "dispute_rate": 0.04,
        "partial_rate": 0.02,
        "writeoff_rate": 0.005,
        "qe_bunch": 0.15,
        "q1_add": 3,
        "q1_std": 3,
        "terms": [45, 60],
        "cl_range": (2e6, 15e6),
        "amt_log_mean": 12.2,
        "amt_log_std": 0.6,
        "inv_range": (1, 3),
        "risk": "Low",
        "responsiveness": 0.75,
        "autocorr": 0.50,
        "deter_prob": 0.03,
    },
    "quarter_end_payer": {
        "weight": 0.10,
        "segments": ["Enterprise", "Mid-Market"],
        "delay_mean": 10,
        "delay_std": 8,
        "dispute_rate": 0.05,
        "partial_rate": 0.03,
        "writeoff_rate": 0.008,
        "qe_bunch": 0.55,
        "q1_add": 5,
        "q1_std": 3,
        "terms": [30, 45, 60],
        "cl_range": (1.5e6, 12e6),
        "amt_log_mean": 11.8,
        "amt_log_std": 0.8,
        "inv_range": (1, 3),
        "risk": "Low",
        "responsiveness": 0.60,
        "autocorr": 0.40,
        "deter_prob": 0.04,
    },
    "variable_midmarket": {
        "weight": 0.20,
        "segments": ["Mid-Market"],
        "delay_mean": 18,
        "delay_std": 15,
        "dispute_rate": 0.08,
        "partial_rate": 0.06,
        "writeoff_rate": 0.02,
        "qe_bunch": 0.10,
        "q1_add": 6,
        "q1_std": 4,
        "terms": [30, 45],
        "cl_range": (5e5, 5e6),
        "amt_log_mean": 11.2,
        "amt_log_std": 0.9,
        "inv_range": (1, 3),
        "risk": "Medium",
        "responsiveness": 0.55,
        "autocorr": 0.35,
        "deter_prob": 0.06,
    },
    "cash_strapped_smb": {
        "weight": 0.15,
        "segments": ["SMB"],
        "delay_mean": 22,
        "delay_std": 14,
        "dispute_rate": 0.10,
        "partial_rate": 0.10,
        "writeoff_rate": 0.04,
        "qe_bunch": 0.05,
        "q1_add": 8,
        "q1_std": 5,
        "terms": [30],
        "cl_range": (1e5, 1.5e6),
        "amt_log_mean": 10.5,
        "amt_log_std": 0.8,
        "inv_range": (1, 3),
        "risk": "Medium",
        "responsiveness": 0.45,
        "autocorr": 0.50,
        "deter_prob": 0.08,
    },
    "dispute_prone": {
        "weight": 0.08,
        "segments": ["Mid-Market", "SMB"],
        "delay_mean": 25,
        "delay_std": 12,
        "dispute_rate": 0.20,
        "partial_rate": 0.05,
        "writeoff_rate": 0.03,
        "qe_bunch": 0.05,
        "q1_add": 5,
        "q1_std": 4,
        "terms": [30, 45],
        "cl_range": (3e5, 3e6),
        "amt_log_mean": 11.0,
        "amt_log_std": 0.85,
        "inv_range": (1, 4),
        "risk": "Medium",
        "responsiveness": 0.40,
        "autocorr": 0.30,
        "deter_prob": 0.05,
    },
    "chronic_late_payer": {
        "weight": 0.10,
        "segments": ["Mid-Market", "SMB"],
        "delay_mean": 42,
        "delay_std": 15,
        "dispute_rate": 0.07,
        "partial_rate": 0.08,
        "writeoff_rate": 0.05,
        "qe_bunch": 0.05,
        "q1_add": 8,
        "q1_std": 5,
        "terms": [30],
        "cl_range": (2e5, 2e6),
        "amt_log_mean": 10.8,
        "amt_log_std": 0.9,
        "inv_range": (1, 3),
        "risk": "High",
        "responsiveness": 0.35,
        "autocorr": 0.60,
        "deter_prob": 0.04,
    },
    "high_risk_deteriorating": {
        "weight": 0.05,
        "segments": ["Mid-Market", "SMB"],
        "delay_mean": 20,
        "delay_std": 12,
        "dispute_rate": 0.10,
        "partial_rate": 0.12,
        "writeoff_rate": 0.08,
        "qe_bunch": 0.05,
        "q1_add": 7,
        "q1_std": 5,
        "terms": [30, 45],
        "cl_range": (2e5, 3e6),
        "amt_log_mean": 11.0,
        "amt_log_std": 0.85,
        "inv_range": (1, 3),
        "risk": "High",
        "responsiveness": 0.40,
        "autocorr": 0.50,
        "deter_prob": 0.90,
    },
    "new_customer_uncertain": {
        "weight": 0.05,
        "segments": ["Mid-Market", "SMB"],
        "delay_mean": 12,
        "delay_std": 18,
        "dispute_rate": 0.08,
        "partial_rate": 0.05,
        "writeoff_rate": 0.03,
        "qe_bunch": 0.05,
        "q1_add": 4,
        "q1_std": 3,
        "terms": [30],
        "cl_range": (1e5, 1e6),
        "amt_log_mean": 10.3,
        "amt_log_std": 0.7,
        "inv_range": (1, 2),
        "risk": "Medium",
        "responsiveness": 0.50,
        "autocorr": 0.20,
        "deter_prob": 0.10,
    },
    "seasonal_payer": {
        "weight": 0.02,
        "segments": ["Mid-Market", "SMB"],
        "delay_mean": 8,
        "delay_std": 6,
        "dispute_rate": 0.05,
        "partial_rate": 0.04,
        "writeoff_rate": 0.02,
        "qe_bunch": 0.10,
        "q1_add": 25,
        "q1_std": 10,
        "terms": [30, 45],
        "cl_range": (3e5, 3e6),
        "amt_log_mean": 11.0,
        "amt_log_std": 0.8,
        "inv_range": (1, 3),
        "risk": "Medium",
        "responsiveness": 0.55,
        "autocorr": 0.30,
        "deter_prob": 0.04,
    },
}

INDUSTRIES = [
    "Manufacturing", "IT Services", "Retail", "Healthcare", "Financial Services",
    "Logistics", "Construction", "Telecom", "Pharma", "Energy"
]
INDUSTRY_W = [0.15, 0.18, 0.12, 0.10, 0.10, 0.08, 0.08, 0.07, 0.07, 0.05]

REGIONS = ["North", "South", "East", "West"]
REGION_W = [0.22, 0.32, 0.20, 0.26]

PRODUCTS = [
    "Software License", "IT Hardware", "Consulting Services", "Maintenance Contract",
    "Raw Materials", "Equipment Lease", "Logistics Services", "Professional Services",
    "Subscription", "Project Delivery"
]

PAY_METHODS = ["Wire", "ACH", "Check", "Online"]
PAY_METHOD_W = [0.30, 0.35, 0.15, 0.20]

DISPUTE_REASONS = ["Pricing", "Quality", "Delivery", "Duplicate", "Documentation", "Other"]
DISPUTE_REASON_W = [0.25, 0.20, 0.20, 0.10, 0.15, 0.10]

COLLECTORS = [f"COLL_{i:02d}" for i in range(1, 21)]

PREFIXES = [
    "Tata", "Reliance", "Infosys", "Mahindra", "Godrej", "Wipro", "Bajaj", "Birla",
    "Hero", "Adani", "JSW", "Vedanta", "HDFC", "Kotak", "Sun", "Cipla", "Lupin",
    "Havells", "Titan", "Zenith", "Pinnacle", "Nexus", "Orbit", "Vanguard", "Synergy",
    "Prism", "Apex", "Kiran", "Sagar", "Lakshmi", "Patel", "Sharma", "Gupta", "Mehta",
    "Joshi", "Rao", "Reddy", "Nair", "Desai", "Verma", "Singh", "Kumar", "Bhatt",
    "Kapoor", "Arora", "Malhotra", "Bhatia", "Saxena", "Trivedi", "Pandey", "Mishra",
    "Agarwal", "Bansal", "Mittal", "Jain", "Sethi", "Khanna", "Dutta", "Bose", "Ghosh",
    "Das", "Roy", "Saha", "Thakur", "Rawat", "Chauhan", "Tiwari", "Shukla", "Kulkarni",
    "Patil", "Pawar", "Shinde", "Amara", "Iyer"
]

SUFFIXES = [
    "Industries", "Technologies", "Solutions", "Enterprises", "Systems", "Services",
    "Corporation", "Group", "Pvt Ltd", "Infra", "Pharma", "Chemicals", "Textiles",
    "Engineers", "Traders", "Logistics", "Healthcare", "Power", "Steels", "Foods",
    "Electronics", "Polymers", "Exports", "Innovations", "Ventures", "Capital",
    "Associates", "Infotech", "Global"
]

# ============================================================
# HELPERS
# ============================================================

def snap_bday(dt):
    dow = dt.dayofweek if isinstance(dt, pd.Timestamp) else dt.weekday()
    if dow == 5:
        return dt - timedelta(days=1)
    if dow == 6:
        return dt + timedelta(days=1)
    return dt

def snap_payday(dt):
    dt = snap_bday(dt)
    dow = dt.dayofweek if isinstance(dt, pd.Timestamp) else dt.weekday()
    if dow in [0, 4] and np.random.random() < 0.4:
        shift = int(np.random.choice([1, 2]))
        dt = dt + timedelta(days=shift) if dow == 0 else dt - timedelta(days=shift)
    return dt

def get_qe(dt):
    qe_months = {1:3, 2:3, 3:3, 4:6, 5:6, 6:6, 7:9, 8:9, 9:9, 10:12, 11:12, 12:12}
    m = qe_months[dt.month]
    if m == 12:
        last = pd.Timestamp(dt.year, 12, 31)
    else:
        last = pd.Timestamp(dt.year, m + 1, 1) - timedelta(days=1)
    return snap_bday(last)

def recovery_prob(dpd):
    if dpd <= 0:
        return 0.95
    if dpd <= 30:
        return 0.85
    if dpd <= 60:
        return 0.70
    if dpd <= 90:
        return 0.50
    if dpd <= 120:
        return 0.30
    if dpd <= 180:
        return 0.15
    return 0.05

print(f"[{time.time() - t0:.1f}s] Setup complete")

# ============================================================
# TABLE 1: CUSTOMERS
# ============================================================

print("Generating customers...")

arch_names = list(ARCHETYPES.keys())
arch_weights = [ARCHETYPES[a]["weight"] for a in arch_names]
assigned = np.random.choice(arch_names, size=NUM_CUSTOMERS, p=arch_weights)

used_names = set()
customers = []

for i in range(NUM_CUSTOMERS):
    archetype_name = assigned[i]
    archetype = ARCHETYPES[archetype_name]

    seg = np.random.choice(archetype["segments"])
    industry = np.random.choice(INDUSTRIES, p=INDUSTRY_W)
    region = np.random.choice(REGIONS, p=REGION_W)
    terms = int(np.random.choice(archetype["terms"]))

    credit_limit = np.random.uniform(*archetype["cl_range"])
    credit_limit = round(credit_limit / 50000) * 50000 + np.random.randint(-25000, 25001)
    credit_limit = max(50000, credit_limit)

    if archetype_name == "new_customer_uncertain":
        customer_since = START_DATE + timedelta(days=np.random.randint(480, 700))
    elif seg == "Enterprise":
        customer_since = START_DATE - timedelta(days=np.random.randint(0, 365))
    else:
        customer_since = START_DATE + timedelta(days=np.random.randint(0, 360))

    customer_since = snap_bday(customer_since)

    for _ in range(50):
        customer_name = f"{np.random.choice(PREFIXES)} {np.random.choice(SUFFIXES)}"
        if customer_name not in used_names:
            used_names.add(customer_name)
            break
    else:
        customer_name = f"{np.random.choice(PREFIXES)} {np.random.choice(SUFFIXES)} {i}"

    customers.append({
        "customer_id": f"CUST_{i:04d}",
        "customer_name": customer_name,
        "customer_segment": seg,
        "industry": industry,
        "region": region,
        "payment_terms_days": terms,
        "credit_limit": round(credit_limit, 2),
        "customer_since": customer_since,
        "customer_risk_tier": archetype["risk"],
        "archetype": archetype_name,
    })

customers_df = pd.DataFrame(customers)
customers_df["customer_since"] = pd.to_datetime(customers_df["customer_since"])

print(f"[{time.time() - t0:.1f}s] Customers: {len(customers_df)}")
print(f"  Segments: {dict(customers_df['customer_segment'].value_counts())}")
print(f"  Archetypes: {dict(customers_df['archetype'].value_counts())}")

# ============================================================
# TABLE 2: INVOICES
# ============================================================

print("\nGenerating invoices...")

invoices = []
invoice_counter = 0

for _, customer in customers_df.iterrows():
    archetype = ARCHETYPES[customer["archetype"]]
    inv_lo, inv_hi = archetype["inv_range"]

    customer_start = max(START_DATE, customer["customer_since"])
    current_month = pd.Timestamp(customer_start.year, customer_start.month, 1)

    while current_month <= END_DATE:
        skip_prob = 0.05 if customer["customer_segment"] == "SMB" else 0.02
        if np.random.random() < skip_prob:
            current_month += pd.DateOffset(months=1)
            continue

        n_invoices = np.random.randint(inv_lo, inv_hi + 1)
        days_in_month = (current_month + pd.DateOffset(months=1) - timedelta(days=1)).day

        for _ in range(n_invoices):
            invoice_counter += 1
            d = np.random.randint(1, days_in_month + 1)

            try:
                invoice_date = pd.Timestamp(current_month.year, current_month.month, min(d, days_in_month))
            except Exception:
                invoice_date = pd.Timestamp(current_month.year, current_month.month, days_in_month)

            invoice_date = snap_bday(invoice_date)

            if invoice_date < customer_start:
                invoice_counter -= 1
                continue

            due_date = snap_bday(invoice_date + timedelta(days=int(customer["payment_terms_days"])))

            amount = np.random.lognormal(archetype["amt_log_mean"], archetype["amt_log_std"])
            if amount > 100000:
                amount = round(amount / 1000) * 1000 + np.random.randint(-500, 501)
            elif amount > 10000:
                amount = round(amount / 100) * 100 + np.random.randint(-50, 51)
            else:
                amount = round(amount / 10) * 10 + np.random.randint(-5, 6)

            amount = max(500, round(amount, 2))

            product_category = np.random.choice(PRODUCTS)
            is_recurring = (
                product_category in ["Subscription", "Maintenance Contract", "Software License"]
                or np.random.random() < 0.15
            )

            if customer["customer_segment"] == "Enterprise":
                has_po = np.random.random() < 0.92
            elif customer["customer_segment"] == "Mid-Market":
                has_po = np.random.random() < 0.70
            else:
                has_po = np.random.random() < 0.40

            po_number = (
                f"PO-{np.random.randint(2022, 2024)}-{np.random.randint(100, 9999)}"
                if has_po else None
            )

            invoices.append({
                "invoice_id": f"INV_{invoice_counter:06d}",
                "customer_id": customer["customer_id"],
                "invoice_date": invoice_date,
                "due_date": due_date,
                "invoice_amount": amount,
                "invoice_status": "Open",
                "payment_terms_days": customer["payment_terms_days"],
                "product_category": product_category,
                "is_recurring": is_recurring,
                "po_number": po_number,
                "line_item_count": int(np.random.choice([1, 1, 2, 2, 3, 3, 4, 5, 6, 8])),
                "sales_rep_id": f"SR_{np.random.randint(1, 31):03d}",
            })

        current_month += pd.DateOffset(months=1)

invoices_df = pd.DataFrame(invoices)
invoices_df["invoice_date"] = pd.to_datetime(invoices_df["invoice_date"])
invoices_df["due_date"] = pd.to_datetime(invoices_df["due_date"])

print(f"[{time.time() - t0:.1f}s] Invoices: {len(invoices_df)}")
print(f"  Amount: mean={invoices_df['invoice_amount'].mean():,.0f}, median={invoices_df['invoice_amount'].median():,.0f}")
print(f"  Date range: {invoices_df['invoice_date'].min().date()} to {invoices_df['invoice_date'].max().date()}")

# ============================================================
# TABLE 3: PAYMENTS
# ============================================================

print("\nSimulating payments...")

inv_sim = invoices_df.merge(
    customers_df[["customer_id", "archetype", "customer_segment"]],
    on="customer_id",
    how="left"
).sort_values("due_date").reset_index(drop=True)

N = len(inv_sim)

arch_delay_mean = inv_sim["archetype"].map(lambda x: ARCHETYPES[x]["delay_mean"]).values
arch_delay_std = inv_sim["archetype"].map(lambda x: ARCHETYPES[x]["delay_std"]).values
arch_partial = inv_sim["archetype"].map(lambda x: ARCHETYPES[x]["partial_rate"]).values
arch_writeoff = inv_sim["archetype"].map(lambda x: ARCHETYPES[x]["writeoff_rate"]).values
arch_qe = inv_sim["archetype"].map(lambda x: ARCHETYPES[x]["qe_bunch"]).values
arch_q1add = inv_sim["archetype"].map(lambda x: ARCHETYPES[x]["q1_add"]).values
arch_q1std = inv_sim["archetype"].map(lambda x: ARCHETYPES[x]["q1_std"]).values
arch_autocorr = inv_sim["archetype"].map(lambda x: ARCHETYPES[x]["autocorr"]).values

base_delays = np.random.normal(arch_delay_mean, arch_delay_std)

# Q1 seasonal effect
due_months = inv_sim["due_date"].dt.month.values
is_q1 = np.isin(due_months, [1, 2, 3])
q1_bump = np.maximum(0, np.random.normal(arch_q1add, arch_q1std))
base_delays = np.where(is_q1, base_delays + q1_bump, base_delays)

# Quarter-end bunching
qe_roll = np.random.random(N)
qe_mask = qe_roll < arch_qe

for i in np.where(qe_mask)[0]:
    qe_date = get_qe(inv_sim.loc[i, "due_date"])
    days_to_qe = (qe_date - inv_sim.loc[i, "due_date"]).days
    if days_to_qe > 0:
        base_delays[i] = days_to_qe + np.random.randint(-3, 3)

# Customer-level autocorrelation + deterioration
cust_last_delay = {}
cust_inv_count = {}
cust_deteriorating = {}

for _, c in customers_df.iterrows():
    cust_inv_count[c["customer_id"]] = 0
    cust_deteriorating[c["customer_id"]] = (
        np.random.random() < ARCHETYPES[c["archetype"]]["deter_prob"]
    )

for i in range(N):
    cid = inv_sim.loc[i, "customer_id"]
    cust_inv_count[cid] += 1

    if cid in cust_last_delay:
        alpha = arch_autocorr[i]
        base_delays[i] = alpha * cust_last_delay[cid] + (1 - alpha) * base_delays[i]

    # stabilized deterioration factor
    if cust_deteriorating.get(cid, False):
        deterioration_factor = min(2.5, 1 + cust_inv_count[cid] * 0.01)
        base_delays[i] *= deterioration_factor

    cust_last_delay[cid] = base_delays[i]

# hard safety cleanup
base_delays = np.nan_to_num(base_delays, nan=0.0, posinf=365.0, neginf=-5.0)
base_delays = np.clip(base_delays, -5, 365)

eff_dpd = np.maximum(0, base_delays)
recovery_probs = np.vectorize(recovery_prob)(eff_dpd)

# Archetype adjustments
for i in range(N):
    archetype_name = inv_sim.loc[i, "archetype"]
    cid = inv_sim.loc[i, "customer_id"]

    if archetype_name == "reliable_enterprise":
        recovery_probs[i] = min(0.99, recovery_probs[i] + 0.10)
    elif archetype_name == "chronic_late_payer":
        recovery_probs[i] = max(0.10, recovery_probs[i] - 0.10)
    elif archetype_name == "high_risk_deteriorating" and cust_deteriorating.get(cid, False):
        recovery_probs[i] = max(0.05, recovery_probs[i] - 0.15)

will_pay = np.random.random(N) < recovery_probs
will_writeoff = (np.random.random(N) < arch_writeoff) & (eff_dpd > 90)
is_partial = np.random.random(N) < arch_partial

due_dates = inv_sim["due_date"].values
inv_dates = inv_sim["invoice_date"].values
invoice_amounts = inv_sim["invoice_amount"].values

print(f"[{time.time() - t0:.1f}s] Building payment records...")

payments = []
statuses = np.full(N, "Open", dtype=object)
payment_counter = 0

for i in range(N):
    if will_writeoff[i]:
        statuses[i] = "Written Off"
        continue

    if not will_pay[i]:
        if eff_dpd[i] > 150:
            statuses[i] = "Written Off"
        continue

    delay_days = int(round(base_delays[i]))
    delay_days = max(-5, min(delay_days, 365))

    try:
        payment_date = pd.Timestamp(due_dates[i]) + timedelta(days=delay_days)
    except Exception:
        payment_date = pd.Timestamp(due_dates[i]) + timedelta(days=30)

    payment_date = snap_payday(payment_date)

    invoice_date = pd.Timestamp(inv_dates[i])
    if payment_date < invoice_date:
        payment_date = invoice_date + timedelta(days=max(3, abs(int(np.random.normal(7, 3)))))
        payment_date = snap_bday(payment_date)

    if payment_date > OBSERVATION_END:
        continue

    amount = invoice_amounts[i]
    dpd = (payment_date - pd.Timestamp(due_dates[i])).days

    if is_partial[i]:
        partial_pct = np.random.uniform(0.30, 0.85)
        paid_amount = round(amount * partial_pct, 2)
        statuses[i] = "Partially Paid"

        payment_counter += 1
        payments.append({
            "payment_id": f"PAY_{payment_counter:06d}",
            "invoice_id": inv_sim.loc[i, "invoice_id"],
            "customer_id": inv_sim.loc[i, "customer_id"],
            "payment_date": payment_date,
            "payment_amount": paid_amount,
            "payment_method": np.random.choice(PAY_METHODS, p=PAY_METHOD_W),
            "days_past_due": dpd,
            "is_partial": True,
        })

        # possible second payment
        if np.random.random() < 0.60:
            remaining_amt = round(amount - paid_amount, 2)
            payment_date_2 = payment_date + timedelta(days=np.random.randint(10, 45))
            payment_date_2 = snap_payday(payment_date_2)

            if payment_date_2 <= OBSERVATION_END:
                payment_counter += 1
                payments.append({
                    "payment_id": f"PAY_{payment_counter:06d}",
                    "invoice_id": inv_sim.loc[i, "invoice_id"],
                    "customer_id": inv_sim.loc[i, "customer_id"],
                    "payment_date": payment_date_2,
                    "payment_amount": remaining_amt,
                    "payment_method": np.random.choice(PAY_METHODS, p=PAY_METHOD_W),
                    "days_past_due": (payment_date_2 - pd.Timestamp(due_dates[i])).days,
                    "is_partial": False,
                })
                statuses[i] = "Paid"
    else:
        statuses[i] = "Paid"
        payment_counter += 1
        payments.append({
            "payment_id": f"PAY_{payment_counter:06d}",
            "invoice_id": inv_sim.loc[i, "invoice_id"],
            "customer_id": inv_sim.loc[i, "customer_id"],
            "payment_date": payment_date,
            "payment_amount": amount,
            "payment_method": np.random.choice(PAY_METHODS, p=PAY_METHOD_W),
            "days_past_due": dpd,
            "is_partial": False,
        })

invoices_df["invoice_status"] = statuses
payments_df = pd.DataFrame(payments)

if len(payments_df) > 0:
    payments_df["payment_date"] = pd.to_datetime(payments_df["payment_date"])

print(f"[{time.time() - t0:.1f}s] Payments: {len(payments_df)}")
print(f"  Status distribution: {dict(invoices_df['invoice_status'].value_counts())}")
if len(payments_df) > 0:
    print(f"  Avg DPD: {payments_df['days_past_due'].mean():.1f}")
    print(f"  Partial rate: {payments_df['is_partial'].mean():.2%}")
else:
    print("  Avg DPD: No payments generated")
    print("  Partial rate: No payments generated")

# ============================================================
# TABLE 4: DISPUTES
# ============================================================

print("\nGenerating disputes...")

inv_arch = invoices_df.merge(customers_df[["customer_id", "archetype"]], on="customer_id", how="left")
dispute_rates = inv_arch["archetype"].map(lambda x: ARCHETYPES[x]["dispute_rate"]).values

amt_p90 = invoices_df["invoice_amount"].quantile(0.90)
large_adj = np.where(invoices_df["invoice_amount"] > amt_p90, 1.3, 1.0)
product_adj = np.where(
    invoices_df["product_category"].isin(["Raw Materials", "IT Hardware", "Equipment Lease"]),
    1.2,
    1.0
)

final_dispute_probs = dispute_rates * large_adj * product_adj
will_dispute = np.random.random(len(invoices_df)) < final_dispute_probs
disputed_idx = np.where(will_dispute)[0]

disputes = []
dispute_counter = 0
disputed_invoice_ids = set()

for idx in disputed_idx:
    row = invoices_df.iloc[idx]
    dispute_counter += 1
    disputed_invoice_ids.add(row["invoice_id"])

    dispute_date = row["invoice_date"] + timedelta(days=np.random.randint(5, 26))
    dispute_date = snap_bday(dispute_date)

    dispute_amount = round(row["invoice_amount"] * np.random.uniform(0.15, 1.0), 2)
    dispute_reason = np.random.choice(DISPUTE_REASONS, p=DISPUTE_REASON_W)

    roll = np.random.random()
    if roll < 0.65:
        resolution_days = np.random.randint(7, 31)
    elif roll < 0.85:
        resolution_days = np.random.randint(31, 61)
    elif roll < 0.95:
        resolution_days = np.random.randint(61, 91)
    else:
        resolution_days = None

    if resolution_days is not None:
        resolution_date = snap_bday(dispute_date + timedelta(days=resolution_days))

        if resolution_date > OBSERVATION_END:
            resolution_date = None
            dispute_status = "Open"
            resolution_outcome = None
            resolution_amount = None
        else:
            dispute_status = "Escalated" if resolution_days > 45 and np.random.random() < 0.30 else "Resolved"
            outcome_roll = np.random.random()

            if outcome_roll < 0.30:
                resolution_outcome = "Credited"
                resolution_amount = dispute_amount
            elif outcome_roll < 0.55:
                resolution_outcome = "Adjusted"
                resolution_amount = round(dispute_amount * np.random.uniform(0.3, 0.7), 2)
            elif outcome_roll < 0.80:
                resolution_outcome = "Upheld"
                resolution_amount = 0.0
            else:
                resolution_outcome = "Withdrawn"
                resolution_amount = 0.0
    else:
        resolution_date = None
        dispute_status = "Open"
        resolution_outcome = None
        resolution_amount = None

    disputes.append({
        "dispute_id": f"DISP_{dispute_counter:05d}",
        "invoice_id": row["invoice_id"],
        "customer_id": row["customer_id"],
        "dispute_date": dispute_date,
        "dispute_reason": dispute_reason,
        "dispute_amount": dispute_amount,
        "dispute_status": dispute_status,
        "resolution_date": resolution_date,
        "resolution_outcome": resolution_outcome,
        "resolution_amount": resolution_amount,
    })

disputes_df = pd.DataFrame(disputes)

if len(disputes_df) > 0:
    disputes_df["dispute_date"] = pd.to_datetime(disputes_df["dispute_date"])
    disputes_df["resolution_date"] = pd.to_datetime(disputes_df["resolution_date"])

    open_dispute_invoices = set(
        disputes_df.loc[disputes_df["dispute_status"].isin(["Open", "Escalated"]), "invoice_id"]
    )

    invoices_df.loc[
        (invoices_df["invoice_id"].isin(open_dispute_invoices)) &
        (invoices_df["invoice_status"] == "Open"),
        "invoice_status"
    ] = "Disputed"

print(f"[{time.time() - t0:.1f}s] Disputes: {len(disputes_df)}")
print(f"  Dispute rate: {len(disputes_df) / len(invoices_df):.2%}")
if len(disputes_df) > 0:
    print(f"  Status: {dict(disputes_df['dispute_status'].value_counts())}")
else:
    print("  Status: No disputes generated")

# ============================================================
# TABLE 5: DUNNING CONTACTS
# ============================================================

print("\nGenerating dunning contacts...")

if len(payments_df) > 0:
    first_pay = payments_df.groupby("invoice_id")["payment_date"].min().reset_index()
    first_pay.columns = ["invoice_id", "first_pay_date"]
else:
    first_pay = pd.DataFrame(columns=["invoice_id", "first_pay_date"])

inv_dun = invoices_df.merge(first_pay, on="invoice_id", how="left")
inv_dun = inv_dun.merge(customers_df[["customer_id", "archetype"]], on="customer_id", how="left")

inv_dun["is_overdue"] = (
    inv_dun["first_pay_date"].isna() |
    (inv_dun["first_pay_date"] > inv_dun["due_date"] + timedelta(days=3))
)

overdue_df = inv_dun[inv_dun["is_overdue"]].reset_index(drop=True)
print(f"  Overdue invoices to generate contacts for: {len(overdue_df)}")

CONTACT_SEQ = [
    ("Auto-Reminder", 1),
    ("Email", 7),
    ("Phone", 15),
    ("Phone", 25),
    ("Email", 35),
    ("Escalation", 45),
    ("Escalation", 60),
    ("Letter", 75),
]

notes_ptp = [
    "Spoke with AP manager, committed to payment",
    "Customer confirmed wire transfer scheduled",
    "Payment approved, pending finance sign-off",
    "Will pay after resolving internal approval",
    "Customer requested 1 week extension",
]

notes_dispute = [
    "Customer claims pricing discrepancy",
    "Delivery issues reported",
    "Customer says invoice not received",
    "Duplicate billing claimed",
]

contacts = []
contact_counter = 0

for _, row in overdue_df.iterrows():
    arch = ARCHETYPES[row["archetype"]]
    due_date = row["due_date"]
    first_payment_date = row["first_pay_date"]

    if pd.notna(first_payment_date):
        days_outstanding = (first_payment_date - due_date).days
    elif row["invoice_status"] in ["Open", "Disputed", "Partially Paid"]:
        days_outstanding = (OBSERVATION_END - due_date).days
    else:
        days_outstanding = min(180, (OBSERVATION_END - due_date).days)

    if days_outstanding <= 0:
        continue

    responsiveness = arch["responsiveness"]

    for seq_idx, (contact_type, day_offset) in enumerate(CONTACT_SEQ):
        if day_offset > days_outstanding:
            break

        contact_date = due_date + timedelta(days=day_offset + np.random.randint(-2, 3))
        contact_date = snap_bday(contact_date)

        if contact_date < due_date:
            contact_date = snap_bday(due_date + timedelta(days=1))
        if pd.notna(first_payment_date) and contact_date >= first_payment_date:
            break
        if contact_date > OBSERVATION_END:
            break

        if seq_idx > 0 and np.random.random() < 0.15:
            continue

        if contact_type == "Auto-Reminder":
            outcome = "Reached"
        else:
            r = np.random.random()
            if r < responsiveness * 0.6:
                if np.random.random() < 0.35:
                    outcome = "Promise to Pay"
                elif row["invoice_id"] in disputed_invoice_ids and np.random.random() < 0.30:
                    outcome = "Disputed"
                else:
                    outcome = "Reached"
            elif r < responsiveness:
                outcome = "Left Message"
            else:
                outcome = np.random.choice(["No Answer", "No Answer", "Refused"])

        promised_pay_date = None
        promised_pay_amount = None
        notes = None

        if outcome == "Promise to Pay":
            promised_pay_date = snap_bday(contact_date + timedelta(days=np.random.randint(3, 21)))
            promised_pay_amount = row["invoice_amount"]
            if np.random.random() < 0.20:
                promised_pay_amount = round(promised_pay_amount * np.random.uniform(0.5, 0.9), 2)
            notes = np.random.choice(notes_ptp)

        elif outcome == "Disputed" and np.random.random() < 0.50:
            notes = np.random.choice(notes_dispute)

        collector = np.random.choice(COLLECTORS[:5]) if contact_type == "Escalation" else np.random.choice(COLLECTORS)

        contact_counter += 1
        contacts.append({
            "contact_id": f"CONT_{contact_counter:06d}",
            "invoice_id": row["invoice_id"],
            "customer_id": row["customer_id"],
            "collector_id": collector,
            "contact_date": contact_date,
            "contact_type": contact_type,
            "contact_sequence": seq_idx + 1,
            "contact_outcome": outcome,
            "promised_pay_date": promised_pay_date,
            "promised_pay_amount": promised_pay_amount,
            "notes": notes,
        })

dunning_df = pd.DataFrame(contacts)

if len(dunning_df) > 0:
    dunning_df["contact_date"] = pd.to_datetime(dunning_df["contact_date"])
    dunning_df["promised_pay_date"] = pd.to_datetime(dunning_df["promised_pay_date"])

print(f"[{time.time() - t0:.1f}s] Dunning contacts: {len(dunning_df)}")
if len(dunning_df) > 0:
    print(f"  Contact types: {dict(dunning_df['contact_type'].value_counts())}")
    print(f"  Outcomes: {dict(dunning_df['contact_outcome'].value_counts())}")
else:
    print("  Contact types: No contacts generated")
    print("  Outcomes: No contacts generated")

# ============================================================
# DATA QUALITY ISSUES
# ============================================================

print("\nAdding data quality issues...")

po_idx = invoices_df[invoices_df["po_number"].notna()].index
n_po_remove = int(0.02 * len(po_idx))
if n_po_remove > 0:
    remove_idx = np.random.choice(po_idx, n_po_remove, replace=False)
    invoices_df.loc[remove_idx, "po_number"] = None

n_rounding = int(0.01 * len(payments_df))
if n_rounding > 0:
    rounding_idx = np.random.choice(payments_df.index, n_rounding, replace=False)
    payments_df.loc[rounding_idx, "payment_amount"] += np.random.uniform(-5, 5, n_rounding).round(2)

print(f"  Removed {n_po_remove} PO numbers, added {n_rounding} rounding discrepancies")

# ============================================================
# FINAL VALIDATION
# ============================================================

print("\n" + "=" * 60)
print("VALIDATION REPORT")
print("=" * 60)

print("\n1. REFERENTIAL INTEGRITY")
print(f"  Invoice→Customer FK valid: {set(invoices_df['customer_id']).issubset(set(customers_df['customer_id']))}")
print(f"  Payment→Invoice FK valid: {set(payments_df['invoice_id']).issubset(set(invoices_df['invoice_id']))}")
print(f"  Dispute→Invoice FK valid: {set(disputes_df['invoice_id']).issubset(set(invoices_df['invoice_id']))}")
print(f"  Dunning→Invoice FK valid: {set(dunning_df['invoice_id']).issubset(set(invoices_df['invoice_id']))}")

print("\n2. DUPLICATE PKs")
for name, df, col in [
    ("customers", customers_df, "customer_id"),
    ("invoices", invoices_df, "invoice_id"),
    ("payments", payments_df, "payment_id"),
    ("disputes", disputes_df, "dispute_id"),
    ("dunning", dunning_df, "contact_id"),
]:
    print(f"  {name}: {df[col].duplicated().sum()} duplicates")

if len(payments_df) > 0:
    pay_inv = payments_df.merge(invoices_df[["invoice_id", "invoice_date"]], on="invoice_id", how="left")
    bad_payments = (pay_inv["payment_date"] < pay_inv["invoice_date"]).sum()
else:
    bad_payments = 0

print(f"\n3. TEMPORAL: Payments before invoice date: {bad_payments}")

total_invoiced = invoices_df["invoice_amount"].sum()
total_collected = payments_df["payment_amount"].sum() if len(payments_df) > 0 else 0.0
recovery_rate = total_collected / total_invoiced if total_invoiced > 0 else 0.0

print(f"\n4. BUSINESS METRICS")
print(f"  Total invoiced: INR {total_invoiced:,.0f}")
print(f"  Total collected: INR {total_collected:,.0f}")
print(f"  Recovery rate: {recovery_rate:.1%}")
print(f"  Dispute rate: {len(disputes_df) / len(invoices_df):.2%}")

if len(payments_df) > 0:
    print(f"  Avg payment delay: {payments_df['days_past_due'].mean():.1f} DPD")
    print(f"  Median payment delay: {payments_df['days_past_due'].median():.0f} DPD")
else:
    print("  Avg payment delay: N/A")
    print("  Median payment delay: N/A")

print(f"\n5. INVOICE STATUS")
status_counts = invoices_df["invoice_status"].value_counts()
status_pcts = (status_counts / len(invoices_df) * 100).round(1)
for status in status_counts.index:
    print(f"  {status}: {status_counts[status]:,} ({status_pcts[status]}%)")

print(f"\n6. PAYMENT DELAY BY ARCHETYPE")
if len(payments_df) > 0:
    pay_arch = payments_df.merge(customers_df[["customer_id", "archetype"]], on="customer_id", how="left")
    delay_by_arch = pay_arch.groupby("archetype")["days_past_due"].agg(["mean", "median", "std"]).round(1)
    delay_by_arch.columns = ["Mean", "Median", "Std"]
    print(delay_by_arch.sort_values("Mean").to_string())
else:
    print("No payment records available")

print(f"\n7. AGING BUCKETS (open invoices)")
open_invoices = invoices_df[invoices_df["invoice_status"].isin(["Open", "Partially Paid", "Disputed"])].copy()
open_invoices["dpd"] = (OBSERVATION_END - open_invoices["due_date"]).dt.days
bins = [-999, 0, 30, 60, 90, 120, 999]
labels = ["Current", "1-30", "31-60", "61-90", "91-120", "120+"]
open_invoices["bucket"] = pd.cut(open_invoices["dpd"], bins=bins, labels=labels)
print(open_invoices["bucket"].value_counts().sort_index())

print(f"\n8. TABLE SIZES")
tables = {
    "customers": customers_df,
    "invoices": invoices_df,
    "payments": payments_df,
    "disputes": disputes_df,
    "dunning_contacts": dunning_df,
}
for name, df in tables.items():
    print(f"  {name}: {len(df):>9,} rows")

# ============================================================
# SAVE
# ============================================================

print(f"\n{'=' * 60}")
print("SAVING FILES")
print(f"{'=' * 60}")

for name, df in tables.items():
    csv_path = f"{OUTPUT_DIR}{name}.csv"
    df.to_csv(csv_path, index=False)

    try:
        parquet_path = f"{OUTPUT_DIR}{name}.parquet"
        df.to_parquet(parquet_path, index=False)
    except ImportError:
        pass

    size_mb = os.path.getsize(csv_path) / (1024 * 1024)
    print(f"  {name}: csv={size_mb:.1f}MB, {len(df):,} rows")

print(f"\n[{time.time() - t0:.1f}s] COMPLETE!")
print(f"""
Next steps:
  1. Open these files in 02_eda.ipynb
  2. Load with: df = pd.read_parquet('{OUTPUT_DIR}<table>.parquet')
  3. Or CSV:    df = pd.read_csv('{OUTPUT_DIR}<table>.csv')
""")