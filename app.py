
"""
AR Collections Intelligence Engine — Streamlit Dashboard
==========================================================
Run with: streamlit run app.py
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import json
import os

# ════════════════════════════════════════════════════════════════
# PAGE CONFIG
# ════════════════════════════════════════════════════════════════

st.set_page_config(
    page_title="AR Collections Intelligence Engine",
    page_icon="💰",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ── Custom CSS ──
st.markdown("""
<style>
    .main-header {
        font-size: 2.2rem;
        font-weight: 700;
        color: #E2E8F0 !important;
        margin-bottom: 0.5rem;
    }
    .sub-header {
        font-size: 1.0rem;
        color: #94A3B8 !important;
        margin-bottom: 2rem;
    }
    /* Force bright text on metric cards for dark theme */
    [data-testid="stMetricValue"] > div {
        color: #FFFFFF !important;
        font-weight: 700 !important;
    }
    [data-testid="stMetricLabel"] > div > div > p {
        color: #93C5FD !important;
        font-weight: 600 !important;
    }
    [data-testid="metric-container"] {
        background-color: #1E293B !important;
        border: 1px solid #3B82F6 !important;
        border-radius: 10px !important;
        padding: 1rem !important;
    }
</style>
""", unsafe_allow_html=True)

# ════════════════════════════════════════════════════════════════
# DATA LOADING
# ════════════════════════════════════════════════════════════════

DATA_DIR = 'C:/Users/Manu/Desktop/Project 2 AR Collections Intelligence Engine/'
RAW_DIR = f'{DATA_DIR}Raw Data/'
PROC_DIR = f'{DATA_DIR}processed/'

@st.cache_data
def load_data():
    """Load all data files. Cached so it only runs once."""
    data = {}
    data['customers'] = pd.read_csv(f'{RAW_DIR}customers.csv', parse_dates=['customer_since'])
    data['invoices'] = pd.read_csv(f'{RAW_DIR}invoices.csv', parse_dates=['invoice_date', 'due_date'])
    data['payments'] = pd.read_csv(f'{RAW_DIR}payments.csv', parse_dates=['payment_date'])
    data['disputes'] = pd.read_csv(f'{RAW_DIR}disputes.csv', parse_dates=['dispute_date', 'resolution_date'])
    data['dunning'] = pd.read_csv(f'{RAW_DIR}dunning_contacts.csv', parse_dates=['contact_date'])

    # Processed files
    if os.path.exists(f'{PROC_DIR}priority_scores.csv'):
        data['priority'] = pd.read_csv(f'{PROC_DIR}priority_scores.csv')
    if os.path.exists(f'{PROC_DIR}strategy_segments.csv'):
        data['segments'] = pd.read_csv(f'{PROC_DIR}strategy_segments.csv')
    if os.path.exists(f'{PROC_DIR}simulation_results.csv'):
        data['simulation'] = pd.read_csv(f'{PROC_DIR}simulation_results.csv')
    if os.path.exists(f'{PROC_DIR}test_predictions.csv'):
        data['predictions'] = pd.read_csv(f'{PROC_DIR}test_predictions.csv',
                                           parse_dates=['observation_date'])
    if os.path.exists(f'{PROC_DIR}survival_predictions.csv'):
        data['survival'] = pd.read_csv(f'{PROC_DIR}survival_predictions.csv',
                                        parse_dates=['due_date'])

    # Model metadata
    meta_paths = [
        f'{DATA_DIR}models/model_metadata.json',
        f'{DATA_DIR}Raw Datamodels/model_metadata.json',
    ]
    for mp in meta_paths:
        if os.path.exists(mp):
            with open(mp, 'r') as f:
                data['model_meta'] = json.load(f)
            break

    return data

data = load_data()
customers = data['customers']
invoices = data['invoices']
payments = data['payments']
disputes = data['disputes']
dunning = data['dunning']

OBSERVATION_END = pd.Timestamp('2023-12-31')

# ════════════════════════════════════════════════════════════════
# SIDEBAR NAVIGATION
# ════════════════════════════════════════════════════════════════

st.sidebar.markdown("## 💰 AR Collections Engine")
st.sidebar.markdown("---")

page = st.sidebar.radio(
    "Navigate to:",
    [
        "📊 Portfolio Overview",
        "📋 Collector Work Queue",
        "🔍 Customer Deep Dive",
        "🎯 Strategy Segments",
        "📈 Business Impact",
        "🧠 Model Explainability",
    ]
)

# Sidebar filters
st.sidebar.markdown("---")
st.sidebar.markdown("### Filters")
selected_segments = st.sidebar.multiselect(
    "Customer Segment",
    options=['Enterprise', 'Mid-Market', 'SMB'],
    default=['Enterprise', 'Mid-Market', 'SMB']
)
selected_risk = st.sidebar.multiselect(
    "Risk Tier",
    options=['Low', 'Medium', 'High'],
    default=['Low', 'Medium', 'High']
)

# Apply filters
inv_filtered = invoices.merge(
    customers[['customer_id', 'customer_segment', 'customer_risk_tier']],
    on='customer_id', how='left'
)
inv_filtered = inv_filtered[
    (inv_filtered['customer_segment'].isin(selected_segments)) &
    (inv_filtered['customer_risk_tier'].isin(selected_risk))
]


# ════════════════════════════════════════════════════════════════
# PAGE 1: PORTFOLIO OVERVIEW
# ════════════════════════════════════════════════════════════════

if page == "📊 Portfolio Overview":
    st.markdown('<p class="main-header">Portfolio Overview</p>', unsafe_allow_html=True)
    st.markdown('<p class="sub-header">High-level view of Accounts Receivable health</p>', unsafe_allow_html=True)

    # ── Key Metrics Row ──
    total_ar = inv_filtered['invoice_amount'].sum()
    total_paid = payments[payments['invoice_id'].isin(inv_filtered['invoice_id'])]['payment_amount'].sum()
    recovery_rate = total_paid / total_ar if total_ar > 0 else 0
    total_customers = inv_filtered['customer_id'].nunique()
    total_invoices = len(inv_filtered)
    avg_dpd = payments.merge(invoices[['invoice_id', 'due_date']], on='invoice_id')
    avg_dpd = (avg_dpd['payment_date'] - avg_dpd['due_date']).dt.days.mean()

    col1, col2, col3, col4, col5 = st.columns(5)
    col1.metric("Total AR", f"₹{total_ar/1e7:.0f} Cr")
    col2.metric("Recovery Rate", f"{recovery_rate:.1%}")
    col3.metric("Customers", f"{total_customers:,}")
    col4.metric("Invoices", f"{total_invoices:,}")
    col5.metric("Avg Payment Delay", f"{avg_dpd:.0f} days")

    st.markdown("---")

    # ── Row 2: Charts ──
    col1, col2 = st.columns(2)

    with col1:
        # Invoice Status Distribution
        status_counts = inv_filtered['invoice_status'].value_counts()
        fig = px.pie(
            values=status_counts.values,
            names=status_counts.index,
            title="Invoice Status Distribution",
            color=status_counts.index,
            color_discrete_map={
                'Paid': '#059669', 'Open': '#2563EB',
                'Partially Paid': '#F59E0B', 'Written Off': '#DC2626',
                'Disputed': '#7C3AED'
            }
        )
        fig.update_layout(height=400)
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        # AR by Segment
        seg_ar = inv_filtered.groupby('customer_segment')['invoice_amount'].sum().reset_index()
        fig = px.bar(
            seg_ar, x='customer_segment', y='invoice_amount',
            title="AR Balance by Segment",
            color='customer_segment',
            color_discrete_map={'Enterprise': '#2563EB', 'Mid-Market': '#0891B2', 'SMB': '#F59E0B'},
            labels={'invoice_amount': 'Amount (₹)', 'customer_segment': 'Segment'}
        )
        fig.update_layout(height=400, showlegend=False)
        fig.update_layout(yaxis_tickformat=',.0f')
        st.plotly_chart(fig, use_container_width=True)

    # ── Row 3: Aging Analysis ──
    st.subheader("Aging Bucket Distribution")

    open_inv = inv_filtered[inv_filtered['invoice_status'].isin(['Open', 'Partially Paid', 'Disputed'])].copy()
    open_inv['dpd'] = (OBSERVATION_END - open_inv['due_date']).dt.days
    bins = [-999, 0, 30, 60, 90, 120, 999]
    labels = ['Current', '1-30 DPD', '31-60 DPD', '61-90 DPD', '91-120 DPD', '120+ DPD']
    open_inv['aging_bucket'] = pd.cut(open_inv['dpd'], bins=bins, labels=labels)

    col1, col2 = st.columns(2)

    with col1:
        aging_amount = open_inv.groupby('aging_bucket', observed=False)['invoice_amount'].sum().reset_index()
        fig = px.bar(
            aging_amount, x='aging_bucket', y='invoice_amount',
            title="Open AR by Aging Bucket (Amount)",
            color='aging_bucket',
            color_discrete_sequence=['#059669', '#0891B2', '#2563EB', '#F59E0B', '#DC2626', '#7C3AED'],
            labels={'invoice_amount': 'Amount (₹)', 'aging_bucket': 'Aging Bucket'}
        )
        fig.update_layout(height=400, showlegend=False)
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        aging_count = open_inv.groupby('aging_bucket', observed=False)['invoice_id'].count().reset_index()
        aging_count.columns = ['aging_bucket', 'count']
        fig = px.bar(
            aging_count, x='aging_bucket', y='count',
            title="Open AR by Aging Bucket (Count)",
            color='aging_bucket',
            color_discrete_sequence=['#059669', '#0891B2', '#2563EB', '#F59E0B', '#DC2626', '#7C3AED'],
            labels={'count': 'Invoice Count', 'aging_bucket': 'Aging Bucket'}
        )
        fig.update_layout(height=400, showlegend=False)
        st.plotly_chart(fig, use_container_width=True)

    # ── Top Overdue Accounts ──
    st.subheader("Top 10 Overdue Accounts")
    if len(open_inv) > 0:
        top_overdue = open_inv.groupby('customer_id').agg(
            total_overdue=('invoice_amount', 'sum'),
            invoice_count=('invoice_id', 'count'),
            max_dpd=('dpd', 'max'),
        ).nlargest(10, 'total_overdue').reset_index()
        top_overdue = top_overdue.merge(
            customers[['customer_id', 'customer_name', 'customer_segment']],
            on='customer_id', how='left'
        )
        top_overdue['total_overdue'] = top_overdue['total_overdue'].apply(lambda x: f"₹{x:,.0f}")
        st.dataframe(
            top_overdue[['customer_name', 'customer_segment', 'total_overdue', 'invoice_count', 'max_dpd']],
            use_container_width=True, hide_index=True
        )


# ════════════════════════════════════════════════════════════════
# PAGE 2: COLLECTOR WORK QUEUE
# ════════════════════════════════════════════════════════════════

elif page == "📋 Collector Work Queue":
    st.markdown('<p class="main-header">Collector Work Queue</p>', unsafe_allow_html=True)
    st.markdown('<p class="sub-header">AI-prioritized daily worklist for collection agents</p>', unsafe_allow_html=True)

    if 'priority' in data:
        priority = data['priority'].copy()

        # Filters
        col1, col2, col3 = st.columns(3)
        with col1:
            max_invoices = st.slider("Worklist Size", 10, 100, 50, step=10)
        with col2:
            min_amount = st.number_input("Min Invoice Amount (₹)", value=0, step=10000)
        with col3:
            status_filter = st.multiselect(
                "Invoice Status",
                options=priority['invoice_status'].unique().tolist(),
                default=[s for s in ['Open', 'Partially Paid', 'Disputed']
                         if s in priority['invoice_status'].values]
            )

        # Filter
        worklist = priority[
            (priority['invoice_status'].isin(status_filter)) &
            (priority['invoice_amount'] >= min_amount)
        ].nlargest(max_invoices, 'priority_score').copy()

        # Add recommended action
        def recommend_action(row):
            if row.get('has_dispute', 0) == 1:
                return '🔴 Resolve Dispute'
            elif row.get('total_contacts', 0) == 0:
                return '🟢 First Contact'
            elif row.get('total_contacts', 0) <= 2:
                return '🟡 Phone Follow-up'
            elif row.get('customer_segment', '') == 'Enterprise':
                return '🔵 Escalation'
            else:
                return '🟠 Final Notice'

        worklist['recommended_action'] = worklist.apply(recommend_action, axis=1)

        # Summary metrics
        col1, col2, col3, col4 = st.columns(4)
        total_expected = (worklist['p_recovery_if_contacted'] * worklist['remaining_balance']).sum()
        col1.metric("Invoices in Queue", f"{len(worklist):,}")
        col2.metric("Total Amount", f"₹{worklist['remaining_balance'].sum():,.0f}")
        col3.metric("Expected Recovery", f"₹{total_expected:,.0f}")
        col4.metric("Avg Priority Score", f"{worklist['priority_score'].mean():,.0f}")

        st.markdown("---")

        # Display worklist
        display_cols = ['invoice_id', 'customer_id', 'invoice_amount', 'days_past_due',
                       'priority_score', 'recommended_action']
        available_cols = [c for c in display_cols if c in worklist.columns]

        # Format for display
        display_df = worklist[available_cols].copy()
        if 'invoice_amount' in display_df.columns:
            display_df['invoice_amount'] = display_df['invoice_amount'].apply(lambda x: f"₹{x:,.0f}")
        if 'priority_score' in display_df.columns:
            display_df['priority_score'] = display_df['priority_score'].apply(lambda x: f"{x:,.0f}")
        if 'days_past_due' in display_df.columns:
            display_df['days_past_due'] = display_df['days_past_due'].apply(lambda x: f"{x:.0f}")

        st.dataframe(display_df, use_container_width=True, hide_index=True, height=500)

        # Priority score distribution
        st.subheader("Priority Score Distribution")
        fig = px.histogram(
            worklist, x='priority_score', nbins=30,
            title="Priority Score Distribution (Current Worklist)",
            color_discrete_sequence=['#2563EB']
        )
        fig.update_layout(height=300)
        st.plotly_chart(fig, use_container_width=True)

    else:
        st.warning("Priority scores not found. Run Notebook 06 first.")


# ════════════════════════════════════════════════════════════════
# PAGE 3: CUSTOMER DEEP DIVE
# ════════════════════════════════════════════════════════════════

elif page == "🔍 Customer Deep Dive":
    st.markdown('<p class="main-header">Customer Deep Dive</p>', unsafe_allow_html=True)
    st.markdown('<p class="sub-header">Detailed view of individual customer payment behavior</p>', unsafe_allow_html=True)

    # Customer selector
    cust_list = customers['customer_id'].tolist()
    selected_cust = st.selectbox("Select Customer", cust_list)

    if selected_cust:
        cust_info = customers[customers['customer_id'] == selected_cust].iloc[0]
        cust_invoices = invoices[invoices['customer_id'] == selected_cust].copy()
        cust_payments = payments[payments['customer_id'] == selected_cust].copy()
        cust_disputes = disputes[disputes['customer_id'] == selected_cust].copy()
        cust_dunning = dunning[dunning['customer_id'] == selected_cust].copy()

        # ── Customer Info Card ──
        col1, col2, col3, col4 = st.columns(4)
        col1.metric("Segment", cust_info['customer_segment'])
        col2.metric("Risk Tier", cust_info['customer_risk_tier'])
        col3.metric("Credit Limit", f"₹{cust_info['credit_limit']:,.0f}")
        col4.metric("Terms", f"{cust_info['payment_terms_days']} days")

        col1, col2, col3, col4 = st.columns(4)
        col1.metric("Total Invoices", f"{len(cust_invoices):,}")
        col2.metric("Total Amount", f"₹{cust_invoices['invoice_amount'].sum():,.0f}")
        col3.metric("Disputes", f"{len(cust_disputes)}")
        col4.metric("Contacts Made", f"{len(cust_dunning)}")

        st.markdown("---")

        # ── Payment History ──
        col1, col2 = st.columns(2)

        with col1:
            st.subheader("Invoice Status")
            status_counts = cust_invoices['invoice_status'].value_counts()
            fig = px.pie(
                values=status_counts.values, names=status_counts.index,
                title=f"Invoice Status — {cust_info['customer_name']}",
                color=status_counts.index,
                color_discrete_map={
                    'Paid': '#059669', 'Open': '#2563EB',
                    'Partially Paid': '#F59E0B', 'Written Off': '#DC2626',
                    'Disputed': '#7C3AED'
                }
            )
            fig.update_layout(height=350)
            st.plotly_chart(fig, use_container_width=True)

        with col2:
            st.subheader("Payment Timeline")
            if len(cust_payments) > 0:
                pay_inv = cust_payments.merge(
                    cust_invoices[['invoice_id', 'due_date']], on='invoice_id', how='left'
                )
                pay_inv['delay'] = (pay_inv['payment_date'] - pay_inv['due_date']).dt.days

                fig = px.scatter(
                    pay_inv, x='payment_date', y='delay',
                    size='payment_amount', color='delay',
                    color_continuous_scale=['#059669', '#F59E0B', '#DC2626'],
                    title="Payment Delay Over Time",
                    labels={'delay': 'Days Past Due', 'payment_date': 'Payment Date'}
                )
                fig.add_hline(y=0, line_dash="dash", line_color="gray")
                fig.update_layout(height=350)
                st.plotly_chart(fig, use_container_width=True)
            else:
                st.info("No payments recorded for this customer.")

        # ── Invoice Table ──
        st.subheader("All Invoices")
        inv_display = cust_invoices[['invoice_id', 'invoice_date', 'due_date',
                                      'invoice_amount', 'invoice_status']].copy()
        inv_display['invoice_amount'] = inv_display['invoice_amount'].apply(lambda x: f"₹{x:,.0f}")
        inv_display = inv_display.sort_values('invoice_date', ascending=False)
        st.dataframe(inv_display, use_container_width=True, hide_index=True, height=300)

        # ── Contact History ──
        if len(cust_dunning) > 0:
            st.subheader("Contact History")
            contact_display = cust_dunning[['contact_date', 'contact_type', 'contact_outcome',
                                             'invoice_id']].sort_values('contact_date', ascending=False).head(20)
            st.dataframe(contact_display, use_container_width=True, hide_index=True)


# ════════════════════════════════════════════════════════════════
# PAGE 4: STRATEGY SEGMENTS
# ════════════════════════════════════════════════════════════════

elif page == "🎯 Strategy Segments":
    st.markdown('<p class="main-header">Collection Strategy Segments</p>', unsafe_allow_html=True)
    st.markdown('<p class="sub-header">Customer segmentation with recommended collection playbooks</p>', unsafe_allow_html=True)

    if 'segments' in data:
        segments = data['segments']

        # Segment distribution
        col1, col2 = st.columns(2)

        with col1:
            seg_counts = segments['strategy_segment'].value_counts().reset_index()
            seg_counts.columns = ['segment', 'count']
            fig = px.bar(
                seg_counts, x='segment', y='count',
                title="Customer Count by Segment",
                color='segment',
                color_discrete_sequence=['#059669', '#0891B2', '#F59E0B', '#DC2626', '#94A3B8'],
            )
            fig.update_layout(height=400, showlegend=False)
            st.plotly_chart(fig, use_container_width=True)

        with col2:
            seg_ar = segments.groupby('strategy_segment')['total_open_ar'].sum().reset_index()
            fig = px.bar(
                seg_ar, x='strategy_segment', y='total_open_ar',
                title="Total Open AR by Segment",
                color='strategy_segment',
                color_discrete_sequence=['#059669', '#0891B2', '#F59E0B', '#DC2626', '#94A3B8'],
            )
            fig.update_layout(height=400, showlegend=False)
            fig.update_yaxis(tickformat=',.0f')
            st.plotly_chart(fig, use_container_width=True)

        # Strategy playbook
        st.subheader("Collection Playbook")

        strategies = {
            'A — Reliable Payer': {
                'color': '🟢',
                'action': 'Automated reminder only',
                'contact': 'Email at DPD 1 and DPD 7',
                'escalation': 'Only if 30+ DPD',
            },
            'B — Variable Payer': {
                'color': '🔵',
                'action': 'Structured escalation sequence',
                'contact': 'Email → Phone → Senior, weekly',
                'escalation': 'At DPD 30',
            },
            'C — Chronic Late Payer': {
                'color': '🟡',
                'action': 'Immediate human contact',
                'contact': 'Phone from day 1, every 5 days',
                'escalation': 'At DPD 45',
            },
            'D — High-Risk High-Value': {
                'color': '🔴',
                'action': 'Senior collector from start',
                'contact': 'Phone every 3 days + email',
                'escalation': 'At DPD 15',
            },
            'E — Likely Write-off': {
                'color': '⚫',
                'action': 'Minimize effort, consider external agency',
                'contact': 'One final notice',
                'escalation': 'External agency at DPD 60',
            },
        }

        for seg_name, strat in strategies.items():
            count = len(segments[segments['strategy_segment'] == seg_name])
            with st.expander(f"{strat['color']} {seg_name} ({count} customers)"):
                st.write(f"**Action:** {strat['action']}")
                st.write(f"**Contact:** {strat['contact']}")
                st.write(f"**Escalation:** {strat['escalation']}")

        # Segment profile table
        st.subheader("Segment Profiles")
        seg_profile = segments.groupby('strategy_segment').agg(
            customers=('customer_id', 'count'),
            avg_open_ar=('total_open_ar', 'mean'),
            avg_dpd=('avg_dpd', 'mean'),
            avg_risk=('avg_p_recovery', lambda x: (1-x).mean()),
        ).round(2).reset_index()
        seg_profile.columns = ['Segment', 'Customers', 'Avg Open AR', 'Avg DPD', 'Avg Risk']
        seg_profile['Avg Open AR'] = seg_profile['Avg Open AR'].apply(lambda x: f"₹{x:,.0f}")
        seg_profile['Avg Risk'] = seg_profile['Avg Risk'].apply(lambda x: f"{x:.1%}")
        st.dataframe(seg_profile, use_container_width=True, hide_index=True)

    else:
        st.warning("Strategy segments not found. Run Notebook 06 first.")


# ════════════════════════════════════════════════════════════════
# PAGE 5: BUSINESS IMPACT
# ════════════════════════════════════════════════════════════════

elif page == "📈 Business Impact":
    st.markdown('<p class="main-header">Business Impact</p>', unsafe_allow_html=True)
    st.markdown('<p class="sub-header">Model-driven prioritization vs traditional approaches</p>', unsafe_allow_html=True)

    if 'simulation' in data:
        simulation = data['simulation']

        # Load strategy summaries
        summary_path = f'{PROC_DIR}strategy_summaries.json'
        if os.path.exists(summary_path):
            with open(summary_path, 'r') as f:
                strategy_summaries = json.load(f)

            # Key comparison metrics
            model_data = strategy_summaries.get('Model-Driven', {})
            fifo_data = strategy_summaries.get('FIFO (Oldest First)', {})

            col1, col2, col3 = st.columns(3)
            col1.metric(
                "Model Cash Collected",
                f"₹{model_data.get('total_cash', 0)/1e7:.1f} Cr",
            )
            col2.metric(
                "FIFO Cash Collected",
                f"₹{fifo_data.get('total_cash', 0)/1e7:.1f} Cr",
            )
            model_eff = model_data.get('cash_per_contact', 0)
            fifo_eff = fifo_data.get('cash_per_contact', 0)
            col3.metric(
                "Efficiency Gain",
                f"₹{model_eff:,.0f}/contact",
                f"vs ₹{fifo_eff:,.0f} FIFO"
            )

        st.markdown("---")

        # Cumulative cash chart
        st.subheader("Cash Collection Over Time")
        fig = go.Figure()
        strategy_colors = {
            'Model-Driven': '#059669',
            'FIFO (Oldest First)': '#2563EB',
            'Amount-Based': '#F59E0B',
            'Aging-Based': '#0891B2',
            'Random': '#94A3B8',
        }
        for strategy in simulation['strategy'].unique():
            strat_data = simulation[simulation['strategy'] == strategy]
            fig.add_trace(go.Scatter(
                x=strat_data['day'], y=strat_data['cumulative_cash'] / 1e7,
                mode='lines', name=strategy,
                line=dict(color=strategy_colors.get(strategy, '#94A3B8'), width=3)
            ))
        fig.update_layout(
            xaxis_title="Simulation Day",
            yaxis_title="Cumulative Cash (₹ Crore)",
            height=450,
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
        )
        st.plotly_chart(fig, use_container_width=True)

        # Strategy comparison table
        st.subheader("Strategy Comparison Summary")
        if os.path.exists(summary_path):
            comparison = []
            for strat, vals in strategy_summaries.items():
                comparison.append({
                    'Strategy': strat,
                    'Cash Collected': f"₹{vals['total_cash']/1e7:.1f} Cr",
                    'Total Contacts': f"{vals['total_contacts']:,}",
                    '₹ per Contact': f"₹{vals['cash_per_contact']:,.0f}",
                    'Recovery Rate': f"{vals['recovery_rate']:.1%}",
                })
            st.dataframe(pd.DataFrame(comparison), use_container_width=True, hide_index=True)

        # Assumptions
        st.subheader("Simulation Assumptions")
        st.info("""
        **Stated transparently:**
        1. Contact accelerates payment by ~20% for invoices that would be paid
        2. Contact adds ~10% recovery probability for unpaid invoices
        3. Collector capacity: 10 collectors × 40 contacts/day
        4. Simulation window: 60 days
        5. Results are simulated on test data, not from production deployment
        6. Recommend A/B test for production validation
        """)

    else:
        st.warning("Simulation results not found. Run Notebook 06 first.")


# ════════════════════════════════════════════════════════════════
# PAGE 6: MODEL EXPLAINABILITY
# ════════════════════════════════════════════════════════════════

elif page == "🧠 Model Explainability":
    st.markdown('<p class="main-header">Model Explainability</p>', unsafe_allow_html=True)
    st.markdown('<p class="sub-header">How the prediction model works and what drives its decisions</p>', unsafe_allow_html=True)

    # Model performance metrics
    if 'model_meta' in data:
        meta = data['model_meta']
        metrics = meta.get('test_metrics', {})

        st.subheader("Model Performance (Test Set)")
        col1, col2, col3, col4, col5 = st.columns(5)
        col1.metric("AUC-ROC", f"{metrics.get('auc_roc', 0):.4f}")
        col2.metric("AUC-PR", f"{metrics.get('auc_pr', 0):.4f}")
        col3.metric("F1 Score", f"{metrics.get('f1', 0):.4f}")
        col4.metric("Precision", f"{metrics.get('precision', 0):.4f}")
        col5.metric("Recall", f"{metrics.get('recall', 0):.4f}")

        st.markdown("---")

        # Feature list
        st.subheader("Model Features")
        features = meta.get('features', [])
        col1, col2, col3 = st.columns(3)
        third = len(features) // 3
        with col1:
            for f in features[:third]:
                st.write(f"• {f}")
        with col2:
            for f in features[third:2*third]:
                st.write(f"• {f}")
        with col3:
            for f in features[2*third:]:
                st.write(f"• {f}")

        st.markdown("---")

        # Model architecture
        st.subheader("System Architecture")
        st.markdown("""
```
        ┌──────────────┐     ┌──────────────┐
        │ Classification│     │   Survival   │
        │   LightGBM   │     │   Cox PH     │
        │  P(paid 90d)  │     │ Payment Time │
        └──────┬───────┘     └──────┬───────┘
               │                     │
               └──────────┬──────────┘
                          ▼
               ┌─────────────────────┐
               │   Prioritization    │
               │ Score = P × Amt ×   │
               │ Urgency / Effort    │
               └──────────┬──────────┘
                          ▼
               ┌─────────────────────┐
               │  Strategy Segments  │
               │  5 action playbooks │
               └─────────────────────┘
```
        """)

    else:
        st.warning("Model metadata not found.")

    # SHAP plots (if saved as images)
    st.subheader("SHAP Feature Importance")
    shap_path = f'{DATA_DIR}shap_global.png'
    if os.path.exists(shap_path):
        st.image(shap_path, caption="SHAP Global Feature Importance")
    else:
        st.info("SHAP plots not found. They are generated in Notebook 04.")

    # Prediction lookup
    if 'predictions' in data:
        st.subheader("Invoice Prediction Lookup")
        inv_id = st.text_input("Enter Invoice ID (e.g., INV_000100)")
        if inv_id:
            pred_row = data['predictions'][data['predictions']['invoice_id'] == inv_id]
            if len(pred_row) > 0:
                row = pred_row.iloc[-1]  # Latest observation
                col1, col2, col3 = st.columns(3)
                col1.metric("P(Paid within 90d)", f"{row['pred_prob_calibrated']:.1%}")
                col2.metric("Amount", f"₹{row['invoice_amount']:,.0f}")
                col3.metric("Days Past Due", f"{row['days_past_due']:.0f}")

                risk_level = "🟢 Low" if row['pred_prob_calibrated'] > 0.7 else (
                    "🟡 Medium" if row['pred_prob_calibrated'] > 0.4 else "🔴 High")
                st.write(f"**Risk Level:** {risk_level}")
                st.write(f"**Actual Outcome:** {'Paid' if row['actual'] == 1 else 'Not Paid'}")
            else:
                st.warning(f"Invoice {inv_id} not found in test predictions.")


# ════════════════════════════════════════════════════════════════
# FOOTER
# ════════════════════════════════════════════════════════════════

st.sidebar.markdown("---")
st.sidebar.markdown(
    """
    <div style="text-align: center; color: #94A3B8; font-size: 0.8rem;">
        AR Collections Intelligence Engine<br>
        Portfolio Project — Tier 1 Flagship<br>
        Built with Python, LightGBM, Lifelines, Streamlit
    </div>
    """,
    unsafe_allow_html=True
)