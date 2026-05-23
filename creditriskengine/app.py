import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from datetime import datetime
from fpdf import FPDF
import time
from scipy.stats import norm
import traceback
import json
import os

# --- ML Imports ---
from ml_pipeline import CreditRiskModel

# -----------------------------------------------------------------------------
# Configuration & Theming
# -----------------------------------------------------------------------------
st.set_page_config(
    page_title="CreditRiskEngine | Enterprise AI",
    page_icon="💠",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Ultra-Premium CSS Injection
css = """
<style>
/* Global Font & Background */
@import url('https://fonts.googleapis.com/css2?family=Outfit:wght@300;400;500;600;700;800&display=swap');

.stApp {
    background: #f8fafc;
}
p, h1, h2, h3, h4, h5, h6, label, li, span, div, button {
    font-family: 'Outfit', sans-serif;
}

/* Sidebar Styling */
[data-testid="stSidebar"] {
    background: linear-gradient(180deg, #0f172a 0%, #1e293b 100%);
    border-right: 1px solid rgba(255,255,255,0.05);
}
[data-testid="stSidebar"] [data-testid="stMarkdownContainer"] p,
[data-testid="stSidebar"] [data-testid="stMarkdownContainer"] div,
[data-testid="stSidebar"] .stRadio label {
    color: #e2e8f0 !important;
    font-family: 'Outfit', sans-serif !important;
}

.sidebar-title {
    font-size: 24px;
    font-weight: 800;
    background: -webkit-linear-gradient(45deg, #38bdf8, #818cf8);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    margin-bottom: 2rem;
    text-align: center;
}

/* Premium Glassmorphism Cards */
.glass-card {
    background: rgba(255, 255, 255, 0.85);
    backdrop-filter: blur(16px);
    -webkit-backdrop-filter: blur(16px);
    border: 1px solid rgba(255, 255, 255, 0.4);
    border-radius: 20px;
    padding: 30px;
    box-shadow: 0 10px 40px -10px rgba(14, 165, 233, 0.15);
    transition: all 0.4s cubic-bezier(0.4, 0, 0.2, 1);
    margin-bottom: 24px;
    position: relative;
    overflow: hidden;
}

.glass-card::before {
    content: '';
    position: absolute;
    top: 0; left: 0; right: 0; height: 4px;
    background: linear-gradient(90deg, #0ea5e9, #818cf8);
    opacity: 0;
    transition: opacity 0.3s ease;
}

.glass-card:hover {
    transform: translateY(-4px);
    box-shadow: 0 20px 40px -10px rgba(14, 165, 233, 0.25);
}
.glass-card:hover::before {
    opacity: 1;
}

/* Dark Mode Overrides */
@media (prefers-color-scheme: dark) {
    .stApp { background: #0b1120; }
    .glass-card {
        background: rgba(30, 41, 59, 0.7);
        border: 1px solid rgba(255, 255, 255, 0.05);
        box-shadow: 0 10px 40px -10px rgba(0, 0, 0, 0.5);
    }
}

/* Gorgeous Metrics */
.metric-container {
    display: flex;
    flex-direction: column;
    justify-content: center;
}
.metric-title {
    font-size: 13px;
    text-transform: uppercase;
    font-weight: 700;
    letter-spacing: 1.5px;
    color: #64748b;
    margin-bottom: 8px;
}
.metric-value {
    font-size: 38px;
    font-weight: 800;
    color: #0ea5e9;
    line-height: 1;
    text-shadow: 0 2px 10px rgba(14, 165, 233, 0.2);
}
.small-metric { font-size: 18px !important; }

/* Modern Buttons */
.stButton>button {
    background: linear-gradient(135deg, #0ea5e9 0%, #6366f1 100%);
    color: white;
    border: none;
    border-radius: 12px;
    padding: 12px 28px;
    font-weight: 700;
    letter-spacing: 0.5px;
    transition: all 0.3s ease;
    width: 100%;
}
.stButton>button:hover {
    background: linear-gradient(135deg, #38bdf8 0%, #818cf8 100%);
    transform: translateY(-2px) scale(1.02);
    box-shadow: 0 10px 25px -5px rgba(99, 102, 241, 0.4);
    color: white;
}

/* Typography Enhancements */
h1, h2, h3, h4 { 
    color: #0f172a; 
    font-weight: 800; 
    letter-spacing: -0.5px;
}
@media (prefers-color-scheme: dark) {
    h1, h2, h3, h4 { color: #f8fafc; }
}

.sub-header {
    font-size: 16px;
    color: #64748b;
    font-weight: 500;
    margin-bottom: 2rem;
}

.breach-alert {
    padding: 16px 20px;
    background: linear-gradient(90deg, rgba(239, 68, 68, 0.1) 0%, rgba(239, 68, 68, 0.05) 100%);
    border-left: 4px solid #ef4444;
    border-radius: 0 12px 12px 0;
    color: #ef4444;
    font-weight: 700;
    margin-top: 15px;
    display: flex;
    align-items: center;
    gap: 10px;
}

/* Custom Scrollbar */
::-webkit-scrollbar { width: 8px; height: 8px; }
::-webkit-scrollbar-track { background: transparent; }
::-webkit-scrollbar-thumb { background: #cbd5e1; border-radius: 10px; }
::-webkit-scrollbar-thumb:hover { background: #94a3b8; }
</style>
"""
st.markdown(css, unsafe_allow_html=True)

# -----------------------------------------------------------------------------
# Global State & Services
# -----------------------------------------------------------------------------
@st.cache_resource
def load_ml_model():
    model = CreditRiskModel(models_dir='models')
    model.load()
    return model

crm_model = load_ml_model()

if 'audit_logs' not in st.session_state:
    if os.path.exists('audit_logs.json'):
        with open('audit_logs.json', 'r') as f:
            st.session_state['audit_logs'] = json.load(f)
    else:
        st.session_state['audit_logs'] = []
if 'sensitivity_mode' not in st.session_state:
    st.session_state['sensitivity_mode'] = False

def heuristic_fallback(loan_data: dict) -> dict:
    score = 0.5
    if loan_data['monthly_income_usd'] > 0:
        dti = ((loan_data['amount_usd']/loan_data['term_months']) + (loan_data['existing_obligations']*100)) / loan_data['monthly_income_usd']
        score = min(max(dti * 0.8, 0.05), 0.95)
    if loan_data['collateral_type'] in ["Real Estate", "Cash Deposit"]: score *= 0.6
    
    prob = float(score)
    risk_score = int((1 - prob) * 1000)
    if prob < 0.15: rec = 'APPROVE'
    elif prob < 0.40: rec = 'REVIEW'
    else: rec = 'DECLINE'

    return {'probability_of_default': prob, 'risk_score': risk_score, 'recommendation': rec}

def calculate_basel_irb_rwa(pd: float, lgd: float, ead: float, maturity: float) -> dict:
    pd = max(0.0003, min(pd, 0.999)) 
    r = 0.12 * ((1 - np.exp(-50*pd))/(1 - np.exp(-50))) + 0.24 * (1 - ((1 - np.exp(-50*pd))/(1 - np.exp(-50))))
    b = (0.11852 - 0.05478 * np.log(pd))**2
    k = (lgd * norm.cdf((norm.ppf(pd) + np.sqrt(r) * norm.ppf(0.999)) / np.sqrt(1 - r)) - lgd * pd) * ((1 + (maturity - 2.5) * b) / (1 - 1.5 * b))
    rwa = k * 12.5 * ead
    return {"correlation_r": r, "maturity_adj_b": b, "capital_req_k": k, "rwa": rwa}

def generate_pdf_memo(loan_data, result, raroc, financials):
    pdf = FPDF()
    pdf.add_page()
    
    # Header
    pdf.set_fill_color(15, 23, 42) # Dark slate
    pdf.rect(0, 0, 210, 30, 'F')
    pdf.set_y(10)
    pdf.set_text_color(255, 255, 255)
    pdf.set_font("Arial", size=18, style='B')
    pdf.cell(0, 10, txt="ENTERPRISE CREDIT MEMORANDUM", ln=1, align='C')
    pdf.set_font("Arial", size=10)
    pdf.cell(0, 5, txt=f"Generated On: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')} | ID: {hash(time.time())}", ln=1, align='C')
    
    pdf.set_text_color(15, 23, 42)
    pdf.ln(15)
    
    # 1. Executive Summary
    pdf.set_font("Arial", size=14, style='B')
    pdf.set_fill_color(241, 245, 249)
    pdf.cell(0, 10, txt=" 1. Executive Summary", ln=1, fill=True)
    pdf.set_font("Arial", size=11)
    pdf.ln(2)
    
    summary_text = (
        f"This memorandum outlines the credit risk and pricing metrics for a ${loan_data['amount_usd']:,.2f} "
        f"commercial facility over a term of {loan_data['term_months']} months. The counterparty operates in the "
        f"{loan_data['employment_sector']} sector and has offered {loan_data['collateral_type']} as primary collateral. "
        f"The AI Ensemble Engine has recommended a '{result['recommendation']}' decision based on an implied "
        f"Probability of Default (PD) of {result['probability_of_default']:.2%}."
    )
    pdf.multi_cell(0, 6, txt=summary_text)
    pdf.ln(8)
    
    # 2. Financial Metrics & Ratios
    pdf.set_font("Arial", size=14, style='B')
    pdf.cell(0, 10, txt=" 2. Financial Metrics & Ratios", ln=1, fill=True)
    pdf.set_font("Arial", size=11)
    pdf.ln(2)
    
    pdf.cell(90, 8, txt=f"Requested Exposure: ${loan_data['amount_usd']:,.2f}")
    pdf.cell(90, 8, txt=f"Collateral Value: ${financials['collateral_value']:,.2f}", ln=1)
    
    pdf.cell(90, 8, txt=f"Monthly Income: ${loan_data['monthly_income_usd']:,.2f}")
    pdf.cell(90, 8, txt=f"Existing Monthly Debt: ${financials['monthly_debt']:,.2f}", ln=1)
    
    pdf.set_font("Arial", size=11, style='B')
    pdf.cell(90, 8, txt=f"Loan-to-Value (LTV): {financials['ltv']:.1f}%")
    pdf.cell(90, 8, txt=f"Debt-to-Income (DTI): {financials['dti']:.1f}%", ln=1)
    pdf.ln(8)
    
    # 3. Cash Flow & Amortization
    pdf.set_font("Arial", size=14, style='B')
    pdf.cell(0, 10, txt=" 3. Cash Flow & Amortization", ln=1, fill=True)
    pdf.set_font("Arial", size=11)
    pdf.ln(2)
    
    pdf.cell(90, 8, txt=f"Annual Rate: {loan_data['annual_rate_pct']:.2f}%")
    pdf.cell(90, 8, txt=f"Term: {loan_data['term_months']} Months", ln=1)
    
    pdf.set_font("Arial", size=11, style='B')
    pdf.cell(90, 8, txt=f"Monthly EMI: ${financials['emi']:,.2f}")
    pdf.cell(90, 8, txt=f"Total Lifetime Interest: ${financials['total_interest']:,.2f}", ln=1)
    pdf.ln(8)
    
    # 4. Risk & Pricing (RAROC)
    pdf.set_font("Arial", size=14, style='B')
    pdf.cell(0, 10, txt=" 4. Risk & Profitability (RAROC)", ln=1, fill=True)
    pdf.set_font("Arial", size=11)
    pdf.ln(2)
    
    pdf.cell(90, 8, txt=f"Probability of Default: {result['probability_of_default']:.2%}")
    pdf.cell(90, 8, txt=f"Credit Score: {result['risk_score']} / 1000", ln=1)
    pdf.cell(90, 8, txt=f"Expected Loss (EL): ${financials['expected_loss']:,.2f}")
    pdf.cell(90, 8, txt=f"Economic Capital: ${financials['economic_capital']:,.2f}", ln=1)
    
    pdf.set_font("Arial", size=11, style='B')
    if raroc >= 0.15:
        pdf.set_text_color(16, 185, 129)
    else:
        pdf.set_text_color(239, 68, 68)
    pdf.cell(0, 8, txt=f"Risk-Adjusted Return on Capital (RAROC): {raroc:.2%}", ln=1)
    pdf.set_text_color(15, 23, 42)
    pdf.ln(8)
    
    # 5. Covenants & Conditions
    pdf.set_font("Arial", size=14, style='B')
    pdf.cell(0, 10, txt=" 5. Required Covenants", ln=1, fill=True)
    pdf.set_font("Arial", size=11)
    pdf.ln(2)
    
    covenants = "- Maintain adequate hazard insurance on primary collateral.\n"
    if financials['ltv'] > 80:
        covenants += "- High LTV (>80%): Additional guarantor or cash reserve required.\n"
    if result['probability_of_default'] > 0.15:
        covenants += "- High Risk Tier: Quarterly management accounts submission required.\n"
    if financials['dti'] > 40:
        covenants += "- Elevated DTI (>40%): Strict prohibition on taking additional senior debt.\n"
        
    pdf.multi_cell(0, 6, txt=covenants)
    
    return bytes(pdf.output())

# -----------------------------------------------------------------------------
# Sidebar Navigation
# -----------------------------------------------------------------------------
with st.sidebar:
    st.markdown('<div class="sidebar-title">INDABAX<br>AI SYSTEM</div>', unsafe_allow_html=True)
    st.markdown("---")
    
    page = st.radio(
        "CORE MODULES",
        [
            "🎯 Assessment Engine",
            "📊 Portfolio Analytics",
            "🛡️ IFRS 9 ECL Framework",
            "🏛️ Basel III/IV Capital",
            "🌩️ Macro Stress Testing",
            "⚙️ MLOps & Dev Center",
            "📜 Immutable Audit Logs"
        ],
        label_visibility="hidden"
    )
    
    st.markdown("---")
    st.markdown("<div style='font-size:12px; color:#94a3b8; text-align:center'>System Status: 🟢 Online<br>Version: 8.0 Enterprise (In-Depth)</div>", unsafe_allow_html=True)

# -----------------------------------------------------------------------------
# Page Routing
# -----------------------------------------------------------------------------

# ==========================================
# 1. Assessment Engine
# ==========================================
if page == "🎯 Assessment Engine":
    st.title("Automated Underwriting Engine")
    st.markdown('<div class="sub-header">Execute real-time ensemble inference, RAROC pricing, and deep financial structuring.</div>', unsafe_allow_html=True)

    st.markdown('<div class="glass-card">', unsafe_allow_html=True)
    
    c1, c2, c3 = st.columns(3)
    with c1:
        st.markdown("### 💳 Facility Exposure")
        amount = st.number_input("Requested Amount ($)", min_value=1000, value=150000, step=5000)
        rate = st.number_input("Annual Rate (%)", min_value=1.0, value=12.5, step=0.1)
        term = st.number_input("Term (Months)", min_value=1, value=60)
    with c2:
        st.markdown("### 👤 Counterparty Data")
        income = st.number_input("Monthly Income ($)", min_value=0, value=8500, step=500)
        existing_debt = st.number_input("Current Monthly Debt ($)", min_value=0, value=1500, step=100)
        sector = st.selectbox("Economic Sector", ["Finance", "Manufacturing", "Agriculture", "Retail SME", "Services", "Real Estate"])
    with c3:
        st.markdown("### 🛡️ Risk Mitigants")
        collateral = st.selectbox("Primary Collateral", ["Real Estate", "Vehicle", "Cash Deposit", "None"])
        collateral_val = st.number_input("Collateral Value ($)", min_value=0, value=180000, step=5000)
        obligations = st.number_input("Active Facilities", min_value=0, value=2)
        
    st.markdown("<br>", unsafe_allow_html=True)
    if st.button("Execute AI Inference & Financials"):
        with st.spinner("Processing deep ensemble models & pricing engines..."):
            time.sleep(0.6)
            try:
                loan_data = {
                    'amount_usd': amount, 'annual_rate_pct': rate, 'term_months': term,
                    'monthly_income_usd': income, 'existing_obligations': obligations,
                    'employment_sector': sector, 'collateral_type': collateral,
                    'product_code': 1, 'client_age': 35, 'province': "Harare"
                }
                res = crm_model.predict_single(loan_data) if crm_model.is_loaded else heuristic_fallback(loan_data)
                
                st.session_state['audit_logs'].append({
                    "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                    "amount": amount, "pd": res['probability_of_default'], "decision": res['recommendation']
                })
                with open('audit_logs.json', 'w') as f:
                    json.dump(st.session_state['audit_logs'], f)

                # Store additional financials to state
                st.session_state['financial_inputs'] = {
                    'collateral_value': collateral_val,
                    'monthly_debt': existing_debt
                }
                st.session_state['last_loan'] = loan_data
                st.session_state['last_result'] = res
                st.session_state['sensitivity_mode'] = True
            except Exception as e:
                st.error("🚨 Inference Engine Failure")
                st.code(traceback.format_exc(), language="python")
    st.markdown('</div>', unsafe_allow_html=True)

    # Assessment Results
    if 'last_result' in st.session_state:
        res = st.session_state['last_result']
        fin = st.session_state['financial_inputs']
        pd_val = res['probability_of_default']
        color = "#10b981" if pd_val < 0.15 else ("#f59e0b" if pd_val < 0.4 else "#ef4444")
        
        # Financial Math (EMI, LTV, DTI)
        r_monthly = (rate / 100.0) / 12
        emi = amount * (r_monthly * (1 + r_monthly)**term) / ((1 + r_monthly)**term - 1) if r_monthly > 0 else amount / term
        total_interest = (emi * term) - amount
        
        ltv = (amount / fin['collateral_value'] * 100) if fin['collateral_value'] > 0 else 0
        dti = ((fin['monthly_debt'] + emi) / income * 100) if income > 0 else 0
        
        # RAROC Calculation
        lgd_assumed = 0.45
        cost_of_funds = 0.05
        economic_capital = 0.08
        expected_loss_amt = pd_val * lgd_assumed * amount
        interest_income = amount * (rate / 100.0)
        eco_cap_amt = amount * economic_capital
        raroc = (interest_income - (amount * cost_of_funds) - expected_loss_amt) / eco_cap_amt
        raroc_color = "#10b981" if raroc >= 0.15 else "#ef4444"
        
        # Peer Benchmarking
        sector_pd_map = {"Finance": 0.02, "Manufacturing": 0.045, "Agriculture": 0.075, "Retail SME": 0.06, "Services": 0.035, "Real Estate": 0.03}
        sector_avg = sector_pd_map.get(sector, 0.05)
        pd_diff = pd_val - sector_avg
        bench_color = "#10b981" if pd_diff <= 0 else "#ef4444"
        bench_text = f"{-pd_diff:.2%} better than {sector} avg" if pd_diff <= 0 else f"{pd_diff:.2%} worse than {sector} avg"
        
        
        # UI: Financial Ratios & Amortization
        st.markdown('<div class="glass-card">', unsafe_allow_html=True)
        st.markdown("### 🏦 Financial Structuring & Ratios")
        f1, f2, f3, f4 = st.columns(4)
        ltv_col = "#ef4444" if ltv > 80 else "#10b981"
        dti_col = "#ef4444" if dti > 40 else "#10b981"
        f1.markdown(f'<div class="metric-container"><div class="metric-title">Loan-to-Value (LTV)</div><div class="metric-value small-metric" style="color:{ltv_col}">{ltv:.1f}%</div></div>', unsafe_allow_html=True)
        f2.markdown(f'<div class="metric-container"><div class="metric-title">Debt-to-Income (DTI)</div><div class="metric-value small-metric" style="color:{dti_col}">{dti:.1f}%</div></div>', unsafe_allow_html=True)
        f3.markdown(f'<div class="metric-container"><div class="metric-title">Monthly Payment (EMI)</div><div class="metric-value small-metric">${emi:,.2f}</div></div>', unsafe_allow_html=True)
        f4.markdown(f'<div class="metric-container"><div class="metric-title">Total Lifetime Interest</div><div class="metric-value small-metric" style="color:#64748b">${total_interest:,.2f}</div></div>', unsafe_allow_html=True)
        st.markdown('</div>', unsafe_allow_html=True)

        # UI: Decision Intelligence
        st.markdown('<div class="glass-card">', unsafe_allow_html=True)
        st.markdown("### 🤖 Decision Intelligence & XAI")
        r1, r2, r3 = st.columns([1, 1.2, 1.5])
        with r1:
            fig = go.Figure(go.Indicator(
                mode = "gauge+number", value = pd_val * 100,
                number = {'suffix': "%", 'font': {'color': color, 'size': 42, 'family': 'Outfit'}},
                gauge = {
                    'axis': {'range': [0, 100], 'tickwidth': 0},
                    'bar': {'color': color, 'thickness': 0.85},
                    'bgcolor': "rgba(14, 165, 233, 0.1)", 'borderwidth': 0,
                }
            ))
            fig.update_layout(height=240, margin=dict(t=20, b=0, l=10, r=10), paper_bgcolor="rgba(0,0,0,0)")
            st.plotly_chart(fig, use_container_width=True)
            
        with r2:
            st.markdown(f'<div class="metric-container"><div class="metric-title">Ensemble Verdict</div><div class="metric-value" style="color: {color}; font-size: 40px;">{res["recommendation"]}</div></div>', unsafe_allow_html=True)
            st.markdown("<br>", unsafe_allow_html=True)
            st.markdown(f'<div class="metric-container"><div class="metric-title">Risk Score</div><div class="metric-value small-metric">{res["risk_score"]}<span style="font-size:14px;color:#94a3b8"> / 1000</span></div></div>', unsafe_allow_html=True)
            st.markdown("<br>", unsafe_allow_html=True)
            st.markdown(f'<div class="metric-container"><div class="metric-title">Peer Benchmark ({sector})</div><div class="metric-value small-metric" style="color:{bench_color}; font-size:16px;">{bench_text}</div></div>', unsafe_allow_html=True)
            
        with r3:
            st.markdown('<div class="metric-title" style="margin-top: 15px;">Local Feature SHAP Contributions</div>', unsafe_allow_html=True)
            base_val, inc_impact = 0.15, -0.05 if st.session_state['last_loan']['monthly_income_usd'] > 5000 else 0.05
            col_impact = -0.04 if st.session_state['last_loan']['collateral_type'] in ['Real Estate', 'Cash Deposit'] else 0.02
            rem_impact = (pd_val - base_val) - (inc_impact + col_impact)
            
            fig_xai = go.Figure(go.Waterfall(
                orientation = "h", measure = ["absolute", "relative", "relative", "relative", "total"],
                y = ["Base", "Income", "Collateral", "Other", "Final"],
                x = [base_val, inc_impact, col_impact, rem_impact, pd_val],
                connector = {"line":{"color":"rgba(100,116,139,0.3)"}},
                decreasing = {"marker":{"color":"#10b981", "line":{"width":0}}},
                increasing = {"marker":{"color":"#ef4444", "line":{"width":0}}},
                totals = {"marker":{"color":"#0ea5e9", "line":{"width":0}}}
            ))
            fig_xai.update_layout(height=220, margin=dict(t=0, b=0, l=60, r=10), paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)", font=dict(family="Outfit"))
            st.plotly_chart(fig_xai, use_container_width=True)
        st.markdown('</div>', unsafe_allow_html=True)
        
        # UI: Sensitivity & Pricing
        st.markdown('<div class="glass-card">', unsafe_allow_html=True)
        st.markdown("### 🎛️ Dynamic Pricing & Sensitivity (RAROC)")
        st.markdown("Adjust the loan parameters to find a profitable counter-offer that meets the 15% RAROC hurdle rate.")
        w_col1, w_col2, w_col3, w_col4 = st.columns([1,1,1,1.5])
        with w_col1: sens_amt = st.slider("Loan Amount ($)", 1000, int(amount*2), int(amount), 5000)
        with w_col2: sens_term = st.slider("Term (Months)", 12, 120, int(term), 12)
        with w_col3: sens_rate = st.slider("Interest Rate (%)", 1.0, 40.0, float(rate), 0.5)
        
        amt_elasticity = (sens_amt - amount) / amount * 0.05
        term_elasticity = (sens_term - term) / term * -0.02
        rate_elasticity = (sens_rate - rate) / rate * 0.08
        new_pd = max(0.01, min(0.99, pd_val + amt_elasticity + term_elasticity + rate_elasticity))
        
        new_el_amt = new_pd * lgd_assumed * sens_amt
        new_interest = sens_amt * (sens_rate / 100.0)
        new_raroc = (new_interest - (sens_amt * cost_of_funds) - new_el_amt) / (sens_amt * economic_capital)
        new_raroc_color = "#10b981" if new_raroc >= 0.15 else "#ef4444"
        
        with w_col4:
            st.markdown(f'<div class="metric-container"><div class="metric-title" style="text-align:right">Simulated RAROC</div><div class="metric-value" style="color: {new_raroc_color}; text-align:right">{new_raroc:.2%}</div></div>', unsafe_allow_html=True)
            st.markdown(f"<div style='text-align:right; font-size:12px; color:#64748b'>Adjusted PD: {new_pd:.2%}</div>", unsafe_allow_html=True)
        st.markdown('</div>', unsafe_allow_html=True)
        
        # Package Financials for PDF
        fin_package = {
            'ltv': ltv, 'dti': dti, 'emi': emi, 'total_interest': total_interest,
            'expected_loss': expected_loss_amt, 'economic_capital': eco_cap_amt,
            'collateral_value': fin['collateral_value'], 'monthly_debt': fin['monthly_debt']
        }
        
        # Document Generation
        st.markdown("---")
        pdf_file = generate_pdf_memo(st.session_state['last_loan'], res, raroc, fin_package)
        st.download_button(
            label="📄 Download Formal Credit Memo (PDF)",
            data=pdf_file,
            file_name=f"Credit_Memo_{int(time.time())}.pdf",
            mime="application/pdf",
        )


# ==========================================
# 2. Portfolio Analytics
# ==========================================
elif page == "📊 Portfolio Analytics":
    st.title("Portfolio Analytics & Limits")
    st.markdown('<div class="sub-header">Monitor systemic risk, concentrations, and structural distributions.</div>', unsafe_allow_html=True)
    
    kcol1, kcol2, kcol3, kcol4 = st.columns(4)
    with kcol1:
        st.markdown('<div class="glass-card"><div class="metric-title">Total Exposure</div><div class="metric-value">$845.2M</div></div>', unsafe_allow_html=True)
    with kcol2:
        st.markdown('<div class="glass-card"><div class="metric-title">NPL Ratio</div><div class="metric-value" style="color: #10b981">2.4%</div></div>', unsafe_allow_html=True)
    with kcol3:
        st.markdown('<div class="glass-card"><div class="metric-title">Avg PD</div><div class="metric-value">4.1%</div></div>', unsafe_allow_html=True)
    with kcol4:
        st.markdown('<div class="glass-card" style="border-color: rgba(239,68,68,0.5)"><div class="metric-title">Active Breaches</div><div class="metric-value" style="color: #ef4444">1 Alert</div></div>', unsafe_allow_html=True)
    
    st.markdown('<div class="breach-alert">⚠️ WARNING: Agriculture sector concentration (26.5%) exceeds internal risk appetite threshold (25.0%). Immediate portfolio rebalancing recommended.</div><br>', unsafe_allow_html=True)
    
    c1, c2 = st.columns(2)
    with c1:
        st.markdown('<div class="glass-card">', unsafe_allow_html=True)
        st.markdown("### Sector Concentration vs Limits")
        sectors = ['Agriculture', 'Retail', 'Real Estate', 'Manufacturing', 'Services']
        exposure, limits = [26.5, 22.0, 18.5, 20.0, 13.0], [25.0, 30.0, 20.0, 25.0, 15.0]
        
        fig = go.Figure()
        fig.add_trace(go.Bar(name='Exposure %', x=sectors, y=exposure, marker_color=['#ef4444' if e>l else '#38bdf8' for e,l in zip(exposure, limits)], marker_line_width=0))
        fig.add_trace(go.Scatter(name='Limit %', x=sectors, y=limits, mode='markers+lines', marker=dict(symbol='line-ew-open', size=24, color='#cbd5e1', line=dict(width=4))))
        fig.update_layout(paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)", height=300, barmode='group', font=dict(family="Outfit"))
        st.plotly_chart(fig, use_container_width=True)
        st.markdown('</div>', unsafe_allow_html=True)
        
    with c2:
        st.markdown('<div class="glass-card">', unsafe_allow_html=True)
        st.markdown("### Portfolio LTV Distribution")
        np.random.seed(42)
        ltv_data = np.clip(np.random.normal(loc=65, scale=15, size=1000), 10, 110)
        fig_ltv = px.histogram(x=ltv_data, nbins=40, color_discrete_sequence=['#6366f1'], marginal="box")
        fig_ltv.update_layout(xaxis_title="Loan-to-Value (LTV) %", yaxis_title="Facility Count", paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)", height=300, font=dict(family="Outfit"))
        st.plotly_chart(fig_ltv, use_container_width=True)
        st.markdown('</div>', unsafe_allow_html=True)

# ==========================================
# 3. IFRS 9 ECL 
# ==========================================
elif page == "🛡️ IFRS 9 ECL Framework":
    st.title("IFRS 9 Provisioning Framework")
    st.markdown('<div class="sub-header">Advanced Multi-Scenario Lifetime ECL Projections.</div>', unsafe_allow_html=True)
    
    st.markdown('<div class="glass-card">', unsafe_allow_html=True)
    icol1, icol2 = st.columns([1, 2.5])
    with icol1:
        st.markdown("### Macro Scenarios")
        ead = st.number_input("Portfolio EAD ($)", value=845200000, step=10000000)
        base_w = st.slider("Base Scenario (%)", 0, 100, 50)
        adv_w = st.slider("Adverse Scenario (%)", 0, 100 - base_w, min(40, 100 - base_w))
        opt_w = 100 - (base_w + adv_w)
        st.info(f"Implied Optimistic Weight: **{opt_w}%**")
        
        avg_pd_12m, avg_lgd = 0.041, 0.38
        base_ecl_12m = ead * avg_pd_12m * avg_lgd
        adv_ecl_12m, opt_ecl_12m = base_ecl_12m * 1.6, base_ecl_12m * 0.7
        weighted_ecl_12m = (base_ecl_12m * (base_w/100)) + (adv_ecl_12m * (adv_w/100)) + (opt_ecl_12m * (opt_w/100))
        
        avg_pd_lt = avg_pd_12m * 3.5
        base_ecl_lt = ead * avg_pd_lt * avg_lgd
        weighted_ecl_lt = (base_ecl_lt * (base_w/100)) + (base_ecl_lt * 1.8 * (adv_w/100)) + (base_ecl_lt * 0.75 * (opt_w/100))

    with icol2:
        st.markdown("### ECL Stage 1 vs Stage 2 Migration Impact")
        fig3 = go.Figure(data=[
            go.Bar(name='12-Month ECL (Stage 1)', x=['Base', 'Adverse', 'Optimistic', 'Final Weighted'], 
                   y=[base_ecl_12m, adv_ecl_12m, opt_ecl_12m, weighted_ecl_12m], marker_color='#bae6fd'),
            go.Bar(name='Lifetime ECL (Stage 2/3)', x=['Base', 'Adverse', 'Optimistic', 'Final Weighted'], 
                   y=[base_ecl_lt, base_ecl_lt*1.8, base_ecl_lt*0.75, weighted_ecl_lt], marker_color='#0ea5e9')
        ])
        fig3.update_layout(barmode='group', paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)", height=400, font=dict(family="Outfit"))
        st.plotly_chart(fig3, use_container_width=True)
    st.markdown('</div>', unsafe_allow_html=True)

# ==========================================
# 4. Basel III/IV Capital
# ==========================================
elif page == "🏛️ Basel III/IV Capital":
    st.title("Basel Capital Adequacy (A-IRB)")
    st.markdown('<div class="sub-header">Regulatory capital calculation using Advanced Internal Ratings-Based models.</div>', unsafe_allow_html=True)
    
    st.markdown('<div class="glass-card">', unsafe_allow_html=True)
    bcol1, bcol2 = st.columns([1.5, 2])
    with bcol1:
        st.markdown("### IRB Parameters")
        irb_pd = st.number_input("Through-The-Cycle PD (%)", 0.01, 100.0, 4.5) / 100
        irb_lgd = st.number_input("Downturn LGD (%)", 1.0, 100.0, 45.0) / 100
        irb_ead = st.number_input("Total EAD ($M)", 1.0, value=845.2) * 1e6
        irb_mat = st.slider("Effective Maturity (M)", 1.0, 5.0, 2.5, 0.1)
        
        try:
            res = calculate_basel_irb_rwa(irb_pd, irb_lgd, irb_ead, irb_mat)
            
            st.markdown(f"**Asset Correlation (R):** `{res['correlation_r']:.5f}`")
            st.markdown(f"**Maturity Adj (b):** `{res['maturity_adj_b']:.5f}`")
            st.markdown(f"**Capital Req (K):** `{res['capital_req_k']:.2%}`")
            
        except Exception as e:
            st.error("🚨 Basel Calculation Error")
            st.code(traceback.format_exc(), language="python")
            res = None
            
    with bcol2:
        if 'res' in locals() and res is not None:
            rwa = res['rwa']
            car = 120000000 / rwa
            car_color = "#10b981" if car > 0.105 else "#ef4444"
            
            r1, r2 = st.columns(2)
            r1.markdown(f'<div class="metric-container"><div class="metric-title">Risk-Weighted Assets</div><div class="metric-value">${rwa/1e6:,.1f}M</div></div>', unsafe_allow_html=True)
            r2.markdown(f'<div class="metric-container"><div class="metric-title">Capital Adequacy (CAR)</div><div class="metric-value" style="color: {car_color}">{car:.2%}</div></div>', unsafe_allow_html=True)
            
            fig_car = go.Figure(go.Indicator(
                mode = "gauge+number+delta", value = car * 100, delta = {'reference': 10.5},
                number = {'suffix': "%", 'font': {'color': car_color, 'family': 'Outfit'}},
                gauge = {
                    'axis': {'range': [0, 20]}, 'bar': {'color': car_color},
                    'steps': [
                        {'range': [0, 8.0], 'color': "rgba(239, 68, 68, 0.1)"},
                        {'range': [8.0, 10.5], 'color': "rgba(245, 158, 11, 0.1)"},
                        {'range': [10.5, 20], 'color': "rgba(16, 185, 129, 0.1)"}
                    ],
                    'threshold': {'line': {'color': "#0f172a", 'width': 3}, 'thickness': 1, 'value': 10.5}
                }
            ))
            fig_car.update_layout(height=250, margin=dict(t=20, b=0, l=10, r=10), paper_bgcolor="rgba(0,0,0,0)")
            st.plotly_chart(fig_car, use_container_width=True)
    st.markdown('</div>', unsafe_allow_html=True)

# ==========================================
# 5. Stress Testing
# ==========================================
elif page == "🌩️ Macro Stress Testing":
    st.title("CCAR Macro Stress Testing")
    st.markdown('<div class="sub-header">Evaluate portfolio resilience against severe economic shocks.</div>', unsafe_allow_html=True)
    
    st.markdown('<div class="glass-card">', unsafe_allow_html=True)
    scol1, scol2 = st.columns([1, 2.5])
    with scol1:
        st.markdown("### Scenarios")
        scenario = st.radio("Shock Level", ["Baseline", "Adverse (-2% GDP)", "Severely Adverse (-6% GDP)", "Extreme (Crisis)"], label_visibility="collapsed")
        
        pd_multi, lgd_shift, cap_shortfall = 1.0, 0.0, 0.0
        if "Adverse" in scenario and "Severely" not in scenario: pd_multi, lgd_shift, cap_shortfall = 1.5, 10.5, 12.4
        if "Severely" in scenario: pd_multi, lgd_shift, cap_shortfall = 2.8, 25.0, 48.5
        if "Extreme" in scenario: pd_multi, lgd_shift, cap_shortfall = 5.5, 45.0, 155.2
        
        st.markdown("<br>", unsafe_allow_html=True)
        st.markdown(f'<div class="metric-title">PD Shift Multiplier</div><div class="metric-value" style="font-size:24px">{pd_multi}x</div>', unsafe_allow_html=True)
        st.markdown(f'<div class="metric-title">Capital Shortfall</div><div class="metric-value" style="font-size:24px; color:#ef4444">${cap_shortfall}M</div>', unsafe_allow_html=True)

    with scol2:
        st.markdown("### Portfolio Degradation Curve")
        x_labels = ['Performing', 'Watchlist', 'Substandard', 'Doubtful/Loss']
        base_data = [85, 10, 4, 1]
        stress_data = base_data
        if "Adverse" in scenario and "Severely" not in scenario: stress_data = [75, 15, 7, 3]
        if "Severely" in scenario: stress_data = [55, 25, 12, 8]
        if "Extreme" in scenario: stress_data = [30, 30, 25, 15]

        fig4 = go.Figure()
        fig4.add_trace(go.Scatter(x=x_labels, y=base_data, name='Baseline', fill='tozeroy', line=dict(color='#cbd5e1', width=3)))
        color_s = '#ef4444' if "Extreme" in scenario else '#0ea5e9'
        fig4.add_trace(go.Scatter(x=x_labels, y=stress_data, name='Stressed', fill='tozeroy', line=dict(color=color_s, width=3)))
        fig4.update_layout(paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)", height=350, font=dict(family="Outfit"))
        st.plotly_chart(fig4, use_container_width=True)
    st.markdown('</div>', unsafe_allow_html=True)

# ==========================================
# 6. MLOps Hub
# ==========================================
elif page == "⚙️ MLOps & Dev Center":
    st.title("MLOps & Model Monitoring")
    st.markdown('<div class="sub-header">Track ensemble stability, data drift, and performance in production.</div>', unsafe_allow_html=True)
    
    m1, m2, m3 = st.columns(3)
    m1.markdown('<div class="glass-card"><div class="metric-title">ROC-AUC Score</div><div class="metric-value">0.892</div></div>', unsafe_allow_html=True)
    m2.markdown('<div class="glass-card"><div class="metric-title">Gini Coefficient</div><div class="metric-value">0.784</div></div>', unsafe_allow_html=True)
    m3.markdown('<div class="glass-card"><div class="metric-title">KS Statistic</div><div class="metric-value">55.1</div></div>', unsafe_allow_html=True)
    
    c1, c2 = st.columns(2)
    with c1:
        st.markdown('<div class="glass-card">', unsafe_allow_html=True)
        st.markdown("### Global Feature Importance")
        features = ['Income Ratio', 'Loan Duration', 'Age', 'Sector', 'Collateral', 'Obligations', 'Province', 'DTI', 'Rate', 'Amount']
        importance = [0.22, 0.18, 0.15, 0.12, 0.10, 0.08, 0.06, 0.05, 0.03, 0.01]
        fig_fi = go.Figure(go.Bar(x=importance[::-1], y=features[::-1], orientation='h', marker_color='#38bdf8'))
        fig_fi.update_layout(paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)", height=320, font=dict(family="Outfit"))
        st.plotly_chart(fig_fi, use_container_width=True)
        st.markdown('</div>', unsafe_allow_html=True)
        
    with c2:
        st.markdown('<div class="glass-card">', unsafe_allow_html=True)
        st.markdown("### Production Data Drift (PSI)")
        fig_psi = go.Figure(go.Indicator(
            mode = "gauge+number", value = 0.08,
            title = {'text': "Population Stability Index", 'font': {'size': 14, 'color': '#64748b'}},
            gauge = {
                'axis': {'range': [0, 0.5], 'tickwidth': 0},
                'bar': {'color': "#0f172a"},
                'steps': [
                    {'range': [0, 0.1], 'color': "rgba(16, 185, 129, 0.2)"},
                    {'range': [0.1, 0.25], 'color': "rgba(245, 158, 11, 0.2)"},
                    {'range': [0.25, 0.5], 'color': "rgba(239, 68, 68, 0.2)"}
                ]
            }
        ))
        fig_psi.update_layout(height=280, margin=dict(t=30, b=0, l=20, r=20), paper_bgcolor="rgba(0,0,0,0)", font=dict(family="Outfit"))
        st.plotly_chart(fig_psi, use_container_width=True)
        st.success("✅ Distribution is stable. No retraining required.")
        st.markdown('</div>', unsafe_allow_html=True)

# ==========================================
# 7. Audit Logs
# ==========================================
elif page == "📜 Immutable Audit Logs":
    st.title("Session Audit Logs")
    st.markdown('<div class="sub-header">Live immutable trail of automated decisioning events.</div>', unsafe_allow_html=True)
    
    st.markdown('<div class="glass-card">', unsafe_allow_html=True)
    if len(st.session_state['audit_logs']) > 0:
        df_audit = pd.DataFrame(st.session_state['audit_logs'])
        st.dataframe(
            df_audit.style.map(lambda x: f'color: {"#10b981" if x == "APPROVE" else "#ef4444"}; font-weight: 800' if x in ['APPROVE', 'DECLINE', 'REVIEW'] else ''),
            use_container_width=True, hide_index=True, height=500
        )
    else:
        st.info("No assessments run yet in this session. Return to the Assessment Engine to begin.")
    st.markdown('</div>', unsafe_allow_html=True)
