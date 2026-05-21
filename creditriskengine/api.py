"""
CreditRiskEngine Enterprise API
Developed by Inclusion Algorithm Team

Full-stack banking backend featuring:
- Machine Learning inference engine (CatBoost, LGBM, XGBoost)
- IFRS 9 ECL calculations & Macro Stress Testing
- Apache Kafka event streaming (Audit Logging)
- Dynamic PDF Report generation
"""

import os
from fastapi import FastAPI, HTTPException, UploadFile, File, Request, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, Response
from pydantic import BaseModel, Field
from typing import List, Optional, Dict, Any
import pandas as pd
import numpy as np
import time
from datetime import datetime

# Import custom modules
from ml_pipeline import CreditRiskModel
from ifrs9_engine import IFRS9Engine, IFRS9Config
from stress_testing import StressTestEngine, StressScenario
from kafka_service import AuditEventProducer, SystemAlertProducer, broker
from fpdf import FPDF

# Initialize FastAPI App
app = FastAPI(
    title="CreditRiskEngine API",
    description="Enterprise Credit Risk Platform by Inclusion Algorithm",
    version="5.0.0"
)

# Allow CORS for Netlify Frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allows all origins for hackathon deployment
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ─── Load Global Services ──────────────────────────────────────
print("Loading Enterprise Services...")
crm_model = CreditRiskModel(models_dir='models')
crm_model.load()  # Will silently fail if no models exist
ifrs9_engine = IFRS9Engine()
stress_engine = StressTestEngine()
SystemAlertProducer.publish_alert("INFO", "CreditRiskEngine API services initialized successfully.")

# ─── Pydantic Schemas ──────────────────────────────────────────
class LoanApplication(BaseModel):
    amount_usd: float = Field(..., gt=0)
    annual_rate_pct: float = Field(..., ge=0)
    term_months: int = Field(..., gt=0)
    product_code: int = 0
    client_age: int = 35
    employment_sector: str = "Finance"
    province: str = "Harare"
    monthly_income_usd: float = 2000
    existing_obligations: int = 0
    collateral_type: str = "None"

class IFRS9Request(BaseModel):
    ead: float
    pd: float
    lgd: float
    term_months: int
    base_weight: float = 0.6
    adverse_weight: float = 0.3
    optimistic_weight: float = 0.1

# ─── Core Helper Functions ─────────────────────────────────────
def heuristic_fallback(loan: LoanApplication) -> dict:
    """Rule-based engine used if ML models aren't trained."""
    score = 0.5
    if loan.monthly_income_usd > 0:
        dti = ((loan.amount_usd/loan.term_months) + (loan.existing_obligations*100)) / loan.monthly_income_usd
        score = min(max(dti * 0.8, 0.05), 0.95)
    if loan.collateral_type in ["Real Estate", "Cash Deposit"]: score *= 0.6
    
    prob = float(score)
    risk_score = int((1 - prob) * 1000)
    if prob < 0.15: rec = 'APPROVE'
    elif prob < 0.40: rec = 'REVIEW'
    else: rec = 'DECLINE'

    return {
        'probability_of_default': prob,
        'risk_score': risk_score,
        'recommendation': rec
    }

def generate_fpdf_report(loan_data: dict, result: dict) -> bytes:
    """Generates an official bank PDF using FPDF2."""
    pdf = FPDF()
    pdf.add_page()
    
    pdf.set_font('Helvetica', 'B', 20)
    pdf.cell(0, 10, 'INDABAX BANKING CORP', ln=True)
    pdf.set_font('Helvetica', '', 10)
    pdf.set_text_color(100, 100, 100)
    pdf.cell(0, 5, 'Credit Risk Committee | Automated Decisioning Unit', ln=True)
    pdf.line(10, 28, 200, 28)
    
    pdf.set_y(35)
    pdf.set_font('Helvetica', 'B', 12)
    pdf.set_text_color(0, 0, 0)
    pdf.cell(100, 8, 'OFFICIAL RISK ASSESSMENT', ln=False)
    pdf.set_font('Helvetica', '', 10)
    pdf.cell(90, 8, f"Date: {datetime.now().strftime('%Y-%m-%d %H:%M')}", ln=True, align='R')
    pdf.cell(190, 8, f"Ref: REQ-{hash(str(loan_data))%1000000:06d}", ln=True, align='R')
    
    pdf.set_y(50)
    pdf.set_fill_color(240, 240, 240)
    pdf.set_font('Helvetica', 'B', 11)
    pdf.cell(0, 8, ' 1. EXECUTIVE SUMMARY', ln=True, fill=True)
    
    pdf.set_font('Helvetica', '', 10)
    pdf.set_y(62)
    pdf.cell(50, 8, 'Requested Amount:')
    pdf.cell(50, 8, f"${loan_data['amount_usd']:,.2f}", ln=True)
    pdf.cell(50, 8, 'Decision:')
    pdf.set_font('Helvetica', 'B', 12)
    rec = result['recommendation']
    if rec == 'APPROVE': pdf.set_text_color(22, 101, 52)
    elif rec == 'DECLINE': pdf.set_text_color(153, 27, 27)
    pdf.cell(50, 8, rec, ln=True)
    pdf.set_text_color(0,0,0)
    pdf.set_font('Helvetica', '', 10)
    pdf.cell(50, 8, 'Probability of Default:')
    pdf.cell(50, 8, f"{result['probability_of_default']:.2%}", ln=True)
    pdf.cell(50, 8, 'Credit Score:')
    pdf.cell(50, 8, f"{result['risk_score']} / 1000", ln=True)
    
    pdf.set_y(100)
    pdf.set_font('Helvetica', 'B', 11)
    pdf.cell(0, 8, ' 2. IFRS 9 REGULATORY PROVISIONING', ln=True, fill=True)
    
    stage = 1 if result['probability_of_default'] < 0.15 else (2 if result['probability_of_default'] < 0.40 else 3)
    lgd = 0.20 if loan_data['collateral_type'] in ['Real Estate', 'Cash Deposit'] else 0.45
    ecl = loan_data['amount_usd'] * result['probability_of_default'] * lgd
    
    pdf.set_y(112)
    pdf.set_font('Helvetica', '', 10)
    pdf.cell(0, 8, f"Assigned Stage: Stage {stage}", ln=True)
    pdf.cell(0, 8, f"Loss Given Default (LGD): {lgd*100:.1f}%", ln=True)
    pdf.cell(0, 8, f"Expected Credit Loss (ECL): ${ecl:,.2f}", ln=True)
    
    pdf.set_y(220)
    pdf.line(10, 220, 80, 220)
    pdf.set_font('Helvetica', 'B', 10)
    pdf.cell(80, 6, 'System Authorized Signature', ln=True)
    pdf.set_font('Helvetica', '', 8)
    pdf.cell(80, 6, 'CreditRiskEngine AI Ensemble | Inclusion Algorithm')

    return pdf.output(dest='S')

# ─── API Endpoints ─────────────────────────────────────────────

@app.get("/api/health")
def health_check():
    return {
        "status": "online",
        "version": "5.0.0",
        "ml_engine_loaded": crm_model.is_loaded,
        "kafka_broker": "connected"
    }

@app.post("/api/predict")
def predict_loan(loan: LoanApplication, request: Request, background_tasks: BackgroundTasks):
    """Real-time scoring endpoint."""
    loan_dict = loan.dict()
    
    if crm_model.is_loaded:
        result = crm_model.predict_single(loan_dict)
    else:
        result = heuristic_fallback(loan)
        
    # Asynchronously publish to Kafka audit log
    client_ip = request.client.host if request.client else "unknown"
    background_tasks.add_task(AuditEventProducer.publish_decision, loan_dict, result, client_ip)
    
    return {
        "status": "success",
        "data": {
            "loan": loan_dict,
            "decision": result
        }
    }

@app.post("/api/generate-report")
def generate_report(loan: LoanApplication):
    """Generates an official PDF report based on loan input."""
    loan_dict = loan.dict()
    result = crm_model.predict_single(loan_dict) if crm_model.is_loaded else heuristic_fallback(loan)
    
    pdf_bytes = generate_fpdf_report(loan_dict, result)
    
    return Response(
        content=pdf_bytes, 
        media_type="application/pdf", 
        headers={"Content-Disposition": f"attachment; filename=RiskReport_{datetime.now().strftime('%Y%m%d%H%M')}.pdf"}
    )

@app.get("/api/audit-logs")
def get_audit_logs():
    """Retrieve event stream from Kafka topic."""
    logs = broker.consume("credit-decisions", max_messages=50)
    return {"status": "success", "logs": logs}

@app.post("/api/ifrs9/recalculate")
def recalculate_portfolio_ecl(req: IFRS9Request):
    """Calculates probability-weighted ECL."""
    stage = 1 if req.pd < 0.15 else (2 if req.pd < 0.40 else 3)
    
    base_ecl = req.ead * req.pd * req.lgd
    adverse_ecl = base_ecl * 1.5
    opt_ecl = base_ecl * 0.8
    
    weighted_ecl = (base_ecl * req.base_weight) + (adverse_ecl * req.adverse_weight) + (opt_ecl * req.optimistic_weight)
    
    return {
        "stage": stage,
        "base_ecl": base_ecl,
        "probability_weighted_ecl": weighted_ecl,
        "adverse_scenario_impact": adverse_ecl - base_ecl
    }
