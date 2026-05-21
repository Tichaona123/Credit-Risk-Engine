const API_BASE_URL = 'http://localhost:8000/api';
let currentLoanData = null;
let currentECLData = null;

// ==========================================
// Tab Navigation
// ==========================================
function switchTab(tabId) {
    ['assessment', 'analytics', 'ifrs9', 'stress', 'audit'].forEach(t => {
        const el = document.getElementById(`tab-${t}`);
        if(el) el.style.display = 'none';
        
        const nav = document.getElementById(`nav-${t}`);
        if(nav) nav.classList.remove('nav-active', 'font-semibold', 'text-blue-600', 'border-r-2', 'border-blue-600');
    });
    
    const activeTab = document.getElementById(`tab-${tabId}`);
    if(activeTab) activeTab.style.display = 'block';
    
    const activeNav = document.getElementById(`nav-${tabId}`);
    if(activeNav) activeNav.classList.add('nav-active', 'font-semibold', 'text-blue-600', 'border-r-2', 'border-blue-600');

    if(tabId === 'audit') loadAuditLogs();
    if(tabId === 'ifrs9') initIFRS9();
    if(tabId === 'stress') initStressTest();
    if(tabId === 'analytics') initAnalytics();
}

// ==========================================
// Core Utilities
// ==========================================
async function checkHealth() {
    try {
        const res = await fetch(`${API_BASE_URL}/health`);
        const data = await res.json();
        document.getElementById('api-status').innerText = 'Connected (v5.0)';
        document.getElementById('api-status').previousElementSibling.className = 'w-2 h-2 rounded-full bg-green-500 shadow-[0_0_8px_rgba(34,197,94,0.8)]';
    } catch (e) {
        document.getElementById('api-status').innerText = 'Backend Offline';
        document.getElementById('api-status').previousElementSibling.className = 'w-2 h-2 rounded-full bg-red-500 shadow-[0_0_8px_rgba(239,68,68,0.8)] animate-pulse';
    }
}

// ==========================================
// 1. Underwriting Module
// ==========================================
function renderGauge(prob) {
    let color = "#16a34a"; let text = "LOW RISK";
    if (prob >= 0.15) { color = "#2563eb"; text = "MODERATE"; }
    if (prob >= 0.40) { color = "#d97706"; text = "ELEVATED"; }
    if (prob >= 0.70) { color = "#dc2626"; text = "HIGH RISK"; }

    const data = [{
        type: "indicator",
        mode: "gauge+number",
        value: prob * 100,
        number: { suffix: "%", font: { size: 30, color: "#0f172a", family: 'Inter' } },
        title: { text: `<span style='color:#64748b;font-size:10px;font-weight:bold;letter-spacing:1px'>DEFAULT PROBABILITY</span><br><span style='color:${color};font-size:14px;font-weight:bold'>${text}</span>` },
        gauge: {
            axis: { range: [0, 100], tickwidth: 1, tickcolor: "#e2e8f0" },
            bar: { color: color, thickness: 0.75 },
            bgcolor: "#f1f5f9",
            borderwidth: 0,
            threshold: { line: { color: "#0f172a", width: 2 }, thickness: 0.75, value: prob * 100 }
        }
    }];
    Plotly.newPlot('gauge-chart', data, { margin: { t: 40, b: 20, l: 30, r: 30 } }, {displayModeBar: false});
}

document.getElementById('loan-form')?.addEventListener('submit', async (e) => {
    e.preventDefault();
    const btn = document.getElementById('btn-score');
    btn.innerHTML = '<span class="animate-pulse">Running Ensemble...</span>';
    btn.disabled = true;

    currentLoanData = {
        amount_usd: parseFloat(document.getElementById('f_amount').value),
        annual_rate_pct: parseFloat(document.getElementById('f_rate').value),
        term_months: parseInt(document.getElementById('f_term').value),
        monthly_income_usd: parseFloat(document.getElementById('f_income').value),
        existing_obligations: parseInt(document.getElementById('f_obligations').value),
        employment_sector: document.getElementById('f_sector').value,
        collateral_type: document.getElementById('f_collateral').value,
        product_code: 1, client_age: 35, province: "Harare"
    };

    try {
        const res = await fetch(`${API_BASE_URL}/predict`, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify(currentLoanData)
        });
        const data = await res.json();
        const decision = data.data.decision;

        document.getElementById('results-area').classList.remove('hidden');
        document.getElementById('res_score').innerText = decision.risk_score;
        
        const recEl = document.getElementById('res_rec');
        recEl.innerText = decision.recommendation;
        recEl.className = `text-3xl font-black ${decision.recommendation === 'APPROVE' ? 'text-green-600' : 'text-red-600'}`;
        
        // IFRS 9 Assignment
        const pd = decision.probability_of_default;
        let stage = 1; let stageColor = "bg-green-100 text-green-800";
        if(pd > 0.15) { stage = 2; stageColor = "bg-yellow-100 text-yellow-800"; }
        if(pd > 0.40) { stage = 3; stageColor = "bg-red-100 text-red-800"; }
        
        const stageEl = document.getElementById('res_stage');
        stageEl.innerText = `Stage ${stage}`;
        stageEl.className = `px-2 py-1 rounded font-bold text-sm ${stageColor}`;

        const lgd = currentLoanData.collateral_type === 'None' ? 0.45 : 0.20;
        const ecl = currentLoanData.amount_usd * pd * lgd;
        document.getElementById('res_ecl').innerText = `$${ecl.toLocaleString(undefined, {minimumFractionDigits: 2, maximumFractionDigits: 2})}`;

        renderGauge(pd);
    } catch (e) {
        alert("API Error: Ensure the FastAPI backend is running.");
    } finally {
        btn.innerHTML = 'Execute AI Ensemble';
        btn.disabled = false;
    }
});

async function downloadPDF() {
    if(!currentLoanData) return;
    try {
        const res = await fetch(`${API_BASE_URL}/generate-report`, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify(currentLoanData)
        });
        const blob = await res.blob();
        const url = window.URL.createObjectURL(blob);
        const a = document.createElement('a');
        a.href = url;
        a.download = `Indabax_Risk_Dossier_${Date.now()}.pdf`;
        document.body.appendChild(a);
        a.click();
        window.URL.revokeObjectURL(url);
    } catch(e) { alert("Error generating PDF."); }
}

// ==========================================
// 1.5 Portfolio Analytics
// ==========================================
let analyticsRendered = false;
function initAnalytics() {
    if (analyticsRendered) return;
    
    // 1. Vintage Analysis (Cumulative Default Curves)
    const trace2023 = { x: [1,2,3,4,5,6,7,8,9,10,11,12], y: [0.1, 0.4, 0.8, 1.2, 1.5, 1.9, 2.1, 2.3, 2.4, 2.5, 2.6, 2.6], name: '2023 Vintage', type: 'scatter', line: {color: '#94a3b8'} };
    const trace2024 = { x: [1,2,3,4,5,6,7,8], y: [0.2, 0.6, 1.1, 1.6, 2.2, 2.8, 3.1, 3.4], name: '2024 Vintage', type: 'scatter', line: {color: '#2563eb'} };
    Plotly.newPlot('chart-vintage', [trace2023, trace2024], {
        margin: {t:20, b:40, l:40, r:10},
        yaxis: {ticksuffix: '%', title: 'Cumulative Default Rate'},
        xaxis: {title: 'Months on Book'},
        legend: {orientation: 'h', y: 1.1}
    }, {displayModeBar: false});

    // 2. Sector Exposure
    const dataSector = [{
        values: [45, 25, 15, 10, 5],
        labels: ['Agriculture', 'Retail SME', 'Manufacturing', 'Services', 'Real Estate'],
        type: 'pie',
        hole: .6,
        marker: {colors: ['#2563eb', '#3b82f6', '#60a5fa', '#93c5fd', '#bfdbfe']}
    }];
    Plotly.newPlot('chart-sector', dataSector, {
        margin: {t:10, b:10, l:10, r:10},
        showlegend: true, legend: {orientation: 'v'}
    }, {displayModeBar: false});

    // 3. Score vs Default Rate (Bar + Line combo)
    const traceScores = { x: ['300-500', '501-600', '601-700', '701-800', '801-900'], y: [15, 30, 35, 15, 5], name: '% of Portfolio', type: 'bar', marker: {color: '#e2e8f0'}, yaxis: 'y' };
    const traceDefaults = { x: ['300-500', '501-600', '601-700', '701-800', '801-900'], y: [12.4, 6.2, 2.1, 0.5, 0.1], name: 'Default Rate', type: 'scatter', line: {color: '#dc2626', width: 3}, yaxis: 'y2' };
    Plotly.newPlot('chart-score-dist', [traceScores, traceDefaults], {
        margin: {t:20, b:40, l:40, r:40},
        yaxis: {title: '% of Portfolio', range: [0, 50]},
        yaxis2: {title: 'Default Rate (%)', overlaying: 'y', side: 'right', range: [0, 15], ticksuffix: '%'},
        legend: {orientation: 'h', y: 1.2}
    }, {displayModeBar: false});

    // 4. Loan Purpose
    const dataPurpose = [{
        x: [42, 28, 15, 10, 5],
        y: ['Working Capital', 'Asset Purchase', 'Expansion', 'Refinancing', 'Other'],
        type: 'bar', orientation: 'h',
        marker: {color: '#cbd5e1'}
    }];
    Plotly.newPlot('chart-purpose', dataPurpose, {
        margin: {t:10, b:40, l:100, r:10},
        xaxis: {ticksuffix: '%'}
    }, {displayModeBar: false});

    analyticsRendered = true;
}


// ==========================================
// 2. IFRS 9 Macro Configurator
// ==========================================
function updateScenarios() {
    let base = parseInt(document.getElementById('w_base').value);
    let adv = parseInt(document.getElementById('w_adv').value);
    
    if (base + adv > 100) { adv = 100 - base; document.getElementById('w_adv').value = adv; }
    let opt = 100 - (base + adv);
    
    document.getElementById('lbl_w_base').innerText = `${base}%`;
    document.getElementById('lbl_w_adv').innerText = `${adv}%`;
    document.getElementById('lbl_w_opt').innerText = `${opt}%`;
}

async function calculateIFRS9() {
    const ead = parseFloat(document.getElementById('portfolio_ead').value) || 150000000;
    const baseW = parseInt(document.getElementById('w_base').value) / 100;
    const advW = parseInt(document.getElementById('w_adv').value) / 100;
    const optW = 1 - (baseW + advW);

    try {
        const res = await fetch(`${API_BASE_URL}/ifrs9/recalculate`, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({
                ead: ead,
                pd: 0.05, // Assumed portfolio average PD for demo
                lgd: 0.45,
                term_months: 12,
                base_weight: baseW,
                adverse_weight: advW,
                optimistic_weight: optW
            })
        });
        const data = await res.json();
        
        // Render Bar Chart
        const trace1 = {
            x: ['Base ECL', 'Adverse ECL', 'Optimistic ECL', 'Weighted ECL'],
            y: [data.base_ecl, data.base_ecl * 1.5, data.base_ecl * 0.8, data.probability_weighted_ecl],
            type: 'bar',
            marker: { color: ['#cbd5e1', '#fca5a5', '#bbf7d0', '#2563eb'] }
        };
        const layout = {
            margin: { t: 10, b: 30, l: 40, r: 10 },
            yaxis: { tickprefix: '$', tickformat: '.2s' },
            paper_bgcolor: 'transparent',
            plot_bgcolor: 'transparent'
        };
        Plotly.newPlot('ecl-bar-chart', [trace1], layout, {displayModeBar: false});

    } catch (e) { console.error("Failed to calculate IFRS 9"); }
}

function initIFRS9() {
    calculateIFRS9(); // Initial render
}

// ==========================================
// 3. Macro Stress Testing
// ==========================================
function initStressTest() {
    runStressTest('baseline');
}

function runStressTest(scenario) {
    // Reset styling
    ['baseline', 'adverse', 'severe', 'extreme'].forEach(s => {
        document.getElementById(`scen-${s}`).className = 'flex items-center p-3 border border-gray-200 rounded cursor-pointer hover:bg-gray-50 transition';
    });
    
    const activeEl = document.getElementById(`scen-${scenario}`);
    if(scenario === 'baseline') activeEl.className = 'flex items-center p-3 border rounded cursor-pointer transition border-blue-500 bg-blue-50';
    else if(scenario === 'extreme') activeEl.className = 'flex items-center p-3 border rounded cursor-pointer transition border-red-500 bg-red-50';
    else activeEl.className = 'flex items-center p-3 border rounded cursor-pointer transition border-orange-500 bg-orange-50';

    // Mock Impact Logic based on CCAR
    let pdMulti = 1.0; let lgdShift = 0; let capital = 0;
    
    if(scenario === 'adverse') { pdMulti = 1.5; lgdShift = 10.5; capital = 2.4; }
    if(scenario === 'severe') { pdMulti = 2.8; lgdShift = 25.0; capital = 18.5; }
    if(scenario === 'extreme') { pdMulti = 5.5; lgdShift = 45.0; capital = 55.2; }

    document.getElementById('stress-pd').innerText = `${pdMulti}x`;
    document.getElementById('stress-lgd').innerText = `+${lgdShift}%`;
    document.getElementById('stress-cap').innerText = `$${capital}M`;
    
    if(scenario === 'extreme') {
        document.getElementById('stress-pd').className = "text-2xl font-bold text-red-600 mt-1";
        document.getElementById('stress-lgd').className = "text-2xl font-bold text-red-600 mt-1";
    } else {
        document.getElementById('stress-pd').className = "text-2xl font-bold text-gray-900 mt-1";
        document.getElementById('stress-lgd').className = "text-2xl font-bold text-gray-900 mt-1";
    }

    // Render Stress Chart
    const xLabels = ['Performing', 'Watchlist', 'Substandard', 'Doubtful/Loss'];
    let baseData = [85, 10, 4, 1];
    let stressData = baseData;
    
    if(scenario === 'adverse') stressData = [75, 15, 7, 3];
    if(scenario === 'severe') stressData = [55, 25, 12, 8];
    if(scenario === 'extreme') stressData = [30, 30, 25, 15];

    const trace1 = { x: xLabels, y: baseData, name: 'Baseline', type: 'scatter', fill: 'tozeroy', line: {color: '#cbd5e1'} };
    const trace2 = { x: xLabels, y: stressData, name: 'Stressed', type: 'scatter', fill: 'tozeroy', line: {color: scenario === 'extreme' ? '#dc2626' : '#2563eb'} };

    const layout = {
        title: 'Portfolio Quality Degradation curve',
        margin: { t: 30, b: 30, l: 40, r: 10 },
        yaxis: { ticksuffix: '%', range: [0, 100] },
        paper_bgcolor: 'transparent',
        plot_bgcolor: 'transparent',
        legend: { orientation: 'h', y: 1.1 }
    };
    Plotly.newPlot('stress-dist-chart', [trace1, trace2], layout, {displayModeBar: false});
}

// ==========================================
// 4. Kafka Audit Logs
// ==========================================
async function loadAuditLogs() {
    try {
        const res = await fetch(`${API_BASE_URL}/audit-logs`);
        const data = await res.json();
        
        const tbody = document.getElementById('audit-table-body');
        if(!data.logs || data.logs.length === 0) {
            tbody.innerHTML = '<tr><td colspan="5" class="px-6 py-8 text-center text-sm text-gray-500">No events in stream yet. Run an assessment!</td></tr>';
            return;
        }
        
        tbody.innerHTML = '';
        data.logs.forEach(log => {
            const date = new Date(log.timestamp).toLocaleTimeString();
            const payload = log.payload;
            const rowClass = payload.decision === 'APPROVE' ? 'bg-green-50 text-green-800 border-green-200' : 'bg-red-50 text-red-800 border-red-200';
            
            tbody.   innerHTML += `
                <tr class="hover:bg-slate-50 transition cursor-pointer">
                    <td class="px-6 py-3 whitespace-nowrap text-xs text-gray-500 font-mono">${date}</td>
                    <td class="px-6 py-3 whitespace-nowrap text-xs font-mono text-gray-900 bg-slate-100 rounded">OFFSET-${log.offset}</td>
                    <td class="px-6 py-3 whitespace-nowrap text-xs font-semibold text-gray-900">$${payload.loan_amount.toLocaleString()}</td>
                    <td class="px-6 py-3 whitespace-nowrap text-xs text-gray-900">${(payload.probability_of_default * 100).toFixed(2)}%</td>
                    <td class="px-6 py-3 whitespace-nowrap text-xs font-semibold">
                        <span class="px-2 py-0.5 rounded border ${rowClass} uppercase tracking-wider">${payload.decision}</span>
                    </td>
                </tr>
            `;
        });
    } catch(e) {
        document.getElementById('audit-table-body').innerHTML = '<tr><td colspan="5" class="px-6 py-8 text-center text-sm text-red-500">Failed to connect to Kafka Stream API.</td></tr>';
    }
}

// Initialize System
checkHealth();
switchTab('assessment');
