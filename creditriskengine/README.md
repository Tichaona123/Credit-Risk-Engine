# CreditRiskEngine (Enterprise Banking Edition)

Developed by the **Inclusion Algorithm** team.

This is a true full-stack Enterprise Banking application, specifically decoupled to allow for Netlify frontend hosting while maintaining a powerful Python machine learning backend.

## Architecture

1. **Frontend (Netlify / GitHub Pages)**
   - Built with Vanilla JavaScript, HTML5, and Tailwind CSS.
   - Deploys as an ultra-fast Single Page Application (SPA).
   - Features: Real-time Credit Assessment, Plotly Visualizations, IFRS 9 Dashboards, and Kafka Event Logs.

2. **Backend (Render / Railway / AWS)**
   - Built with **FastAPI** (Python 3.11).
   - Houses the CatBoost/LGBM/XGBoost Ensemble.
   - Event-driven architecture utilizing a simulated **Apache Kafka** service (`kafka_service.py`) for Audit Trail logging.
   - Real-time **FPDF2** PDF official report generation.

---

## 🚀 How to Deploy

### Step 1: Deploy Frontend to Netlify
1. Push this entire repository to GitHub.
2. Go to [Netlify](https://www.netlify.com/) and click **Add New Site** > **Import an existing project**.
3. Select your GitHub repository.
4. Netlify will automatically detect the `netlify.toml` file.
5. Click **Deploy**. Your frontend is now live!

### Step 2: Deploy Backend to Render (Free)
1. Go to [Render](https://render.com/) and click **New Web Service**.
2. Connect the same GitHub repository.
3. Render will detect the `render.yaml` file (or just specify the start command: `uvicorn api:app --host 0.0.0.0 --port 10000`).
4. Click **Create Web Service**.

### Step 3: Connect Frontend to Backend
Once Render gives you a live URL (e.g., `https://creditrisk-api.onrender.com`), open `frontend/app.js` and change line 1:
```javascript
const API_BASE_URL = 'https://creditrisk-api.onrender.com/api';
```
Commit and push that change. Netlify will automatically rebuild and your system is completely live!

---

## 💻 Local Testing

1. **Start the FastAPI Backend**
```bash
pip install -r requirements.txt
uvicorn api:app --reload
```

2. **Open the Frontend**
Simply double-click the `frontend/index.html` file to open it in your browser. It will automatically connect to the local backend.
