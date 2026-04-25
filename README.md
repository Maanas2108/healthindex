# HealthSensex AI
### Kraken'X 2026 · Team GridMind

Real-Time National Health Intelligence & Fraud Detection System  
Built on Python (Flask) + Scikit-learn (IsolationForest) + Vanilla JS

---

## Setup & Run (2 minutes)

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Start the server
python app.py

# 3. Open in browser
http://localhost:5000
```

---

## Three Pages

| Page | URL | Access |
|------|-----|--------|
| ABHA Data Entry Portal | `/` → click ABHA Portal | Public (hospitals) |
| Fraud Detection Dashboard | `/` → click Fraud Monitor | Login required |
| Public HealthSensex | `/` → click HealthSensex | Public |

### Fraud Dashboard Login
- **Username:** `NHA_AUDITOR`
- **Password:** `HealthSensex@2026`

---

## How the ML Works

1. **On startup:** ~42,000 synthetic hospital records are generated for 2024 (12 hospitals, 60 doctors, 8 diseases)
2. **Fraud injection:** HOSP-007 Dengue billing ×10 in June, DR-0042 signed 30+ death certs in October
3. **Feature engineering:** 7-day rolling billing average, deviation ratio, disease-level z-score, daily death cert count per doctor
4. **IsolationForest** (200 trees, 3% contamination): trained on the historical data, flags statistical impossibilities
5. **Anomaly score:** Raw IF score normalized to 0–100
6. **New submissions:** Scored in real-time against the trained model using the same feature pipeline

## HealthSensex Index Formula
```
score = 100 - (anomaly_rate × 1500) - spike_penalty - death_cert_penalty
clamped to [0, 100]
```

---

## Project Structure
```
healthsensex/
├── app.py              # Flask backend + ML
├── requirements.txt    # Dependencies
├── README.md
└── templates/
    └── index.html      # Full 3-page frontend
```
