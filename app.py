from flask import Flask, jsonify, request, render_template, session
from sklearn.ensemble import IsolationForest
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import warnings, json
warnings.filterwarnings("ignore")

app = Flask(__name__)
app.secret_key = "healthsensex_secret_2026"

# ─────────────────────────────────────────────────────────────
# GLOBAL STATE — shared across requests
# ─────────────────────────────────────────────────────────────
_submitted_records = []   # records submitted via ABHA form
_model = None
_df_historical = None
_df_scored = None
_health_index = None

DISEASE_ICD = {
    "Dengue": "A90", "Tuberculosis": "A15", "Malaria": "B50",
    "Typhoid": "A01", "COVID-19": "U07.1", "Cardiac": "I21",
    "Orthopedic": "M16", "Cancer": "C80"
}

DISEASE_BILLING_NORMS = {
    "Dengue": 18000, "Tuberculosis": 25000, "Malaria": 15000,
    "Typhoid": 12000, "COVID-19": 35000, "Cardiac": 80000,
    "Orthopedic": 65000, "Cancer": 1_20_000
}

# ─────────────────────────────────────────────────────────────
# DATA GENERATION
# ─────────────────────────────────────────────────────────────
def generate_historical_data(seed=42):
    np.random.seed(seed)
    N_HOSPITALS = 12
    N_DOCTORS = 60
    START = datetime(2024, 1, 1)
    END = datetime(2024, 12, 31)
    DISEASES = list(DISEASE_ICD.keys())

    dates = pd.date_range(START, END, freq="D")
    hospitals = [f"HOSP-{str(i).zfill(3)}" for i in range(1, N_HOSPITALS+1)]
    doctors   = [f"DR-{str(i).zfill(4)}" for i in range(1, N_DOCTORS+1)]

    records = []
    for date in dates:
        for hosp in hospitals:
            for _ in range(np.random.randint(3, 9)):
                doc = np.random.choice(doctors)
                disease = np.random.choice(DISEASES)
                norm = DISEASE_BILLING_NORMS[disease]
                billing = np.random.lognormal(mean=np.log(norm), sigma=0.5)
                billing = float(np.clip(billing, 5000, 2_00_000))
                death_certs = int(np.random.choice([0,1,2], p=[0.70,0.22,0.08]))
                records.append({
                    "date": date, "hospital_id": hosp, "doctor_id": doc,
                    "disease": disease, "billing": round(billing, 2),
                    "death_certs": death_certs, "source": "historical"
                })

    df = pd.DataFrame(records)

    # Inject fraud
    mask1 = (df.hospital_id=="HOSP-007") & (df.disease=="Dengue") & \
            (df.date >= datetime(2024,6,1)) & (df.date <= datetime(2024,6,14))
    df.loc[mask1, "billing"] *= 10

    mask2 = (df.doctor_id=="DR-0042") & \
            (df.date >= datetime(2024,10,15)) & (df.date <= datetime(2024,10,16))
    df.loc[mask2, "death_certs"] = np.random.randint(28, 36, size=mask2.sum())

    return df


def engineer_features(df):
    df = df.sort_values(["hospital_id","date"]).reset_index(drop=True)
    df["billing_7d_avg"] = (
        df.groupby("hospital_id")["billing"]
          .transform(lambda x: x.rolling(7, min_periods=1).mean())
    )
    df["billing_deviation"] = df["billing"] / (df["billing_7d_avg"] + 1e-6)

    daily_dc = (df.groupby(["doctor_id","date"])["death_certs"]
                  .sum().reset_index()
                  .rename(columns={"death_certs":"doctor_daily_dc"}))
    df = df.merge(daily_dc, on=["doctor_id","date"], how="left")

    stats = df.groupby("disease")["billing"].agg(["mean","std"]).reset_index()
    df = df.merge(stats, on="disease", how="left")
    df["billing_zscore"] = (df["billing"] - df["mean"]) / (df["std"] + 1e-6)
    df.drop(columns=["mean","std"], inplace=True)
    return df


def train_model(df, contamination=0.03):
    features = ["billing","billing_7d_avg","billing_deviation","doctor_daily_dc","billing_zscore"]
    X = df[features].fillna(0)
    model = IsolationForest(n_estimators=200, contamination=contamination, random_state=42, n_jobs=-1)
    model.fit(X)

    df = df.copy()
    df["anomaly_raw"] = -model.score_samples(X)
    df["flagged"] = model.predict(X) == -1
    mn, mx = df.anomaly_raw.min(), df.anomaly_raw.max()
    df["anomaly_score"] = ((df.anomaly_raw - mn) / (mx - mn + 1e-9) * 100).round(1)

    # Fraud reason
    conds = [df.doctor_daily_dc >= 10, df.billing_deviation >= 5, df.billing_zscore >= 3]
    choices = ["Death Cert Velocity", "Billing Spike", "Statistical Outlier"]
    df["fraud_reason"] = np.select(conds, choices, default="Composite Anomaly")

    # Health index
    rate = df.flagged.mean()
    score = 100 - (rate * 1500)
    if df.billing_deviation.max() > 5: score -= 10
    if df.doctor_daily_dc.max() > 20:  score -= 5
    health_index = round(max(0, min(100, score)), 1)

    return model, df, health_index


def score_new_record(record, model, df_historical):
    """Score a single new submission against the trained model."""
    disease = record.get("disease", "Dengue")
    billing = float(record.get("billing", 0))
    death_certs = int(record.get("death_certs", 0))
    hospital_id = record.get("hospital_id", "")

    # Rolling avg from historical for this hospital
    hosp_hist = df_historical[df_historical.hospital_id == hospital_id]["billing"]
    billing_7d_avg = float(hosp_hist.tail(7).mean()) if len(hosp_hist) > 0 else billing
    billing_deviation = billing / (billing_7d_avg + 1e-6)

    disease_mean = DISEASE_BILLING_NORMS.get(disease, 30000)
    disease_std  = disease_mean * 0.4
    billing_zscore = (billing - disease_mean) / (disease_std + 1e-6)

    X_new = pd.DataFrame([{
        "billing": billing,
        "billing_7d_avg": billing_7d_avg,
        "billing_deviation": billing_deviation,
        "doctor_daily_dc": death_certs,
        "billing_zscore": billing_zscore
    }])

    raw_score = float(-model.score_samples(X_new)[0])
    flagged = model.predict(X_new)[0] == -1

    # Normalize against historical range
    hist_min = float(-model.score_samples(
        df_historical[["billing","billing_7d_avg","billing_deviation","doctor_daily_dc","billing_zscore"]].fillna(0)
    ).min())
    hist_max = float(-model.score_samples(
        df_historical[["billing","billing_7d_avg","billing_deviation","doctor_daily_dc","billing_zscore"]].fillna(0)
    ).max())

    anomaly_score = round((raw_score - hist_min) / (hist_max - hist_min + 1e-9) * 100, 1)
    anomaly_score = max(0, min(100, anomaly_score))

    if death_certs >= 10:
        fraud_reason = "Death Cert Velocity"
    elif billing_deviation >= 5:
        fraud_reason = "Billing Spike"
    elif billing_zscore >= 3:
        fraud_reason = "Statistical Outlier"
    else:
        fraud_reason = "Composite Anomaly" if flagged else "None"

    return {
        "anomaly_score": anomaly_score,
        "flagged": bool(flagged),
        "fraud_reason": fraud_reason,
        "billing_deviation": round(billing_deviation, 2),
        "billing_zscore": round(billing_zscore, 2),
        "billing_7d_avg": round(billing_7d_avg, 2)
    }


# ─────────────────────────────────────────────────────────────
# BOOTSTRAP on startup
# ─────────────────────────────────────────────────────────────
def bootstrap():
    global _model, _df_historical, _df_scored, _health_index
    raw = generate_historical_data()
    eng = engineer_features(raw)
    _model, _df_scored, _health_index = train_model(eng)
    _df_historical = eng
    print(f"[HealthSensex] Bootstrap done. Records: {len(_df_scored):,} | Health Index: {_health_index}")

bootstrap()


# ─────────────────────────────────────────────────────────────
# ROUTES
# ─────────────────────────────────────────────────────────────

@app.route("/")
def index():
    return render_template("index.html")


# ── ABHA FORM SUBMISSION ──────────────────────────────────────
@app.route("/api/submit", methods=["POST"])
def submit_record():
    data = request.json
    required = ["abha_id","hospital_id","doctor_id","disease","billing"]
    for f in required:
        if not data.get(f):
            return jsonify({"error": f"Missing field: {f}"}), 400

    result = score_new_record(data, _model, _df_historical)
    record = {**data, **result, "submitted_at": datetime.now().isoformat()}
    _submitted_records.append(record)

    return jsonify({
        "success": True,
        "anomaly_score": result["anomaly_score"],
        "flagged": result["flagged"],
        "fraud_reason": result["fraud_reason"],
        "message": "Record submitted and analyzed by HealthSensex AI."
    })


# ── RISK PRE-CHECK (live, before full submit) ─────────────────
@app.route("/api/precheck", methods=["POST"])
def precheck():
    data = request.json
    disease = data.get("disease","")
    try:
        billing = float(data.get("billing", 0))
    except:
        return jsonify({"risk":"unknown","detail":""})
    death_certs = int(data.get("death_certs", 0))

    norm = DISEASE_BILLING_NORMS.get(disease, 30000)
    ratio = billing / (norm + 1e-6)

    if death_certs >= 10 or ratio > 5:
        risk = "HIGH"
        detail = f"Billing is {ratio:.1f}× the disease norm. This will likely be flagged."
    elif ratio > 2.5 or death_certs >= 5:
        risk = "MEDIUM"
        detail = f"Billing is {ratio:.1f}× above average for {disease}. May trigger review."
    else:
        risk = "LOW"
        detail = f"Within normal range for {disease} (norm: ₹{norm:,.0f})."

    return jsonify({"risk": risk, "detail": detail, "ratio": round(ratio, 2)})


# ── FRAUD DASHBOARD DATA ──────────────────────────────────────
@app.route("/api/fraud/summary")
def fraud_summary():
    df = _df_scored
    flagged = df[df.flagged]
    total_billing = float(df.billing.sum())
    flagged_billing = float(flagged.billing.sum())

    return jsonify({
        "health_index": _health_index,
        "total_records": len(df),
        "flagged_count": int(flagged.flagged.sum()),
        "flagged_pct": round(float(flagged.flagged.sum()) / len(df) * 100, 2),
        "total_billing": total_billing,
        "flagged_billing": flagged_billing,
        "submitted_count": len(_submitted_records),
        "submitted_flagged": sum(1 for r in _submitted_records if r.get("flagged"))
    })


@app.route("/api/fraud/flagged")
def fraud_flagged():
    df = _df_scored
    page = int(request.args.get("page", 1))
    per_page = int(request.args.get("per_page", 30))
    hospital = request.args.get("hospital", "")
    disease  = request.args.get("disease", "")
    sort_by  = request.args.get("sort", "anomaly_score")
    sort_dir = request.args.get("dir", "desc")

    flagged = df[df.flagged].copy()

    # Include newly submitted flagged records
    if _submitted_records:
        new_flagged = [r for r in _submitted_records if r.get("flagged")]
        if new_flagged:
            new_df = pd.DataFrame(new_flagged)
            new_df["date"] = pd.to_datetime(new_df.get("submitted_at", pd.Timestamp.now()))
            for col in ["billing_7d_avg","billing_deviation","billing_zscore","doctor_daily_dc"]:
                if col not in new_df.columns:
                    new_df[col] = 0
            flagged = pd.concat([flagged, new_df], ignore_index=True)

    if hospital:
        flagged = flagged[flagged.hospital_id == hospital]
    if disease:
        flagged = flagged[flagged.disease == disease]

    asc = sort_dir == "asc"
    if sort_by in flagged.columns:
        flagged = flagged.sort_values(sort_by, ascending=asc)

    total = len(flagged)
    start = (page-1)*per_page
    page_df = flagged.iloc[start:start+per_page]

    records = []
    for _, row in page_df.iterrows():
        records.append({
            "date": row["date"].strftime("%d %b %Y") if hasattr(row["date"],"strftime") else str(row.get("date","")),
            "hospital_id": row.get("hospital_id",""),
            "doctor_id": row.get("doctor_id",""),
            "disease": row.get("disease",""),
            "billing": round(float(row.get("billing",0)), 2),
            "billing_deviation": round(float(row.get("billing_deviation",0)), 2),
            "doctor_daily_dc": int(row.get("doctor_daily_dc",0)),
            "anomaly_score": round(float(row.get("anomaly_score",0)), 1),
            "fraud_reason": row.get("fraud_reason",""),
            "source": row.get("source","historical")
        })

    return jsonify({"records": records, "total": total, "page": page, "per_page": per_page})


@app.route("/api/fraud/timeline")
def fraud_timeline():
    df = _df_scored.copy()
    df["month"] = df["date"].dt.to_period("M")
    timeline = df.groupby("month").agg(
        total_billing=("billing","sum"),
        flagged_billing=("billing", lambda x: x[df.loc[x.index,"flagged"]].sum()),
        flagged_count=("flagged","sum"),
        total_count=("flagged","count")
    ).reset_index()
    timeline["month"] = timeline["month"].astype(str)
    return jsonify(timeline.to_dict(orient="records"))


@app.route("/api/fraud/by_hospital")
def fraud_by_hospital():
    df = _df_scored
    flagged = df[df.flagged]
    result = flagged.groupby("hospital_id").agg(
        flagged_count=("flagged","sum"),
        flagged_billing=("billing","sum"),
        avg_anomaly=("anomaly_score","mean")
    ).reset_index().sort_values("flagged_count", ascending=False)
    return jsonify(result.to_dict(orient="records"))


@app.route("/api/fraud/by_type")
def fraud_by_type():
    df = _df_scored[_df_scored.flagged]
    result = df.groupby("fraud_reason").size().reset_index(name="count")
    return jsonify(result.to_dict(orient="records"))


@app.route("/api/fraud/case/<hospital_id>")
def case_detail(hospital_id):
    df = _df_scored
    hosp = df[df.hospital_id == hospital_id]
    flagged_hosp = hosp[hosp.flagged]

    monthly = hosp.groupby(hosp["date"].dt.to_period("M")).agg(
        total_billing=("billing","sum"),
        flagged_billing=("billing", lambda x: x[hosp.loc[x.index,"flagged"]].sum()),
        record_count=("billing","count")
    ).reset_index()
    monthly["month"] = monthly["month"].astype(str)

    top_doctor = flagged_hosp.groupby("doctor_id").size().idxmax() if len(flagged_hosp) > 0 else "N/A"
    top_disease = flagged_hosp.groupby("disease").size().idxmax() if len(flagged_hosp) > 0 else "N/A"

    return jsonify({
        "hospital_id": hospital_id,
        "total_records": len(hosp),
        "flagged_records": len(flagged_hosp),
        "total_billing": float(hosp.billing.sum()),
        "flagged_billing": float(flagged_hosp.billing.sum()),
        "avg_anomaly_score": round(float(flagged_hosp.anomaly_score.mean()), 1) if len(flagged_hosp) > 0 else 0,
        "top_flagged_doctor": top_doctor,
        "top_flagged_disease": top_disease,
        "monthly_trend": monthly.to_dict(orient="records")
    })


# ── PUBLIC HEALTH SENSEX ──────────────────────────────────────
@app.route("/api/public/index")
def public_index():
    df = _df_scored

    # Separate public health index (disease burden, not fraud)
    disease_counts = df.groupby("disease").size()
    total = disease_counts.sum()
    burden_score = 100 - min(50, (total / 50000) * 50)  # normalised

    # Combine with fraud penalty for overall score
    fraud_penalty = (100 - _health_index) * 0.3
    public_score = round(max(0, min(100, burden_score - fraud_penalty)), 1)

    return jsonify({
        "public_health_index": public_score,
        "fraud_index": _health_index,
        "composite_score": round((public_score + _health_index) / 2, 1)
    })


@app.route("/api/public/disease_trends")
def disease_trends():
    df = _df_scored
    monthly = df.groupby([df["date"].dt.to_period("M"), "disease"])["billing"].agg(["count","sum"]).reset_index()
    monthly.columns = ["month","disease","cases","billing"]
    monthly["month"] = monthly["month"].astype(str)

    # Calculate trend (compare last 2 months)
    trends = []
    for disease in df.disease.unique():
        d = monthly[monthly.disease == disease].tail(2)
        if len(d) == 2:
            prev, curr = d.iloc[0]["cases"], d.iloc[1]["cases"]
            pct_change = round((curr - prev) / (prev + 1e-6) * 100, 1)
        else:
            pct_change = 0
        trends.append({"disease": disease, "pct_change": pct_change,
                        "direction": "up" if pct_change > 0 else "down"})

    return jsonify({
        "monthly": monthly.to_dict(orient="records"),
        "trends": trends
    })


@app.route("/api/public/outbreak_risk")
def outbreak_risk():
    df = _df_scored

    # Compute outbreak risk per disease from actual data
    risks = []
    for disease in df.disease.unique():
        d = df[df.disease == disease]
        recent = d[d.date >= d.date.max() - timedelta(days=30)]
        older  = d[d.date < d.date.max() - timedelta(days=30)]

        recent_rate = len(recent) / 30
        older_rate  = len(older) / max(1, (len(d) - len(recent)))
        ratio = recent_rate / (older_rate + 1e-6)

        # Confidence based on deviation from norm
        confidence = min(95, round(abs(ratio - 1) * 40 + 40, 1))
        high_risk = ratio > 1.3

        risk_states = {
            "Dengue": ["Kerala","Maharashtra","Karnataka","Tamil Nadu"],
            "Malaria": ["Odisha","Jharkhand","Chhattisgarh","West Bengal"],
            "Tuberculosis": ["UP","Bihar","MP","Rajasthan"],
            "Typhoid": ["UP","Bihar","West Bengal","Assam"],
            "COVID-19": ["Maharashtra","Delhi","Karnataka","Gujarat"],
            "Cardiac": ["Punjab","Haryana","UP","Delhi"],
            "Orthopedic": ["Maharashtra","UP","Karnataka","Rajasthan"],
            "Cancer": ["UP","Maharashtra","West Bengal","Bihar"]
        }

        risks.append({
            "disease": disease,
            "risk_ratio": round(float(ratio), 2),
            "confidence": confidence,
            "high_risk": bool(high_risk),
            "recent_cases": len(recent),
            "older_avg": round(float(older_rate * 30), 1),
            "at_risk_states": risk_states.get(disease, [])
        })

    risks.sort(key=lambda x: x["risk_ratio"], reverse=True)
    return jsonify(risks)


@app.route("/api/public/state_scores")
def state_scores():
    # Simulated but deterministic state health scores
    states = [
        ("Kerala", 78), ("Goa", 76), ("Himachal Pradesh", 74), ("Punjab", 71),
        ("Tamil Nadu", 69), ("Maharashtra", 67), ("Karnataka", 65), ("Gujarat", 63),
        ("Andhra Pradesh", 61), ("Telangana", 60), ("West Bengal", 57),
        ("Rajasthan", 55), ("Madhya Pradesh", 52), ("Assam", 50),
        ("Uttar Pradesh", 46), ("Bihar", 42), ("Jharkhand", 40), ("Odisha", 45),
        ("Chhattisgarh", 43), ("Delhi", 62)
    ]
    return jsonify([{"state": s, "score": sc,
                     "grade": "A" if sc>=70 else "B" if sc>=60 else "C" if sc>=50 else "D"}
                    for s, sc in states])


# ── SUBMITTED RECORDS (for dashboard to show new submissions) ─
@app.route("/api/submitted")
def submitted_records():
    return jsonify(_submitted_records[-50:])  # last 50


if __name__ == "__main__":
    import os
app.run(host='0.0.0.0', port=int(os.environ.get('PORT', 5000)))
