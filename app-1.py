"""
Web interface: AI-Based Payroll Anomaly Detection
Local Government Administration, Nigeria

Deploy: Streamlit Community Cloud (GitHub repo with app.py + requirements.txt)
Libs: streamlit numpy pandas matplotlib seaborn scikit-learn
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import streamlit as st
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

LGAS = ["Adavi", "Ajaokuta", "Ankpa", "Bassa", "Dekina", "Ibaji", "Idah",
        "Igalamela-Odolu", "Ijumu", "Kabba/Bunu", "Kogi (Koton Karfe)", "Lokoja",
        "Mopa-Muro", "Ofu", "Ogori/Magongo", "Okehi", "Okene", "Olamaboro",
        "Omala", "Yagba East", "Yagba West"]       # the 21 LGAs of Kogi State
LGA_WEIGHTS = np.array([6, 4, 5, 4, 6, 3, 3, 3, 3, 4, 3, 9, 2, 4, 2, 4, 8, 4, 3, 3, 3], float)
LGA_WEIGHTS /= LGA_WEIGHTS.sum()
GRADE_BASE = {gl: 30000 + (gl - 1) * 9000 for gl in range(1, 16)}  # illustrative
REQUIRED = ["staff_id", "lga", "grade_level", "age", "days_present",
            "overtime_hours", "basic_salary", "allowances", "bank_account"]
FEATURES = ["salary_ratio", "allowance_ratio", "account_share_count",
            "days_present", "age", "net_pay", "pay_per_day_present",
            "overtime_hours"]


# ---------------------------------------------------------------- data
@st.cache_data(show_spinner=False)
def make_data(n=1500, seed=42):
    rng = np.random.default_rng(seed)
    df = pd.DataFrame({
        "staff_id": [f"KG{200000 + i}" for i in range(n)],
        "lga": rng.choice(LGAS, n, p=LGA_WEIGHTS),
        "grade_level": rng.integers(1, 16, n),
        "age": rng.integers(22, 60, n),
        "days_present": rng.integers(18, 23, n),
        "overtime_hours": rng.integers(0, 15, n),
    })
    base = df["grade_level"].map(GRADE_BASE)
    df["basic_salary"] = base * rng.normal(1.0, 0.02, n)
    df["allowances"] = df["basic_salary"] * rng.normal(0.35, 0.03, n)
    df["bank_account"] = rng.integers(10**9, 10**10, n)

    k = max(5, int(n * 0.02))
    pool = list(df.index)
    rng.shuffle(pool)
    g, inf, du, rt, al = (pool[i * k:(i + 1) * k] for i in range(5))
    df.loc[g, ["days_present", "overtime_hours"]] = 0           # ghost workers
    df.loc[inf, "basic_salary"] *= rng.uniform(1.8, 3.0, len(inf))
    half = len(du) // 2
    df.loc[du[half:half * 2], "bank_account"] = df.loc[du[:half], "bank_account"].values
    df.loc[rt, "age"] = rng.integers(61, 72, len(rt))           # retired
    df.loc[al, "allowances"] *= rng.uniform(2.5, 4.0, len(al))  # padding
    return df


def detect(df, contamination):
    d = df.copy()
    d["expected_basic"] = d["grade_level"].map(GRADE_BASE)
    d["gross_pay"] = d["basic_salary"] + d["allowances"]
    d["deductions"] = d["basic_salary"] * 0.105 + d["gross_pay"] * 0.07
    d["net_pay"] = d["gross_pay"] - d["deductions"]
    d["salary_ratio"] = d["basic_salary"] / d["expected_basic"]
    d["allowance_ratio"] = d["allowances"] / d["basic_salary"]
    d["account_share_count"] = d.groupby("bank_account")["staff_id"].transform("count")
    d["pay_per_day_present"] = d["gross_pay"] / (d["days_present"] + 1)

    X = StandardScaler().fit_transform(d[FEATURES])
    iso = IsolationForest(n_estimators=300, contamination=contamination,
                          random_state=42).fit(X)
    d["flagged"] = iso.predict(X) == -1
    d["risk_score"] = -iso.score_samples(X)
    d[["pc1", "pc2"]] = PCA(n_components=2).fit_transform(X)

    def reasons(r):
        out = []
        if r.days_present == 0: out.append("No attendance (possible ghost worker)")
        if r.salary_ratio > 1.3: out.append("Salary far above grade level")
        if r.account_share_count > 1: out.append("Bank account shared with another staff")
        if r.age > 60: out.append("Past retirement age")
        if r.allowance_ratio > 0.7: out.append("Unusually high allowances")
        return "; ".join(out) if out else "Statistical outlier - review manually"

    d["reason"] = ""
    d.loc[d.flagged, "reason"] = d[d.flagged].apply(reasons, axis=1)
    return d


# ---------------------------------------------------------------- charts
def chart_pay(d):
    fig, ax = plt.subplots(figsize=(6, 4))
    sns.histplot(d, x="gross_pay", hue="flagged", bins=40,
                 palette={False: "#2a9d8f", True: "#e63946"}, ax=ax)
    ax.set_title("Gross pay distribution (red = flagged)")
    ax.set_xlabel("Gross pay (₦)")
    return fig


def chart_grade(d):
    fig, ax = plt.subplots(figsize=(6, 4))
    sns.boxplot(d, x="grade_level", y="gross_pay", color="#a8dadc", ax=ax)
    ax.set_title("Gross pay by grade level")
    ax.set_ylabel("Gross pay (₦)")
    return fig


def chart_pca(d):
    fig, ax = plt.subplots(figsize=(6, 4))
    n, f = d[~d.flagged], d[d.flagged]
    ax.scatter(n.pc1, n.pc2, s=8, c="#2a9d8f", label="Normal")
    ax.scatter(f.pc1, f.pc2, s=26, c="#e63946", marker="x", label="Flagged")
    ax.set_title("Payroll records (PCA view)")
    ax.legend()
    return fig


def chart_lga(d):
    fig, ax = plt.subplots(figsize=(6, 4))
    c = d[d.flagged]["lga"].value_counts()
    sns.barplot(x=c.values, y=c.index, color="#457b9d", ax=ax)
    ax.set_title("Flagged staff per local government")
    ax.set_xlabel("Flagged staff")
    return fig


# ---------------------------------------------------------------- app
def main():
    st.set_page_config(page_title="Payroll Anomaly Detection", page_icon="🔍")
    st.title("🔍 Payroll Anomaly Detection")
    st.caption("AI-based audit tool for Local Government Administration, Nigeria - Kogi State (simulated data)")

    with st.sidebar:
        st.header("Settings")
        source = st.radio("Data source", ["Simulated data", "Upload CSV"])
        sensitivity = st.slider("Sensitivity (% of staff to flag)", 1, 15, 7)
        n = 1500
        if source == "Simulated data":
            n = st.slider("Number of staff", 300, 3000, 1500, step=100)

    if source == "Upload CSV":
        st.info("CSV must have these columns: " + ", ".join(REQUIRED))
        template = make_data(20).loc[:, REQUIRED].to_csv(index=False)
        st.download_button("Download CSV template", template, "payroll_template.csv")
        up = st.file_uploader("Upload payroll CSV", type="csv")
        if up is None:
            st.stop()
        raw = pd.read_csv(up)
        missing = [c for c in REQUIRED if c not in raw.columns]
        if missing:
            st.error("Missing columns: " + ", ".join(missing))
            st.stop()
        raw = raw[raw["grade_level"].between(1, 15)]
    else:
        raw = make_data(n)

    d = detect(raw, sensitivity / 100)
    flagged = d[d.flagged]

    c1, c2 = st.columns(2)
    c1.metric("Staff analysed", f"{len(d):,}")
    c2.metric("Flagged for audit", f"{len(flagged):,}")
    c3, c4 = st.columns(2)
    c3.metric("% flagged", f"{len(flagged) / len(d) * 100:.1f}%")
    c4.metric("Monthly pay at risk", f"₦{flagged.gross_pay.sum():,.0f}")

    tab1, tab2, tab3 = st.tabs(["📊 Charts", "🚩 Flagged staff", "📋 All data"])

    with tab1:
        for fn in (chart_pay, chart_grade, chart_pca, chart_lga):
            st.pyplot(fn(d))
            plt.close("all")

    with tab2:
        cols = ["staff_id", "lga", "grade_level", "gross_pay",
                "days_present", "age", "risk_score", "reason"]
        table = flagged.sort_values("risk_score", ascending=False)[cols].round(2)
        st.dataframe(table, hide_index=True)
        st.download_button("⬇️ Download flagged list (CSV)",
                           table.to_csv(index=False), "flagged_payroll.csv")

    with tab3:
        st.dataframe(d.drop(columns=["pc1", "pc2"]).round(2), hide_index=True)


if __name__ == "__main__":
    main()
