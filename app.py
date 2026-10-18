"""
Payroll Anomaly Detection - Web interface (Gradio, works in Google Colab)
Local Government Administration, Nigeria

Deploy: Hugging Face Spaces (SDK: Gradio). Files: app.py + requirements.txt
"""

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
import gradio as gr
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


def fig_pay(d):
    fig, ax = plt.subplots(figsize=(6, 4))
    sns.histplot(d, x="gross_pay", hue="flagged", bins=40,
                 palette={False: "#2a9d8f", True: "#e63946"}, ax=ax)
    ax.set_title("Gross pay distribution (red = flagged)")
    ax.set_xlabel("Gross pay (NGN)")
    return fig


def fig_grade(d):
    fig, ax = plt.subplots(figsize=(6, 4))
    sns.boxplot(d, x="grade_level", y="gross_pay", color="#a8dadc", ax=ax)
    ax.set_title("Gross pay by grade level")
    return fig


def fig_pca(d):
    fig, ax = plt.subplots(figsize=(6, 4))
    n, f = d[~d.flagged], d[d.flagged]
    ax.scatter(n.pc1, n.pc2, s=8, c="#2a9d8f", label="Normal")
    ax.scatter(f.pc1, f.pc2, s=26, c="#e63946", marker="x", label="Flagged")
    ax.set_title("Payroll records (PCA view)")
    ax.legend()
    return fig


def fig_lga(d):
    fig, ax = plt.subplots(figsize=(6, 4))
    c = d[d.flagged]["lga"].value_counts()
    sns.barplot(x=c.values, y=c.index, color="#457b9d", ax=ax)
    ax.set_title("Flagged staff per local government")
    ax.set_xlabel("Flagged staff")
    return fig


def analyze(source, n_staff, sensitivity, file):
    if source == "Upload CSV":
        if file is None:
            raise gr.Error("Please upload a CSV file first.")
        raw = pd.read_csv(getattr(file, "name", file))
        missing = [c for c in REQUIRED if c not in raw.columns]
        if missing:
            raise gr.Error("Missing columns: " + ", ".join(missing))
        raw = raw[raw["grade_level"].between(1, 15)]
    else:
        raw = make_data(int(n_staff))

    d = detect(raw, sensitivity / 100)
    fl = d[d.flagged].sort_values("risk_score", ascending=False)

    summary = (f"### Results\n"
               f"- **Staff analysed:** {len(d):,}\n"
               f"- **Flagged for audit:** {len(fl):,} ({len(fl) / len(d) * 100:.1f}%)\n"
               f"- **Monthly pay at risk:** ₦{fl.gross_pay.sum():,.0f}")

    cols = ["staff_id", "lga", "grade_level", "gross_pay",
            "days_present", "age", "risk_score", "reason"]
    table = fl[cols].round(2)
    table.to_csv("flagged_payroll.csv", index=False)

    figs = [fig_pay(d), fig_grade(d), fig_pca(d), fig_lga(d)]
    return summary, *figs, table, "flagged_payroll.csv"


with gr.Blocks(title="Payroll Anomaly Detection") as demo:
    gr.Markdown("# 🔍 AI-Based Payroll Anomaly Detection\n"
                "Local Government Administration, Nigeria - Kogi State (simulated data)")
    source = gr.Radio(["Simulated data", "Upload CSV"], value="Simulated data",
                      label="Data source")
    n_staff = gr.Slider(300, 3000, value=1500, step=100, label="Number of staff (simulated)")
    sens = gr.Slider(1, 15, value=7, step=1, label="Sensitivity (% of staff to flag)")
    file = gr.File(label="Payroll CSV (columns: " + ", ".join(REQUIRED) + ")",
                   file_types=[".csv"])
    btn = gr.Button("Run analysis", variant="primary")

    out_md = gr.Markdown()
    p1, p2, p3, p4 = gr.Plot(), gr.Plot(), gr.Plot(), gr.Plot()
    out_tbl = gr.Dataframe(label="Flagged staff for audit")
    out_file = gr.File(label="Download flagged list (CSV)")

    btn.click(analyze, [source, n_staff, sens, file],
              [out_md, p1, p2, p3, p4, out_tbl, out_file])

if __name__ == "__main__":
    demo.launch()
