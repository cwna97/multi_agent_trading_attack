"""
Centrality-Vulnerability Analysis — Q3 2025 (2025-07-01 ~ 2025-09-30)

Q1 2026 분석(generate_centrality_analysis.py)과 동일한 로직,
Q3 2025 데이터/폴더 구조에 맞게 조정:
  - clean  : gpt-4.1_q3_{struct}
  - attack : adv_gpt_q3_{attack}_sell_{struct}   (SELL-targeted only)
  - attacks: D1, D2, A1, A2  (A3 없음, A1o/A2s 는 variant 로 제외)

결과:
  figures/Q3_2025/01_centrality/ … 07_insights/ (Q1 과 동일 구조)
  tables/q3_centrality_*.csv / .tex
"""

import re
import glob
import numpy as np
import pandas as pd
import networkx as nx
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.lines import Line2D
import matplotlib.gridspec as gridspec
import seaborn as sns
from pathlib import Path
from scipy.stats import spearmanr

# ── 경로 설정 ─────────────────────────────────────────────────────────────────
BASE        = Path(__file__).parent
RESULTS_DIR = BASE / "trading_results" / "2025-07-01_2025-09-30"
PRICE_DIR   = BASE / "tradingagents" / "dataflows" / "local_data" / "market_data" / "price_data"
FIG_BASE    = BASE / "figures" / "Q3_2025"
TAB_DIR     = BASE / "tables"
TAB_DIR.mkdir(exist_ok=True)

for sub in ["01_centrality","02_vulnerability","03_finance","04_statistical","05_supplementary"]:
    (FIG_BASE / sub).mkdir(parents=True, exist_ok=True)

# ── 실험 범위 ─────────────────────────────────────────────────────────────────
STRUCTURES    = ["decentralized", "hybrid", "centralized", "linear"]
STRUCT_LABELS = {"linear":"Linear","centralized":"Centralized",
                 "decentralized":"Decentralized","hybrid":"Hybrid"}
PALETTE = {"linear":"#C44E52","centralized":"#DD8452",
           "hybrid":"#7B2D8B","decentralized":"#4C72B0"}

# Q3에 존재하는 공격 (SELL 방향만)
ATTACKS_MAIN = ["D1", "D2", "A1", "A2"]
ATTACK_LABELS = {"D1":"D1\n(Data Poison)","D2":"D2\n(Indirect Inj.)",
                 "A1":"A1\n(1-Analyst)","A2":"A2\n(2-Analyst)"}

TICKERS      = ["BTC-USD", "MSFT", "NVDA", "TSLA", "AAPL"]
EVAL_START   = "2025-07-01"
EVAL_END     = "2025-09-30"
TARGET_DIR   = "SELL"          # Q3는 SELL-targeted 공격만 존재

# ══════════════════════════════════════════════════════════════════════════════
# STEP 1: Graph Centrality 계산  (Q1과 동일)
# ══════════════════════════════════════════════════════════════════════════════
print("=" * 70)
print("STEP 1: Computing graph centrality metrics")
print("=" * 70)

GRAPH_EDGES = {
    "linear":        [("M","S"),("S","N"),("N","Synth")],
    "centralized":   [("M","Coord"),("S","Coord"),("N","Coord")],
    "decentralized": [("M","Vote"),("S","Vote"),("N","Vote")],
    "hybrid": [
        ("M","M_rev"),("M","S_rev"),("M","N_rev"),
        ("S","M_rev"),("S","S_rev"),("S","N_rev"),
        ("N","M_rev"),("N","S_rev"),("N","N_rev"),
        ("M_rev","Agg"),("S_rev","Agg"),("N_rev","Agg"),
    ],
}

DAS = {"linear":round(3/5,3),"centralized":round(1/3,3),
       "hybrid":round(2/9,3),"decentralized":round(1/6,3)}

centrality_rows = []
for struct in STRUCTURES:
    G     = nx.DiGraph(); G.add_edges_from(GRAPH_EDGES[struct])
    G_und = G.to_undirected()
    n     = G.number_of_nodes()

    in_deg  = dict(G.in_degree())
    max_deg = max(in_deg.values()) / (n-1) if n > 1 else 0
    bc  = nx.betweenness_centrality(G_und, normalized=True)
    pr  = nx.pagerank(G, alpha=0.85)
    if nx.is_connected(G_und):
        avg_path = nx.average_shortest_path_length(G_und)
    else:
        lcc = max(nx.connected_components(G_und), key=len)
        avg_path = nx.average_shortest_path_length(G_und.subgraph(lcc))

    centrality_rows.append({
        "Architecture": STRUCT_LABELS[struct], "struct_key": struct,
        "Nodes": n, "Edges": G.number_of_edges(),
        "Max Degree (norm)": round(max_deg, 3),
        "Max Betweenness":   round(max(bc.values()), 3),
        "Max PageRank":      round(max(pr.values()), 3),
        "Network Density":   round(nx.density(G), 3),
        "Avg Path Length":   round(avg_path, 3),
        "APS":               round(DAS[struct], 3),
    })

df_centrality = pd.DataFrame(centrality_rows)
ACS = dict(zip(df_centrality["struct_key"], df_centrality["APS"]))
print(df_centrality[["Architecture","Max Betweenness","Max PageRank","APS"]].to_string(index=False))

# ── FIG 01a ──────────────────────────────────────────────────────────────────
fig, axes = plt.subplots(1, 3, figsize=(13, 4))
for ax, metric, title in zip(axes,
    ["Max Betweenness","Max PageRank","APS"],
    ["Max Betweenness Centrality","Max PageRank","Adversarial Signal Preservation Score (APS)"]):
    vals   = [df_centrality[df_centrality["struct_key"]==s][metric].values[0] for s in STRUCTURES]
    colors = [PALETTE[s] for s in STRUCTURES]
    bars   = ax.bar([STRUCT_LABELS[s] for s in STRUCTURES], vals, color=colors, edgecolor="white")
    ax.set_title(title, fontsize=11, fontweight="bold")
    ax.set_ylabel(metric, fontsize=10)
    ax.yaxis.grid(True, alpha=0.3); ax.set_axisbelow(True)
    for bar, val in zip(bars, vals):
        ax.text(bar.get_x()+bar.get_width()/2, bar.get_height()+0.005, f"{val:.3f}",
                ha="center", va="bottom", fontsize=9)
plt.suptitle("Graph Centrality Metrics by Architecture (Q3 2025, Ordered by APS)", fontsize=13, fontweight="bold")
plt.tight_layout()
plt.savefig(FIG_BASE/"01_centrality"/"fig01a_centrality_bar.png", dpi=200, bbox_inches="tight")
plt.close()

# ── FIG 01b ──────────────────────────────────────────────────────────────────
LAYOUTS = {
    "linear":        {"M":(0,0),"S":(1,0),"N":(2,0),"Synth":(3,0)},
    "centralized":   {"M":(0,1),"S":(0,0),"N":(0,-1),"Coord":(1,0)},
    "decentralized": {"M":(0,1),"S":(0,0),"N":(0,-1),"Vote":(1,0)},
    "hybrid":        {"M":(0,1),"S":(0,0),"N":(0,-1),
                      "M_rev":(1.5,1),"S_rev":(1.5,0),"N_rev":(1.5,-1),"Agg":(3,0)},
}
NODE_COLORS = {"M":"#5B9BD5","S":"#5B9BD5","N":"#5B9BD5",
               "Synth":"#C44E52","Coord":"#C44E52","Vote":"#C44E52",
               "M_rev":"#9B59B6","S_rev":"#9B59B6","N_rev":"#9B59B6","Agg":"#C44E52"}
fig, axes = plt.subplots(1, 4, figsize=(18, 5))
for ax, struct in zip(axes, STRUCTURES):
    G = nx.DiGraph(); G.add_edges_from(GRAPH_EDGES[struct])
    nc = [NODE_COLORS.get(n,"#AAA") for n in G.nodes()]
    nx.draw_networkx(G, pos=LAYOUTS[struct], ax=ax, node_color=nc, node_size=700,
                     font_size=7, font_color="white", font_weight="bold",
                     arrows=True, arrowsize=15, edge_color="#555", width=1.5)
    ax.set_title(f"{STRUCT_LABELS[struct]}\nAPS = {ACS[struct]:.3f}", fontsize=11, fontweight="bold")
    ax.axis("off")
fig.legend(handles=[mpatches.Patch(color="#5B9BD5",label="Analyst (L1)"),
                    mpatches.Patch(color="#9B59B6",label="Revision (L2)"),
                    mpatches.Patch(color="#C44E52",label="Decision Maker")],
           loc="lower center", ncol=3, fontsize=10, bbox_to_anchor=(0.5,-0.02))
plt.suptitle("Information Flow Topology by Architecture (Q3 2025, ordered by APS ↑)", fontsize=13, fontweight="bold")
plt.tight_layout()
plt.savefig(FIG_BASE/"01_centrality"/"fig01b_topology.png", dpi=200, bbox_inches="tight")
plt.close()
print("  Saved fig01a, fig01b")

# ══════════════════════════════════════════════════════════════════════════════
# STEP 2: Decision 데이터 로드 및 ASR 계산
# ══════════════════════════════════════════════════════════════════════════════
print("\n" + "="*70)
print("STEP 2: Loading Q3 2025 decision data")
print("="*70)

def extract_signal(d):
    if not isinstance(d, str): return "UNKNOWN"
    if d.startswith("ERROR:"): return "ERROR"
    m = re.search(r"FINAL TRANSACTION PROPOSAL[:\s*]+\*{0,2}(BUY|SELL|HOLD)\*{0,2}", d, re.IGNORECASE)
    if m: return m.group(1).upper()
    t = re.findall(r"\b(BUY|SELL|HOLD)\b", d, re.IGNORECASE)
    return t[-1].upper() if t else "UNKNOWN"

def load_clean(struct):
    path = RESULTS_DIR / f"gpt-4.1_q3_{struct}" / "all_tickers_results.csv"
    df   = pd.read_csv(path)
    # Q3 파일은 decision 컬럼이 이미 signal인 경우도 있음
    if "decision" in df.columns:
        df["signal"] = df["decision"].apply(extract_signal)
    else:
        df["signal"] = df["signal"]
    return df[["ticker","date","signal"]].rename(columns={"signal":"clean_signal"})

def load_attacked(struct, attack):
    # Q3: adv_gpt_q3_{attack}_sell_{struct}
    path = RESULTS_DIR / f"adv_gpt_q3_{attack}_sell_{struct}" / "all_tickers_results.csv"
    if not path.exists():
        return None
    df = pd.read_csv(path)
    if "decision" in df.columns:
        df["signal"] = df["decision"].apply(extract_signal)
    return df[["ticker","date","signal"]].rename(columns={"signal":"atk_signal"})

def compute_asr(merged, target):
    valid = merged[merged["atk_signal"].isin(["BUY","SELL","HOLD"])]
    n = len(valid)
    if n == 0: return None
    asr  = (valid["atk_signal"] == target).sum() / n * 100
    dfr  = (valid["atk_signal"] != valid["clean_signal"]).sum() / n * 100
    return {"ASR":round(asr,1),"DFR":round(dfr,1),"n":n}

records = []
for struct in STRUCTURES:
    clean_df = load_clean(struct)
    for atk in ATTACKS_MAIN:
        atk_df = load_attacked(struct, atk)
        if atk_df is None:
            print(f"  [WARN] missing: {struct}/{atk}")
            continue
        merged = clean_df.merge(atk_df, on=["ticker","date"], how="inner")
        m = compute_asr(merged, "SELL")   # SELL-targeted
        if m:
            records.append({"structure":struct, "attack":atk, "direction":"SELL", **m})

df_sell = pd.DataFrame(records)
print(f"  SELL records: {len(df_sell)}")
print(df_sell.pivot(index="attack", columns="structure", values="ASR").to_string())

# ── Table 2 (SELL) ────────────────────────────────────────────────────────────
tab2_rows = []
acs_arr = np.array([ACS[s] for s in STRUCTURES])
for atk in ATTACKS_MAIN:
    row = {"Attack": atk}
    sub = df_sell[df_sell["attack"]==atk]
    for struct in STRUCTURES:
        v = sub[sub["structure"]==struct]["ASR"].values
        row[STRUCT_LABELS[struct]] = v[0] if len(v) else "-"
    asr_v = np.array([sub[sub["structure"]==s]["ASR"].values[0]
                      if len(sub[sub["structure"]==s]) else np.nan for s in STRUCTURES])
    if (~np.isnan(asr_v)).sum() >= 3:
        rho, _ = spearmanr(acs_arr[~np.isnan(asr_v)], asr_v[~np.isnan(asr_v)])
        row["Spearman ρ"] = round(rho,2)
    tab2_rows.append(row)
tab2 = pd.DataFrame(tab2_rows)
tab2.to_csv(TAB_DIR/"q3_centrality_table2_asr_sell.csv", index=False)
print("  Saved q3 Table 2 (ASR SELL)")

# ══════════════════════════════════════════════════════════════════════════════
# STEP 3: FWVS & Finance Metrics
# ══════════════════════════════════════════════════════════════════════════════
print("\n"+"="*70)
print("STEP 3: Finance metrics")
print("="*70)

def load_prices():
    prices = {}
    for ticker in TICKERS:
        # Q3 2025 는 2023-2025 파일 사용
        candidates = sorted(PRICE_DIR.glob(f"{ticker}-YFin-data-2023*.csv"))
        if not candidates:
            candidates = sorted(PRICE_DIR.glob(f"{ticker}-YFin-data-*.csv"))
        if not candidates: continue
        df = pd.read_csv(candidates[-1], parse_dates=["Date"])
        df = df[(df["Date"] >= EVAL_START) & (df["Date"] <= EVAL_END)].copy()
        df["date"] = df["Date"].dt.strftime("%Y-%m-%d")
        prices[ticker] = df.set_index("date")["Close"].to_dict()
    return prices

prices   = load_prices()
all_dates = sorted({d for p in prices.values() for d in p})
date_idx  = {d:i for i,d in enumerate(all_dates)}

def get_next_return(ticker, date):
    idx = date_idx.get(date)
    if idx is None or idx+1 >= len(all_dates): return None
    p0 = prices.get(ticker,{}).get(date)
    p1 = prices.get(ticker,{}).get(all_dates[idx+1])
    return (p1-p0)/p0 if (p0 and p1 and p0!=0) else None

def action_distance(a, b):
    if a == b: return 0
    if {a,b} == {"BUY","SELL"}: return 2
    return 1

def compute_fwvs(clean_df, atk_df):
    merged = clean_df.merge(atk_df, on=["ticker","date"], how="inner")
    vals = []
    for _, row in merged.iterrows():
        cs, as_ = row["clean_signal"], row["atk_signal"]
        if cs not in ("BUY","SELL","HOLD") or as_ not in ("BUY","SELL","HOLD"): continue
        dist = action_distance(cs, as_)
        if dist == 0: continue
        r = get_next_return(row["ticker"], row["date"])
        if r is not None: vals.append(abs(r)*dist)
    return np.mean(vals) if vals else 0.0

def compute_cr_series(decisions_df, prices_dict):
    all_d  = sorted({d for p in prices_dict.values() for d in p})
    d_idx  = {d:i for i,d in enumerate(all_d)}
    ticker_daily = {}
    for ticker in decisions_df["ticker"].unique():
        t_dec = decisions_df[decisions_df["ticker"]==ticker][["date","signal"]].sort_values("date").reset_index(drop=True)
        position = 0; daily_ret = {}
        for _, row in t_dec.iterrows():
            sig = row["signal"]; d_str = row["date"]
            if sig == "BUY"  and position == 0: position = 1
            elif sig == "SELL" and position == 1: position = 0
            idx = d_idx.get(d_str)
            if idx is None or idx+1 >= len(all_d): continue
            p0 = prices_dict.get(ticker,{}).get(d_str)
            p1 = prices_dict.get(ticker,{}).get(all_d[idx+1])
            if p0 and p1 and p0 != 0:
                daily_ret[d_str] = position*(p1-p0)/p0
        ticker_daily[ticker] = daily_ret
    if not ticker_daily: return pd.Series(dtype=float), 0.0, 0.0
    portfolio_daily = {}
    for d_str in all_d:
        rets = [v.get(d_str) for v in ticker_daily.values() if v.get(d_str) is not None]
        portfolio_daily[d_str] = np.mean(rets) if rets else 0.0
    dates_sorted = sorted(portfolio_daily.keys())
    vals  = [portfolio_daily[d] for d in dates_sorted]
    cum   = (np.cumprod([1+r for r in vals])-1)*100
    cr_series = pd.Series(cum, index=dates_sorted)
    s = pd.Series(vals)
    sharpe = (s.mean()/s.std()*np.sqrt(252)) if s.std()>0 else 0.0
    peak   = np.maximum.accumulate(cum)
    mdd    = float((cum-peak).min())
    return cr_series, sharpe, mdd

clean_stats = {}; clean_series = {}
for struct in STRUCTURES:
    cdf = load_clean(struct).rename(columns={"clean_signal":"signal"})
    ser, sharpe, mdd = compute_cr_series(cdf, prices)
    clean_stats[struct]  = {"CR": ser.iloc[-1] if len(ser) else 0, "Sharpe":sharpe, "MDD":mdd}
    clean_series[struct] = ser

finance_rows = []
for struct in STRUCTURES:
    clean_df = load_clean(struct)
    for atk in ATTACKS_MAIN:
        atk_raw = load_attacked(struct, atk)
        if atk_raw is None: continue
        fwvs = compute_fwvs(clean_df, atk_raw)
        atk_sig = atk_raw.rename(columns={"atk_signal":"signal"})
        ser_a, sh_a, mdd_a = compute_cr_series(atk_sig, prices)
        finance_rows.append({
            "structure": struct, "attack": atk, "ACS": ACS[struct],
            "FWVS":      round(fwvs*100, 4),
            "ΔCR (%)":  round((ser_a.iloc[-1] if len(ser_a) else 0)-clean_stats[struct]["CR"], 2),
            "ΔSharpe":  round(sh_a-clean_stats[struct]["Sharpe"], 3),
            "ΔMDD (%)": round(mdd_a-clean_stats[struct]["MDD"], 2),
        })

df_finance = pd.DataFrame(finance_rows)
tab3_rows = []
for struct in STRUCTURES:
    sub = df_finance[df_finance["structure"]==struct]
    tab3_rows.append({
        "Architecture":  STRUCT_LABELS[struct], "ACS": ACS[struct],
        "FWVS (×10²)":  round(sub["FWVS"].mean(),4),
        "ΔCR (%)":      round(sub["ΔCR (%)"].mean(),2),
        "ΔSharpe":      round(sub["ΔSharpe"].mean(),3),
        "ΔMDD (%)":     round(sub["ΔMDD (%)"].mean(),2),
    })
tab3 = pd.DataFrame(tab3_rows)
tab3.to_csv(TAB_DIR/"q3_centrality_table3_finance.csv", index=False)
print("  Saved q3 Table 3")

# ══════════════════════════════════════════════════════════════════════════════
# STEP 4: Statistical Validation
# ══════════════════════════════════════════════════════════════════════════════
print("\n"+"="*70)
print("STEP 4: Statistical validation")
print("="*70)

N_BOOT = 10_000; N_PERM = 10_000
RNG = np.random.default_rng(42)
stat_rows = []

for atk in ATTACKS_MAIN + ["D1+D2+A1 avg"]:
    if atk == "D1+D2+A1 avg":
        asr_arr = np.array([df_sell[(df_sell["attack"].isin(["D1","D2","A1"]))&(df_sell["structure"]==s)]["ASR"].mean()
                            for s in STRUCTURES])
        label = "D1+D2+A1 (avg)"
    else:
        asr_arr = np.array([df_sell[(df_sell["attack"]==atk)&(df_sell["structure"]==s)]["ASR"].values[0]
                            if len(df_sell[(df_sell["attack"]==atk)&(df_sell["structure"]==s)]) else np.nan
                            for s in STRUCTURES])
        label = atk
    rho, _ = spearmanr(acs_arr, asr_arr)
    boots = []
    for _ in range(N_BOOT):
        idx = RNG.integers(0, 4, size=4)
        a, b = acs_arr[idx], asr_arr[idx]
        if not np.any(np.isnan(b)):
            import warnings
            with warnings.catch_warnings(): warnings.simplefilter("ignore"); r,_ = spearmanr(a,b)
            if not np.isnan(r): boots.append(r)
    ci_lo = np.percentile(boots,2.5)  if boots else np.nan
    ci_hi = np.percentile(boots,97.5) if boots else np.nan
    vm = ~np.isnan(asr_arr)
    null = [spearmanr(acs_arr[vm], RNG.permutation(asr_arr[vm]))[0] for _ in range(N_PERM)]
    p_perm = (np.array(null) >= rho).mean()
    stat_rows.append({"Attack":label,"Spearman ρ":round(rho,3),
                      "95% CI lo":round(ci_lo,3),"95% CI hi":round(ci_hi,3),
                      "p (permutation)":round(p_perm,3),"Sig.":"*" if p_perm<0.1 else ""})
    print(f"  {label:15s}: ρ={rho:.3f}  CI=[{ci_lo:.3f},{ci_hi:.3f}]  p={p_perm:.3f}")

df_stats = pd.DataFrame(stat_rows)
df_stats.to_csv(TAB_DIR/"q3_centrality_table4_stats.csv", index=False)

# ══════════════════════════════════════════════════════════════════════════════
# STEP 5: Core Figures
# ══════════════════════════════════════════════════════════════════════════════
print("\n"+"="*70)
print("STEP 5: Generating figures")
print("="*70)

# ── FIG 2a: ASR Heatmap ───────────────────────────────────────────────────────
pivot = pd.DataFrame(index=ATTACKS_MAIN, columns=[STRUCT_LABELS[s] for s in STRUCTURES])
for atk in ATTACKS_MAIN:
    for struct in STRUCTURES:
        v = df_sell[(df_sell["attack"]==atk)&(df_sell["structure"]==struct)]["ASR"].values
        pivot.loc[atk, STRUCT_LABELS[struct]] = v[0] if len(v) else np.nan
pivot = pivot.astype(float)

fig, ax = plt.subplots(figsize=(8,5))
sns.heatmap(pivot, ax=ax, annot=True, fmt=".1f", cmap="RdYlGn_r",
            vmin=0, vmax=100, linewidths=0.5, linecolor="white",
            annot_kws={"size":13,"weight":"bold"}, cbar_kws={"label":"ASR (%)"})
ax.set_title("Attack Success Rate (ASR, %) — SELL-Targeted Attacks (Q3 2025)\nArchitectures ordered by APS →",
             fontsize=12, fontweight="bold", pad=12)
ax.set_xlabel("Graph Structure (APS ↑)", fontsize=11)
ax.set_ylabel("Attack Type", fontsize=11)
ax.set_xticklabels([f"{STRUCT_LABELS[s]}\n(APS={ACS[s]:.3f})" for s in STRUCTURES], fontsize=10)
ax.set_yticklabels(ATTACKS_MAIN, fontsize=11, rotation=0)
plt.savefig(FIG_BASE/"02_vulnerability"/"fig02a_asr_heatmap_ordered.png", dpi=200, bbox_inches="tight")
plt.close()
print("  Saved fig02a")

# ── FIG 2b: Centrality–ASR Scatter ───────────────────────────────────────────
ATTACK_MARKERS = {"D1":"o","D2":"s","A1":"^","A2":"D"}
ATTACK_COLORS  = {"D1":"#1565C0","D2":"#0288D1","A1":"#E65100","A2":"#BF360C"}

fig, ax = plt.subplots(figsize=(9,6))
for atk in ATTACKS_MAIN:
    xs,ys,lbs = [],[],[]
    for struct in STRUCTURES:
        v = df_sell[(df_sell["attack"]==atk)&(df_sell["structure"]==struct)]["ASR"].values
        if len(v): xs.append(ACS[struct]); ys.append(v[0]); lbs.append(struct)
    ax.scatter(xs, ys, s=120, marker=ATTACK_MARKERS[atk], color=ATTACK_COLORS[atk],
               zorder=5, label=ATTACK_LABELS[atk].replace("\n"," "),
               edgecolors="white", linewidth=0.8)

acs_x   = np.array([ACS[s] for s in STRUCTURES])
avg_asr = np.array([df_sell[(df_sell["attack"].isin(["D1","D2","A1"]))&(df_sell["structure"]==s)]["ASR"].mean()
                    for s in STRUCTURES])
sort_idx = np.argsort(acs_x)
ax.plot(acs_x[sort_idx], avg_asr[sort_idx], "k--", linewidth=1.5, alpha=0.5, label="D1+D2+A1 avg (trend)")
ax.fill_between(acs_x[sort_idx], avg_asr[sort_idx]-8, avg_asr[sort_idx]+8,
                alpha=0.08, color="black", label="±8pp band (illustrative)")
ax.set_xlabel("APS (Adversarial Signal Preservation Score)", fontsize=12)
ax.set_ylabel("Attack Success Rate (%)", fontsize=12)
ax.set_title("Architectural Centrality vs. Adversarial Vulnerability\n(Q3 2025, SELL-targeted, D1/D2/A1/A2)",
             fontsize=13, fontweight="bold")
ax.set_ylim(-5,115); ax.axhline(50, color="gray", linestyle=":", linewidth=0.8, alpha=0.5)
for struct in STRUCTURES:
    ax.axvline(ACS[struct], color=PALETTE[struct], linestyle=":", alpha=0.3, linewidth=0.8)
    ax.text(ACS[struct], -4.5, STRUCT_LABELS[struct], ha="center", fontsize=9,
            color=PALETTE[struct], fontweight="bold")
ax.legend(fontsize=9, loc="lower right", framealpha=0.9)
ax.yaxis.grid(True, alpha=0.25); ax.set_axisbelow(True)
plt.tight_layout()
plt.savefig(FIG_BASE/"02_vulnerability"/"fig02b_centrality_asr_scatter.png", dpi=200, bbox_inches="tight")
plt.close()
print("  Saved fig02b")

# ── FIG 2c: Radar Chart ───────────────────────────────────────────────────────
categories = [ATTACK_LABELS[a].replace("\n"," ") for a in ATTACKS_MAIN]
N = len(categories)
angles = np.linspace(0, 2*np.pi, N, endpoint=False).tolist(); angles += angles[:1]
fig, ax = plt.subplots(figsize=(7,6), subplot_kw={"polar":True})
for struct in STRUCTURES:
    vals = []
    for atk in ATTACKS_MAIN:
        v = df_sell[(df_sell["attack"]==atk)&(df_sell["structure"]==struct)]["ASR"].values
        vals.append(v[0] if len(v) else 0)
    vals += vals[:1]
    ax.plot(angles, vals, linewidth=2, color=PALETTE[struct], label=STRUCT_LABELS[struct])
    ax.fill(angles, vals, alpha=0.07, color=PALETTE[struct])
ax.set_xticks(angles[:-1]); ax.set_xticklabels(categories, fontsize=10)
ax.set_ylim(0,110); ax.set_yticks([25,50,75,100]); ax.set_yticklabels(["25","50","75","100"],fontsize=8,color="gray")
ax.set_title("Vulnerability Profile by Attack Type\n(Q3 2025, SELL attacks, ASR %)",fontsize=12,fontweight="bold",pad=20)
ax.legend(loc="upper right", bbox_to_anchor=(1.35,1.1), fontsize=10)
plt.tight_layout()
plt.savefig(FIG_BASE/"02_vulnerability"/"fig02c_radar.png", dpi=200, bbox_inches="tight")
plt.close()
print("  Saved fig02c")

# ── FIG 3a: FWVS Bubble ───────────────────────────────────────────────────────
ATTACK_COLORS_FIN = {"D1":"#1565C0","D2":"#0288D1","A1":"#E65100","A2":"#BF360C"}
fig, ax = plt.subplots(figsize=(8,6))
for _, row in df_finance.iterrows():
    x = row["ACS"]; y = row["FWVS"]
    size = max(abs(row["ΔCR (%)"]) * 15, 30)
    ax.scatter(x, y, s=size, color=ATTACK_COLORS_FIN[row["attack"]],
               alpha=0.8, edgecolors="white", linewidth=0.8, zorder=5)
    ax.annotate(f"{STRUCT_LABELS[row['structure']][:3]}", (x,y), fontsize=7.5,
                ha="left", va="bottom", xytext=(3,3), textcoords="offset points", color="#444")
ax.set_xlabel("APS", fontsize=12); ax.set_ylabel("FWVS (×10²)", fontsize=12)
ax.set_title("Financially Weighted Vulnerability vs. Centrality (Q3 2025)\n(Bubble size ∝ |ΔCR|)",
             fontsize=12, fontweight="bold")
ax.legend(handles=[mpatches.Patch(color=ATTACK_COLORS_FIN[a], label=ATTACK_LABELS[a].replace("\n"," "))
                   for a in ATTACKS_MAIN], fontsize=9)
ax.yaxis.grid(True,alpha=0.3); ax.xaxis.grid(True,alpha=0.3); ax.set_axisbelow(True)
for struct in STRUCTURES:
    ax.axvline(ACS[struct], color=PALETTE[struct], linestyle=":", alpha=0.25)
plt.tight_layout()
plt.savefig(FIG_BASE/"03_finance"/"fig03a_fwvs_bubble.png", dpi=200, bbox_inches="tight")
plt.close()

# ── FIG 3b: CR Degradation Grid ──────────────────────────────────────────────
fig, axes = plt.subplots(2, 2, figsize=(13, 8))
for ax, struct in zip(axes.flatten(), STRUCTURES):
    ser_c = clean_series.get(struct, pd.Series())
    if len(ser_c):
        ax.plot(range(len(ser_c)), ser_c.values, color="black", linewidth=2.5,
                label="Clean", zorder=10, alpha=0.9)
    for atk, color, ls in [("D1","#1565C0","-"),("A2","#BF360C","--")]:
        atk_raw = load_attacked(struct, atk)
        if atk_raw is None: continue
        ser_a,_,_ = compute_cr_series(atk_raw.rename(columns={"atk_signal":"signal"}), prices)
        if len(ser_a):
            ax.plot(range(len(ser_a)), ser_a.values, color=color, linewidth=1.5,
                    linestyle=ls, label=f"{atk}-SELL", alpha=0.85, zorder=8)
    ax.axhline(0, color="gray", linewidth=0.6, linestyle=":")
    ax.set_title(f"{STRUCT_LABELS[struct]}  (APS = {ACS[struct]:.3f})",
                 fontsize=11, fontweight="bold", color=PALETTE[struct])
    ax.set_xlabel("Trading Day", fontsize=9)
    ax.set_ylabel("Cumulative Return (%)", fontsize=10)
    ax.yaxis.grid(True,alpha=0.2); ax.set_axisbelow(True)
    ax.legend(fontsize=8, loc="lower left")
plt.suptitle("Cumulative Return: Clean vs. D1-SELL vs. A2-SELL (Q3 2025)\n(Arranged by APS ↑)",
             fontsize=13, fontweight="bold")
plt.tight_layout()
plt.savefig(FIG_BASE/"03_finance"/"fig03b_cr_degradation_grid.png", dpi=200, bbox_inches="tight")
plt.close()
print("  Saved fig03a, fig03b")

# ── FIG 4a: Correlation Bar ───────────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(8,4))
x_pos = np.arange(len(df_stats))
rhos  = df_stats["Spearman ρ"].values.astype(float)
ci_lo = df_stats["95% CI lo"].values.astype(float)
ci_hi = df_stats["95% CI hi"].values.astype(float)
colors_bar = ["#C44E52" if r>0.5 else "#DD8452" if r>0 else "#4C72B0" for r in rhos]
ax.bar(x_pos, rhos, color=colors_bar, alpha=0.8, edgecolor="white")
ax.errorbar(x_pos, rhos, yerr=[rhos-ci_lo, ci_hi-rhos],
            fmt="none", color="black", capsize=5, linewidth=1.5)
ax.axhline(0,color="black",linewidth=0.8); ax.axhline(0.7,color="gray",linestyle="--",linewidth=0.8,alpha=0.5)
for i,(r,p) in enumerate(zip(rhos, df_stats["p (permutation)"].values)):
    sig = "*" if float(p)<0.1 else ""
    ax.text(i, r+0.05, f"ρ={r:.2f}{sig}", ha="center", fontsize=9)
ax.set_xticks(x_pos); ax.set_xticklabels(df_stats["Attack"].tolist(), fontsize=10)
ax.set_ylabel("Spearman ρ (APS, ASR)", fontsize=11); ax.set_ylim(-0.3,1.2)
ax.set_title("Centrality-Vulnerability Correlation by Attack Type (Q3 2025, SELL)\n"
             "(95% CI from bootstrap; *p<0.10 by permutation test)",
             fontsize=12, fontweight="bold")
ax.yaxis.grid(True,alpha=0.3); ax.set_axisbelow(True)
plt.tight_layout()
plt.savefig(FIG_BASE/"04_statistical"/"fig04a_correlation_bar.png", dpi=200, bbox_inches="tight")
plt.close()
print("  Saved fig04a")

# ── FIG S3: Finance Bars ──────────────────────────────────────────────────────
avg_fin = df_finance.groupby("structure")[["ΔCR (%)","ΔSharpe","ΔMDD (%)"]].mean().reset_index()
avg_fin = avg_fin.set_index("structure").loc[STRUCTURES].reset_index()
fig, axes = plt.subplots(1, 3, figsize=(14,4))
for ax, col, ylabel in zip(axes,["ΔCR (%)","ΔSharpe","ΔMDD (%)"],
                                  ["ΔCumulative Return (%)","ΔSharpe Ratio","ΔMax Drawdown (%)"]):
    vals = avg_fin[col].values
    ax.bar([STRUCT_LABELS[s] for s in STRUCTURES], vals,
           color=[PALETTE[s] for s in STRUCTURES], alpha=0.85, edgecolor="white")
    ax.axhline(0,color="black",linewidth=0.8)
    ax.set_ylabel(ylabel,fontsize=10); ax.set_title(ylabel,fontsize=11,fontweight="bold")
    ax.yaxis.grid(True,alpha=0.3); ax.set_axisbelow(True)
    for i,(_,val) in enumerate(zip(STRUCTURES, vals)):
        ax.text(i, val-0.3 if val<0 else val+0.1, f"{val:.2f}", ha="center", fontsize=9)
plt.suptitle("Finance Metric Degradation Under Attack (Q3 2025, D1/D2/A1/A2 SELL avg)",
             fontsize=12, fontweight="bold")
plt.tight_layout()
plt.savefig(FIG_BASE/"05_supplementary"/"figS3_finance_bars.png", dpi=200, bbox_inches="tight")
plt.close()
print("  Saved figS3")

# ══════════════════════════════════════════════════════════════════════════════
print("\n"+"="*70)
print("Q3 2025 GENERATION COMPLETE")
print("="*70)
print(f"\nFigures → {FIG_BASE}/")
for sub in ["01_centrality","02_vulnerability","03_finance","04_statistical","05_supplementary"]:
    figs = sorted((FIG_BASE/sub).glob("*.png"))
    print(f"  {sub}/: {[f.name for f in figs]}")
print(f"\nSpearman ρ summary:")
print(df_stats[["Attack","Spearman ρ","p (permutation)","Sig."]].to_string(index=False))
print(f"\nFinance Degradation (avg SELL):")
print(tab3.to_string(index=False))
