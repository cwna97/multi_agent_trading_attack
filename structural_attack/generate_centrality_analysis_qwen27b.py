"""
Centrality-Vulnerability Analysis
핵심 가설: graph centrality가 높을수록 adversarial 공격에 더 취약하다.

생성 결과 (figures/ 서브폴더):
  01_centrality/   — centrality bar chart, network topology diagrams
  02_vulnerability/ — ASR heatmap (centrality-ordered), scatter plot, radar
  03_finance/       — FWVS bubble chart, CR degradation grid
  04_statistical/   — correlation plot, permutation test
  05_supplementary/ — SELL attacks, A3, per-ticker

Core Tables (tables/qwen27b_centrality_*.csv / .tex):
  centrality_table1_acs.csv          — Table 1: Architecture Centrality Profile
  centrality_table2_asr_buy.csv      — Table 2: Main ASR (BUY only, centrality-ordered)
  centrality_table3_finance.csv      — Table 3: FWVS + Finance metrics
  centrality_table4_stats.csv        — Table 4: Spearman ρ + p-value + bootstrap CI
  centrality_table5_asr_sell.csv     — Table 5 (Supp): SELL attack ASR
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
RESULTS_DIR = BASE / "trading_results" / "qwen27b_2026-01-01_2026-03-31"
PRICE_DIR   = BASE / "tradingagents" / "dataflows" / "local_data" / "market_data" / "price_data"
FIG_BASE    = BASE / "figures" / "qwen27b"
TAB_DIR     = BASE / "tables"
TAB_DIR.mkdir(exist_ok=True)

for sub in ["01_centrality", "02_vulnerability", "03_finance", "04_statistical", "05_supplementary"]:
    (FIG_BASE / sub).mkdir(parents=True, exist_ok=True)

# ── 실험 범위 정의 (MoA 제외) ─────────────────────────────────────────────────
STRUCTURES  = ["decentralized", "hybrid", "centralized", "linear"]   # DAS ascending: 0.33, 0.60, 1.00, 1.00
STRUCT_LABELS = {
    "linear":        "Linear",
    "centralized":   "Centralized",
    "decentralized": "Decentralized",
    "hybrid":        "Hybrid",
}
PALETTE = {
    "linear":        "#C44E52",
    "centralized":   "#DD8452",
    "hybrid":        "#7B2D8B",
    "decentralized": "#4C72B0",
}

ATTACKS_MAIN = ["D1", "D2", "A1", "A2"]   # A3는 supplementary
ATTACKS_ALL  = ["D1", "D2", "A1", "A2", "A3"]
ATTACK_LABELS = {
    "D1": "D1\n(Data Poison)",
    "D2": "D2\n(Indirect Inj.)",
    "A1": "A1\n(1-Analyst)",
    "A2": "A2\n(2-Analyst)",
    "A3": "A3\n(Decision Hijack)",
}
TICKERS = ["BTC-USD", "MSFT", "NVDA", "TSLA", "AAPL"]

# ══════════════════════════════════════════════════════════════════════════════
# STEP 1: Graph Centrality 계산
# ══════════════════════════════════════════════════════════════════════════════
print("=" * 70)
print("STEP 1: Computing graph centrality metrics")
print("=" * 70)

# 정보 흐름 DAG 정의 (3 analysts: market=M, social=S, news=N)
GRAPH_EDGES = {
    "linear": [
        ("M", "S"), ("S", "N"), ("N", "Synth"),
    ],
    "centralized": [
        ("M", "Coord"), ("S", "Coord"), ("N", "Coord"),
    ],
    "decentralized": [
        ("M", "Vote"), ("S", "Vote"), ("N", "Vote"),
    ],
    "hybrid": [
        # L1 → L2 (peer-review: each L1 node informs all L2 revision nodes)
        ("M", "M_rev"), ("M", "S_rev"), ("M", "N_rev"),
        ("S", "M_rev"), ("S", "S_rev"), ("S", "N_rev"),
        ("N", "M_rev"), ("N", "S_rev"), ("N", "N_rev"),
        # L2 → L3
        ("M_rev", "Agg"), ("S_rev", "Agg"), ("N_rev", "Agg"),
    ],
}

# Decision Authority Score (DAS): 단일 노드 타협 시 최종 결정에 미치는 영향 비율
# N=3 (market, social, news), M=2 (과반수 임계값 = floor(N/2)+1)
#
# - Linear:        DAS = 3/(N+2) = 3/5 = 0.600
#   Step N = "Final Integration" 으로 응답 예산 3배 할당 (프롬프트 코드 기반)
# - Centralized:   DAS = 1/N = 1/3 ≈ 0.333
#   N개 리포트 동시 합성, 균등 가중치
# - Hybrid:        DAS = (N-1)/N² = 2/9 ≈ 0.222
#   자기교정된 adversarial L2(=0) + (N-1)개 피어 L2(각 1/N) → 집계
# - Decentralized: DAS = 1/(N×M) = 1/6 ≈ 0.167
#   1/N 투표 가중치 × 과반수 임계값 M으로 이중 희석
DAS = {
    "linear":        round(3 / (3 + 2), 3),          # 0.600
    "centralized":   round(1 / 3, 3),                 # 0.333
    "hybrid":        round(2 / 9, 3),                 # 0.222
    "decentralized": round(1 / (3 * 2), 3),           # 0.167
}

# 다수결 통제에 필요한 최소 analyst 수 (graph metric 보완용)
MIN_ATTACK_COST = {
    "linear":        1,
    "centralized":   1,
    "decentralized": 2,
    "hybrid":        1,
}

centrality_rows = []
for struct in STRUCTURES:
    G = nx.DiGraph()
    G.add_edges_from(GRAPH_EDGES[struct])
    G_und = G.to_undirected()

    n_nodes = G.number_of_nodes()
    n_edges = G.number_of_edges()

    # Degree centrality (max over all nodes, directed in-degree normalized)
    in_deg = dict(G.in_degree())
    max_in_deg = max(in_deg.values())
    max_deg_centrality = max_in_deg / (n_nodes - 1) if n_nodes > 1 else 0

    # Betweenness centrality (undirected, normalized)
    bc = nx.betweenness_centrality(G_und, normalized=True)
    max_betweenness = max(bc.values())

    # PageRank (directed)
    pr = nx.pagerank(G, alpha=0.85)
    max_pagerank = max(pr.values())

    # Network density (directed)
    density = nx.density(G)

    # Average shortest path length (undirected)
    if nx.is_connected(G_und):
        avg_path = nx.average_shortest_path_length(G_und)
    else:
        # 연결 컴포넌트 중 가장 큰 것의 평균 경로
        largest_cc = max(nx.connected_components(G_und), key=len)
        sub = G_und.subgraph(largest_cc)
        avg_path = nx.average_shortest_path_length(sub)

    # Architecture Centrality Score (ACS) = Decision Authority Score
    # 구조적 betweenness는 Decentralized와 Centralized를 구분 못함 (동일 star topology).
    # DAS가 의미론적 차이(다수결 vs. 총체적 합성)를 올바르게 반영.
    acs = DAS[struct]

    centrality_rows.append({
        "Architecture": STRUCT_LABELS[struct],
        "struct_key":   struct,
        "Nodes": n_nodes,
        "Edges": n_edges,
        "Max Degree (norm)": round(max_deg_centrality, 3),
        "Max Betweenness": round(max_betweenness, 3),
        "Max PageRank": round(max_pagerank, 3),
        "Network Density": round(density, 3),
        "Avg Path Length": round(avg_path, 3),
        "Min Attack Cost": MIN_ATTACK_COST[struct],
        "APS": round(acs, 3),
    })

df_centrality = pd.DataFrame(centrality_rows)
ACS = dict(zip(df_centrality["struct_key"], df_centrality["APS"]))

print(df_centrality[["Architecture", "Max Betweenness", "Max PageRank", "Network Density", "Min Attack Cost", "APS"]].to_string(index=False))

# Table 1 저장 (STRUCTURES 순서 = ACS 오름차순)
tab1_out = df_centrality.drop(columns=["struct_key"])
tab1_out.to_csv(TAB_DIR / "qwen27b_centrality_table1_acs.csv", index=False)
with open(TAB_DIR / "qwen27b_centrality_table1_acs.tex", "w") as f:
    f.write(tab1_out.to_latex(index=False, escape=False,
        caption="Architecture Centrality Profile. Decision Authority Score (DAS) = the proportion of final decision controlled by a single-node compromise. Standard graph metrics (betweenness, PageRank) are shown for structural characterization; DAS is used as the primary centrality variable (ACS) for correlation analysis, as structural betweenness alone cannot distinguish Decentralized (majority vote) from Centralized (holistic synthesis).",
        label="tab:centrality"))
print("  Saved Table 1 (centrality profile)")

# ── FIG 01_a: Centrality Bar Chart ──────────────────────────────────────────
fig, axes = plt.subplots(1, 3, figsize=(13, 4))
metrics = ["Max Betweenness", "Max PageRank", "APS"]
titles  = ["Max Betweenness Centrality", "Max PageRank", "Adversarial Signal Preservation Score (APS)"]

# centrality 오름차순 순서 유지 (STRUCTURES 순서)
ordered_labels = [STRUCT_LABELS[s] for s in STRUCTURES]
ordered_colors = [PALETTE[s] for s in STRUCTURES]

for ax, metric, title in zip(axes, metrics, titles):
    vals = [df_centrality[df_centrality["struct_key"] == s][metric].values[0] for s in STRUCTURES]
    bars = ax.bar(ordered_labels, vals, color=ordered_colors, edgecolor="white", linewidth=0.8)
    ax.set_title(title, fontsize=11, fontweight="bold")
    ax.set_ylabel(metric, fontsize=10)
    ax.yaxis.grid(True, alpha=0.3); ax.set_axisbelow(True)
    for bar, val in zip(bars, vals):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.005, f"{val:.3f}",
                ha="center", va="bottom", fontsize=9)

plt.suptitle("Graph Centrality Metrics by Architecture (Ordered by APS)", fontsize=13, fontweight="bold")
plt.tight_layout()
plt.savefig(FIG_BASE / "01_centrality" / "fig01a_centrality_bar.pdf", dpi=200, bbox_inches="tight")
plt.close()
print("  Saved fig01a_centrality_bar.png")

# ── FIG 01_b: Architecture Network Topology ──────────────────────────────────
fig, axes = plt.subplots(1, 4, figsize=(18, 5))
layout_opts = {
    "linear":        {"prog": "dot"},
    "centralized":   {"prog": "dot"},
    "decentralized": {"prog": "dot"},
    "hybrid":        {"prog": "dot"},
}

LAYOUTS = {
    "linear":        {"M": (0,0), "S": (1,0), "N": (2,0), "Synth": (3,0)},
    "centralized":   {"M": (0,1), "S": (0,0), "N": (0,-1), "Coord": (1,0)},
    "decentralized": {"M": (0,1), "S": (0,0), "N": (0,-1), "Vote": (1,0)},
    "hybrid":        {
        "M": (0, 1), "S": (0, 0), "N": (0,-1),
        "M_rev": (1.5, 1), "S_rev": (1.5, 0), "N_rev": (1.5,-1),
        "Agg": (3, 0),
    },
}

NODE_COLORS_GRAPH = {
    "M": "#5B9BD5", "S": "#5B9BD5", "N": "#5B9BD5",
    "Synth": "#C44E52", "Coord": "#C44E52", "Vote": "#C44E52",
    "M_rev": "#9B59B6", "S_rev": "#9B59B6", "N_rev": "#9B59B6",
    "Agg": "#C44E52",
}

for ax, struct in zip(axes, STRUCTURES):
    G = nx.DiGraph()
    G.add_edges_from(GRAPH_EDGES[struct])
    pos = LAYOUTS[struct]
    node_colors = [NODE_COLORS_GRAPH.get(n, "#AAA") for n in G.nodes()]
    nx.draw_networkx(G, pos=pos, ax=ax,
                     node_color=node_colors, node_size=700,
                     font_size=7, font_color="white", font_weight="bold",
                     arrows=True, arrowsize=15,
                     edge_color="#555", width=1.5)
    ax.set_title(f"{STRUCT_LABELS[struct]}\nAPS = {ACS[struct]:.3f}",
                 fontsize=11, fontweight="bold")
    ax.axis("off")

legend_h = [
    mpatches.Patch(color="#5B9BD5", label="Analyst (L1)"),
    mpatches.Patch(color="#9B59B6", label="Revision (L2)"),
    mpatches.Patch(color="#C44E52", label="Decision Maker"),
]
fig.legend(handles=legend_h, loc="lower center", ncol=3, fontsize=10, bbox_to_anchor=(0.5, -0.02))
plt.suptitle("Information Flow Topology by Architecture (ordered by APS ↑)", fontsize=13, fontweight="bold")
plt.tight_layout()
plt.savefig(FIG_BASE / "01_centrality" / "fig01b_topology.pdf", dpi=200, bbox_inches="tight")
plt.close()
print("  Saved fig01b_topology.png")

# ══════════════════════════════════════════════════════════════════════════════
# STEP 2: Decision 데이터 로드 및 ASR 계산
# ══════════════════════════════════════════════════════════════════════════════
print("\n" + "=" * 70)
print("STEP 2: Loading decision data and computing ASR/DFR")
print("=" * 70)

def extract_signal(d):
    if not isinstance(d, str): return "UNKNOWN"
    if d.startswith("ERROR:"): return "ERROR"
    m = re.search(r"FINAL TRANSACTION PROPOSAL[:\s*]+\*{0,2}(BUY|SELL|HOLD)\*{0,2}", d, re.IGNORECASE)
    if m: return m.group(1).upper()
    t = re.findall(r"\b(BUY|SELL|HOLD)\b", d, re.IGNORECASE)
    return t[-1].upper() if t else "UNKNOWN"

def load_clean(struct):
    path = RESULTS_DIR / f"qwen3.6-27b-fp8_{struct}" / "all_tickers_results.csv"
    df = pd.read_csv(path)
    df["signal"] = df["decision"].apply(extract_signal)
    return df[["ticker", "date", "signal"]].rename(columns={"signal": "clean_signal"})

def load_attacked(struct, attack, direction):
    path = RESULTS_DIR / f"adv_qwen_{attack}_{direction.lower()}_{struct}" / "all_tickers_results.csv"
    if not path.exists():
        return None
    df = pd.read_csv(path)
    df["signal"] = df["decision"].apply(extract_signal)
    return df[["ticker", "date", "signal"]].rename(columns={"signal": "atk_signal"})

def compute_asr_dfr(merged, target_signal):
    valid = merged[merged["atk_signal"].isin(["BUY", "SELL", "HOLD"])]
    n = len(valid)
    if n == 0: return None
    asr = (valid["atk_signal"] == target_signal).sum() / n * 100
    dfr = (valid["atk_signal"] != valid["clean_signal"]).sum() / n * 100
    buy_pct  = valid["atk_signal"].eq("BUY").sum()  / n * 100
    sell_pct = valid["atk_signal"].eq("SELL").sum() / n * 100
    hold_pct = valid["atk_signal"].eq("HOLD").sum() / n * 100
    return {"ASR": round(asr, 1), "DFR": round(dfr, 1),
            "BUY%": round(buy_pct, 1), "SELL%": round(sell_pct, 1), "HOLD%": round(hold_pct, 1), "n": n}

# BUY 공격 메인 데이터 수집
records_buy  = []   # main analysis
records_sell = []   # supplementary

for struct in STRUCTURES:
    clean_df = load_clean(struct)
    for atk in ATTACKS_ALL:
        for direction in ["BUY", "SELL"]:
            atk_df = load_attacked(struct, atk, direction)
            if atk_df is None:
                continue
            merged = clean_df.merge(atk_df, on=["ticker", "date"], how="inner")
            m = compute_asr_dfr(merged, direction)
            if m:
                row = {"structure": struct, "attack": atk, "direction": direction, **m}
                if direction == "BUY":
                    records_buy.append(row)
                else:
                    records_sell.append(row)

df_buy  = pd.DataFrame(records_buy)
df_sell = pd.DataFrame(records_sell)
print(f"  BUY records: {len(df_buy)}, SELL records: {len(df_sell)}")

# ── Table 2: Main ASR (BUY, centrality-ordered columns, A3 별도) ─────────────
def build_asr_table(df, attacks, direction_label):
    rows = []
    for atk in attacks:
        row = {"Attack": f"{atk} ({ATTACK_LABELS[atk].replace(chr(10),' ')})"}
        sub = df[(df["attack"] == atk) & (df["direction"] == direction_label)]
        for struct in STRUCTURES:
            val = sub[sub["structure"] == struct]["ASR"].values
            row[STRUCT_LABELS[struct]] = val[0] if len(val) else "-"
        # Spearman ρ(ACS, ASR)
        acs_vals = [ACS[s] for s in STRUCTURES]
        asr_vals = [sub[sub["structure"] == s]["ASR"].values[0]
                    if len(sub[sub["structure"] == s]) else np.nan for s in STRUCTURES]
        valid_mask = ~np.isnan(asr_vals)
        if valid_mask.sum() >= 3:
            acs_v = np.array(acs_vals)[valid_mask]
            asr_v = np.array(asr_vals)[valid_mask]
            rho, _ = spearmanr(acs_v, asr_v)
            row["Spearman ρ"] = round(rho, 2)
        else:
            row["Spearman ρ"] = "-"
        rows.append(row)
    # Average row
    avg_row = {"Attack": "Average (D1–A2 excl. A3)"}
    for struct in STRUCTURES:
        sub = df[(df["attack"].isin(["D1","D2","A1","A2"])) & (df["direction"] == direction_label)]
        val = sub[sub["structure"] == struct]["ASR"].values
        avg_row[STRUCT_LABELS[struct]] = round(val.mean(), 1) if len(val) else "-"
    acs_vals = [ACS[s] for s in STRUCTURES]
    asr_avg  = [df[(df["attack"].isin(["D1","D2","A1","A2"])) &
                   (df["direction"] == direction_label) &
                   (df["structure"] == s)]["ASR"].mean() for s in STRUCTURES]
    rho, _ = spearmanr(acs_vals, asr_avg)
    avg_row["Spearman ρ"] = round(rho, 2)
    rows.append(avg_row)
    return pd.DataFrame(rows)

tab2 = build_asr_table(df_buy, ATTACKS_ALL, "BUY")
tab2.to_csv(TAB_DIR / "qwen27b_centrality_table2_asr_buy.csv", index=False)
with open(TAB_DIR / "qwen27b_centrality_table2_asr_buy.tex", "w") as f:
    f.write(tab2.to_latex(index=False, escape=False,
        caption="Attack Success Rate (ASR, \\%) under BUY-targeted attacks. Architectures ordered left-to-right by ascending centrality score (ACS). A3 achieves 100\\% across all structures (output-level hijacking; see Supplementary). Spearman $\\rho$ computed between ACS and ASR across four architectures.",
        label="tab:asr_buy"))
print("  Saved Table 2 (main ASR BUY)")

tab5 = build_asr_table(df_sell, ATTACKS_ALL, "SELL")
tab5.to_csv(TAB_DIR / "qwen27b_centrality_table5_asr_sell.csv", index=False)
print("  Saved Table 5 (supplementary ASR SELL)")

# ══════════════════════════════════════════════════════════════════════════════
# STEP 3: FWVS & Finance Metrics
# ══════════════════════════════════════════════════════════════════════════════
print("\n" + "=" * 70)
print("STEP 3: Computing FWVS and finance metrics")
print("=" * 70)

def load_prices():
    prices = {}
    for ticker in TICKERS:
        candidates = sorted(PRICE_DIR.glob(f"{ticker}-YFin-data-2025*.csv"))
        if not candidates:
            continue
        df = pd.read_csv(candidates[-1], parse_dates=["Date"])
        df = df[df["Date"] >= "2026-01-01"].copy()
        df["date"] = df["Date"].dt.strftime("%Y-%m-%d")
        prices[ticker] = df.set_index("date")["Close"].to_dict()
    return prices

prices = load_prices()
all_dates = sorted({d for p in prices.values() for d in p})
date_idx  = {d: i for i, d in enumerate(all_dates)}

def action_distance(a, b):
    """a = clean, b = attacked. Returns distance weight."""
    if a == b: return 0
    if {a, b} == {"BUY", "SELL"}: return 2
    return 1

def get_next_return(ticker, date):
    idx = date_idx.get(date)
    if idx is None or idx + 1 >= len(all_dates): return None
    p0 = prices.get(ticker, {}).get(date)
    p1 = prices.get(ticker, {}).get(all_dates[idx + 1])
    if p0 and p1 and p0 != 0:
        return (p1 - p0) / p0
    return None

def compute_fwvs(clean_df, atk_df):
    merged = clean_df.merge(atk_df, on=["ticker", "date"], how="inner")
    fwvs_vals = []
    for _, row in merged.iterrows():
        cs, as_ = row["clean_signal"], row["atk_signal"]
        if cs not in ("BUY","SELL","HOLD") or as_ not in ("BUY","SELL","HOLD"):
            continue
        dist = action_distance(cs, as_)
        if dist == 0:
            continue
        r = get_next_return(row["ticker"], row["date"])
        if r is not None:
            fwvs_vals.append(abs(r) * dist)
    return np.mean(fwvs_vals) if fwvs_vals else 0.0

def compute_cr_series(decisions_df, prices_dict):
    """
    Long-only equal-weight portfolio, 1-day signal lag.
    decisions_df must have columns: ticker, date, signal
    Returns: (cr_series indexed by date-string, annualized Sharpe, MDD%)
    """
    all_d  = sorted({d for p in prices_dict.values() for d in p})
    d_idx  = {d: i for i, d in enumerate(all_d)}

    # Per-ticker daily strategy returns (1-day lag: signal on day t → position day t+1)
    ticker_daily: dict[str, dict] = {}   # ticker → {date: return}

    for ticker in decisions_df["ticker"].unique():
        t_dec = (decisions_df[decisions_df["ticker"] == ticker]
                 [["date", "signal"]]
                 .sort_values("date")
                 .reset_index(drop=True))
        position = 0
        daily_ret: dict = {}
        for _, row in t_dec.iterrows():
            sig   = row["signal"]
            d_str = row["date"]
            # Update position based on signal
            if sig == "BUY"  and position == 0:
                position = 1
            elif sig == "SELL" and position == 1:
                position = 0
            # Position held from d_str → d_str+1 (1-day lag already implicit: next open)
            idx = d_idx.get(d_str)
            if idx is None or idx + 1 >= len(all_d):
                continue
            p0 = prices_dict.get(ticker, {}).get(d_str)
            p1 = prices_dict.get(ticker, {}).get(all_d[idx + 1])
            if p0 and p1 and p0 != 0:
                daily_ret[d_str] = position * (p1 - p0) / p0
        ticker_daily[ticker] = daily_ret

    if not ticker_daily:
        return pd.Series(dtype=float), 0.0, 0.0

    # Equal-weight portfolio: average across tickers that have data on each date
    portfolio_daily: dict = {}
    for d_str in all_d:
        rets = [v.get(d_str) for v in ticker_daily.values() if v.get(d_str) is not None]
        portfolio_daily[d_str] = np.mean(rets) if rets else 0.0

    dates_sorted = sorted(portfolio_daily.keys())
    vals  = [portfolio_daily[d] for d in dates_sorted]
    cum   = (np.cumprod([1 + r for r in vals]) - 1) * 100
    cr_series = pd.Series(cum, index=dates_sorted)

    # Sharpe (annualized)
    s = pd.Series(vals)
    sharpe = (s.mean() / s.std() * np.sqrt(252)) if s.std() > 0 else 0.0

    # Max Drawdown
    peak     = np.maximum.accumulate(cum)
    drawdown = cum - peak
    mdd      = float(drawdown.min())

    return cr_series, sharpe, mdd

# CR / Sharpe / MDD for clean baselines
print("  Computing clean baselines...")
clean_stats = {}
clean_series = {}
for struct in STRUCTURES:
    clean_df = load_clean(struct)
    clean_df = clean_df.rename(columns={"clean_signal": "signal"})
    ser, sharpe, mdd = compute_cr_series(clean_df, prices)
    clean_stats[struct]  = {"CR": ser.iloc[-1] if len(ser) else 0, "Sharpe": sharpe, "MDD": mdd}
    clean_series[struct] = ser

# Finance metrics per (struct, attack, BUY)
finance_rows = []
print("  Computing FWVS per attack...")
for struct in STRUCTURES:
    clean_df = load_clean(struct)
    for atk in ATTACKS_MAIN:
        atk_raw = load_attacked(struct, atk, "BUY")
        if atk_raw is None:
            continue
        # FWVS
        fwvs = compute_fwvs(clean_df, atk_raw)
        # CR degradation
        atk_sig = atk_raw.rename(columns={"atk_signal": "signal"})
        atk_series, atk_sharpe, atk_mdd = compute_cr_series(atk_sig, prices)
        delta_cr     = (atk_series.iloc[-1] if len(atk_series) else 0) - clean_stats[struct]["CR"]
        delta_sharpe = atk_sharpe - clean_stats[struct]["Sharpe"]
        delta_mdd    = atk_mdd    - clean_stats[struct]["MDD"]
        finance_rows.append({
            "structure":    struct,
            "attack":       atk,
            "ACS":          ACS[struct],
            "FWVS":         round(fwvs * 100, 4),   # ×100 for percentage-point interpretation
            "ΔCR (%)":     round(delta_cr, 2),
            "ΔSharpe":     round(delta_sharpe, 3),
            "ΔMDD (%)":    round(delta_mdd, 2),
        })

df_finance = pd.DataFrame(finance_rows)

# Table 3: per-architecture average (D1/D2/A1/A2)
tab3_rows = []
for struct in STRUCTURES:
    sub = df_finance[df_finance["structure"] == struct]
    tab3_rows.append({
        "Architecture (ACS ↑)": STRUCT_LABELS[struct],
        "ACS":           ACS[struct],
        "FWVS (×10²)":  round(sub["FWVS"].mean(), 4),
        "ΔCR (%)":      round(sub["ΔCR (%)"].mean(), 2),
        "ΔSharpe":      round(sub["ΔSharpe"].mean(), 3),
        "ΔMDD (%)":     round(sub["ΔMDD (%)"].mean(), 2),
    })
tab3 = pd.DataFrame(tab3_rows)
tab3.to_csv(TAB_DIR / "qwen27b_centrality_table3_finance.csv", index=False)
with open(TAB_DIR / "qwen27b_centrality_table3_finance.tex", "w") as f:
    f.write(tab3.to_latex(index=False, escape=False,
        caption="Finance-aware vulnerability metrics averaged across D1, D2, A1, A2 BUY-targeted attacks. FWVS $= \\overline{\\mathbf{1}[a_{clean} \\neq a_{atk}] \\cdot |r_{t+1}| \\cdot d_{action}}$ where $d_{action}=2$ (BUY$\\leftrightarrow$SELL), $1$ otherwise. $\\Delta$CR, $\\Delta$Sharpe, $\\Delta$MDD are relative to the clean baseline.",
        label="tab:finance"))
print("  Saved Table 3 (finance metrics)")

# ══════════════════════════════════════════════════════════════════════════════
# STEP 4: Statistical Validation
# ══════════════════════════════════════════════════════════════════════════════
print("\n" + "=" * 70)
print("STEP 4: Statistical validation (Spearman + bootstrap + permutation)")
print("=" * 70)

N_BOOTSTRAP   = 10_000
N_PERMUTATION = 10_000
RNG = np.random.default_rng(42)

# Per-attack ASR values across structures (BUY, ATTACKS_MAIN)
stat_rows = []
acs_arr = np.array([ACS[s] for s in STRUCTURES])

for atk in ATTACKS_MAIN + ["D1+D2+A1 avg"]:
    if atk == "D1+D2+A1 avg":
        asr_arr = np.array([
            df_buy[(df_buy["attack"].isin(["D1","D2","A1"])) & (df_buy["structure"] == s)]["ASR"].mean()
            for s in STRUCTURES
        ])
        label = "D1+D2+A1 (avg)"
    else:
        asr_arr = np.array([
            df_buy[(df_buy["attack"] == atk) & (df_buy["structure"] == s)]["ASR"].values[0]
            if len(df_buy[(df_buy["attack"] == atk) & (df_buy["structure"] == s)]) else np.nan
            for s in STRUCTURES
        ])
        label = atk

    rho, _ = spearmanr(acs_arr, asr_arr)

    # Bootstrap CI (resample architectures with replacement — limited but informative)
    boot_rhos = []
    for _ in range(N_BOOTSTRAP):
        idx = RNG.integers(0, len(STRUCTURES), size=len(STRUCTURES))
        acs_b = acs_arr[idx]
        asr_b = asr_arr[idx]
        if not np.any(np.isnan(asr_b)):
            import warnings
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                r, _ = spearmanr(acs_b, asr_b)
            if not np.isnan(r):
                boot_rhos.append(r)
    valid_boots = [r for r in boot_rhos if not np.isnan(r)]
    ci_lo = np.percentile(valid_boots, 2.5)  if valid_boots else np.nan
    ci_hi = np.percentile(valid_boots, 97.5) if valid_boots else np.nan

    # Permutation test (null: shuffle architecture labels)
    null_dist = []
    valid_mask = ~np.isnan(asr_arr)
    acs_v = acs_arr[valid_mask]
    asr_v = asr_arr[valid_mask]
    for _ in range(N_PERMUTATION):
        perm = RNG.permutation(asr_v)
        r_null, _ = spearmanr(acs_v, perm)
        null_dist.append(r_null)
    p_perm = (np.array(null_dist) >= rho).mean()

    stat_rows.append({
        "Attack": label,
        "Spearman ρ": round(rho, 3),
        "95% CI lo":  round(ci_lo, 3),
        "95% CI hi":  round(ci_hi, 3),
        "p (permutation)": round(p_perm, 3),
        "Sig.": "*" if p_perm < 0.1 else "",
    })
    print(f"  {label:15s}: ρ={rho:.3f}  CI=[{ci_lo:.3f},{ci_hi:.3f}]  p={p_perm:.3f}")

df_stats = pd.DataFrame(stat_rows)
df_stats.to_csv(TAB_DIR / "qwen27b_centrality_table4_stats.csv", index=False)
with open(TAB_DIR / "qwen27b_centrality_table4_stats.tex", "w") as f:
    f.write(df_stats.to_latex(index=False, escape=False,
        caption="Spearman rank correlation between Architecture Centrality Score (ACS) and Attack Success Rate (ASR) under BUY-targeted attacks. 95\\% CI from 10,000 bootstrap resamples. $p$-value from 10,000 permutation tests. Note: $n=4$ architectures; results interpreted as directional evidence only. *$p < 0.10$.",
        label="tab:stats"))
print("  Saved Table 4 (statistical validation)")

# ══════════════════════════════════════════════════════════════════════════════
# STEP 5: Core Figures
# ══════════════════════════════════════════════════════════════════════════════
print("\n" + "=" * 70)
print("STEP 5: Generating core figures")
print("=" * 70)

# ── FIG 2a: ASR Heatmap (centrality-ordered, BUY only) ──────────────────────
print("  Fig 2a: ASR heatmap (centrality-ordered)...")
struct_order = STRUCTURES          # already centrality-ascending
atk_order    = ATTACKS_MAIN        # D1, D2, A1, A2 (no A3)

pivot_data = pd.DataFrame(index=atk_order, columns=[STRUCT_LABELS[s] for s in struct_order])
for atk in atk_order:
    for struct in struct_order:
        val = df_buy[(df_buy["attack"] == atk) & (df_buy["structure"] == struct)]["ASR"].values
        pivot_data.loc[atk, STRUCT_LABELS[struct]] = val[0] if len(val) else np.nan
pivot_data = pivot_data.astype(float)

fig, ax = plt.subplots(figsize=(8, 5))
sns.heatmap(pivot_data, ax=ax, annot=True, fmt=".1f", cmap="RdYlGn_r",
            vmin=0, vmax=100, linewidths=0.5, linecolor="white",
            annot_kws={"size": 13, "weight": "bold"},
            cbar_kws={"label": "ASR (%)"})
ax.set_title("Attack Success Rate (ASR, %) — BUY-Targeted Attacks\n"
             "Architectures ordered by APS →",
             fontsize=12, fontweight="bold", pad=12)
ax.set_xlabel("Graph Structure (APS ↑)", fontsize=11)
ax.set_ylabel("Attack Type", fontsize=11)
ax.set_xticklabels([f"{STRUCT_LABELS[s]}\n(APS={ACS[s]:.3f})" for s in struct_order], fontsize=10)
ax.set_yticklabels(atk_order, fontsize=11, rotation=0)

plt.savefig(FIG_BASE / "02_vulnerability" / "fig02a_asr_heatmap_ordered.pdf", dpi=200, bbox_inches="tight")
plt.close()
print("    Saved fig02a")

# ── FIG 2b: Centrality–ASR Scatter (Main Figure) ────────────────────────────
print("  Fig 2b: Centrality-ASR scatter (main)...")
ATTACK_MARKERS = {"D1": "o", "D2": "s", "A1": "^", "A2": "D"}
ATTACK_COLORS_SCATTER = {
    "D1": "#1565C0", "D2": "#0288D1", "A1": "#E65100", "A2": "#BF360C"
}

fig, ax = plt.subplots(figsize=(9, 6))

for atk in ATTACKS_MAIN:
    xs, ys, labels_pt = [], [], []
    for struct in STRUCTURES:
        val = df_buy[(df_buy["attack"] == atk) & (df_buy["structure"] == struct)]["ASR"].values
        if len(val):
            xs.append(ACS[struct])
            ys.append(val[0])
            labels_pt.append(struct)

    ax.scatter(xs, ys, s=120, marker=ATTACK_MARKERS[atk],
               color=ATTACK_COLORS_SCATTER[atk], zorder=5,
               label=ATTACK_LABELS[atk].replace("\n", " "), edgecolors="white", linewidth=0.8)
    for x, y, lbl in zip(xs, ys, labels_pt):
        if atk == "A2" and lbl == "hybrid":
            ax.annotate("A2-Hybrid\n(peer-review\namplification)",
                        (x, y), fontsize=7.5, color="#BF360C",
                        xytext=(x - 0.07, y + 6),
                        arrowprops=dict(arrowstyle="->", color="#BF360C", lw=1.2))

# Trend line (D1+D2+A1 average, excluding A2 anomaly)
acs_x   = np.array([ACS[s] for s in STRUCTURES])
avg_asr = np.array([
    df_buy[(df_buy["attack"].isin(["D1","D2","A1"])) & (df_buy["structure"] == s)]["ASR"].mean()
    for s in STRUCTURES
])
# Monotone line using sorted ranks
sort_idx   = np.argsort(acs_x)
x_sorted   = acs_x[sort_idx]
y_sorted   = avg_asr[sort_idx]
ax.plot(x_sorted, y_sorted, "k--", linewidth=1.5, alpha=0.5, zorder=2,
        label="D1+D2+A1 avg (trend)")

# Bootstrap CI band for the trend (architecture-level, illustration)
ax.fill_between(x_sorted, y_sorted - 8, y_sorted + 8,
                alpha=0.08, color="black", label="±8pp band (illustrative)")

ax.set_xlabel("APS (Adversarial Signal Preservation Score)", fontsize=12)
ax.set_ylabel("Attack Success Rate (%)", fontsize=12)
ax.set_title("Architectural Centrality vs. Adversarial Vulnerability\n"
             "(BUY-targeted attacks, D1/D2/A1/A2)",
             fontsize=13, fontweight="bold")
ax.set_ylim(-5, 115)
ax.axhline(50, color="gray", linestyle=":", linewidth=0.8, alpha=0.5)

# Architecture labels on x-axis
for struct in STRUCTURES:
    ax.axvline(ACS[struct], color=PALETTE[struct], linestyle=":", alpha=0.3, linewidth=0.8)
    ax.text(ACS[struct], -4.5, STRUCT_LABELS[struct], ha="center", fontsize=9,
            color=PALETTE[struct], fontweight="bold")

ax.legend(fontsize=9, loc="lower right", framealpha=0.9)
ax.yaxis.grid(True, alpha=0.25); ax.set_axisbelow(True)

plt.tight_layout()
plt.savefig(FIG_BASE / "02_vulnerability" / "fig02b_centrality_asr_scatter.pdf", dpi=200, bbox_inches="tight")
plt.close()
print("    Saved fig02b (MAIN FIGURE)")

# ── FIG 2c: Vulnerability Radar Chart ────────────────────────────────────────
print("  Fig 2c: Vulnerability radar chart...")
categories = [ATTACK_LABELS[a].replace("\n"," ") for a in ATTACKS_MAIN]
N = len(categories)
angles = np.linspace(0, 2 * np.pi, N, endpoint=False).tolist()
angles += angles[:1]  # close

fig, ax = plt.subplots(figsize=(7, 6), subplot_kw={"polar": True})
for struct in STRUCTURES:
    vals = []
    for atk in ATTACKS_MAIN:
        v = df_buy[(df_buy["attack"] == atk) & (df_buy["structure"] == struct)]["ASR"].values
        vals.append(v[0] if len(v) else 0)
    vals += vals[:1]
    ax.plot(angles, vals, linewidth=2, color=PALETTE[struct],
            label=STRUCT_LABELS[struct])
    ax.fill(angles, vals, alpha=0.07, color=PALETTE[struct])

ax.set_xticks(angles[:-1])
ax.set_xticklabels(categories, fontsize=10)
ax.set_ylim(0, 110)
ax.set_yticks([25, 50, 75, 100])
ax.set_yticklabels(["25", "50", "75", "100"], fontsize=8, color="gray")
ax.set_title("Vulnerability Profile by Attack Type\n(BUY attacks, ASR %)",
             fontsize=12, fontweight="bold", pad=20)
ax.legend(loc="upper right", bbox_to_anchor=(1.35, 1.1), fontsize=10)

plt.tight_layout()
plt.savefig(FIG_BASE / "02_vulnerability" / "fig02c_radar.pdf", dpi=200, bbox_inches="tight")
plt.close()
print("    Saved fig02c")

# ── FIG 3a: FWVS vs Centrality Bubble Chart ──────────────────────────────────
print("  Fig 3a: FWVS bubble chart...")
ATTACK_COLORS_FIN = {
    "D1": "#1565C0", "D2": "#0288D1",
    "A1": "#E65100", "A2": "#BF360C",
}

fig, ax = plt.subplots(figsize=(8, 6))
for _, row in df_finance.iterrows():
    struct = row["structure"]
    atk    = row["attack"]
    x = row["ACS"]
    y = row["FWVS"]
    size = max(abs(row["ΔCR (%)"]) * 15, 30)
    ax.scatter(x, y, s=size, color=ATTACK_COLORS_FIN[atk],
               alpha=0.8, edgecolors="white", linewidth=0.8, zorder=5)
    ax.annotate(f"{STRUCT_LABELS[struct][:3]}",
                (x, y), fontsize=7.5, ha="left", va="bottom",
                xytext=(3, 3), textcoords="offset points", color="#444")

ax.set_xlabel("APS (Adversarial Signal Preservation Score)", fontsize=12)
ax.set_ylabel("FWVS (×10²)", fontsize=12)
ax.set_title("Financially Weighted Vulnerability vs. Centrality\n"
             "(Bubble size ∝ |ΔCR| — cumulative return degradation)",
             fontsize=12, fontweight="bold")

legend_h = [mpatches.Patch(color=ATTACK_COLORS_FIN[a], label=ATTACK_LABELS[a].replace("\n"," "))
            for a in ATTACKS_MAIN]
ax.legend(handles=legend_h, fontsize=9, loc="upper left")
ax.yaxis.grid(True, alpha=0.3); ax.xaxis.grid(True, alpha=0.3)
ax.set_axisbelow(True)

for struct in STRUCTURES:
    ax.axvline(ACS[struct], color=PALETTE[struct], linestyle=":", alpha=0.25, linewidth=0.8)

plt.tight_layout()
plt.savefig(FIG_BASE / "03_finance" / "fig03a_fwvs_bubble.pdf", dpi=200, bbox_inches="tight")
plt.close()
print("    Saved fig03a")

# ── FIG 3b: Cumulative Return Degradation Grid ───────────────────────────────
print("  Fig 3b: CR degradation grid (2×2)...")
fig, axes = plt.subplots(2, 2, figsize=(13, 8), sharey=False)
axes_flat = axes.flatten()

# centrality-ascending order for grid
STRUCT_GRID_ORDER = STRUCTURES  # Dec, Hybrid, Cent, Linear

centrality_alpha = {
    "decentralized": 0.10,
    "hybrid":        0.18,
    "centralized":   0.22,
    "linear":        0.28,
}

for ax, struct in zip(axes_flat, STRUCT_GRID_ORDER):
    bg_alpha = centrality_alpha[struct]
    ax.set_facecolor((*[1 - bg_alpha * 0.5] * 3, 1))   # Light gray gradient

    # Clean line
    ser_c = clean_series.get(struct, pd.Series())
    if len(ser_c):
        ax.plot(range(len(ser_c)), ser_c.values, color="black", linewidth=2.5,
                label="Clean", zorder=10, alpha=0.9)

    # D1-BUY (worst impact) and A2-BUY (Hybrid anomaly)
    for atk, color, ls in [("D1", "#1565C0", "-"), ("A2", "#BF360C", "--")]:
        atk_raw = load_attacked(struct, atk, "BUY")
        if atk_raw is None:
            continue
        atk_sig = atk_raw.rename(columns={"atk_signal": "signal"})
        ser_a, _, _ = compute_cr_series(atk_sig, prices)
        if len(ser_a):
            ax.plot(range(len(ser_a)), ser_a.values, color=color, linewidth=1.5,
                    linestyle=ls, label=f"{atk}-BUY", alpha=0.85, zorder=8)

    ax.axhline(0, color="gray", linewidth=0.6, linestyle=":")
    acs_val = ACS[struct]
    ax.set_title(f"{STRUCT_LABELS[struct]}  (APS = {acs_val:.3f})",
                 fontsize=11, fontweight="bold", color=PALETTE[struct])
    ax.set_xlabel("Trading Day", fontsize=9)
    if ax in axes[:, 0]:
        ax.set_ylabel("Cumulative Return (%)", fontsize=10)
    ax.yaxis.grid(True, alpha=0.2); ax.set_axisbelow(True)
    ax.legend(fontsize=8, loc="lower left")

plt.suptitle("Cumulative Return: Clean vs. D1-BUY vs. A2-BUY\n"
             "(Arranged by Centrality ↑, background shade ∝ APS)",
             fontsize=13, fontweight="bold")
plt.tight_layout()
plt.savefig(FIG_BASE / "03_finance" / "fig03b_cr_degradation_grid.pdf", dpi=200, bbox_inches="tight")
plt.close()
print("    Saved fig03b")

# ── FIG 4a: Statistical Correlation Plot ─────────────────────────────────────
print("  Fig 4a: Statistical correlation bar...")
fig, ax = plt.subplots(figsize=(8, 4))
x_pos = np.arange(len(df_stats))
rhos  = df_stats["Spearman ρ"].values.astype(float)
ci_lo = df_stats["95% CI lo"].values.astype(float)
ci_hi = df_stats["95% CI hi"].values.astype(float)
labels_s = df_stats["Attack"].tolist()

colors_bar = ["#C44E52" if r > 0.5 else "#DD8452" if r > 0 else "#4C72B0" for r in rhos]
ax.bar(x_pos, rhos, color=colors_bar, alpha=0.8, edgecolor="white")
ax.errorbar(x_pos, rhos, yerr=[rhos - ci_lo, ci_hi - rhos],
            fmt="none", color="black", capsize=5, linewidth=1.5)
ax.axhline(0, color="black", linewidth=0.8)
ax.axhline(0.7, color="gray", linestyle="--", linewidth=0.8, alpha=0.5)

for i, (r, p) in enumerate(zip(rhos, df_stats["p (permutation)"].values)):
    sig = "*" if float(p) < 0.1 else ""
    ax.text(i, r + 0.05, f"ρ={r:.2f}{sig}", ha="center", fontsize=9)

ax.set_xticks(x_pos)
ax.set_xticklabels(labels_s, fontsize=10)
ax.set_ylabel("Spearman ρ (APS, ASR)", fontsize=11)
ax.set_ylim(-0.3, 1.2)
ax.set_title("Centrality-Vulnerability Correlation by Attack Type\n"
             "(95% CI from bootstrap; *p<0.10 by permutation test)",
             fontsize=12, fontweight="bold")
ax.yaxis.grid(True, alpha=0.3); ax.set_axisbelow(True)

plt.tight_layout()
plt.savefig(FIG_BASE / "04_statistical" / "fig04a_correlation_bar.pdf", dpi=200, bbox_inches="tight")
plt.close()
print("    Saved fig04a")

# ══════════════════════════════════════════════════════════════════════════════
# STEP 6: Supplementary Figures
# ══════════════════════════════════════════════════════════════════════════════
print("\n" + "=" * 70)
print("STEP 6: Generating supplementary figures")
print("=" * 70)

# ── FIG S1: SELL attack ASR heatmap ──────────────────────────────────────────
print("  Fig S1: SELL attack heatmap...")
pivot_sell = pd.DataFrame(index=ATTACKS_MAIN, columns=[STRUCT_LABELS[s] for s in STRUCTURES])
for atk in ATTACKS_MAIN:
    for struct in STRUCTURES:
        val = df_sell[(df_sell["attack"] == atk) & (df_sell["structure"] == struct)]["ASR"].values
        pivot_sell.loc[atk, STRUCT_LABELS[struct]] = val[0] if len(val) else np.nan
pivot_sell = pivot_sell.astype(float)

fig, axes = plt.subplots(1, 2, figsize=(14, 5))
for ax, (data, title) in zip(axes, [(pivot_data, "BUY-Targeted ASR (%)"), (pivot_sell, "SELL-Targeted ASR (%)")]):
    sns.heatmap(data, ax=ax, annot=True, fmt=".1f", cmap="RdYlGn_r",
                vmin=0, vmax=100, linewidths=0.5, linecolor="white",
                annot_kws={"size": 12, "weight": "bold"},
                cbar_kws={"label": "ASR (%)"})
    ax.set_title(title, fontsize=12, fontweight="bold")
    ax.set_xlabel("Architecture (APS ↑)", fontsize=10)
    ax.set_ylabel("Attack Type", fontsize=10)
    ax.set_xticklabels([f"{STRUCT_LABELS[s]}\n({ACS[s]:.3f})" for s in STRUCTURES], fontsize=9)
    ax.set_yticklabels(ATTACKS_MAIN, rotation=0, fontsize=10)

plt.suptitle("BUY vs. SELL Targeted ASR: Market Direction Effects\n"
             "(Q1 2026 was a bearish quarter — SELL attacks align with market trend)",
             fontsize=12, fontweight="bold")
plt.tight_layout()
plt.savefig(FIG_BASE / "05_supplementary" / "figS1_sell_heatmap.pdf", dpi=200, bbox_inches="tight")
plt.close()
print("    Saved figS1")

# ── FIG S2: A3 Supplementary ─────────────────────────────────────────────────
print("  Fig S2: A3 (Decision Hijacking) analysis...")
a3_data = {"Architecture": [], "BUY-ASR": [], "SELL-ASR": []}
for struct in STRUCTURES:
    v_buy  = df_buy[(df_buy["attack"] == "A3") & (df_buy["structure"] == struct)]["ASR"].values
    v_sell = df_sell[(df_sell["attack"] == "A3") & (df_sell["structure"] == struct)]["ASR"].values
    a3_data["Architecture"].append(STRUCT_LABELS[struct])
    a3_data["BUY-ASR"].append(v_buy[0] if len(v_buy) else np.nan)
    a3_data["SELL-ASR"].append(v_sell[0] if len(v_sell) else np.nan)

df_a3 = pd.DataFrame(a3_data)
fig, ax = plt.subplots(figsize=(7, 4))
x = np.arange(len(STRUCTURES))
w = 0.35
ax.bar(x - w/2, df_a3["BUY-ASR"],  w, color="#1565C0", alpha=0.85, label="BUY")
ax.bar(x + w/2, df_a3["SELL-ASR"], w, color="#C44E52", alpha=0.85, label="SELL")
ax.set_xticks(x)
ax.set_xticklabels([f"{STRUCT_LABELS[s]}\n(APS={ACS[s]:.3f})" for s in STRUCTURES], fontsize=9)
ax.set_ylim(0, 115)
ax.set_ylabel("ASR (%)", fontsize=11)
ax.axhline(100, color="gray", linestyle="--", linewidth=0.8, alpha=0.5)
ax.set_title("A3 (Decision-Maker Hijacking) ASR — Output-Level Attack\n"
             "Achieves ~100% regardless of architectural centrality",
             fontsize=11, fontweight="bold")
ax.legend(fontsize=10)
ax.yaxis.grid(True, alpha=0.3); ax.set_axisbelow(True)
plt.tight_layout()
plt.savefig(FIG_BASE / "05_supplementary" / "figS2_a3_hijacking.pdf", dpi=200, bbox_inches="tight")
plt.close()
print("    Saved figS2")

# ── FIG S3: Finance Metrics Bar (Sharpe + MDD) ───────────────────────────────
print("  Fig S3: Finance degradation bars...")
avg_fin = df_finance.groupby("structure")[["ΔCR (%)", "ΔSharpe", "ΔMDD (%)"]].mean().reset_index()
avg_fin["Architecture"] = avg_fin["structure"].map(STRUCT_LABELS)
avg_fin = avg_fin.set_index("structure").loc[STRUCTURES].reset_index()

fig, axes = plt.subplots(1, 3, figsize=(14, 4))
for ax, col, ylabel in zip(axes,
    ["ΔCR (%)", "ΔSharpe", "ΔMDD (%)"],
    ["ΔCumulative Return (%)", "ΔSharpe Ratio", "ΔMax Drawdown (%)"]
):
    vals = avg_fin[col].values
    colors_b = [PALETTE[s] for s in STRUCTURES]
    ax.bar([STRUCT_LABELS[s] for s in STRUCTURES], vals, color=colors_b, alpha=0.85, edgecolor="white")
    ax.axhline(0, color="black", linewidth=0.8)
    ax.set_ylabel(ylabel, fontsize=10)
    ax.set_title(ylabel, fontsize=11, fontweight="bold")
    ax.yaxis.grid(True, alpha=0.3); ax.set_axisbelow(True)
    for i, (lbl, val) in enumerate(zip([STRUCT_LABELS[s] for s in STRUCTURES], vals)):
        ax.text(i, val - 0.3 if val < 0 else val + 0.1, f"{val:.2f}",
                ha="center", fontsize=9)

plt.suptitle("Finance Metric Degradation Under Attack (D1/D2/A1/A2 BUY avg)",
             fontsize=12, fontweight="bold")
plt.tight_layout()
plt.savefig(FIG_BASE / "05_supplementary" / "figS3_finance_bars.pdf", dpi=200, bbox_inches="tight")
plt.close()
print("    Saved figS3")

# ══════════════════════════════════════════════════════════════════════════════
# SUMMARY
# ══════════════════════════════════════════════════════════════════════════════
print("\n" + "=" * 70)
print("GENERATION COMPLETE")
print("=" * 70)
print("\nArchitecture Centrality Scores (ACS):")
for s in STRUCTURES:
    print(f"  {STRUCT_LABELS[s]:15s} ACS = {ACS[s]:.3f}")

print("\nSpearman ρ (ACS vs ASR, BUY attacks):")
print(df_stats[["Attack", "Spearman ρ", "p (permutation)", "Sig."]].to_string(index=False))

print("\nFinance Degradation (avg D1/D2/A1/A2 BUY):")
print(tab3.to_string(index=False))

print(f"\nFigures → {FIG_BASE}/")
for sub in ["01_centrality", "02_vulnerability", "03_finance", "04_statistical", "05_supplementary"]:
    figs = sorted((FIG_BASE / sub).glob("*.pdf"))
    print(f"  {sub}/: {[f.name for f in figs]}")

print(f"\nTables → {TAB_DIR}/")
print(f"  {[f.name for f in sorted(TAB_DIR.glob('qwen27b_centrality_*.csv'))]}")
