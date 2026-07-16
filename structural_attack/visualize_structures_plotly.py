"""
Interactive agent structure diagram via Plotly → HTML.
Opens in VSCode Simple Browser or system browser.
"""

import plotly.graph_objects as go
import webbrowser
from pathlib import Path

OUT = Path(__file__).parent / "figures" / "agent_structures.html"
OUT.parent.mkdir(exist_ok=True)

# ── Layout constants ──────────────────────────────────────────────────────────
# 4 columns, each centered at X_CENTERS[i]
X_CENTERS   = [1.5, 4.5, 7.5, 10.5]
COL_W       = 2.2        # analyst box width
COL_H       = 0.55       # analyst box height
DEC_H       = 0.70       # decision node height
TOOL_W      = 0.70
TOOL_H      = 0.32
SE_R        = 0.22       # start/end circle radius

Y_TITLE     = 9.6
Y_START     = 9.0
Y_ANALYSTS  = [7.8, 6.7, 5.6, 4.5]   # Market, Social, News, Fundamentals
Y_DEC       = 3.2
Y_END       = 2.0
Y_LABEL     = 1.55

# ── Colors ────────────────────────────────────────────────────────────────────
C_ANALYST   = "#1565C0"
C_TOOL      = "#388E3C"
C_SE        = "#455A64"
C_TEXT      = "white"
C_DARK      = "#212121"
C_DEC = {
    0: "#6A1B9A",   # Linear  → purple
    1: "#E65100",   # Centralized → orange
    2: "#00695C",   # Decentralized → teal
    3: "#AD1457",   # MoA → pink
}

ANALYST_LABELS = ["Market\nAnalyst", "Social\nAnalyst",
                  "News\nAnalyst", "Fund.\nAnalyst"]

STRUCTURES = [
    {"title": "(a) Linear",        "dec": "Linear\nSynthesizer",   "sub": "sequential accumulation"},
    {"title": "(b) Centralized",   "dec": "Central\nCoordinator",  "sub": "holistic synthesis"},
    {"title": "(c) Decentralized", "dec": "Vote\nAggregator",      "sub": "majority voting"},
    {"title": "(d) MoA",           "dec": "MoA\nAggregator",       "sub": "diversity-aware"},
]

shapes      = []
annotations = []


# ── Helper: rounded rect shape ────────────────────────────────────────────────
def box(cx, cy, w, h, fc, lc="#FFFFFF", lw=1.5):
    shapes.append(dict(
        type="rect",
        x0=cx - w/2, x1=cx + w/2,
        y0=cy - h/2, y1=cy + h/2,
        fillcolor=fc, line=dict(color=lc, width=lw),
        layer="below",
    ))


def label(cx, cy, text, size=9, color=C_TEXT, bold=False):
    annotations.append(dict(
        x=cx, y=cy, text=text.replace("\n", "<br>"),
        showarrow=False,
        font=dict(size=size, color=color,
                  family="Arial Black" if bold else "Arial"),
        xanchor="center", yanchor="middle",
        align="center",
    ))


def arrow_v(x, y1, y2, color="#455A64", width=1.5):
    """Vertical downward arrow."""
    annotations.append(dict(
        x=x, y=y2, ax=x, ay=y1,
        xref="x", yref="y", axref="x", ayref="y",
        showarrow=True,
        arrowhead=2, arrowsize=1.0, arrowwidth=width,
        arrowcolor=color,
    ))


def circle(cx, cy, r, fc):
    shapes.append(dict(
        type="circle",
        x0=cx - r, x1=cx + r,
        y0=cy - r, y1=cy + r,
        fillcolor=fc, line=dict(color="white", width=1.5),
        layer="below",
    ))


# ── Draw each structure ───────────────────────────────────────────────────────
for i, (cx, s) in enumerate(zip(X_CENTERS, STRUCTURES)):
    dc = C_DEC[i]

    # Title
    label(cx, Y_TITLE, f"<b>{s['title']}</b>", size=12, color=C_DARK, bold=True)

    # START
    circle(cx, Y_START, SE_R, C_SE)
    label(cx, Y_START, "<b>START</b>", size=7.5)

    # START → Market
    arrow_v(cx, Y_START - SE_R, Y_ANALYSTS[0] + COL_H/2)

    # Analysts + tool loops
    for j, (ay, al) in enumerate(zip(Y_ANALYSTS, ANALYST_LABELS)):
        box(cx, ay, COL_W, COL_H, C_ANALYST)
        label(cx, ay, f"<b>{al}</b>", size=8.5)

        # Tool box (right side)
        tx = cx + COL_W/2 + TOOL_W/2 + 0.05
        box(tx, ay, TOOL_W, TOOL_H, C_TOOL, lw=1.2)
        label(tx, ay, "<b>Tools</b>", size=7.5)

        # Analyst → Tool  (right arrow)
        annotations.append(dict(
            x=tx - TOOL_W/2, y=ay + 0.07,
            ax=cx + COL_W/2, ay=ay + 0.07,
            xref="x", yref="y", axref="x", ayref="y",
            showarrow=True, arrowhead=2, arrowsize=0.8,
            arrowwidth=1.2, arrowcolor=C_TOOL,
        ))
        # Tool → Analyst  (left return arrow)
        annotations.append(dict(
            x=cx + COL_W/2, y=ay - 0.07,
            ax=tx - TOOL_W/2, ay=ay - 0.07,
            xref="x", yref="y", axref="x", ayref="y",
            showarrow=True, arrowhead=2, arrowsize=0.8,
            arrowwidth=1.2, arrowcolor=C_TOOL,
        ))

        # Chain arrows between analysts
        if j < len(Y_ANALYSTS) - 1:
            arrow_v(cx, ay - COL_H/2, Y_ANALYSTS[j+1] + COL_H/2)

    # Last analyst → Decision node
    arrow_v(cx, Y_ANALYSTS[-1] - COL_H/2, Y_DEC + DEC_H/2, width=2.0)

    # Decision node
    box(cx, Y_DEC, COL_W, DEC_H, dc, lc="white", lw=2)
    label(cx, Y_DEC + 0.13, f"<b>{s['dec']}</b>", size=9.5)
    label(cx, Y_DEC - 0.18, f"<i>{s['sub']}</i>", size=7.5)

    # Decision → END
    arrow_v(cx, Y_DEC - DEC_H/2, Y_END + SE_R, width=2.0)

    # END
    circle(cx, Y_END, SE_R, C_SE)
    label(cx, Y_END, "<b>END</b>", size=7.5)

    # BUY/SELL/HOLD label
    label(cx, Y_LABEL, "<i>BUY / SELL / HOLD</i>", size=8, color="#546E7A")


# ── Shared-layer brace label ──────────────────────────────────────────────────
mid_x = (X_CENTERS[0] + X_CENTERS[-1]) / 2
annotations.append(dict(
    x=mid_x, y=(Y_ANALYSTS[0] + Y_ANALYSTS[-1]) / 2,
    text="← identical analyst layer across all structures →",
    showarrow=False,
    font=dict(size=8.5, color="#1565C0", family="Arial"),
    xanchor="center", yanchor="middle",
    bgcolor="#E3F2FD", bordercolor="#90CAF9", borderwidth=1,
    borderpad=4, opacity=0.85,
))

# ── Vertical dividers ─────────────────────────────────────────────────────────
for div_x in [3.0, 6.0, 9.0]:
    shapes.append(dict(
        type="line",
        x0=div_x, x1=div_x, y0=1.2, y1=9.4,
        line=dict(color="#CFD8DC", width=1.0, dash="dot"),
    ))

# ── Legend ────────────────────────────────────────────────────────────────────
legend_items = [
    (C_ANALYST,        "Analyst Agent  (shared LLM)"),
    (C_TOOL,           "Tool Node  (data retrieval)"),
    (C_DEC[0],         "Linear Synthesizer  – sequential accumulation"),
    (C_DEC[1],         "Central Coordinator  – holistic synthesis"),
    (C_DEC[2],         "Vote Aggregator  – majority voting"),
    (C_DEC[3],         "MoA Aggregator  – diversity-aware"),
    (C_SE,             "START / END"),
]

leg_x0, leg_y0 = 0.2, 1.0
for k, (fc, txt) in enumerate(legend_items):
    col, row = divmod(k, 4)
    lx = leg_x0 + col * 4.0
    ly = leg_y0 - row * 0.0   # single row
    # We'll put them all on one row using a table-style annotation
    pass

# Simpler: one annotation block
leg_text = "  ".join(
    f'<span style="color:{fc}">■</span> {txt}'
    for fc, txt in legend_items
)
annotations.append(dict(
    x=6.0, y=0.75,
    text=leg_text,
    showarrow=False,
    font=dict(size=8, family="Arial"),
    xanchor="center", yanchor="middle",
    bgcolor="#F5F5F5", bordercolor="#BDBDBD", borderwidth=1,
    borderpad=6,
))

# ── Figure layout ─────────────────────────────────────────────────────────────
fig = go.Figure()
fig.update_layout(
    shapes=shapes,
    annotations=annotations,
    xaxis=dict(range=[-0.2, 12.2], visible=False, fixedrange=True),
    yaxis=dict(range=[0.4, 10.2],  visible=False, fixedrange=True,
               scaleanchor="x", scaleratio=1),
    width=1200, height=820,
    margin=dict(l=10, r=10, t=50, b=10),
    paper_bgcolor="#F7F9FC",
    plot_bgcolor="#F7F9FC",
    title=dict(
        text="Multi-Agent Graph Structures for Financial Decision-Making",
        x=0.5, xanchor="center",
        font=dict(size=16, family="Arial Black"),
    ),
    dragmode=False,
)

fig.write_html(str(OUT), include_plotlyjs="cdn", full_html=True)
print(f"Saved → {OUT}")

# Try to open in browser / VSCode
try:
    webbrowser.open(f"file://{OUT.resolve()}")
except Exception:
    pass
