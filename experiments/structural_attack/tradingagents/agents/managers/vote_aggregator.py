"""
Decentralized Architecture: Each analyst votes independently.
Each analyst already states its own BUY/HOLD/SELL vote at the end of its report
(the "MY VOTE: ..." line, appended to its prompt in setup_decentralized.py).
This node parses those lines with a regex and tallies the majority in Python.
If an analyst's report is missing the exact "MY VOTE: ..." line, the vote isn't
just defaulted to HOLD anymore - the same kind of LLM-based recovery that
SignalProcessor.process_signal() already uses for the other topologies'
final decision (tradingagents/graph/signal_processing.py) is applied here too,
so a formatting slip doesn't silently count as a HOLD vote in the tally.
"""

import re
from collections import Counter

_VOTE_OPTIONS = ("BUY", "HOLD", "SELL")
_VOTE_LINE_RE = re.compile(r"MY VOTE:\s*(BUY|HOLD|SELL)", re.IGNORECASE)

# analyst_type (as used in selected_analysts) -> (display name, AgentState report key)
_ANALYST_META = {
    "market":       ("Market",       "market_report"),
    "social":       ("Sentiment",    "sentiment_report"),
    "news":         ("News",         "news_report"),
    "fundamentals": ("Fundamentals", "fundamentals_report"),
}

_FALLBACK_SYSTEM_PROMPT = (
    "You are an efficient assistant designed to analyze a financial analyst's report. "
    "The analyst was instructed to end their report with a line reading 'MY VOTE: BUY' "
    "(or HOLD, or SELL) but their report below does not contain that line in the "
    "expected format. Based on the analysis and conclusion in the report, determine "
    "which of BUY, SELL, or HOLD this analyst's findings support. Provide only that "
    "single word (BUY, SELL, or HOLD) as your output, without adding any additional "
    "text or information."
)


def _parse_vote(report_text: str) -> tuple[str, bool]:
    """
    Pure string parsing - no LLM call.
    Returns (vote, matched). matched=False means the analyst's report was missing
    the required 'MY VOTE: ...' line (or it wasn't one of BUY/HOLD/SELL).
    """
    match = _VOTE_LINE_RE.search(report_text or "")
    if match:
        return match.group(1).upper(), True
    return "HOLD", False


def _llm_fallback_vote(llm, report_text: str) -> str:
    """
    Recovery path for when _parse_vote finds no 'MY VOTE: ...' line: ask the LLM
    to read the analyst's own report and extract the decision it supports, the
    same way SignalProcessor.process_signal() does for every other topology's
    final decision text. Falls back to HOLD only if the LLM's own reply somehow
    doesn't contain any of BUY/HOLD/SELL either.
    """
    messages = [
        ("system", _FALLBACK_SYSTEM_PROMPT),
        ("human", report_text or ""),
    ]
    reply = llm.invoke(messages).content.strip().upper()
    for opt in _VOTE_OPTIONS:
        if opt in reply:
            return opt
    return "HOLD"


def create_vote_aggregator(llm, selected_analysts=None):
    """
    Vote aggregator for the Decentralized architecture.
    Each vote is read straight out of that analyst's own report text via regex.
    If the exact 'MY VOTE: ...' line is missing, `llm` is used to recover the
    vote from the report's actual content (see _llm_fallback_vote) instead of
    silently defaulting to HOLD.

    selected_analysts: analyst_type list actually wired into the graph for this run
        (e.g. ["market", "social", "news"]). Only these are tallied - an analyst
        that was never run has an empty report and would otherwise be counted as
        a guaranteed, phantom "parse failure -> HOLD" vote every single time.
        Defaults to all four for backward compatibility.
    """
    active = selected_analysts or list(_ANALYST_META)
    active_meta = {atype: _ANALYST_META[atype] for atype in active if atype in _ANALYST_META}
    n_voters = len(active_meta)

    def vote_node(state) -> dict:
        ticker = state["company_of_interest"]
        trade_date = state["trade_date"]
        reports = {
            display_name: state[report_key]
            for display_name, report_key in active_meta.values()
        }

        # 1. Each vote is read straight out of that analyst's own report text.
        #    On a regex miss, fall back to an LLM read of the same report text
        #    instead of defaulting straight to HOLD.
        votes: dict[str, str] = {}
        recovered: list[str] = []   # regex missed, LLM fallback recovered a real vote
        for name, text in reports.items():
            vote, ok = _parse_vote(text)
            if not ok:
                vote = _llm_fallback_vote(llm, text)
                recovered.append(name)
                print(
                    f"[VoteAggregator] {name} analyst report for {ticker} on "
                    f"{trade_date} had no valid 'MY VOTE: BUY/HOLD/SELL' line - "
                    f"recovered vote={vote} via LLM fallback."
                )
            votes[name] = vote

        # 2. Majority is tallied in code. Any tie for the top spot (2-2, or any
        #    other split with no single plurality winner) defaults to HOLD.
        tally = Counter(votes.values())
        top_count = max(tally.values())
        winners = [opt for opt in _VOTE_OPTIONS if tally.get(opt, 0) == top_count]
        decision = winners[0] if len(winners) == 1 else "HOLD"

        vote_lines = "\n".join(
            f"{name} Vote: {vote}" + ("  [RECOVERED via LLM fallback]" if name in recovered else "")
            for name, vote in votes.items()
        )
        tally_line = ", ".join(f"{opt}={tally.get(opt, 0)}" for opt in _VOTE_OPTIONS)
        recovery_line = (
            f"LLM-recovered votes: {len(recovered)}/{n_voters} analysts had no valid "
            f"MY VOTE line ({', '.join(recovered)}) - their vote was recovered via LLM "
            f"fallback, not read from a literal MY VOTE line."
            if recovered
            else f"LLM-recovered votes: 0/{n_voters} - all analysts emitted a valid MY VOTE line."
        )

        summary = f"""Decentralized Vote Result for {ticker} on {trade_date}:

{vote_lines}

Vote Tally: {tally_line}
Majority Decision: {decision}
{recovery_line}

FINAL TRANSACTION PROPOSAL: **{decision}**"""

        empty_invest_debate = {
            "bull_history": "",
            "bear_history": "",
            "history": "",
            "current_response": summary,
            "judge_decision": summary,
            "count": 0,
        }
        empty_risk_debate = {
            "risky_history": "",
            "safe_history": "",
            "neutral_history": "",
            "history": "",
            "latest_speaker": "",
            "current_risky_response": "",
            "current_safe_response": "",
            "current_neutral_response": "",
            "judge_decision": summary,
            "count": 0,
        }

        return {
            "investment_plan": summary,
            "trader_investment_plan": summary,
            "investment_debate_state": empty_invest_debate,
            "risk_debate_state": empty_risk_debate,
            "final_trade_decision": summary,
        }

    return vote_node
