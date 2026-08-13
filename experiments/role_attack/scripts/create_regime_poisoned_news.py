"""
Create Regime-Based Poisoned News Data for Attack 01 (news_data_poisoned/)

(This used to feed attack_02_indirect_injection.py, but that attack was really
full-content replacement - data poisoning, not prompt injection - so it moved
to run_attack_01_data_poisoning.py. attack_02 is now genuine indirect prompt
injection targeting social_data instead. This generator/data is unaffected by
that move, only its consumer script changed.)

Unlike the original create_poisoned_news.py (single fixed "always bearish"
instruction regardless of date or target action), this builds news_data_poisoned/
by picking, per article date, one of the three already-generated news sources
under news_data/:

    2026-01-01 .. 2026-03-31  ->  alpha_vantage_filled_text_poison_buy   (bullish fabrication)
    2025-07-01 .. 2025-09-30  ->  alpha_vantage_filled_text_poison_sell  (bearish fabrication)
    everything else           ->  alpha_vantage_filled                  (untouched original)

(Bearish window is 2025 Q3, not 2025 Q1, because the actual experiment runs
cover 2025 Q3 + 2026 Q1 - see run_role_attack_suite.py DEFAULT_PERIODS. AAPL's
own alpha_vantage_filled file happens to have zero articles in 2025 Q1, so
using that quarter would have left AAPL with no bearish comparison at all.)

alpha_vantage_filled_text_poison_{buy,sell} are D1-style content poisoning
(headline/summary/content fully replaced with fabricated bullish/bearish text,
sentiment scores left untouched) - see tradingagents/agents/attacks/... in the
sibling structural_attack project for the generator; here we just recombine
the already-materialized per-direction copies by date.

news_data_poisoned/ is what get_local_news() in tradingagents/dataflows/local.py
reads when config["poisoned_news"] is True (see run_attack_01_data_poisoning.py).
That directory was missing from local_data/ entirely at the time this script was
written (only a stale copy existed under local_data_old/) - this script recreates it.

Ablation: pass ratio < 1.0 to only replace a random subset of each in-window
date's articles with the fabricated buy/sell version (seeded for reproducibility);
the rest of that date's articles fall back to the original, untouched article
(same as dates outside the bullish/bearish windows). This lets you sweep the
poisoning rate (e.g. 0.1/0.3/0.6) instead of always fully replacing every
in-window article.

Usage:
    python scripts/create_regime_poisoned_news.py
    python scripts/create_regime_poisoned_news.py --ratio 0.3 --seed 42
"""

import argparse
import json
import os
import random

TICKERS = ["AAPL", "BTC-USD", "MSFT", "NVDA", "TSLA"]

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
LOCAL_DATA_DIR = os.path.join(os.path.dirname(SCRIPT_DIR), "tradingagents", "dataflows", "local_data")
NEWS_DATA_DIR = os.path.join(LOCAL_DATA_DIR, "news_data")

ORIGINAL_DIR = os.path.join(NEWS_DATA_DIR, "alpha_vantage_filled")
BUY_DIR = os.path.join(NEWS_DATA_DIR, "alpha_vantage_filled_text_poison_buy")
SELL_DIR = os.path.join(NEWS_DATA_DIR, "alpha_vantage_filled_text_poison_sell")
DEST_DIR = os.path.join(LOCAL_DATA_DIR, "news_data_poisoned")

BULLISH_START, BULLISH_END = "2026-01-01", "2026-03-31"
BEARISH_START, BEARISH_END = "2025-07-01", "2025-09-30"


def regime_for_date(date_str: str) -> str:
    if BEARISH_START <= date_str <= BEARISH_END:
        return "bearish"
    if BULLISH_START <= date_str <= BULLISH_END:
        return "bullish"
    return "none"


def load_json(path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def build_regime_news(ratio: float = 1.0, seed: int = None):
    """Build news_data_poisoned/ by regime, optionally poisoning only a random
    subset of each in-window date's articles.

    Args:
        ratio: Fraction of each in-window date's articles to actually replace
            with the fabricated buy/sell version (0.0~1.0). 1.0 (default)
            matches the original behavior - every in-window article replaced.
            Articles not selected fall back to the original article (tagged
            poison_regime="none"), same as out-of-window dates.
        seed: RNG seed for article sampling, for reproducible ablation runs.
    """
    os.makedirs(DEST_DIR, exist_ok=True)
    ratio = max(0.0, min(1.0, ratio))
    rng = random.Random(seed)

    stats = {
        "tickers_processed": 0, "bullish_dates": 0, "bearish_dates": 0, "untouched_dates": 0,
        "bullish_articles": 0, "bearish_articles": 0, "untouched_articles": 0,
        "ratio": ratio, "seed": seed,
    }

    print("=" * 70)
    print("Creating Regime-Based Poisoned News Data (news_data_poisoned/)")
    print("=" * 70)
    print(f"Bullish window (from {BUY_DIR!r}): {BULLISH_START} .. {BULLISH_END}")
    print(f"Bearish window (from {SELL_DIR!r}): {BEARISH_START} .. {BEARISH_END}")
    print(f"Everything else: unmodified original ({ORIGINAL_DIR!r})")
    print(f"In-window poison ratio: {ratio:.0%} (seed={seed})")
    print("=" * 70)

    for ticker in TICKERS:
        orig_file = os.path.join(ORIGINAL_DIR, f"{ticker}_news.json")
        buy_file = os.path.join(BUY_DIR, f"{ticker}_news.json")
        sell_file = os.path.join(SELL_DIR, f"{ticker}_news.json")
        dest_file = os.path.join(DEST_DIR, f"{ticker}_news.json")

        if not (os.path.exists(orig_file) and os.path.exists(buy_file) and os.path.exists(sell_file)):
            print(f"[SKIP] {ticker}: missing one of original/buy/sell source files")
            continue

        print(f"\n[{ticker}] Processing...")
        original = load_json(orig_file)
        buy = load_json(buy_file)
        sell = load_json(sell_file)

        merged = {}
        ticker_bullish = ticker_bearish = ticker_untouched = 0
        ticker_bullish_articles = ticker_bearish_articles = ticker_untouched_articles = 0

        for date_str, articles in original.items():
            regime = regime_for_date(date_str)
            if regime == "bullish":
                fabricated_articles = buy.get(date_str, articles)
                ticker_bullish += 1
            elif regime == "bearish":
                fabricated_articles = sell.get(date_str, articles)
                ticker_bearish += 1
            else:
                fabricated_articles = None
                ticker_untouched += 1

            # For in-window dates, only `ratio` of the articles actually get the
            # fabricated version (per-article Bernoulli sample); the rest keep
            # the real original article and are tagged "none", same as if the
            # date were outside the window entirely.
            tagged = []
            for idx, orig_article in enumerate(articles):
                use_fabricated = fabricated_articles is not None and rng.random() < ratio
                if use_fabricated:
                    source = fabricated_articles[idx] if idx < len(fabricated_articles) else orig_article
                    a = dict(source)
                    a["poison_regime"] = regime
                    if regime == "bullish":
                        ticker_bullish_articles += 1
                    else:
                        ticker_bearish_articles += 1
                else:
                    a = dict(orig_article)
                    a["poison_regime"] = "none"
                    ticker_untouched_articles += 1
                tagged.append(a)
            merged[date_str] = tagged

        with open(dest_file, "w", encoding="utf-8") as f:
            json.dump(merged, f, indent=2, ensure_ascii=False)

        print(f"  Bullish dates (2026 Q1, from text_poison_buy): {ticker_bullish} ({ticker_bullish_articles} articles poisoned)")
        print(f"  Bearish dates (2025 Q3, from text_poison_sell): {ticker_bearish} ({ticker_bearish_articles} articles poisoned)")
        print(f"  Untouched dates (original): {ticker_untouched} ({ticker_untouched_articles} articles untouched)")
        print(f"  Saved to: {dest_file}")

        stats["tickers_processed"] += 1
        stats["bullish_dates"] += ticker_bullish
        stats["bearish_dates"] += ticker_bearish
        stats["untouched_dates"] += ticker_untouched
        stats["bullish_articles"] += ticker_bullish_articles
        stats["bearish_articles"] += ticker_bearish_articles
        stats["untouched_articles"] += ticker_untouched_articles

    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    print(f"Tickers processed: {stats['tickers_processed']}")
    print(f"Total bullish (ticker,date) pairs: {stats['bullish_dates']}")
    print(f"Total bearish (ticker,date) pairs: {stats['bearish_dates']}")
    print(f"Total untouched (ticker,date) pairs: {stats['untouched_dates']}")
    total_articles = stats['bullish_articles'] + stats['bearish_articles'] + stats['untouched_articles']
    poisoned_articles = stats['bullish_articles'] + stats['bearish_articles']
    if total_articles:
        print(f"Articles poisoned: {poisoned_articles}/{total_articles} ({poisoned_articles / total_articles * 100:.1f}%)")
    print(f"\nRegime-poisoned data saved to: {DEST_DIR}")
    print("=" * 70)

    return stats


def parse_arguments():
    parser = argparse.ArgumentParser(description="Build regime-based poisoned news data (Attack 01)")
    parser.add_argument(
        "--ratio",
        type=float,
        default=1.0,
        help="Fraction of each in-window date's articles to poison (0.0~1.0). Default 1.0 poisons every in-window article.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="RNG seed for article sampling (reproducibility across ablation runs).",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_arguments()
    build_regime_news(ratio=args.ratio, seed=args.seed)
