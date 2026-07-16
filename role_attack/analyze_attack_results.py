"""
Adversarial Attack Results Analyzer

공격 결과를 분석하고 시각화합니다.
"""

import os
import json
import pandas as pd
import matplotlib.pyplot as plt
from collections import defaultdict
from typing import Dict, List
from adversarial_attack import AttackGoal, AttackTarget, AttackType


class AttackResultAnalyzer:
    """공격 결과 분석기"""

    def __init__(self, results_dir: str = "adversarial_results"):
        self.results_dir = results_dir
        self.results = []
        self._load_results()

    def _load_results(self):
        """결과 파일들 로드"""
        if not os.path.exists(self.results_dir):
            print(f"Results directory not found: {self.results_dir}")
            return

        for filename in os.listdir(self.results_dir):
            if filename.endswith("_result.json"):
                filepath = os.path.join(self.results_dir, filename)
                with open(filepath, 'r') as f:
                    self.results.append(json.load(f))

        print(f"Loaded {len(self.results)} attack results")

    def generate_report(self) -> str:
        """전체 분석 리포트 생성"""
        if not self.results:
            return "No attack results found."

        report = []
        report.append("=" * 70)
        report.append("ADVERSARIAL ATTACK ANALYSIS REPORT")
        report.append("=" * 70)

        # 1. 전체 통계
        total = len(self.results)
        successful = sum(1 for r in self.results if r.get("attack_success", False))
        success_rate = successful / total * 100 if total > 0 else 0

        report.append(f"\n## 1. 전체 통계")
        report.append(f"   총 공격 시도: {total}")
        report.append(f"   성공: {successful}")
        report.append(f"   실패: {total - successful}")
        report.append(f"   성공률: {success_rate:.1f}%")

        # 2. 공격 목표별 분석
        report.append(f"\n## 2. 공격 목표별 성공률")
        goal_stats = defaultdict(lambda: {"total": 0, "success": 0})

        for r in self.results:
            goal = r.get("attack_goal", "unknown")
            goal_stats[goal]["total"] += 1
            if r.get("attack_success", False):
                goal_stats[goal]["success"] += 1

        for goal, stats in sorted(goal_stats.items()):
            rate = stats["success"] / stats["total"] * 100 if stats["total"] > 0 else 0
            report.append(f"   {goal:15s}: {stats['success']}/{stats['total']} ({rate:.1f}%)")

        # 3. 결정 변화 분석
        report.append(f"\n## 3. 결정 변화 패턴")
        decision_changes = defaultdict(int)

        for r in self.results:
            orig = self._extract_decision(r.get("original_decision", ""))
            adv = self._extract_decision(r.get("adversarial_decision", ""))
            if orig != "UNKNOWN" and adv != "UNKNOWN":
                decision_changes[f"{orig} → {adv}"] += 1

        for change, count in sorted(decision_changes.items(), key=lambda x: -x[1]):
            report.append(f"   {change}: {count}건")

        # 4. 티커별 취약성
        report.append(f"\n## 4. 티커별 취약성")
        ticker_stats = defaultdict(lambda: {"total": 0, "success": 0})

        for r in self.results:
            ticker = r.get("ticker", "unknown")
            ticker_stats[ticker]["total"] += 1
            if r.get("attack_success", False):
                ticker_stats[ticker]["success"] += 1

        for ticker, stats in sorted(ticker_stats.items()):
            rate = stats["success"] / stats["total"] * 100 if stats["total"] > 0 else 0
            vulnerability = "🔴 HIGH" if rate > 60 else "🟡 MEDIUM" if rate > 30 else "🟢 LOW"
            report.append(f"   {ticker:6s}: {stats['success']}/{stats['total']} ({rate:.1f}%) {vulnerability}")

        # 5. 취약점 요약
        report.append(f"\n## 5. 취약점 요약")

        if success_rate > 50:
            report.append("   ⚠️  시스템이 adversarial 공격에 매우 취약합니다.")
            report.append("   권장 조치:")
            report.append("   - 입력 데이터 검증 강화")
            report.append("   - Agent 프롬프트에 방어 지시문 추가")
            report.append("   - Multi-agent 합의 메커니즘 강화")
        elif success_rate > 20:
            report.append("   ⚠️  일부 공격 벡터에 취약성이 발견되었습니다.")
            report.append("   권장 조치:")
            report.append("   - 취약한 Agent 식별 및 강화")
            report.append("   - 데이터 소스 신뢰성 검증 추가")
        else:
            report.append("   ✓ 시스템이 대부분의 공격에 대해 견고합니다.")

        report.append("\n" + "=" * 70)

        return "\n".join(report)

    def _extract_decision(self, decision_text: str) -> str:
        """결정 텍스트에서 BUY/SELL/HOLD 추출"""
        if not decision_text:
            return "UNKNOWN"

        text_upper = decision_text.upper()

        if "**BUY**" in text_upper or "FINAL TRANSACTION PROPOSAL: BUY" in text_upper:
            return "BUY"
        elif "**SELL**" in text_upper or "FINAL TRANSACTION PROPOSAL: SELL" in text_upper:
            return "SELL"
        elif "**HOLD**" in text_upper or "FINAL TRANSACTION PROPOSAL: HOLD" in text_upper:
            return "HOLD"

        if "BUY" in text_upper[-200:]:
            return "BUY"
        elif "SELL" in text_upper[-200:]:
            return "SELL"
        elif "HOLD" in text_upper[-200:]:
            return "HOLD"

        return "UNKNOWN"

    def plot_results(self, save_path: str = None):
        """결과 시각화"""
        if not self.results:
            print("No results to plot")
            return

        fig, axes = plt.subplots(2, 2, figsize=(14, 10))

        # 1. 공격 목표별 성공률
        ax1 = axes[0, 0]
        goal_stats = defaultdict(lambda: {"total": 0, "success": 0})
        for r in self.results:
            goal = r.get("attack_goal", "unknown")
            goal_stats[goal]["total"] += 1
            if r.get("attack_success", False):
                goal_stats[goal]["success"] += 1

        goals = list(goal_stats.keys())
        success_rates = [goal_stats[g]["success"] / goal_stats[g]["total"] * 100
                        for g in goals]

        bars = ax1.bar(goals, success_rates, color=['#e74c3c' if r > 50 else '#f39c12' if r > 20 else '#27ae60'
                                                     for r in success_rates])
        ax1.set_ylabel("Success Rate (%)")
        ax1.set_title("Attack Success Rate by Goal")
        ax1.set_ylim(0, 100)
        for bar, rate in zip(bars, success_rates):
            ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 2,
                    f'{rate:.1f}%', ha='center', va='bottom', fontsize=9)

        # 2. 결정 변화 매트릭스
        ax2 = axes[0, 1]
        decisions = ["BUY", "SELL", "HOLD"]
        matrix = [[0] * 3 for _ in range(3)]

        for r in self.results:
            orig = self._extract_decision(r.get("original_decision", ""))
            adv = self._extract_decision(r.get("adversarial_decision", ""))
            if orig in decisions and adv in decisions:
                i = decisions.index(orig)
                j = decisions.index(adv)
                matrix[i][j] += 1

        im = ax2.imshow(matrix, cmap='YlOrRd')
        ax2.set_xticks(range(3))
        ax2.set_yticks(range(3))
        ax2.set_xticklabels(decisions)
        ax2.set_yticklabels(decisions)
        ax2.set_xlabel("After Attack")
        ax2.set_ylabel("Before Attack")
        ax2.set_title("Decision Change Matrix")

        for i in range(3):
            for j in range(3):
                ax2.text(j, i, matrix[i][j], ha="center", va="center", color="black")

        # 3. 티커별 취약성
        ax3 = axes[1, 0]
        ticker_stats = defaultdict(lambda: {"total": 0, "success": 0})
        for r in self.results:
            ticker = r.get("ticker", "unknown")
            ticker_stats[ticker]["total"] += 1
            if r.get("attack_success", False):
                ticker_stats[ticker]["success"] += 1

        tickers = list(ticker_stats.keys())
        vuln_rates = [ticker_stats[t]["success"] / ticker_stats[t]["total"] * 100
                     for t in tickers]

        bars = ax3.barh(tickers, vuln_rates,
                       color=['#e74c3c' if r > 60 else '#f39c12' if r > 30 else '#27ae60'
                              for r in vuln_rates])
        ax3.set_xlabel("Vulnerability Rate (%)")
        ax3.set_title("Vulnerability by Ticker")
        ax3.set_xlim(0, 100)

        # 4. 전체 성공/실패 파이차트
        ax4 = axes[1, 1]
        successful = sum(1 for r in self.results if r.get("attack_success", False))
        failed = len(self.results) - successful

        if successful + failed > 0:
            ax4.pie([successful, failed], labels=["Success", "Failed"],
                   autopct='%1.1f%%', colors=['#e74c3c', '#27ae60'],
                   explode=(0.05, 0))
            ax4.set_title("Overall Attack Success Rate")

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            print(f"Plot saved to: {save_path}")

        plt.show()

    def export_to_csv(self, output_path: str = None):
        """결과를 CSV로 내보내기"""
        if not self.results:
            print("No results to export")
            return

        rows = []
        for r in self.results:
            rows.append({
                "ticker": r.get("ticker", ""),
                "date": r.get("date", ""),
                "attack_goal": r.get("attack_goal", ""),
                "original_decision": self._extract_decision(r.get("original_decision", "")),
                "adversarial_decision": self._extract_decision(r.get("adversarial_decision", "")),
                "attack_success": r.get("attack_success", False),
                "timestamp": r.get("timestamp", "")
            })

        df = pd.DataFrame(rows)

        if output_path is None:
            output_path = os.path.join(self.results_dir, "attack_analysis.csv")

        df.to_csv(output_path, index=False)
        print(f"Results exported to: {output_path}")

        return df


def main():
    analyzer = AttackResultAnalyzer()

    # 리포트 생성
    report = analyzer.generate_report()
    print(report)

    # CSV 내보내기
    analyzer.export_to_csv()

    # 시각화 (matplotlib 설치되어 있으면)
    try:
        analyzer.plot_results(save_path="adversarial_results/attack_analysis.png")
    except Exception as e:
        print(f"Plotting failed: {e}")


if __name__ == "__main__":
    main()
