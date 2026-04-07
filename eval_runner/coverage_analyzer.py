"""
Phase 8.4 — Eval Coverage Gap Analyzer.

Compares alarm types / severity levels in the golden eval dataset against
what has historically appeared in production (approximated by
data/synthetic/alarm_history.json).

Identifies gaps where a real-world alarm pattern is under-represented
in the eval suite — these are the highest-risk blind spots.

Gap severity:
    HIGH    historical_rate > 5%  AND eval_rate < 2%
    MEDIUM  historical_rate > 2%  AND eval_rate < historical_rate / 2
    LOW     historical_rate > 1%  AND eval_rate < historical_rate

Output: reports/coverage_gaps.json

Usage:
    from eval_runner.coverage_analyzer import CoverageAnalyzer
    analyzer = CoverageAnalyzer()
    gaps = analyzer.analyze()
    analyzer.save_report(gaps)
"""

import json
import logging
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Optional

log = logging.getLogger(__name__)

_DATA_DIR    = Path(__file__).parents[1] / "data"
_GOLDEN_DIR  = _DATA_DIR / "golden_dataset"
_REPORTS_DIR = Path(__file__).parents[1] / "reports"

_HISTORY_FILE   = _DATA_DIR / "synthetic" / "alarm_history.json"
_GOLDEN_FILES   = [
    _GOLDEN_DIR / "correctness_cases.json",
    _GOLDEN_DIR / "faithfulness_cases.json",
    _GOLDEN_DIR / "robustness_cases.json",
    _GOLDEN_DIR / "safety_cases.json",
    _GOLDEN_DIR / "latency_cases.json",
]


@dataclass
class CoverageGap:
    alarm_type: str
    historical_count: int
    historical_rate: float     # fraction of historical alarms (0.0–1.0)
    eval_count: int
    eval_coverage_rate: float  # fraction of eval cases (0.0–1.0)
    gap_severity: str          # HIGH / MEDIUM / LOW


def _count_alarm_types(records: list[dict], key: str = "alarm_type") -> dict[str, int]:
    counts: dict[str, int] = {}
    for rec in records:
        alarm_type = rec.get(key) or rec.get("input", {}).get("alarm_type", "UNKNOWN")
        counts[alarm_type] = counts.get(alarm_type, 0) + 1
    return counts


def _extract_eval_alarm_type(case: dict) -> str:
    """Extract alarm_type from any golden dataset case format."""
    # Correctness / latency / safety: flat input dict
    inp = case.get("input", {})
    if isinstance(inp, dict):
        alarm_event = inp.get("alarm_event", inp)
        return alarm_event.get("alarm_type", "UNKNOWN")
    # Robustness: canonical_input
    canonical = case.get("canonical_input", {})
    return canonical.get("alarm_type", "UNKNOWN")


def _classify_gap(
    historical_rate: float,
    eval_rate: float,
) -> Optional[str]:
    """Return gap severity or None if no meaningful gap."""
    if historical_rate > 0.05 and eval_rate < 0.02:
        return "HIGH"
    if historical_rate > 0.02 and eval_rate < historical_rate / 2:
        return "MEDIUM"
    if historical_rate > 0.01 and eval_rate < historical_rate:
        return "LOW"
    return None


class CoverageAnalyzer:
    def __init__(
        self,
        history_file: Path = _HISTORY_FILE,
        golden_files: list[Path] = None,
    ):
        self.history_file = history_file
        self.golden_files = golden_files or _GOLDEN_FILES

    def _load_history(self) -> list[dict]:
        if not self.history_file.exists():
            log.warning("alarm_history.json not found at %s", self.history_file)
            return []
        with open(self.history_file, encoding="utf-8") as f:
            return json.load(f)

    def _load_golden_cases(self) -> list[dict]:
        cases = []
        for path in self.golden_files:
            if path.exists():
                with open(path, encoding="utf-8") as f:
                    cases.extend(json.load(f))
        return cases

    def analyze(self) -> list[CoverageGap]:
        """Compute coverage gaps between historical alarms and eval dataset.

        Returns:
            List of CoverageGap sorted by severity (HIGH first) then by gap size.
        """
        history = self._load_history()
        golden  = self._load_golden_cases()

        if not history:
            log.warning("No historical alarm data — cannot compute gaps")
            return []

        # Count alarm types in each corpus
        hist_counts = _count_alarm_types(history)
        eval_counts: dict[str, int] = {}
        for case in golden:
            at = _extract_eval_alarm_type(case)
            eval_counts[at] = eval_counts.get(at, 0) + 1

        total_hist = sum(hist_counts.values())
        total_eval = sum(eval_counts.values())

        if total_hist == 0 or total_eval == 0:
            return []

        # Compute gaps for all historical alarm types
        gaps: list[CoverageGap] = []
        for alarm_type, hist_count in sorted(hist_counts.items(), key=lambda x: -x[1]):
            h_rate = hist_count / total_hist
            e_count = eval_counts.get(alarm_type, 0)
            e_rate  = e_count / total_eval

            severity = _classify_gap(h_rate, e_rate)
            if severity is None:
                continue

            gaps.append(CoverageGap(
                alarm_type=alarm_type,
                historical_count=hist_count,
                historical_rate=round(h_rate, 4),
                eval_count=e_count,
                eval_coverage_rate=round(e_rate, 4),
                gap_severity=severity,
            ))

        # Sort: HIGH first, then by gap size descending
        _sev_order = {"HIGH": 0, "MEDIUM": 1, "LOW": 2}
        gaps.sort(key=lambda g: (_sev_order[g.gap_severity],
                                  -(g.historical_rate - g.eval_coverage_rate)))

        log.info(
            "Coverage analysis: %d historical alarms, %d eval cases, %d gaps found",
            total_hist, total_eval, len(gaps),
        )
        return gaps

    def save_report(self, gaps: list[CoverageGap]) -> Path:
        """Write gaps to reports/coverage_gaps.json and return the path."""
        _REPORTS_DIR.mkdir(exist_ok=True)
        out_path = _REPORTS_DIR / "coverage_gaps.json"

        report = {
            "total_gaps": len(gaps),
            "high_severity": sum(1 for g in gaps if g.gap_severity == "HIGH"),
            "medium_severity": sum(1 for g in gaps if g.gap_severity == "MEDIUM"),
            "low_severity": sum(1 for g in gaps if g.gap_severity == "LOW"),
            "gaps": [asdict(g) for g in gaps],
        }

        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(report, f, indent=2)

        log.info("Coverage gap report saved → %s", out_path)
        return out_path

    def print_report(self, gaps: list[CoverageGap]):
        """Print a human-readable coverage gap table."""
        if not gaps:
            print("No coverage gaps detected.")
            return

        print(f"\n{'='*65}")
        print("EVAL COVERAGE GAP REPORT")
        print(f"{'='*65}")
        print(f"{'Alarm Type':<30} {'Hist%':>6} {'Eval%':>6} {'Gap':>8}  {'Severity':<8}")
        print("-" * 65)

        for g in gaps:
            gap_pct = (g.historical_rate - g.eval_coverage_rate) * 100
            print(
                f"{g.alarm_type:<30} "
                f"{g.historical_rate*100:>5.1f}% "
                f"{g.eval_coverage_rate*100:>5.1f}% "
                f"{gap_pct:>+7.1f}%  "
                f"{g.gap_severity:<8}"
            )
        print(f"{'='*65}")
        print(f"Total: {len(gaps)} gaps  "
              f"HIGH={sum(1 for g in gaps if g.gap_severity=='HIGH')}  "
              f"MEDIUM={sum(1 for g in gaps if g.gap_severity=='MEDIUM')}  "
              f"LOW={sum(1 for g in gaps if g.gap_severity=='LOW')}")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    analyzer = CoverageAnalyzer()
    gaps = analyzer.analyze()
    analyzer.print_report(gaps)
    analyzer.save_report(gaps)
