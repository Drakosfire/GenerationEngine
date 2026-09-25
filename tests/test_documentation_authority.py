"""Documentation-authority fitness checks."""

from __future__ import annotations

from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def test_current_documentation_authority_shape() -> None:
    required = (
        "README.md",
        "docs/README.md",
        "docs/CORE-CONTRACT.md",
        "docs/CURRENT-STATE.md",
        "docs/STRUCTURED-CONFORMANCE.md",
        "docs/Reports/REPORT-document-authority-audit-2026-09-25.md",
    )
    forbidden_active = (
        "docs/COMPATIBILITY.md",
        "docs/E2-SUCCESSOR-SLICES.md",
        "docs/HANDOFF-E5H-generationengine-text-output-token-ceiling.md",
        "docs/HANDOFF-E5J-generationengine-json-object-mode.md",
    )

    for relative in required:
        path = ROOT / relative
        assert path.is_file(), f"missing current documentation authority: {relative}"
        assert path.stat().st_size > 0, f"empty current documentation authority: {relative}"

    for relative in forbidden_active:
        assert not (ROOT / relative).exists(), f"historical transition doc returned to active root: {relative}"


def test_current_contract_docs_do_not_claim_future_cutover_state() -> None:
    core = (ROOT / "docs/CORE-CONTRACT.md").read_text(encoding="utf-8")
    current = (ROOT / "docs/CURRENT-STATE.md").read_text(encoding="utf-8")
    index = (ROOT / "docs/README.md").read_text(encoding="utf-8")

    assert "core contract (target)" not in core
    assert "Branch: `e5/structured-conformance`" not in current
    assert "The next contract refinement is provider-independent structured conformance" not in index
