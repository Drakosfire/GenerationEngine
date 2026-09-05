"""Target-contract fitness checks.

These tests pin accepted design, not current defects. They must not require
provider credentials or network access.
"""

from __future__ import annotations

import importlib.metadata
import re
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
CONTRACT = REPO_ROOT / "docs" / "CORE-CONTRACT.md"

# Product-domain stems that must not appear as GenerationEngine profile names.
FORBIDDEN_PROFILE_STEMS = (
    "statblock",
    "card",
    "character",
    "store",
    "campaign",
    "map_prompt",
    "ruleslawyer",
    "agent_turn",
    "agent",
    "buddy",
    "runbook",
)

REQUIRED_CONTRACT_HEADINGS = (
    "Capability surface",
    "Provider boundary",
    "Model selection boundary",
    "Model / catalog authority",
    "InferenceObservation",
    "Failure semantics",
    "Streaming",
    "Structured output",
    "Image generation vs artifact persistence",
    "Credentials and optional capabilities",
    "Cutover policy",
)


def _accepted_profiles() -> list[str]:
    text = CONTRACT.read_text(encoding="utf-8")
    match = re.search(
        r"<!-- ACCEPTED_PROFILES -->(.*?)<!-- /ACCEPTED_PROFILES -->",
        text,
        flags=re.S,
    )
    assert match, "CORE-CONTRACT.md must mark accepted profiles with ACCEPTED_PROFILES anchors"
    names = re.findall(r"^([a-z][a-z0-9_]+)$", match.group(1), flags=re.M)
    assert names, "accepted profile list must not be empty"
    return names


def test_core_contract_document_exists_with_required_sections() -> None:
    assert CONTRACT.is_file()
    text = CONTRACT.read_text(encoding="utf-8")
    missing = [heading for heading in REQUIRED_CONTRACT_HEADINGS if heading not in text]
    assert missing == [], f"CORE-CONTRACT.md missing sections: {missing}"


def test_accepted_profiles_are_requirement_shaped() -> None:
    profiles = _accepted_profiles()
    for name in profiles:
        lowered = name.lower()
        for stem in FORBIDDEN_PROFILE_STEMS:
            assert stem not in lowered, f"product vocabulary in profile {name!r} ({stem})"


def test_inference_observation_fields_are_specified() -> None:
    text = CONTRACT.read_text(encoding="utf-8")
    required_fields = (
        "provider",
        "requested_profile",
        "requested_model",
        "resolved_model",
        "response_model",
        "provider_request_id",
        "provider_response_id",
        "input_tokens",
        "cached_input_tokens",
        "output_tokens",
        "cost_usd",
        "latency_ms",
        "retry_count",
        "state",
        "failure_code",
    )
    missing = [field for field in required_fields if field not in text]
    assert missing == [], f"InferenceObservation missing fields: {missing}"
    assert "must not retain full prompts" in text.lower()


def test_streaming_is_transport_neutral_in_the_target() -> None:
    text = CONTRACT.read_text(encoding="utf-8")
    assert "transport-neutral" in text.lower()
    assert "[DONE]" in text
    assert "must not emit HTTP/SSE" in text or "must not emit HTTP/SSE framing" in text


def test_image_persistence_is_outside_the_inference_core() -> None:
    text = CONTRACT.read_text(encoding="utf-8")
    assert "does not require Cloudflare" in text
    assert "durable publication is outside the inference core" in text


def test_openai_is_not_a_core_required_dependency() -> None:
    """OpenAI is an optional extra; core install must not require it."""
    requires = importlib.metadata.requires("generationengine") or []
    required_names = []
    for req in requires:
        name = req.split(";")[0].split("[")[0].split(">")[0].split("=")[0].split("<")[0].strip().lower()
        extra = "extra ==" in req or "extra==" in req
        if not extra:
            required_names.append(name)
    assert "openai" not in required_names
    assert "fal-client" not in required_names
    assert "fal_client" not in required_names


def test_built_wheel_imports_without_provider_extras(tmp_path: Path) -> None:
    """Isolated wheel install must import the package without openai or fal-client."""
    import subprocess

    repo = Path(__file__).resolve().parents[1]
    subprocess.run(["uv", "build"], check=True, cwd=repo)
    wheel = next((repo / "dist").glob("*.whl"))
    venv_dir = tmp_path / "venv"
    subprocess.run(["uv", "venv", str(venv_dir)], check=True, cwd=repo)
    py = venv_dir / "bin" / "python"
    subprocess.run(
        ["uv", "pip", "install", "--python", str(py), str(wheel)],
        check=True,
        cwd=repo,
        capture_output=True,
    )
    result = subprocess.run(
        [
            str(py),
            "-c",
            "import generationengine as ge; "
            "assert ge.InferenceObservation; "
            "assert 'GenerationClient' in ge.__all__; "
            "assert 'TextGenerationService' not in ge.__all__",
        ],
        check=True,
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0
