#!/usr/bin/env python3
"""Static checks for the submitted Navigator-Guide communication boundary."""

from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
HOLISTIC = ROOT / "external/RAINbow/holistic"


def main():
    navigator = (HOLISTIC / "ModularNavigator.py").read_text()
    guide = (HOLISTIC / "ModularGuide.py").read_text()
    runner = (HOLISTIC / "main.py").read_text()
    all_source = "\n".join(
        path.read_text(errors="ignore")
        for path in ROOT.rglob("*")
        if path.is_file()
        and path.suffix in {".py", ".sh", ".env"}
        and path.name != "test_agent_boundary.py"
    )

    assert "question_generation_model" in navigator
    assert "question_generation_model" not in guide
    assert "navigator.ask(" in runner
    assert "guide.localize(" in runner
    assert "guide.answer(" in runner
    assert "guide.confirm_goals(" in runner
    assert "navigator.update_instruction(" in runner
    assert "navigator.explicit_stop_indices(" in runner
    assert "ModularGuide(\n        args,\n        answer_model,\n        localization_model," in runner
    assert "ModularNavigator(\n        args, navigation_model, wta_model, question_model" in runner
    assert "select_goal_confirmation_indices" not in runner
    assert "localized_viewpoints[index] in goals[index]" in guide
    assert "answers[index] == confirmation_text" in navigator

    forbidden = (
        "QuestionFingerprint",
        "question_fingerprint",
        "target_description",
        "language_dfs",
        "VisualSignature",
        "episode_selector",
    )
    for marker in forbidden:
        assert marker not in all_source, marker
    print("agent-boundary checks passed")


if __name__ == "__main__":
    main()
