from __future__ import annotations

from collections.abc import Mapping
from typing import Any


def classify_runtime_readiness(
    *,
    runtime: Mapping[str, Any],
    chat_mode: str,
) -> tuple[list[str], list[str]]:
    readiness = runtime.get("readiness")
    blocking: list[str] = []
    optional: list[str] = []

    if isinstance(readiness, Mapping):
        action_key = "ask_deep" if chat_mode == "deep" else "ask_fast"
        actions = readiness.get("actions")
        if isinstance(actions, Mapping):
            action_report = actions.get(action_key)
            if isinstance(action_report, Mapping):
                blocking = _normalize_issue_messages(
                    action_report.get("blocking_issues")
                )

        optional = _normalize_issue_messages(
            readiness.get("optional_capability_issues")
        )

    if not blocking:
        blocking = _legacy_blocking_issues(runtime=runtime, chat_mode=chat_mode)
    if not optional:
        optional = _legacy_optional_issues(runtime=runtime)

    return _dedupe(blocking), _dedupe(optional)


def _normalize_issue_messages(issues: Any) -> list[str]:
    if not isinstance(issues, list):
        return []

    messages: list[str] = []
    for issue in issues:
        if isinstance(issue, str):
            text = issue.strip()
        elif isinstance(issue, Mapping):
            message = issue.get("message")
            reason = issue.get("reason")
            capability = issue.get("capability")
            text = str(message or "").strip()
            if not text and capability:
                text = str(capability).strip()
                if reason:
                    text = f"{text} ({reason})"
        else:
            text = ""

        if text:
            messages.append(text)
    return messages


def _legacy_blocking_issues(
    *,
    runtime: Mapping[str, Any],
    chat_mode: str,
) -> list[str]:
    issues: list[str] = []
    readiness = runtime.get("readiness")
    if isinstance(readiness, Mapping):
        if str(readiness.get("overall")) != "ready":
            issues.append("runtime index readiness is degraded")

    if chat_mode == "deep":
        if not bool(runtime.get("deep_mode_enabled", False)):
            issues.append("deep mode is disabled")
        deep_provider = runtime.get("deep_provider")
        if isinstance(deep_provider, Mapping) and not bool(
            deep_provider.get("ready", False)
        ):
            reason = str(deep_provider.get("reason") or "unknown")
            issues.append(f"deep provider is not ready ({reason})")
    return issues


def _legacy_optional_issues(runtime: Mapping[str, Any]) -> list[str]:
    capabilities = runtime.get("capabilities")
    if not isinstance(capabilities, Mapping):
        return []

    issues: list[str] = []
    for name, capability in capabilities.items():
        if not isinstance(capability, Mapping):
            continue
        if bool(capability.get("enabled")) and not bool(capability.get("ready")):
            reason = str(capability.get("reason") or "unknown")
            issues.append(f"{name} not ready ({reason})")
    return issues


def _dedupe(items: list[str]) -> list[str]:
    seen: set[str] = set()
    ordered: list[str] = []
    for item in items:
        if item in seen:
            continue
        seen.add(item)
        ordered.append(item)
    return ordered
