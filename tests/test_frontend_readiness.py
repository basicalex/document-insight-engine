from __future__ import annotations

from frontend.readiness import classify_runtime_readiness


def test_classify_runtime_readiness_uses_action_specific_blockers() -> None:
    runtime = {
        "readiness": {
            "actions": {
                "ask_fast": {"ready": True, "blocking_issues": []},
                "ask_deep": {
                    "ready": False,
                    "blocking_issues": [
                        {
                            "code": "deep_provider_not_ready",
                            "message": "deep provider is not ready (missing_api_key)",
                        }
                    ],
                },
            },
            "optional_capability_issues": [
                {
                    "capability": "google_parser",
                    "reason": "missing_api_key",
                    "message": "google_parser not ready (missing_api_key)",
                }
            ],
        }
    }

    blocking, optional = classify_runtime_readiness(runtime=runtime, chat_mode="deep")

    assert blocking == ["deep provider is not ready (missing_api_key)"]
    assert optional == ["google_parser not ready (missing_api_key)"]


def test_classify_runtime_readiness_falls_back_to_legacy_payload_shape() -> None:
    runtime = {
        "readiness": {"overall": "degraded"},
        "deep_mode_enabled": True,
        "deep_provider": {"ready": False, "reason": "missing_api_key"},
        "capabilities": {
            "google_parser": {
                "enabled": True,
                "ready": False,
                "reason": "missing_api_key",
            },
            "docling_parser": {
                "enabled": False,
                "ready": False,
                "reason": "disabled_by_config",
            },
        },
    }

    blocking, optional = classify_runtime_readiness(runtime=runtime, chat_mode="deep")

    assert "runtime index readiness is degraded" in blocking
    assert "deep provider is not ready (missing_api_key)" in blocking
    assert optional == ["google_parser not ready (missing_api_key)"]


def test_classify_runtime_readiness_uses_deep_lite_action_when_available() -> None:
    runtime = {
        "readiness": {
            "actions": {
                "ask_fast": {"ready": True, "blocking_issues": []},
                "ask_deep_lite": {
                    "ready": False,
                    "blocking_issues": [
                        {
                            "code": "index_backend_not_ready",
                            "message": "index backend is degraded (redis_unavailable)",
                        }
                    ],
                },
                "ask_deep": {"ready": True, "blocking_issues": []},
            },
            "optional_capability_issues": [],
        }
    }

    blocking, optional = classify_runtime_readiness(
        runtime=runtime,
        chat_mode="deep-lite",
    )

    assert blocking == ["index backend is degraded (redis_unavailable)"]
    assert optional == []
