# Requirements traceability

The automated suite links the main system boundaries to executable evidence.
Representative checks:

| Capability | Verification |
|---|---|
| Hybrid ontology retrieval | `tests/test_retrieval.py::test_search_finds_exact_alias_match_via_sparse_channel` |
| MCP semantic search | `tests/test_mcp_server.py::test_find_semantic_match_happy_path` |
| End-to-end pipeline | `tests/test_pipeline_orchestration.py::test_full_pipeline_end_to_end_produces_a_pending_review_row_with_length_predicted` |
| Publish from the chat | `tests/test_http_pipeline_routes.py::test_publish_route_publishes_and_learns` |
| Groq input/output layer | `tests/test_assistant.py` |
| Consent-gated publishing | `tests/test_publish.py::test_publish_is_refused_without_consent` |
| Prompt-injection guardrail | `tests/test_agent.py::test_injection_classifier_catches_known_patterns` |

Run `just test`, `just test-integration`, and `just test-e2e` for the full
suite.
