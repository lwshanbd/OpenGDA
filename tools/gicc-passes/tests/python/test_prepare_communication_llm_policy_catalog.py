import importlib.util
import itertools
import json
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock


PASS_ROOT = Path(__file__).resolve().parents[2]
EXPERIMENTS = PASS_ROOT / "experiments"
sys.path.insert(0, str(PASS_ROOT / "python"))
sys.path.insert(0, str(EXPERIMENTS))
SCRIPT = EXPERIMENTS / "prepare_communication_llm_policy_catalog.py"
SPEC = importlib.util.spec_from_file_location("communication_catalog", SCRIPT)
catalog = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
sys.modules[SPEC.name] = catalog
SPEC.loader.exec_module(catalog)


def graph():
    return {
        "schema_version": catalog.communication.GRAPH_SCHEMA,
        "graph_id": "sha256:" + "1" * 64,
        "opportunities": [
            {
                "opportunity_id": "op:a",
                "candidates": [
                    {"candidate_id": "candidate:a0"},
                    {"candidate_id": "candidate:a1"},
                ],
            },
            {
                "opportunity_id": "op:b",
                "candidates": [
                    {"candidate_id": "candidate:b0"},
                    {"candidate_id": "candidate:b1"},
                    {"candidate_id": "candidate:b2"},
                ],
            },
        ],
    }


def policy_id(selected):
    return catalog.bridge._fingerprint({
        "selected_ids_by_slot": dict(selected)
    })


def verified_policy(_, selected):
    return {
        "selected_ids_by_slot": dict(selected),
        "policy_id": policy_id(selected),
    }


def decision_to_policy(value, decision):
    if decision is None:
        selected = {"op:a": "candidate:a0", "op:b": "candidate:b0"}
    else:
        selected = {
            key: item["candidate_id"]
            for key, item in decision["selections"].items()
        }
    return {
        "bridge_accepted": True,
        "fallback_applied": False,
        "selected_ids_by_slot": selected,
        "policy_id": policy_id(selected),
        "compiler_hint_id": catalog.bridge._fingerprint(selected),
        "compiler_hint": {
            "schema_version": "gicc-hint-v1", "sites": selected,
        },
    }


class CommunicationLlmPolicyCatalogTests(unittest.TestCase):
    def patches(self):
        return (
            mock.patch.object(
                catalog.policy_bridge, "verified_graph",
                side_effect=lambda value: value,
            ),
            mock.patch.object(
                catalog.policy_bridge, "verified_policy",
                side_effect=verified_policy,
            ),
            mock.patch.object(
                catalog.policy_bridge, "decision_to_policy",
                side_effect=decision_to_policy,
            ),
        )

    def test_enumerates_full_cross_product_and_revalidates_every_policy(self):
        patches = self.patches()
        with patches[0] as verify_graph, patches[1] as verify, patches[2] as bridge:
            result = catalog.enumerate_catalog(graph())
        self.assertEqual(6, len(result))
        expected = set(itertools.product(
            ("candidate:a0", "candidate:a1"),
            ("candidate:b0", "candidate:b1", "candidate:b2"),
        ))
        observed = {
            (item["selected_ids_by_slot"]["op:a"],
             item["selected_ids_by_slot"]["op:b"])
            for item in result
        }
        self.assertEqual(expected, observed)
        self.assertEqual(1, verify_graph.call_count)
        self.assertEqual(6, verify.call_count)
        self.assertEqual(6, bridge.call_count)

    def test_catalog_roles_do_not_use_runtime_oracle_labels(self):
        fake_index = {
            "runs": [{
                "policy_id": policy_id({
                    "op:a": "candidate:a1", "op:b": "candidate:b2",
                }),
                "view": "relational", "trial": 1,
                "bridge_accepted": True,
            }],
        }
        observed = {
            fake_index["runs"][0]["policy_id"]: {
                "op:a": "candidate:a1", "op:b": "candidate:b2",
            }
        }
        patches = self.patches()
        deterministic = ({
            "schema_version": catalog.communication.DECISION_SCHEMA,
            "graph_id": graph()["graph_id"],
            "selections": {
                "op:a": {"candidate_id": "candidate:a0"},
                "op:b": {"candidate_id": "candidate:b1"},
            },
        }, {})
        with patches[0], patches[1], patches[2], mock.patch.object(
            catalog.capability, "observed_policies", return_value=observed,
        ), mock.patch.object(
            catalog.heuristic, "make_decision", return_value=deterministic,
        ):
            result, anchor, compiler = catalog.catalog_with_roles(
                graph(), fake_index
            )
        self.assertEqual(policy_id({
            "op:a": "candidate:a0", "op:b": "candidate:b0",
        }), anchor)
        self.assertEqual(policy_id({
            "op:a": "candidate:a0", "op:b": "candidate:b1",
        }), compiler)
        observed_rows = [
            row for row in result
            if row["roles"]["observed_in_provider_archive"]
        ]
        self.assertEqual(1, len(observed_rows))
        self.assertEqual([{
            "view": "relational", "trial": 1,
            "bridge_accepted": True,
        }], observed_rows[0]["archive_occurrences"])
        self.assertEqual(catalog.BOUNDARY, {
            "application_source_read_or_modified": False,
            "provider_invoked": False,
            "compiler_invoked": False,
            "scheduler_invoked": False,
            "runtime_or_oracle_labels_read": False,
            "model_output_authority": (
                "existing graph-bound candidate IDs only"
            ),
            "private_hints_visible_to_model": False,
            "compiler_revalidates_every_catalog_policy": True,
        })

    def test_adapter_source_has_no_execution_or_application_source_path(self):
        source = SCRIPT.read_text(encoding="utf-8")
        self.assertNotIn("subprocess", source)
        self.assertNotIn("flux ", source)
        self.assertNotIn("source_file", source)
        self.assertNotIn("examples/", source)

    def test_contained_catalog_replays_and_rejects_hint_mutation(self):
        request = {"request_id": "sha256:" + "2" * 64}
        index = {"runs": []}
        selected = {"op:a": "candidate:a0", "op:b": "candidate:b0"}
        policy = {
            "policy_id": policy_id(selected),
            "selected_ids_by_slot": selected,
            "compiler_hint_id": "sha256:" + "3" * 64,
            "compiler_hint": {
                "schema_version": "gicc-hint-v1", "sites": {},
            },
            "name": "policy-0001",
            "roles": {
                "semantic_anchor": True,
                "deterministic_compiler_control": True,
                "observed_in_provider_archive": False,
            },
            "archive_occurrences": [],
        }
        with tempfile.TemporaryDirectory(dir=PASS_ROOT) as temporary:
            root = Path(temporary)
            graph_path = root / "graph.json"
            request_dir = root / "request"
            archive_dir = root / "archive"
            output_dir = root / "catalog"
            request_dir.mkdir()
            archive_dir.mkdir()
            graph_path.write_text(json.dumps(graph()), encoding="utf-8")
            (request_dir / "request.json").write_text(
                json.dumps(request), encoding="utf-8"
            )
            (archive_dir / "authorization.json").write_text(
                "{}", encoding="utf-8"
            )
            (archive_dir / "run-index.json").write_text(
                "{}", encoding="utf-8"
            )
            with mock.patch.object(
                catalog.policy_bridge, "verified_graph",
                side_effect=lambda value: value,
            ), mock.patch.object(
                catalog, "verify_request_archive",
                return_value=(request, index),
            ), mock.patch.object(
                catalog, "catalog_with_roles",
                return_value=([policy], policy["policy_id"], policy["policy_id"]),
            ):
                manifest = catalog.prepare(
                    graph_path, request_dir, archive_dir, output_dir, PASS_ROOT
                )
                self.assertEqual(
                    catalog.CATALOG_SCHEMA, manifest["schema_version"]
                )
                catalog.verify_contained(
                    output_dir / "manifest.json", PASS_ROOT
                )
                hint = output_dir / "policies/policy-0001/hint.json"
                hint.write_text("{}\n", encoding="utf-8")
                with self.assertRaisesRegex(
                    catalog.CommunicationPolicyCatalogError,
                    "policy hint: evidence changed",
                ):
                    catalog.verify_contained(
                        output_dir / "manifest.json", PASS_ROOT
                    )


if __name__ == "__main__":
    unittest.main()
