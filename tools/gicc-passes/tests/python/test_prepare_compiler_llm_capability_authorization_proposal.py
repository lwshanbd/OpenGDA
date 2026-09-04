import importlib.util
import sys
import unittest
from pathlib import Path


PASS_ROOT = Path(__file__).resolve().parents[2]
EXPERIMENTS = PASS_ROOT / "experiments"
sys.path.insert(0, str(PASS_ROOT / "python"))
sys.path.insert(0, str(EXPERIMENTS))
SCRIPT = (
    EXPERIMENTS
    / "prepare_compiler_llm_capability_authorization_proposal.py"
)
SPEC = importlib.util.spec_from_file_location("authorization_proposal", SCRIPT)
proposal = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
sys.modules[SPEC.name] = proposal
SPEC.loader.exec_module(proposal)


def request():
    return {
        "request_id": "sha256:" + "1" * 64,
        "provider_delivery": {
            "system_prompt_sha256": "2" * 64,
            "response_schema_sha256": "3" * 64,
            "views": [{
                "view": view,
                "prompt_sha256": str(index) * 64,
                "independent_responses": 20,
            } for index, view in enumerate(
                ("relational", "descriptors", "opaque"), start=4
            )],
            "trial_order": {
                "kind": "response_index_major_rotating_views",
                "base_view_order": [
                    "relational", "descriptors", "opaque",
                ],
                "rotation_offset_for_trial": (
                    "(trial - 1) modulo view count"
                ),
            },
            "total_conditional_calls": 60,
        },
    }


class CompilerLlmCapabilityAuthorizationProposalTests(unittest.TestCase):
    def build_proposal(self):
        req = request()
        proposed = proposal.authorization(
            req, executable="claude", cli_version="unit-version",
            model="opus", effort="high", maximum_attempts=3,
            timeout_seconds=300,
        )
        body = proposal.proposal_payload(req, proposed)
        value = {
            "proposal_id": proposal.bridge._fingerprint(body), **body,
        }
        return req, proposed, value

    def test_outer_proposal_cannot_be_used_as_runner_authorization(self):
        req, proposed, value = self.build_proposal()
        verified = proposal.verify_proposal(value)
        self.assertEqual(0, verified["boundary"][
            "currently_permitted_provider_calls"
        ])
        self.assertFalse(
            verified["boundary"]["authorization_granted_by_proposal"]
        )
        self.assertEqual({
            "semantic_trials": 60,
            "max_transport_attempts_per_trial": 3,
            "maximum_provider_process_invocations": 180,
        }, verified["proposed_provider_budget"])
        with self.assertRaisesRegex(
            proposal.trials.CapabilityTrialError,
            proposal.trials.AUTHORIZATION_SCHEMA,
        ):
            proposal.trials.authorization_payload(verified)
        accepted = proposal.trials.verify_authorization(proposed, req)
        self.assertEqual("opus", accepted["provider"]["requested_model"])

    def test_grant_requires_both_exact_ids(self):
        _, proposed, value = self.build_proposal()
        with self.assertRaisesRegex(
            proposal.AuthorizationProposalError,
            "explicit user proposal ID does not match",
        ):
            proposal.explicitly_authorized_payload(
                value, authorized_proposal_id="sha256:" + "0" * 64,
                authorized_authorization_id=value[
                    "proposed_authorization_id"
                ],
            )
        with self.assertRaisesRegex(
            proposal.AuthorizationProposalError,
            "explicit user authorization ID does not match",
        ):
            proposal.explicitly_authorized_payload(
                value, authorized_proposal_id=value["proposal_id"],
                authorized_authorization_id="sha256:" + "0" * 64,
            )
        granted = proposal.explicitly_authorized_payload(
            value, authorized_proposal_id=value["proposal_id"],
            authorized_authorization_id=value["proposed_authorization_id"],
        )
        self.assertEqual(proposed, granted)

    def test_verify_rejects_boundary_or_embedded_authorization_changes(self):
        _, _, value = self.build_proposal()
        changed_boundary = {
            **value,
            "boundary": {**value["boundary"], "provider_cli_invoked": True},
        }
        unsigned = dict(changed_boundary)
        unsigned.pop("proposal_id")
        changed_boundary["proposal_id"] = proposal.bridge._fingerprint(unsigned)
        with self.assertRaisesRegex(
            proposal.AuthorizationProposalError, "zero-call boundary",
        ):
            proposal.verify_proposal(changed_boundary)

        changed_budget = {
            **value,
            "proposed_provider_budget": {
                **value["proposed_provider_budget"],
                "maximum_provider_process_invocations": 60,
            },
        }
        unsigned = dict(changed_budget)
        unsigned.pop("proposal_id")
        changed_budget["proposal_id"] = proposal.bridge._fingerprint(unsigned)
        with self.assertRaisesRegex(
            proposal.AuthorizationProposalError, "provider budget",
        ):
            proposal.verify_proposal(changed_budget)

        changed_authorization = {
            **value,
            "proposed_authorization": {
                **value["proposed_authorization"], "request_id": "changed",
            },
        }
        unsigned = dict(changed_authorization)
        unsigned.pop("proposal_id")
        changed_authorization["proposal_id"] = (
            proposal.bridge._fingerprint(unsigned)
        )
        with self.assertRaisesRegex(
            proposal.trials.CapabilityTrialError,
            "authorization ID does not match content",
        ):
            proposal.verify_proposal(changed_authorization)

    def test_proposal_is_offline_and_grant_requires_both_exact_ids(self):
        source = SCRIPT.read_text(encoding="utf-8")
        self.assertNotIn("subprocess", source)
        self.assertNotIn("cli_version(", source)
        self.assertIn("--authorized-proposal-id", source)
        self.assertIn("--authorized-authorization-id", source)
        self.assertIn("explicit user proposal ID does not match", source)
        self.assertIn("explicit user authorization ID does not match", source)


if __name__ == "__main__":
    unittest.main()
