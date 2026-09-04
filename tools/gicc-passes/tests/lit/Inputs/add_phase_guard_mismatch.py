#!/usr/bin/env python3
"""Add an opposite phase-sensitive guard to a producer-frontier fixture."""

import json
import sys


with open(sys.argv[1], encoding="utf-8") as source:
    document = json.load(source)

extra_domain = {
    "operation": "opposite_lane_collective",
    "guard_predicates": [{
        "condition": {"kind": "param", "param": 7, "type": "i1"},
        "required_value": False,
    }],
    "guard_predicates_exact": True,
    "domain_exact": True,
    "reason": "unit opposite phase-sensitive guard",
}
for operation in document["ops"]:
    frontier = operation.get("producer_frontier")
    if not frontier:
        continue
    frontier["producer_phase_sensitive_domains"].append(extra_domain)
    frontier["phase_sensitive_sites"] += 1

with open(sys.argv[2], "w", encoding="utf-8") as destination:
    json.dump(document, destination, indent=2, sort_keys=True)
    destination.write("\n")
