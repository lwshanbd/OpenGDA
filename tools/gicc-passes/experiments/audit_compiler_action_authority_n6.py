#!/usr/bin/env python3
"""Audit compiler action authority for a collective_n6 decision suite."""

import sys

import audit_compiler_action_authority as base


if __name__ == "__main__":
    raise SystemExit(base.main(["--collective-label", "collective_n6", *sys.argv[1:]]))
