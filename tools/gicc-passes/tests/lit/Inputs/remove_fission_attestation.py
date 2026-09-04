#!/usr/bin/env python3
import json
import sys


with open(sys.argv[1], encoding="utf-8") as source:
    value = json.load(source)
value.pop("producer_fission_device_materialized", None)
with open(sys.argv[2], "w", encoding="utf-8") as destination:
    json.dump(value, destination, indent=2, sort_keys=True)
    destination.write("\n")
