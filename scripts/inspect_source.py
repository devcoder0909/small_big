"""Inspect the WinGo 30S source API response structure.

This script makes a single request to the configured endpoint
and prints a sanitized representation of the response structure.

Usage:
    python scripts/inspect_source.py
"""

import json
import time
import httpx
import sys


def main():
    source_url = "https://draw.ar-lottery01.com/WinGo/WinGo_30S/GetHistoryIssuePage.json"
    ts = int(time.time() * 1000)

    print(f"=== WinGo 30S Source API Inspector ===")
    print(f"URL: {source_url}")
    print(f"Timestamp: {ts}")
    print()

    try:
        response = httpx.get(
            source_url,
            params={"ts": ts},
            headers={
                "User-Agent": "WinGo-Inspector/1.0",
                "Accept": "application/json",
            },
            timeout=15.0,
        )

        print(f"HTTP Status: {response.status_code}")
        print(f"Content-Type: {response.headers.get('content-type')}")
        print(f"Response Size: {len(response.content)} bytes")
        print()

        data = response.json()

        # Top-level structure
        print("=== TOP-LEVEL STRUCTURE ===")
        for key in data:
            val = data[key]
            if isinstance(val, dict):
                print(f"  {key}: dict with keys {list(val.keys())}")
            elif isinstance(val, list):
                print(f"  {key}: list with {len(val)} items")
            else:
                print(f"  {key}: {type(val).__name__} = {val}")

        print()

        # Data structure
        if "data" in data and isinstance(data["data"], dict):
            d = data["data"]
            print("=== DATA STRUCTURE ===")
            for key in d:
                val = d[key]
                if isinstance(val, list):
                    print(f"  data.{key}: list with {len(val)} items")
                else:
                    print(f"  data.{key}: {type(val).__name__} = {val}")

            print()

            # First record
            if "list" in d and len(d["list"]) > 0:
                print("=== FIRST RECORD (NEWEST) ===")
                first = d["list"][0]
                for key, val in first.items():
                    print(f"  {key}: {type(val).__name__} = {repr(val)}")

                print()
                print("=== LAST RECORD (OLDEST) ===")
                last = d["list"][-1]
                for key, val in last.items():
                    print(f"  {key}: {type(val).__name__} = {repr(val)}")

                print()
                print("=== ALL RECORDS ===")
                for item in d["list"]:
                    num = item.get("number", "?")
                    issue = item.get("issueNumber", "?")
                    color = item.get("color", "?")
                    size = "SMALL" if int(num) <= 4 else "BIG"
                    print(f"  {issue} | Number: {num} | Color: {color} | Size: {size}")

                print()
                print(f"=== ANALYSIS ===")
                print(f"  Total records: {len(d['list'])}")
                print(f"  Order: {'Newest-first' if d['list'][0]['issueNumber'] > d['list'][-1]['issueNumber'] else 'Oldest-first'}")
                print(f"  Issue ID format: {d['list'][0]['issueNumber']} (length: {len(d['list'][0]['issueNumber'])})")

                # Unique colors
                colors = set()
                for item in d["list"]:
                    colors.add(item.get("color", ""))
                print(f"  Unique colors: {sorted(colors)}")

                # Number range
                numbers = [int(item["number"]) for item in d["list"]]
                print(f"  Number range: {min(numbers)}-{max(numbers)}")
                print(f"  Small (0-4): {sum(1 for n in numbers if n <= 4)}")
                print(f"  Big (5-9): {sum(1 for n in numbers if n >= 5)}")

        print()
        print("=== RAW JSON (PRETTY) ===")
        print(json.dumps(data, indent=2))

    except Exception as e:
        print(f"ERROR: {e}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
