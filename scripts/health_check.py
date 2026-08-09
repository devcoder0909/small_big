"""External health check script."""

import httpx
import sys
import json


def main():
    api_url = sys.argv[1] if len(sys.argv) > 1 else "http://localhost:8000"
    api_key = sys.argv[2] if len(sys.argv) > 2 else "dev-api-key-change-me"

    print(f"Checking health: {api_url}")

    try:
        # Basic health
        resp = httpx.get(f"{api_url}/health", timeout=10)
        health = resp.json()
        print(f"\n=== Basic Health ===")
        print(json.dumps(health, indent=2))

        # Detailed health (requires auth)
        resp = httpx.get(
            f"{api_url}/health/detailed",
            timeout=10,
        )
        detailed = resp.json()
        print(f"\n=== Detailed Health ===")
        print(json.dumps(detailed, indent=2))

        # Overall status
        status = health.get("status", "unknown")
        if status == "healthy":
            print(f"\n✅ System is HEALTHY")
            sys.exit(0)
        else:
            print(f"\n⚠️ System is {status.upper()}")
            sys.exit(1)

    except Exception as e:
        print(f"\n❌ Health check failed: {e}")
        sys.exit(2)


if __name__ == "__main__":
    main()
