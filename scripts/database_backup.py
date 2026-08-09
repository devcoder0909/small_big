"""Database Logical Backup Script.

Generates a timestamped pg_dump logical backup of the PostgreSQL database.
"""

import os
import subprocess
import sys
from datetime import datetime


def run_backup(backup_dir: str = "backups"):
    """Execute pg_dump backup."""
    os.makedirs(backup_dir, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    backup_file = os.path.join(backup_dir, f"wingo_backup_{timestamp}.sql")

    db_url = os.environ.get("DATABASE_SYNC_URL", "postgresql://postgres:postgres@localhost:5432/wingo_db")

    print(f"=== WinGo Database Logical Backup ===")
    print(f"Destination: {backup_file}")

    cmd = [
        "pg_dump",
        "--dbname=" + db_url,
        "--clean",
        "--if-exists",
        "--format=plain",
        "--file=" + backup_file,
    ]

    try:
        res = subprocess.run(cmd, capture_output=True, text=True)
        if res.returncode == 0:
            print(f"✅ Backup successful! Size: {os.path.getsize(backup_file)} bytes")
        else:
            print(f"❌ Backup failed: {res.stderr}")
            sys.exit(1)
    except FileNotFoundError:
        print("⚠️ pg_dump executable not found in system PATH. Ensure PostgreSQL tools are installed.")
        sys.exit(2)


if __name__ == "__main__":
    out_dir = sys.argv[1] if len(sys.argv) > 1 else "backups"
    run_backup(out_dir)
