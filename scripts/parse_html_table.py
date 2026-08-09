"""HTML table parser for manual data ingestion from web UI snippets."""

from bs4 import BeautifulSoup
import re
import json
import sys


def parse_html_table(html_content: str) -> list[dict]:
    """
    Parse HTML table snippet copied from WinGo 30S Web UI.

    Extracts:
    - Period (issue_id)
    - Number
    - Big Small (size)
    - Color

    Args:
        html_content: HTML snippet string.

    Returns:
        List of parsed record dictionaries.
    """
    soup = BeautifulSoup(html_content, "html.parser")
    rows = soup.find_all("div", class_="van-row")

    records = []
    for row in rows:
        cols = row.find_all("div", class_=re.compile(r"van-col"))
        if len(cols) >= 4:
            period_text = cols[0].get_text(strip=True)
            num_text = cols[1].get_text(strip=True)
            size_text = cols[2].get_text(strip=True)
            color_text = cols[3].get_text(strip=True)

            # Skip header row
            if period_text == "Period" or not re.match(r"^\d{17,}$", period_text):
                continue

            try:
                num = int(num_text)
                records.append({
                    "issue_id": period_text,
                    "number": num,
                    "size": size_text.upper(),
                    "color": color_text.lower() if color_text else ("green" if "green" in str(cols[3]) else "red"),
                })
            except ValueError:
                continue

    return records


if __name__ == "__main__":
    if len(sys.argv) > 1:
        with open(sys.argv[1], "r", encoding="utf-8") as f:
            content = f.read()
    else:
        content = sys.stdin.read()

    parsed = parse_html_table(content)
    print(json.dumps(parsed, indent=2))
