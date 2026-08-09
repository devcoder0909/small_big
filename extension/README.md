# WinGo 30S Analytics — Chrome Extension (Manifest V3)

A Chrome extension for displaying real-time historical data analytics and verified observed results from the WinGo 30S platform.

## Features

- **Actual Observed Results**: Direct display of verified results from the FastAPI collector backend.
- **Statistical Analysis Engine**: Displays weighted indicator recommendations for next Small/Big outcomes.
- **Historical Frequency Matrix**: Visual distribution for last 20, 50, 100, and 500 game issues.
- **Streak & Transition Counters**: Current streak and state transition probabilities ($S \rightarrow S$, $S \rightarrow B$, $B \rightarrow S$, $B \rightarrow B$).
- **Live Connection Monitor**: Visual indicator for `● LIVE`, `● STALE`, and `● OFFLINE` data states.

## Installation Instructions

1. Open Google Chrome and navigate to `chrome://extensions/`.
2. Enable **Developer mode** using the toggle switch in the top right corner.
3. Click **Load unpacked**.
4. Select the `extension/` directory from this repository (`c:\Users\tusha\OneDrive\Desktop\smallbig\extension`).
5. Pin the extension icon to your toolbar.

## Configuration

1. Click the extension icon in your toolbar to open the popup.
2. At the bottom of the popup, enter your backend settings:
   - **API URL**: `http://localhost:8000` (or your Northflank domain `https://<service>.northflank.app`)
   - **API Key**: The `API_KEY` configured in your `.env` / Northflank secrets.
3. Click **Save**. The extension will immediately connect and fetch live data.

## Security & Architecture

- **No direct source API requests**: The extension connects ONLY to your secure FastAPI API backend.
- **Minimal Permissions**: No background tabs or web navigation access required.
