/**
 * Ultra-Clean Minimal WinGo 30S Predictor Logic
 *
 * Dedicated to displaying ONLY the upcoming period prediction
 * (BIG or SMALL) and the 30-second draw countdown.
 */

const REFRESH_INTERVAL = 3000; // Poll every 3s for immediate updates
let countdownSeconds = 30;
let countdownTimerInterval = null;
let lastKnownUpcomingPeriod = null;

// === Initialization ===
document.addEventListener('DOMContentLoaded', () => {
    loadSettings();
    setupSettingsToggle();
    startCountdown();
    fetchPrediction();
    setInterval(fetchPrediction, REFRESH_INTERVAL);
});

function loadSettings() {
    const saved = localStorage.getItem('wingo_settings');
    if (saved) {
        try {
            const s = JSON.parse(saved);
            document.getElementById('apiUrl').value = s.apiUrl || 'http://localhost:8000';
            document.getElementById('apiKey').value = s.apiKey || '';
        } catch (e) {}
    }
}

function setupSettingsToggle() {
    const toggle = document.getElementById('settingsToggle');
    const panel = document.getElementById('settingsPanel');
    const saveBtn = document.getElementById('saveSettings');

    toggle.addEventListener('click', () => {
        panel.classList.toggle('hidden');
    });

    saveBtn.addEventListener('click', () => {
        const url = document.getElementById('apiUrl').value.trim();
        const key = document.getElementById('apiKey').value.trim();
        api.saveSettings(url, key);
        panel.classList.add('hidden');
        fetchPrediction();
    });
}

// === 30S Cycle Countdown Timer ===
function startCountdown() {
    if (countdownTimerInterval) clearInterval(countdownTimerInterval);

    const now = new Date();
    const currentSeconds = now.getSeconds();
    countdownSeconds = 30 - (currentSeconds % 30);
    if (countdownSeconds === 0) countdownSeconds = 30;

    updateTimerUI();

    countdownTimerInterval = setInterval(() => {
        countdownSeconds--;
        if (countdownSeconds <= 0) {
            countdownSeconds = 30;
            setTimeout(fetchPrediction, 1000);
        }
        updateTimerUI();
    }, 1000);
}

function updateTimerUI() {
    const timerEl = document.getElementById('timerBadge');
    if (timerEl) {
        timerEl.textContent = `DRAW IN ${countdownSeconds}S`;
        if (countdownSeconds <= 10) {
            timerEl.className = 'timer-badge urgent';
        } else {
            timerEl.className = 'timer-badge';
        }
    }
}

// === Main Data Fetch ===
async function fetchPrediction() {
    try {
        const [predData, latestData] = await Promise.all([
            api.getPrediction().catch(() => null),
            api.getLatest().catch(() => null),
        ]);

        updateStatus(true);

        if (predData) {
            renderPrediction(predData);
        }

        if (latestData) {
            renderLastDraw(latestData);
        }

    } catch (error) {
        console.error('Fetch error:', error);
        updateStatus(false);
    }
}

function renderPrediction(data) {
    const textEl = document.getElementById('predictionText');
    const periodEl = document.getElementById('upcomingPeriod');
    const confEl = document.getElementById('confidenceBadge');

    // Upcoming Period
    if (data.upcoming_issue_id) {
        const fullId = data.upcoming_issue_id;
        const shortId = fullId.length > 10 ? '...' + fullId.slice(-6) : fullId;
        periodEl.textContent = `PERIOD: ${shortId}`;
    } else {
        periodEl.textContent = 'PERIOD: NEXT DRAW';
    }

    // Prediction Result (BIG / SMALL)
    const result = data.prediction;
    if (result) {
        textEl.textContent = result;
        textEl.className = `prediction-text ${result.toLowerCase()}`;

        // Flash if period changed
        if (lastKnownUpcomingPeriod && lastKnownUpcomingPeriod !== data.upcoming_issue_id) {
            document.getElementById('predictionBox').classList.add('updated');
            setTimeout(() => {
                document.getElementById('predictionBox').classList.remove('updated');
            }, 400);
        }
        lastKnownUpcomingPeriod = data.upcoming_issue_id;
    } else {
        textEl.textContent = '—';
        textEl.className = 'prediction-text';
    }

    // Confidence
    const conf = Math.round((data.confidence || 0) * 100);
    confEl.textContent = `${conf}% CONFIDENCE`;
}

function renderLastDraw(data) {
    const stripEl = document.getElementById('lastDrawValue');
    if (!data) return;

    const issue = data.issue_id || '—';
    const shortIssue = issue.length > 10 ? '...' + issue.slice(-5) : issue;
    const num = data.result !== undefined ? data.result : '?';
    const size = data.size || '';

    stripEl.textContent = `${shortIssue} (${num} - ${size})`;
}

function updateStatus(isOnline) {
    const ind = document.getElementById('statusIndicator');
    const text = document.getElementById('statusText');
    if (isOnline) {
        ind.className = 'status-indicator live';
        text.textContent = 'LIVE';
    } else {
        ind.className = 'status-indicator offline';
        text.textContent = 'OFFLINE';
    }
}
