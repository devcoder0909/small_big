/**
 * WinGo 30S API Client Module
 * 
 * Handles all communication with the FastAPI backend.
 * The extension NEVER contacts the source API directly.
 * Architecture: Chrome Extension → FastAPI → PostgreSQL
 */

class WinGoAPI {
    constructor() {
        this.baseUrl = 'http://localhost:8000';
        this.apiKey = '';
        this.loadSettings();
    }

    loadSettings() {
        const saved = localStorage.getItem('wingo_settings');
        if (saved) {
            try {
                const settings = JSON.parse(saved);
                this.baseUrl = settings.apiUrl || this.baseUrl;
                this.apiKey = settings.apiKey || '';
            } catch (e) {
                console.error('Failed to load settings:', e);
            }
        }
    }

    saveSettings(apiUrl, apiKey) {
        this.baseUrl = apiUrl;
        this.apiKey = apiKey;
        localStorage.setItem('wingo_settings', JSON.stringify({ apiUrl, apiKey }));
    }

    async _fetch(endpoint) {
        const url = `${this.baseUrl}${endpoint}`;
        const headers = {
            'Accept': 'application/json',
        };
        if (this.apiKey) {
            headers['Authorization'] = `Bearer ${this.apiKey}`;
        }

        const response = await fetch(url, {
            method: 'GET',
            headers: headers,
            signal: AbortSignal.timeout(10000),
        });

        if (!response.ok) {
            throw new Error(`HTTP ${response.status}: ${response.statusText}`);
        }

        return response.json();
    }

    async getHealth() {
        return this._fetch('/health');
    }

    async getLatest() {
        return this._fetch('/api/v1/latest');
    }

    async getSummary() {
        return this._fetch('/api/v1/stats/summary');
    }

    async getFrequency() {
        return this._fetch('/api/v1/stats/frequency');
    }

    async getStreaks() {
        return this._fetch('/api/v1/stats/streaks');
    }

    async getTransitions() {
        return this._fetch('/api/v1/stats/transitions');
    }

    async getPrediction() {
        return this._fetch('/api/v1/stats/prediction');
    }

    async getAnomalies() {
        return this._fetch('/api/v1/stats/anomalies');
    }
}

// Global API instance
const api = new WinGoAPI();
