---
type: project
created: 2026-07-18
updated: 2026-07-18
---

# Technical Decisions

- Component metadata uses SemVer while the toolkit release keeps CalVer.
- `manifest.json` and `manifest.lock.json` must remain synchronized with component frontmatter.
- AI Provider Multi-Key Rotation: Supports NVIDIA NIM (Nemotron 3 Ultra 550B), OpenRouter, Groq, and Gemini with key rotation (2 NVIDIA, 3 OpenRouter, 2 Groq, 2 Gemini), 3s safety timeout, 60s rate-limit cooldowns, and non-blocking fallback to local statistical ensemble.
