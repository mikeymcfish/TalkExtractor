---
title: Dialogue→Speaker Dataset Builder
emoji: 🎙️
colorFrom: blue
colorTo: indigo
sdk: gradio
sdk_version: 4.44.0
app_file: app.py
pinned: false
license: mit
---

# Dialogue→Speaker Dataset Builder (HF Spaces)
A GUI app (Gradio) that prepares text passages, calls the OpenAI API to structure dialogue into `Speaker N:` lines, lets you review & edit, and exports JSONL or a HF Dataset.

## Quickstart (HF Spaces)
1. Create a new Space → SDK: **Gradio**.
2. Add **Secrets**:
   - `OPENAI_API_KEY` (required)
   - `OPENAI_MODEL` (optional, default `gpt-4o-mini`)
   - `HF_TOKEN` (optional, for push_to_hub)
3. Upload all these files.
4. Launch the Space.
