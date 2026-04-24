"""Quick Gemini connectivity check (uses GEMINI_API_KEY or GOOGLE_API_KEY from env / .env)."""

from __future__ import annotations

import os

from dotenv import load_dotenv
from google import genai

load_dotenv()


def main() -> None:
    key = os.environ.get("GEMINI_API_KEY") or os.environ.get("GOOGLE_API_KEY")
    if not key:
        print("Set GEMINI_API_KEY (or GOOGLE_API_KEY) in .env")
        return
    model = os.environ.get("GEMINI_MODEL", "gemini-3.1-flash-lite-preview")
    client = genai.Client(api_key=key)
    r = client.models.generate_content(
        model=model,
        contents="Reply with exactly: OK",
    )
    print(model, "->", (r.text or "").strip()[:200])


if __name__ == "__main__":
    main()
