"""Optional OpenAI-compatible prompt rewriting on another machine."""

import httpx

from app.config import PromptEnhancerConfig


def enhance_prompt(prompt: str, config: PromptEnhancerConfig) -> str:
    if not config.enabled:
        return prompt
    response = httpx.post(
        f"{config.base_url.rstrip('/')}/chat/completions",
        json={
            "model": config.model,
            "temperature": 0.2,
            "messages": [
                {"role": "system", "content": "Rewrite the user's idea as one detailed LTX-2.5 audiovisual video caption. Describe subject, action, camera, lighting, environment and sound. Preserve intent; return only the caption."},
                {"role": "user", "content": prompt},
            ],
        },
        timeout=config.timeout_seconds,
    )
    response.raise_for_status()
    rewritten = response.json()["choices"][0]["message"]["content"].strip()
    if not rewritten:
        raise ValueError("Prompt enhancer returned an empty caption")
    return rewritten