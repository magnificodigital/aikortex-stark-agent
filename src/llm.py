"""Resolve qual chave/modelo LLM o Stark usa.

Cascade:
  1. user_api_keys.openrouter do dono da sessao (chave propria da agencia)
  2. platform_config.openrouter_default_model (admin pode setar via UI)
  3. OPENROUTER_API_KEY env (chave da Aikortex)
  4. Fallback hardcoded: anthropic/claude-3.5-haiku

OpenRouter e' gateway universal — cobre Claude, GPT, Gemini, Llama etc.
So pre­cisamos de 1 base_url e a chave decide qual conta paga.
"""

from __future__ import annotations

import os
from dataclasses import dataclass

from loguru import logger
from supabase import Client

OPENROUTER_BASE_URL = "https://openrouter.ai/api/v1"
FALLBACK_MODEL = "anthropic/claude-3.5-haiku"


@dataclass
class LlmResolution:
    api_key: str
    model: str
    base_url: str
    source: str  # "user" | "platform"


def resolve_stark_llm(sb_admin: Client, user_id: str) -> LlmResolution:
    """Resolve qual key+modelo o Stark usa pra esse user."""
    api_key = ""
    source: str = "platform"

    # 1) Chave propria do user?
    try:
        res = (
            sb_admin.table("user_api_keys")
            .select("api_key")
            .eq("user_id", user_id)
            .eq("provider", "openrouter")
            .maybe_single()
            .execute()
        )
        user_key = ((res.data or {}).get("api_key") or "").strip()
        if user_key:
            api_key = user_key
            source = "user"
    except Exception as e:
        logger.warning(f"[llm] erro lendo user_api_keys.openrouter: {e}")

    # 2) Fallback: chave da Aikortex
    if not api_key:
        api_key = (os.environ.get("OPENROUTER_API_KEY") or "").strip()

    # 3) Modelo: env var ou platform_config ou fallback
    model = (os.environ.get("STARK_LLM_MODEL") or "").strip()
    if not model:
        try:
            res = (
                sb_admin.table("platform_config")
                .select("value")
                .eq("key", "openrouter_default_model")
                .maybe_single()
                .execute()
            )
            model = ((res.data or {}).get("value") or "").strip()
        except Exception as e:
            logger.warning(f"[llm] erro lendo platform_config: {e}")
    if not model:
        model = FALLBACK_MODEL

    logger.info(f"[llm] resolved user={user_id} source={source} model={model}")
    return LlmResolution(
        api_key=api_key,
        model=model,
        base_url=OPENROUTER_BASE_URL,
        source=source,
    )
