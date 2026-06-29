"""Resolve qual chave/modelo LLM o Stark usa.

Modelo:
  1. available_llms (gerenciado em /admin?tab=llms) — pega o primeiro
     active=true, status!=dead, ordenado por priority ASC
  2. STARK_LLM_MODEL env (override por deploy)
  3. Fallback hardcoded: anthropic/claude-3.5-haiku

Chave (cascade — agencia paga quando configurou):
  1. user_api_keys.openrouter do dono da sessao
  2. OPENROUTER_API_KEY env (chave Aikortex master)

OpenRouter e' gateway universal — cobre Claude/GPT/Gemini/Llama. Stark
sempre roteia por ele. Chaves diretas (Anthropic/OpenAI/Gemini sem
OpenRouter) caem no fallback Aikortex porque LiveKit plugin so suporta
formato OpenAI-compatible.
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
    model_source: str  # "available_llms" | "env" | "fallback"


def _resolve_model(sb_admin: Client) -> tuple[str, str]:
    """Retorna (model_id, source). Source pra debug."""
    # 1) available_llms — admin controla ordem em /admin?tab=llms
    try:
        res = (
            sb_admin.table("available_llms")
            .select("model_id, provider, priority")
            .eq("active", True)
            .neq("status", "dead")
            .order("priority", desc=False)
            .limit(1)
            .execute()
        )
        rows = res.data or []
        if rows:
            mid = (rows[0].get("model_id") or "").strip()
            if mid:
                return mid, "available_llms"
    except Exception as e:
        logger.warning(f"[llm] erro lendo available_llms: {e}")

    # 2) Env override por deploy
    env_model = (os.environ.get("STARK_LLM_MODEL") or "").strip()
    if env_model:
        return env_model, "env"

    # 3) Hardcoded fallback
    return FALLBACK_MODEL, "fallback"


def _resolve_key(sb_admin: Client, user_id: str) -> tuple[str, str]:
    """Retorna (api_key, source). 'user' = agencia configurou, 'platform' = Aikortex."""
    # 1) Chave propria da agencia
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
            return user_key, "user"
    except Exception as e:
        logger.warning(f"[llm] erro lendo user_api_keys.openrouter: {e}")

    # 2) Fallback Aikortex
    platform_key = (os.environ.get("OPENROUTER_API_KEY") or "").strip()
    return platform_key, "platform"


def resolve_stark_llm(sb_admin: Client, user_id: str) -> LlmResolution:
    api_key, key_source = _resolve_key(sb_admin, user_id)
    model, model_source = _resolve_model(sb_admin)
    logger.info(
        f"[llm] resolved user={user_id} key_source={key_source} "
        f"model={model} model_source={model_source}"
    )
    return LlmResolution(
        api_key=api_key,
        model=model,
        base_url=OPENROUTER_BASE_URL,
        source=key_source,
        model_source=model_source,
    )
