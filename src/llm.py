"""Resolve qual LLM (chave + modelo + plugin) o Stark usa.

Cascade da CHAVE (agencia paga quando configurou — qualquer provider):
  1. user_api_keys.openrouter  -> openai plugin com base_url OpenRouter
  2. user_api_keys.openai      -> openai plugin direto
  3. user_api_keys.anthropic   -> anthropic plugin
  4. user_api_keys.gemini      -> google plugin
  5. Platform OPENROUTER_API_KEY (chave Aikortex master) — fallback

Modelo:
  - OpenRouter (user OU platform): available_llms ordenado por priority
    -> STARK_LLM_MODEL_OPENROUTER env -> FALLBACK_MODELS["openrouter"]
  - Outros providers: STARK_LLM_MODEL_<PROVIDER> env -> default por provider

Mesma logica do agent runtime — quando o user troca a chave do Aikortex
pra qualquer outro provider, o Stark muda junto.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Optional

from livekit.agents import llm as lk_llm
from livekit.plugins import openai as lk_openai
from loguru import logger
from supabase import Client


OPENROUTER_BASE_URL = "https://openrouter.ai/api/v1"

# Ordem do cascade — primeiro que tiver chave ganha.
PROVIDER_ORDER = ("openrouter", "openai", "anthropic", "gemini")

FALLBACK_MODELS = {
    "openrouter": "anthropic/claude-3.5-haiku",
    "openai": "gpt-4o-mini",
    "anthropic": "claude-3-5-haiku-20241022",
    "gemini": "gemini-1.5-flash",
}

ENV_MODEL_KEYS = {
    "openrouter": "STARK_LLM_MODEL_OPENROUTER",
    "openai": "STARK_LLM_MODEL_OPENAI",
    "anthropic": "STARK_LLM_MODEL_ANTHROPIC",
    "gemini": "STARK_LLM_MODEL_GEMINI",
}


@dataclass
class LlmResolution:
    """Resultado pra telemetria — o LLM instance vai pro agent direto."""

    provider: str  # openrouter|openai|anthropic|gemini
    model: str
    key_source: str  # "user" | "platform"
    model_source: str  # "available_llms" | "env" | "fallback"


def _read_user_key(sb_admin: Client, user_id: str, provider: str) -> Optional[str]:
    """Le user_api_keys.<provider> do user via service-role."""
    try:
        res = (
            sb_admin.table("user_api_keys")
            .select("api_key")
            .eq("user_id", user_id)
            .eq("provider", provider)
            .limit(1)
            .execute()
        )
        rows = (res.data if res else None) or []
        if rows:
            key = (rows[0].get("api_key") or "").strip()
            return key or None
    except Exception as e:
        logger.warning(f"[llm] erro lendo user_api_keys.{provider}: {e}")
    return None


def _resolve_openrouter_model(sb_admin: Client) -> tuple[str, str]:
    """OpenRouter usa available_llms (admin controla em /admin?tab=llms)."""
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
        rows = (res.data if res else None) or []
        if rows:
            mid = (rows[0].get("model_id") or "").strip()
            if mid:
                return mid, "available_llms"
    except Exception as e:
        logger.warning(f"[llm] erro lendo available_llms: {e}")

    env_model = (os.environ.get(ENV_MODEL_KEYS["openrouter"]) or "").strip()
    if env_model:
        return env_model, "env"

    return FALLBACK_MODELS["openrouter"], "fallback"


def _resolve_direct_model(provider: str) -> tuple[str, str]:
    """Providers diretos (openai/anthropic/gemini): env override -> default."""
    env_model = (os.environ.get(ENV_MODEL_KEYS[provider]) or "").strip()
    if env_model:
        return env_model, "env"
    return FALLBACK_MODELS[provider], "fallback"


def _build_llm(provider: str, api_key: str, model: str) -> lk_llm.LLM:
    """Instancia o plugin LiveKit certo pro provider."""
    if provider == "openrouter":
        return lk_openai.LLM(
            api_key=api_key,
            model=model,
            base_url=OPENROUTER_BASE_URL,
        )
    if provider == "openai":
        return lk_openai.LLM(api_key=api_key, model=model)
    if provider == "anthropic":
        from livekit.plugins import anthropic as lk_anthropic

        return lk_anthropic.LLM(api_key=api_key, model=model)
    if provider == "gemini":
        from livekit.plugins import google as lk_google

        return lk_google.LLM(api_key=api_key, model=model)
    raise ValueError(f"provider desconhecido: {provider}")


def create_stark_llm(
    sb_admin: Client,
    user_id: str,
    keys: dict[str, str] | None = None,
) -> tuple[lk_llm.LLM, LlmResolution]:
    """Resolve cascade e devolve LLM instance pronto pro VoicePipelineAgent.

    ``keys``: mapa {provider: api_key} pre-carregado pelo caller (evita
    4 round-trips ao Supabase no startup da sessao). Se None, consulta
    provider a provider (comportamento legado).
    """
    # 1) Cascade por provider — primeiro com chave vence.
    for provider in PROVIDER_ORDER:
        if keys is not None:
            user_key = (keys.get(provider) or "").strip() or None
        else:
            user_key = _read_user_key(sb_admin, user_id, provider)
        if not user_key:
            continue

        if provider == "openrouter":
            model, model_source = _resolve_openrouter_model(sb_admin)
        else:
            model, model_source = _resolve_direct_model(provider)

        logger.info(
            f"[llm] resolved user={user_id} provider={provider} "
            f"key_source=user model={model} model_source={model_source}"
        )
        return _build_llm(provider, user_key, model), LlmResolution(
            provider=provider,
            model=model,
            key_source="user",
            model_source=model_source,
        )

    # 2) Fallback: Platform OpenRouter (chave Aikortex master).
    platform_key = (os.environ.get("OPENROUTER_API_KEY") or "").strip()
    model, model_source = _resolve_openrouter_model(sb_admin)
    logger.info(
        f"[llm] resolved user={user_id} provider=openrouter "
        f"key_source=platform model={model} model_source={model_source}"
    )
    return _build_llm("openrouter", platform_key, model), LlmResolution(
        provider="openrouter",
        model=model,
        key_source="platform",
        model_source=model_source,
    )
