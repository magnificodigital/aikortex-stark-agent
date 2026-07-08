"""Pipeline Stark Voice — STT (Deepgram) → LLM (OpenRouter) → TTS (ElevenLabs).

LiveKit Agents SDK cuida da bidirectional streaming via WebRTC. Aqui só
montamos o ``VoicePipelineAgent``, plugamos os providers e registramos
as tools (function calling).
"""

from __future__ import annotations

import asyncio
import os
import time
from typing import Optional

from livekit import rtc
from livekit.agents import (
    AutoSubscribe,
    JobContext,
    llm,
)
from livekit.agents.pipeline import VoicePipelineAgent
from livekit.plugins import deepgram, elevenlabs, silero
from loguru import logger

from src.credits import consume_minutes
from src.llm import create_stark_llm
from src.persona import build_system_prompt, load_prefs
from src.supabase_client import supabase_admin, supabase_user
from src.tools import StarkTools


CREDIT_CHECK_INTERVAL_SECONDS = 30


async def run_stark_session(
    ctx: JobContext,
    user_id: str,
    agency_id: Optional[str],
    locale: str,
    participant_jwt: Optional[str],
    page_context: Optional[dict] = None,
) -> None:
    """Sessão completa do Stark até o disconnect."""
    sb_admin = supabase_admin()

    # ── Sistema de prompt baseado em prefs do user + contexto da pagina ──
    prefs = load_prefs(sb_admin, user_id)
    system_prompt = build_system_prompt(prefs, page_context=page_context)

    # ── Tools (mesmas do stark-tools.ts, agora em Python) ──
    # User-scoped client via JWT (se disponivel) — RLS filtra dados do user.
    # Fallback admin se o cliente nao mandou JWT no metadata (modo dev).
    sb_for_tools = supabase_user(participant_jwt) if participant_jwt else sb_admin
    # StarkTools extende llm.FunctionContext — instancia ja e' o context.
    # agency_id   → filtra agency_clients e client_template_subscriptions
    # room        → tools acionam comandos no frontend (ex: abrir wizard)
    # tools_enabled → user controla quais tools o Stark pode chamar (Settings)
    fnc_ctx = StarkTools(
        sb_for_tools,
        user_id=user_id,
        agency_id=agency_id,
        room=ctx.room,
        tools_enabled=(prefs or {}).get("tools_enabled") if prefs else None,
    )

    # ── Pipeline LiveKit Agents ──
    initial_ctx = llm.ChatContext().append(role="system", text=system_prompt)

    # LLM cascade: chave do user em QUALQUER provider (openrouter/openai/
    # anthropic/gemini) > chave Aikortex (OpenRouter). Quando o user troca
    # a chave no /admin?tab=llms, o Stark muda junto via plugin certo.
    llm_instance, llm_res = create_stark_llm(sb_admin, user_id)

    agent = VoicePipelineAgent(
        vad=silero.VAD.load(),
        stt=deepgram.STT(
            model="nova-2-general",
            language=locale,
            punctuate=True,
            interim_results=True,
        ),
        llm=llm_instance,
        tts=_build_tts(sb_admin, user_id, locale),
        chat_ctx=initial_ctx,
        fnc_ctx=fnc_ctx,
        allow_interruptions=True,
    )

    # ── Telemetria + debito incremental de creditos ──
    session_start = time.time()
    last_debit = session_start
    tools_called: list[str] = []

    @agent.on("function_calls_collected")
    def _on_tool_calls(calls):
        for c in calls:
            tools_called.append(c.function_info.name)

    agent.start(ctx.room)
    logger.info(f"[stark-agent] agent started user={user_id} agency={agency_id} locale={locale}")

    # ── Loop de monitoramento: debita a cada 30s ──
    try:
        while ctx.room.connection_state == rtc.ConnectionState.CONN_CONNECTED:
            await asyncio.sleep(CREDIT_CHECK_INTERVAL_SECONDS)
            now = time.time()
            elapsed_minutes = (now - last_debit) / 60.0
            last_debit = now

            result = await consume_minutes(sb_admin, agency_id, elapsed_minutes)
            if not result.get("ok", True):
                logger.warning(f"[stark-agent] creditos acabaram durante sessao — disconnecting")
                await ctx.room.local_participant.publish_data(
                    b'{"type":"no_credits","message":"Creditos esgotados."}',
                    reliable=True,
                )
                break
    finally:
        # ── Grava telemetria da sessao no Supabase ──
        duration_s = int(time.time() - session_start)
        try:
            sb_admin.table("stark_voice_sessions").insert({
                "user_id": user_id,
                "agency_id": agency_id,
                "livekit_room_id": ctx.room.sid,
                "duration_seconds": duration_s,
                "tools_called": tools_called,
                "llm_provider": llm_res.provider,
                "llm_model": llm_res.model,
                "credit_source": "tier",  # TODO: distinguir tier vs pack
                "ended_at": "now()",
            }).execute()
        except Exception as e:
            logger.exception(f"[stark-agent] telemetria falhou: {e}")

        await ctx.shutdown(reason="session_ended")
        logger.info(f"[stark-agent] session ended user={user_id} duration_s={duration_s}")


def _load_voice_settings(sb_admin, user_id: str) -> tuple[str, Optional[float], Optional[float]]:
    """Le voice_id + stability + speed de user_api_keys.

    Salvo em user_api_keys via UI Settings > Stark > Voz (providers
    stark_voice_id / stark_voice_stability / stark_voice_speed).
    Retorna (voice_id, stability_or_None, speed_or_None).
    """
    voice_id = os.environ.get("AIKORTEX_DEFAULT_VOICE_ID", "EXAVITQu4vr4xnSDxMaL")
    stability: Optional[float] = None
    speed: Optional[float] = None
    try:
        res = (
            sb_admin.table("user_api_keys")
            .select("provider,api_key")
            .eq("user_id", user_id)
            .in_("provider", ["stark_voice_id", "stark_voice_stability", "stark_voice_speed"])
            .execute()
        )
        rows = (res.data if res else None) or []
        by = {(r.get("provider") or ""): (r.get("api_key") or "").strip() for r in rows}
        if by.get("stark_voice_id"):
            voice_id = by["stark_voice_id"]
        if by.get("stark_voice_stability"):
            try:
                stability = float(by["stark_voice_stability"])
            except ValueError:
                pass
        if by.get("stark_voice_speed"):
            try:
                speed = float(by["stark_voice_speed"])
            except ValueError:
                pass
    except Exception as e:
        logger.warning(f"[stark-agent] erro lendo voice settings: {e}")
    return voice_id, stability, speed


def _build_tts(sb_admin, user_id: str, locale: str):
    """Instancia elevenlabs.TTS com Voice + VoiceSettings quando o plugin
    suporta. Defensivo: se VoiceSettings nao existe na versao instalada,
    cai pra config minima (so voice_id) sem quebrar."""
    voice_id, stability, speed = _load_voice_settings(sb_admin, user_id)

    voice = elevenlabs.Voice(id=voice_id, name="Stark", category="premade")

    # Speed do ElevenLabs API aceita 0.8..1.2 (o slider frontend vai 0.7..1.3).
    if speed is not None:
        speed = max(0.8, min(1.2, speed))
    if stability is not None:
        stability = max(0.0, min(1.0, stability))

    voice_settings = None
    VoiceSettingsCls = getattr(elevenlabs, "VoiceSettings", None)
    if VoiceSettingsCls is not None and (stability is not None or speed is not None):
        # Argumentos que a versao antiga pode nao aceitar (speed, use_speaker_boost).
        # Monta kwargs dinamicamente e prova com try.
        kwargs = {
            "stability": stability if stability is not None else 0.5,
            "similarity_boost": 0.75,
        }
        if speed is not None:
            kwargs["speed"] = speed
        try:
            voice_settings = VoiceSettingsCls(**kwargs)
        except TypeError:
            # speed nao aceito nessa versao — tenta so stability + similarity
            kwargs.pop("speed", None)
            try:
                voice_settings = VoiceSettingsCls(**kwargs)
            except Exception as e:
                logger.warning(f"[stark-agent] VoiceSettings falhou: {e}")
                voice_settings = None

    base_kwargs = {
        "api_key": os.environ["ELEVENLABS_API_KEY"],
        "voice": voice,
        "model": "eleven_turbo_v2_5",
        "language": locale.split("-")[0],
    }
    if voice_settings is not None:
        # Tenta injetar voice_settings — pode nao existir na versao antiga.
        try:
            return elevenlabs.TTS(voice_settings=voice_settings, **base_kwargs)
        except TypeError:
            logger.warning("[stark-agent] TTS nao aceita voice_settings — usando defaults")

    logger.info(f"[stark-agent] TTS voice={voice_id} stability={stability} speed={speed}")
    return elevenlabs.TTS(**base_kwargs)
