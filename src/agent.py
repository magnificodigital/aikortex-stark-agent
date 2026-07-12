"""Pipeline Stark Voice — STT (Deepgram) → LLM (cascade) → TTS (ElevenLabs).

LiveKit Agents SDK cuida da bidirectional streaming via WebRTC. Aqui só
montamos o ``VoicePipelineAgent``, plugamos os providers e registramos
as tools (function calling).

PERF: supabase-py e' sincrono. Todo I/O de startup roda em paralelo via
asyncio.gather + to_thread (1 query pra user_api_keys inteira em vez de
7 sequenciais). O VAD silero vem prewarmado do worker process
(main.prewarm) — carregar por sessao custava ~1s de CPU.
"""

from __future__ import annotations

import asyncio
import os
import time
from datetime import datetime, timezone
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
from src.llm import PROVIDER_ORDER, create_stark_llm
from src.persona import build_system_prompt, load_prefs
from src.supabase_client import supabase_admin, supabase_user
from src.tools import StarkTools


CREDIT_CHECK_INTERVAL_SECONDS = 30

VOICE_PROVIDERS = ("stark_voice_id", "stark_voice_stability", "stark_voice_speed")


def _fetch_user_api_keys(sb_admin, user_id: str) -> dict[str, str]:
    """Uma query so' pra todas as rows de user_api_keys que a sessao precisa
    (chaves LLM do cascade + voice settings). Retorna {provider: api_key}."""
    wanted = list(PROVIDER_ORDER) + list(VOICE_PROVIDERS)
    try:
        res = (
            sb_admin.table("user_api_keys")
            .select("provider,api_key")
            .eq("user_id", user_id)
            .in_("provider", wanted)
            .execute()
        )
        rows = (res.data if res else None) or []
        return {
            (r.get("provider") or ""): (r.get("api_key") or "").strip()
            for r in rows
        }
    except Exception as e:
        logger.warning(f"[stark-agent] erro lendo user_api_keys: {e}")
        return {}


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

    # ── Startup I/O em paralelo: prefs + todas as chaves numa tacada ──
    prefs, keys = await asyncio.gather(
        asyncio.to_thread(load_prefs, sb_admin, user_id),
        asyncio.to_thread(_fetch_user_api_keys, sb_admin, user_id),
    )
    system_prompt = build_system_prompt(prefs, page_context=page_context)

    # ── Tools (mesmas do stark-tools.ts, agora em Python) ──
    # SEGURANCA: o client e' service-role (o JWT do user nao chega no
    # metadata hoje) — por isso TODAS as tools filtram tenant explicito
    # (user_id / agency_id / agent_ids). Ver src/tools/__init__.py.
    sb_for_tools = supabase_user(participant_jwt) if participant_jwt else sb_admin
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
    # anthropic/gemini) > chave Aikortex (OpenRouter). Chaves ja vieram
    # na query batch; so available_llms pode gerar 1 query extra (openrouter).
    llm_instance, llm_res = await asyncio.to_thread(
        create_stark_llm, sb_admin, user_id, keys
    )

    # VAD prewarmado no worker process (main.prewarm). Fallback local se
    # o prewarm nao rodou (ex: dev sem prewarm_fnc).
    vad = ctx.proc.userdata.get("vad") if ctx.proc.userdata else None
    if vad is None:
        logger.warning("[stark-agent] VAD nao prewarmado — carregando na sessao")
        vad = silero.VAD.load()

    agent = VoicePipelineAgent(
        vad=vad,
        stt=deepgram.STT(
            model="nova-2-general",
            language=locale,
            punctuate=True,
            interim_results=True,
        ),
        llm=llm_instance,
        tts=_build_tts(keys, locale),
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
                logger.warning("[stark-agent] creditos acabaram durante sessao — disconnecting")
                await ctx.room.local_participant.publish_data(
                    b'{"type":"no_credits","message":"Creditos esgotados."}',
                    reliable=True,
                )
                break
    finally:
        # ── Debita o resto da sessao (segundos desde o ultimo debito) ──
        # Sem isso, ate 30s por sessao ficavam de graca (loop so debita
        # em multiplos do intervalo).
        tail_minutes = (time.time() - last_debit) / 60.0
        if tail_minutes > 0:
            await consume_minutes(sb_admin, agency_id, tail_minutes)

        # ── Grava telemetria da sessao no Supabase ──
        duration_s = int(time.time() - session_start)
        try:
            await asyncio.to_thread(
                lambda: sb_admin.table("stark_voice_sessions").insert({
                    "user_id": user_id,
                    "agency_id": agency_id,
                    "livekit_room_id": ctx.room.sid,
                    "duration_seconds": duration_s,
                    "tools_called": tools_called,
                    "llm_provider": llm_res.provider,
                    "llm_model": llm_res.model,
                    "credit_source": "tier",  # TODO: distinguir tier vs pack
                    "ended_at": datetime.now(timezone.utc).isoformat(),
                }).execute()
            )
        except Exception as e:
            logger.exception(f"[stark-agent] telemetria falhou: {e}")

        await ctx.shutdown(reason="session_ended")
        logger.info(f"[stark-agent] session ended user={user_id} duration_s={duration_s}")


def _build_tts(keys: dict[str, str], locale: str):
    """Instancia elevenlabs.TTS com Voice + VoiceSettings quando o plugin
    suporta. ``keys`` vem da query batch de user_api_keys — sem I/O aqui.

    Defensivo: se VoiceSettings nao existe na versao instalada, cai pra
    config minima (so voice_id) sem quebrar."""
    voice_id = keys.get("stark_voice_id") or os.environ.get(
        "AIKORTEX_DEFAULT_VOICE_ID", "EXAVITQu4vr4xnSDxMaL"
    )

    stability: Optional[float] = None
    speed: Optional[float] = None
    try:
        if keys.get("stark_voice_stability"):
            stability = float(keys["stark_voice_stability"])
    except ValueError:
        pass
    try:
        if keys.get("stark_voice_speed"):
            speed = float(keys["stark_voice_speed"])
    except ValueError:
        pass

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
    logger.info(f"[stark-agent] TTS voice={voice_id} stability={stability} speed={speed}")
    if voice_settings is not None:
        # Tenta injetar voice_settings — pode nao existir na versao antiga.
        try:
            return elevenlabs.TTS(voice_settings=voice_settings, **base_kwargs)
        except TypeError:
            logger.warning("[stark-agent] TTS nao aceita voice_settings — usando defaults")

    return elevenlabs.TTS(**base_kwargs)
