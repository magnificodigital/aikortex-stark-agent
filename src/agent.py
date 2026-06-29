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
) -> None:
    """Sessão completa do Stark até o disconnect."""
    sb_admin = supabase_admin()

    # ── Sistema de prompt baseado em prefs do user ──
    prefs = load_prefs(sb_admin, user_id)
    system_prompt = build_system_prompt(prefs)

    # ── Tools (mesmas do stark-tools.ts, agora em Python) ──
    # User-scoped client via JWT (se disponivel) — RLS filtra dados do user.
    # Fallback admin se o cliente nao mandou JWT no metadata (modo dev).
    sb_for_tools = supabase_user(participant_jwt) if participant_jwt else sb_admin
    # StarkTools extende llm.FunctionContext — instancia ja e' o context.
    fnc_ctx = StarkTools(sb_for_tools, user_id=user_id)

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
        tts=elevenlabs.TTS(
            api_key=os.environ["ELEVENLABS_API_KEY"],
            voice=elevenlabs.Voice(
                id=_pick_voice_id(prefs),
                name="Stark",
                category="premade",
            ),
            model="eleven_turbo_v2_5",
            language=locale.split("-")[0],
        ),
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


def _pick_voice_id(prefs: Optional[dict]) -> str:
    """Pega voice_id do user — ou default Sarah.

    NOTE: stark_voice_id fica em user_api_keys (não em stark_user_prefs).
    Pra simplificar a Fase 2 inicial, usa só default. Próxima iteração
    le do user_api_keys.
    """
    return os.environ.get("AIKORTEX_DEFAULT_VOICE_ID", "EXAVITQu4vr4xnSDxMaL")
