"""Entry point pro LiveKit worker.

Cada room que recebe um participante dispara ``entrypoint(ctx)`` em um job
dedicado. ``cli.run_app`` cuida do registro do worker no LiveKit Cloud e
do shutdown gracioso.
"""

from __future__ import annotations

import json
import os

from dotenv import load_dotenv
from livekit.agents import (
    AutoSubscribe,
    JobContext,
    JobProcess,
    WorkerOptions,
    cli,
)
from livekit.plugins import silero
from loguru import logger

from src.agent import run_stark_session
from src.credits import check_credits_or_exit
from src.supabase_client import supabase_admin

load_dotenv()


def prewarm(proc: JobProcess) -> None:
    """Roda uma vez por worker process, ANTES de qualquer sessao.

    Carregar o VAD silero custa ~1s de CPU — fazer por sessao atrasava
    todo connect. Prewarmado, a sessao pega pronto de proc.userdata.
    """
    proc.userdata["vad"] = silero.VAD.load()
    logger.info("[stark-agent] silero VAD prewarmado")


async def entrypoint(ctx: JobContext) -> None:
    """Conexão de um participante → roda o Stark até o disconnect."""
    # SUBSCRIBE_ALL: alem do audio, recebemos o video da camera do user
    # (modulo visao — Stark ve o que o user mostra). Track de video so'
    # existe quando o user liga a camera na UI.
    await ctx.connect(auto_subscribe=AutoSubscribe.SUBSCRIBE_ALL)
    logger.info(f"[stark-agent] participant connected to room={ctx.room.name}")

    # 1) Espera o primeiro participant (o user) ficar visível
    participant = await ctx.wait_for_participant()
    metadata_raw = participant.metadata or "{}"
    try:
        metadata = json.loads(metadata_raw)
    except json.JSONDecodeError:
        logger.error(f"[stark-agent] metadata invalido: {metadata_raw!r}")
        await ctx.shutdown(reason="invalid_metadata")
        return

    user_id = metadata.get("userId")
    agency_id = metadata.get("agencyId")
    locale = metadata.get("locale", "pt-BR")
    page_context = metadata.get("page_context") or None
    if not user_id:
        logger.error("[stark-agent] sem userId no metadata — abortando")
        await ctx.shutdown(reason="missing_user_id")
        return

    # 2) Re-check creditos no inicio da sessao (pode ter mudado entre
    #    stark-token retornar e o user entrar na sala)
    sb = supabase_admin()
    ok = await check_credits_or_exit(sb, agency_id, ctx, user_id=user_id)
    if not ok:
        return

    # 3) Inicia sessao Stark (STT/LLM/TTS pipeline + tools + telemetria)
    await run_stark_session(
        ctx=ctx,
        user_id=user_id,
        agency_id=agency_id,
        locale=locale,
        participant_jwt=metadata.get("jwt"),  # passado pelo cliente pra tools
        page_context=page_context,
    )


if __name__ == "__main__":
    # Railway expoe PORT mas worker do livekit-agents nao precisa
    # (ele conecta OUTBOUND no LiveKit Cloud).
    cli.run_app(WorkerOptions(entrypoint_fnc=entrypoint, prewarm_fnc=prewarm))
