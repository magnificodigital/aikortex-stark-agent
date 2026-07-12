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

# Kill-switch de plataforma: se a row stark_tools_enabled nao existe no
# platform_config, esses defaults valem. Criador de agentes por voz nasce
# BLOQUEADO — admin libera em /admin?tab=stark quando quiser.
DEFAULT_PLATFORM_TOOLS: dict[str, bool] = {"open_agent_creator": False}


def _fetch_platform_tools(sb_admin) -> dict[str, bool]:
    """Le o kill-switch global de tools do admin (platform_config).

    value e' JSON {tool: bool}. Ausente/invalido → DEFAULT_PLATFORM_TOOLS.
    """
    import json as _json
    try:
        res = (
            sb_admin.table("platform_config")
            .select("value")
            .eq("key", "stark_tools_enabled")
            .limit(1)
            .execute()
        )
        rows = (res.data if res else None) or []
        if rows:
            parsed = _json.loads(rows[0].get("value") or "{}")
            if isinstance(parsed, dict):
                return {str(k): bool(v) for k, v in parsed.items()}
    except Exception as e:
        logger.warning(f"[stark-agent] erro lendo platform tools: {e}")
    return dict(DEFAULT_PLATFORM_TOOLS)


def _merge_tools(platform: dict[str, bool], user: Optional[dict]) -> dict[str, bool]:
    """AND logico: tool ativa so se plataforma E user permitirem.

    Admin desligou → morta pra todo mundo (user nao consegue religar).
    Admin ligou (ou omitiu) → vale a escolha do user.
    """
    merged: dict[str, bool] = dict(user or {})
    for name, enabled in platform.items():
        if enabled is False:
            merged[name] = False
    return merged


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

    # ── Startup I/O em paralelo: prefs + chaves + kill-switch admin ──
    prefs, keys, platform_tools = await asyncio.gather(
        asyncio.to_thread(load_prefs, sb_admin, user_id),
        asyncio.to_thread(_fetch_user_api_keys, sb_admin, user_id),
        asyncio.to_thread(_fetch_platform_tools, sb_admin),
    )
    system_prompt = build_system_prompt(prefs, page_context=page_context)

    # ── Tools (mesmas do stark-tools.ts, agora em Python) ──
    # SEGURANCA: o client e' service-role (o JWT do user nao chega no
    # metadata hoje) — por isso TODAS as tools filtram tenant explicito
    # (user_id / agency_id / agent_ids). Ver src/tools/__init__.py.
    sb_for_tools = supabase_user(participant_jwt) if participant_jwt else sb_admin
    # Gating em 2 niveis: admin da plataforma (kill-switch global) AND
    # escolha do user em Settings. Admin OFF vence sempre.
    effective_tools = _merge_tools(
        platform_tools, (prefs or {}).get("tools_enabled") if prefs else None
    )
    fnc_ctx = StarkTools(
        sb_for_tools,
        user_id=user_id,
        agency_id=agency_id,
        room=ctx.room,
        tools_enabled=effective_tools,
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
        # Comeca o TTS assim que a primeira frase do LLM chega, sem esperar
        # a resposta completa — corta ~0.5-1.5s da latencia percebida.
        preemptive_synthesis=True,
        # Quanto de silencio o VAD espera antes de considerar que o user
        # terminou de falar. Default 0.5s; 0.4s responde mais rapido sem
        # cortar fala normal.
        min_endpointing_delay=0.4,
    )

    # ── Metricas de latencia por turno — vao pro log do Railway ──
    # eou = end-of-utterance (VAD), ttft = tempo do 1o token do LLM,
    # ttfb = tempo do 1o byte de audio do TTS. Soma ≈ latencia percebida.
    @agent.on("metrics_collected")
    def _on_metrics(m):
        try:
            name = type(m).__name__
            if "EOU" in name:
                logger.info(f"[metrics] EOU delay={getattr(m, 'end_of_utterance_delay', '?'):.2f}s")
            elif "LLM" in name:
                logger.info(f"[metrics] LLM ttft={getattr(m, 'ttft', 0):.2f}s tokens/s={getattr(m, 'tokens_per_second', 0):.0f}")
            elif "TTS" in name:
                logger.info(f"[metrics] TTS ttfb={getattr(m, 'ttfb', 0):.2f}s")
        except Exception:
            pass

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

    # ── Alertas proativos: fala sozinho ao conectar se tem pendencia ──
    try:
        alerts_text = await _collect_alerts(sb_admin, user_id, agency_id)
        if alerts_text:
            await agent.say(alerts_text, allow_interruptions=True)
    except Exception as e:
        logger.warning(f"[stark-agent] alertas proativos falharam: {e}")

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

        # ── Memoria entre sessoes: salva um resumo curto da conversa ──
        # Injetado no system prompt da PROXIMA sessao (persona.load_prefs).
        try:
            summary = _summarize_chat(agent.chat_ctx)
            if summary:
                await asyncio.to_thread(
                    lambda: sb_admin.table("stark_user_prefs").upsert({
                        "user_id": user_id,
                        "last_session_summary": summary,
                        "updated_at": datetime.now(timezone.utc).isoformat(),
                    }, on_conflict="user_id").execute()
                )
        except Exception as e:
            logger.warning(f"[stark-agent] memoria de sessao falhou: {e}")

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


async def _collect_alerts(sb_admin, user_id: str, agency_id: Optional[str]) -> str:
    """Pendencias que valem falar sozinho ao conectar: faturas atrasadas
    e leads quentes. Retorna "" quando nao tem nada (fica em silencio)."""
    parts: list[str] = []
    try:
        overdue = await asyncio.to_thread(
            lambda: sb_admin.table("invoices")
            .select("amount", count="exact")
            .eq("user_id", user_id)
            .eq("status", "overdue")
            .execute()
        )
        n_over = overdue.count or 0
        if n_over:
            total = sum(float(r.get("amount") or 0) for r in (overdue.data or []))
            parts.append(f"{n_over} faturas atrasadas somando {total:,.0f} reais")
    except Exception:
        pass
    try:
        if agency_id:
            hot = await asyncio.to_thread(
                lambda: sb_admin.table("crm_contacts")
                .select("id", count="exact")
                .eq("agency_id", agency_id)
                .eq("temperature", "hot")
                .execute()
            )
            if hot.count:
                parts.append(f"{hot.count} leads quentes esperando contato")
    except Exception:
        pass
    if not parts:
        return ""
    return "Atenção: " + " e ".join(parts) + "."


def _summarize_chat(chat_ctx, max_chars: int = 600) -> str:
    """Resumo deterministico da conversa (sem custo de LLM): ultimas trocas
    user/stark em texto corrido, truncado. Vira memoria da proxima sessao."""
    try:
        messages = getattr(chat_ctx, "messages", None) or []
        lines: list[str] = []
        for m in messages[-8:]:
            role = getattr(m, "role", "")
            if role not in ("user", "assistant"):
                continue
            content = getattr(m, "content", "")
            if not isinstance(content, str):
                continue
            content = content.replace("\n", " ").strip()
            if not content:
                continue
            who = "user" if role == "user" else "stark"
            lines.append(f"{who}: {content[:150]}")
        if not lines:
            return ""
        return " | ".join(lines)[:max_chars]
    except Exception:
        return ""


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
