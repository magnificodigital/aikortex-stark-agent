"""Gerenciamento de créditos de voz Stark.

Toda lógica de debito é server-side (função SQL ``consume_stark_voice_minutes``).
Aqui temos só wrappers que chamam a RPC + tratam o retorno pro Python.

O Stark Agent chama ``consume_minutes`` a cada N segundos durante a sessão.
Se a RPC retornar ``ok=False`` (sem créditos suficientes), o agent envia
evento de fim de sessão pro cliente e desconecta.
"""

from __future__ import annotations

from typing import Optional

from livekit.agents import JobContext
from loguru import logger
from supabase import Client


async def check_credits_or_exit(
    sb: Client, agency_id: Optional[str], ctx: JobContext
) -> bool:
    """Faz check inicial. Se não tem créditos, fecha o room e retorna False."""
    if not agency_id:
        logger.warning("[credits] sem agency_id — pulando check (modo dev?)")
        return True

    try:
        result = sb.rpc(
            "consume_stark_voice_minutes",
            {"p_agency_id": agency_id, "p_minutes": 0},
        ).execute()
        data = result.data or {}
        remaining = data.get("remaining_tier", 0)
        if remaining < 1:
            # Pode ainda ter pack — checa separado
            packs = (
                sb.table("stark_voice_credit_packs")
                .select("minutes_total,minutes_used")
                .eq("user_id", agency_id)  # NOTE: agency_profiles.user_id != id
                .eq("status", "paid")
                .execute()
            )
            pack_remaining = sum(
                max(0, (p.get("minutes_total") or 0) - (p.get("minutes_used") or 0))
                for p in (packs.data or [])
            )
            if pack_remaining < 1:
                logger.warning(f"[credits] agency={agency_id} sem creditos — disconnecting")
                await ctx.room.local_participant.publish_data(
                    b'{"type":"no_credits","message":"Acabaram os minutos de Stark voz."}',
                    reliable=True,
                )
                await ctx.shutdown(reason="no_credits")
                return False
        return True
    except Exception as e:
        logger.exception(f"[credits] erro checando creditos: {e}")
        # Fail-open em erro (não bloqueia produto) — usage tracking ainda roda
        # depois e o overage fica visível pro user na próxima sessão.
        return True


async def consume_minutes(
    sb: Client, agency_id: Optional[str], minutes: float
) -> dict:
    """Debita N minutos da agência. Retorna dict da RPC."""
    if not agency_id:
        return {"ok": True, "consumed_from_tier": 0, "consumed_from_pack": 0}
    try:
        result = sb.rpc(
            "consume_stark_voice_minutes",
            {"p_agency_id": agency_id, "p_minutes": float(minutes)},
        ).execute()
        return result.data or {}
    except Exception as e:
        logger.exception(f"[credits] consume falhou: {e}")
        return {"ok": False, "short_minutes": minutes, "error": str(e)}
