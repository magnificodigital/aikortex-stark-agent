"""Stark tools — mesma interface do edge function ``stark-tools.ts``.

Cada tool é um método anotado com ``@llm.ai_callable`` direto na classe
(que extende ``llm.FunctionContext``). Esse e' o padrao exigido pelo
livekit-agents SDK — registro em runtime com ``fctx.ai_callable(...)(method)``
nao funciona porque metodos bound em Python sao read-only e o decorator
precisa setar metadata na funcao.

RLS-aware: o client supabase recebido no __init__ esta autenticado com
o JWT do user (Stark Agent forward via metadata da sala LiveKit).
"""

from __future__ import annotations

from datetime import datetime, timedelta, timezone
from typing import Annotated

from livekit.agents import llm
from loguru import logger
from supabase import Client


def _resolve_period(period: str) -> tuple[str, str]:
    """Mapeia 'today' / 'this_week' etc → (from_iso, to_iso). Espelha
    a função homônima em ``_shared/stark-tools.ts``.
    """
    now = datetime.now(timezone.utc)
    today_start = now.replace(hour=0, minute=0, second=0, microsecond=0)

    if period == "today":
        return today_start.isoformat(), now.isoformat()
    if period == "yesterday":
        y_start = today_start - timedelta(days=1)
        return y_start.isoformat(), today_start.isoformat()
    if period == "this_week":
        week_start = today_start - timedelta(days=today_start.weekday())
        return week_start.isoformat(), now.isoformat()
    if period == "last_7_days":
        return (now - timedelta(days=7)).isoformat(), now.isoformat()
    if period == "this_month":
        month_start = today_start.replace(day=1)
        return month_start.isoformat(), now.isoformat()
    if period == "last_30_days":
        return (now - timedelta(days=30)).isoformat(), now.isoformat()
    return (now - timedelta(days=1)).isoformat(), now.isoformat()


class StarkTools(llm.FunctionContext):
    """FunctionContext do livekit-agents com todas as tools do Stark."""

    def __init__(self, sb: Client, user_id: str) -> None:
        super().__init__()
        self.sb = sb
        self.user_id = user_id

    @llm.ai_callable(description="Lista os agentes cadastrados na agência")
    async def list_agents(self) -> str:
        try:
            res = (
                self.sb.table("user_agents")
                .select("name,agent_type,status")
                .eq("user_id", self.user_id)
                .order("updated_at", desc=True)
                .limit(20)
                .execute()
            )
            rows = res.data or []
            if not rows:
                return "Nenhum agente cadastrado ainda."
            lines = [f"{r['name']} ({r['agent_type']}, {r['status']})" for r in rows]
            return f"Você tem {len(rows)} agentes: {', '.join(lines)}."
        except Exception as e:
            logger.exception(f"[tool list_agents] {e}")
            return "Erro consultando agentes."

    @llm.ai_callable(description="Conta quantos outcomes de um tipo aconteceram em um período")
    async def count_outcomes(
        self,
        outcome_tag: Annotated[str, llm.TypeInfo(description="Tag do outcome — ex: qualified, resolved, meeting_booked")],
        period: Annotated[str, llm.TypeInfo(description="today, yesterday, this_week, last_7_days, this_month, last_30_days")] = "today",
    ) -> str:
        from_iso, to_iso = _resolve_period(period)
        try:
            conv = (
                self.sb.table("conversations")
                .select("id", count="exact")
                .contains("outcome_tags", [outcome_tag])
                .gte("created_at", from_iso)
                .lte("created_at", to_iso)
                .execute()
            )
            calls = (
                self.sb.table("call_logs")
                .select("id", count="exact")
                .contains("outcome_tags", [outcome_tag])
                .gte("created_at", from_iso)
                .lte("created_at", to_iso)
                .execute()
            )
            total = (conv.count or 0) + (calls.count or 0)
            return f"{total} {outcome_tag} no período {period}."
        except Exception as e:
            logger.exception(f"[tool count_outcomes] {e}")
            return "Erro consultando outcomes."

    @llm.ai_callable(description="Soma receita das assinaturas Asaas ativas no período")
    async def query_revenue(
        self,
        period: Annotated[str, llm.TypeInfo(description="today, yesterday, this_week, last_7_days, this_month, last_30_days")] = "this_month",
    ) -> str:
        from_iso, to_iso = _resolve_period(period)
        try:
            res = (
                self.sb.table("user_agents")
                .select("subscription_status,published_at")
                .eq("user_id", self.user_id)
                .gte("published_at", from_iso)
                .lte("published_at", to_iso)
                .execute()
            )
            published = sum(1 for r in (res.data or []) if r.get("subscription_status") == "active")
            revenue = published * 997
            return f"{published} agentes ativos no período. Receita: R$ {revenue:,.2f}."
        except Exception as e:
            logger.exception(f"[tool query_revenue] {e}")
            return "Erro consultando receita."

    @llm.ai_callable(description="Conta conversas trocadas no período")
    async def query_messages(
        self,
        period: Annotated[str, llm.TypeInfo(description="today, yesterday, this_week, last_7_days, this_month, last_30_days")] = "today",
    ) -> str:
        from_iso, to_iso = _resolve_period(period)
        try:
            res = (
                self.sb.table("conversations")
                .select("id", count="exact")
                .gte("created_at", from_iso)
                .lte("created_at", to_iso)
                .execute()
            )
            return f"{res.count or 0} conversas no período {period}."
        except Exception as e:
            logger.exception(f"[tool query_messages] {e}")
            return "Erro consultando mensagens."

    @llm.ai_callable(description="Conta ligações telefônicas no período")
    async def query_calls(
        self,
        period: Annotated[str, llm.TypeInfo(description="today, yesterday, this_week, last_7_days, this_month, last_30_days")] = "today",
    ) -> str:
        from_iso, to_iso = _resolve_period(period)
        try:
            res = (
                self.sb.table("call_logs")
                .select("id", count="exact")
                .gte("created_at", from_iso)
                .lte("created_at", to_iso)
                .execute()
            )
            return f"{res.count or 0} ligações no período {period}."
        except Exception as e:
            logger.exception(f"[tool query_calls] {e}")
            return "Erro consultando ligações."

    @llm.ai_callable(description="Conta execuções de cadência no período")
    async def query_cadences(
        self,
        period: Annotated[str, llm.TypeInfo(description="today, yesterday, this_week, last_7_days, this_month, last_30_days")] = "today",
    ) -> str:
        from_iso, to_iso = _resolve_period(period)
        try:
            res = (
                self.sb.table("cadence_executions")
                .select("id", count="exact")
                .gte("created_at", from_iso)
                .lte("created_at", to_iso)
                .execute()
            )
            return f"{res.count or 0} execuções de cadência no período {period}."
        except Exception as e:
            logger.exception(f"[tool query_cadences] {e}")
            return "Erro consultando cadências."
