"""Stark tools — mesma interface do edge function ``stark-tools.ts``.

Cada tool é um método anotado com ``@llm.ai_callable`` no
``StarkTools.build_function_context()``. O LiveKit Agents SDK lê as
docstrings + type hints e expõe pro LLM via function calling.

RLS-aware: todas as queries usam o ``Client`` recebido no __init__, que
no caso de Stark é autenticado com o JWT do user (filtra dados dele).
"""

from __future__ import annotations

from datetime import datetime, timedelta, timezone
from typing import Optional

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
        # Segunda 00:00
        week_start = today_start - timedelta(days=today_start.weekday())
        return week_start.isoformat(), now.isoformat()
    if period == "last_7_days":
        return (now - timedelta(days=7)).isoformat(), now.isoformat()
    if period == "this_month":
        month_start = today_start.replace(day=1)
        return month_start.isoformat(), now.isoformat()
    if period == "last_30_days":
        return (now - timedelta(days=30)).isoformat(), now.isoformat()
    # default: últimas 24h
    return (now - timedelta(days=1)).isoformat(), now.isoformat()


class StarkTools:
    """Container de todas as tools com client supabase já autenticado."""

    def __init__(self, sb: Client, user_id: str) -> None:
        self.sb = sb
        self.user_id = user_id

    def build_function_context(self) -> llm.FunctionContext:
        """Retorna FunctionContext do livekit-agents com todas as tools."""
        fctx = llm.FunctionContext()
        fctx.ai_callable(name="list_agents", description="Lista os agentes da agência")(
            self.list_agents
        )
        fctx.ai_callable(
            name="count_outcomes",
            description="Conta quantos outcomes de um tipo aconteceram no período",
        )(self.count_outcomes)
        fctx.ai_callable(
            name="query_revenue",
            description="Soma receita das assinaturas Asaas no período",
        )(self.query_revenue)
        fctx.ai_callable(
            name="query_messages",
            description="Conta mensagens trocadas no período (entrada e saída)",
        )(self.query_messages)
        fctx.ai_callable(
            name="query_calls",
            description="Conta ligações telefônicas no período",
        )(self.query_calls)
        fctx.ai_callable(
            name="query_cadences",
            description="Conta execuções de cadência no período",
        )(self.query_cadences)
        return fctx

    # ── Implementações ─────────────────────────────────────────────

    async def list_agents(self) -> str:
        """Retorna lista de agentes do user em texto curto."""
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

    async def count_outcomes(
        self, outcome_tag: str, period: str = "today"
    ) -> str:
        """Conta conversations + call_logs com `outcome_tag` no período."""
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

    async def query_revenue(self, period: str = "this_month") -> str:
        """Soma receita das assinaturas Asaas pagas no período."""
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
            # Cada agente publicado = R$ 997 (default Master v7.4)
            revenue = published * 997
            return f"{published} agentes ativos no período. Receita: R$ {revenue:,.2f}."
        except Exception as e:
            logger.exception(f"[tool query_revenue] {e}")
            return "Erro consultando receita."

    async def query_messages(self, period: str = "today") -> str:
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

    async def query_calls(self, period: str = "today") -> str:
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

    async def query_cadences(self, period: str = "today") -> str:
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
