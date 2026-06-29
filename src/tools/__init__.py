"""Stark tools — function calling do livekit-agents.

Cada tool é um método anotado com ``@llm.ai_callable`` direto na classe
(que extende ``llm.FunctionContext``). Esse e' o padrao exigido pelo
livekit-agents SDK — registro em runtime com ``fctx.ai_callable(...)(method)``
nao funciona porque metodos bound em Python sao read-only e o decorator
precisa setar metadata na funcao.

RLS-aware: o client supabase recebido no __init__ esta autenticado com
o JWT do user (Stark Agent forward via metadata da sala LiveKit).

Tools enviam comandos pro frontend (ex: open_agent_creator) via
``room.local_participant.publish_data`` — o useStarkLiveKit captura.
"""

from __future__ import annotations

import json
from datetime import datetime, timedelta, timezone
from typing import Annotated, Optional

from livekit import rtc
from livekit.agents import llm
from loguru import logger
from supabase import Client


def _resolve_period(period: str) -> tuple[str, str]:
    """Mapeia 'today' / 'this_week' etc → (from_iso, to_iso)."""
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
    """FunctionContext com todas as tools do Stark.

    Se ``tools_enabled`` for fornecido, tools com valor False sao
    removidas do registro logo apos o __init__ — o LLM nem chega a
    enxergar a tool. Quando o dict nao tem a key da tool, considera ON
    por default (backward compat).
    """

    def __init__(
        self,
        sb: Client,
        user_id: str,
        agency_id: Optional[str] = None,
        room: Optional[rtc.Room] = None,
        tools_enabled: Optional[dict[str, bool]] = None,
    ) -> None:
        super().__init__()
        self.sb = sb
        self.user_id = user_id
        self.agency_id = agency_id
        self.room = room

        if tools_enabled:
            # Remove tools desabilitadas do FunctionContext.
            disabled = [name for name, on in tools_enabled.items() if on is False]
            for name in disabled:
                # livekit-agents armazena em self._fncs (dict). Tenta varios nomes.
                for attr in ("_fncs", "ai_functions", "_ai_functions"):
                    registry = getattr(self, attr, None)
                    if isinstance(registry, dict) and name in registry:
                        registry.pop(name, None)
            if disabled:
                logger.info(f"[tools] desabilitadas pelo user: {disabled}")

    # ─────────────────────────────────────────────────────────────
    # Aikortex — Agentes, Mensagens, Ligações, Cadências
    # ─────────────────────────────────────────────────────────────

    @llm.ai_callable(description="Lista os agentes cadastrados pelo user")
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

    @llm.ai_callable(description="Conta quantos outcomes de um tipo aconteceram em um período (qualificados, agendamentos, resolvidos)")
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

    @llm.ai_callable(description="Conta conversas (mensagens) trocadas no período")
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

    # ─────────────────────────────────────────────────────────────
    # Gestão — Clientes (agency_clients)
    # ─────────────────────────────────────────────────────────────

    @llm.ai_callable(description="Lista os clientes da agência (CRM interno). Retorna total + alguns nomes.")
    async def list_clients(
        self,
        status: Annotated[str, llm.TypeInfo(description="Filtro de status: 'all', 'active', 'inactive'. Default 'all'.")] = "all",
    ) -> str:
        if not self.agency_id:
            return "Você ainda não tem agência configurada — sem clientes pra listar."
        try:
            q = (
                self.sb.table("agency_clients")
                .select("client_name,status", count="exact")
                .eq("agency_id", self.agency_id)
                .order("created_at", desc=True)
            )
            if status != "all":
                q = q.eq("status", status)
            res = q.limit(10).execute()
            rows = res.data or []
            total = res.count or 0
            if total == 0:
                return "Nenhum cliente cadastrado ainda."
            names = ", ".join(r["client_name"] for r in rows[:5])
            extra = f" e mais {total - 5}" if total > 5 else ""
            return f"{total} clientes. Últimos: {names}{extra}."
        except Exception as e:
            logger.exception(f"[tool list_clients] {e}")
            return "Erro consultando clientes."

    @llm.ai_callable(description="Conta quantos clientes novos foram cadastrados no período")
    async def count_new_clients(
        self,
        period: Annotated[str, llm.TypeInfo(description="today, yesterday, this_week, last_7_days, this_month, last_30_days")] = "this_month",
    ) -> str:
        if not self.agency_id:
            return "Você ainda não tem agência configurada."
        from_iso, to_iso = _resolve_period(period)
        try:
            res = (
                self.sb.table("agency_clients")
                .select("id", count="exact")
                .eq("agency_id", self.agency_id)
                .gte("created_at", from_iso)
                .lte("created_at", to_iso)
                .execute()
            )
            return f"{res.count or 0} clientes novos no período {period}."
        except Exception as e:
            logger.exception(f"[tool count_new_clients] {e}")
            return "Erro consultando clientes novos."

    # ─────────────────────────────────────────────────────────────
    # Gestão — Reuniões (meetings)
    # ─────────────────────────────────────────────────────────────

    @llm.ai_callable(description="Lista reuniões do user — agendadas, em andamento ou encerradas")
    async def list_meetings(
        self,
        status: Annotated[str, llm.TypeInfo(description="'waiting' (agendadas), 'active' (em andamento), 'ended' (encerradas), 'all'")] = "all",
        period: Annotated[str, llm.TypeInfo(description="today, this_week, last_7_days, this_month")] = "this_week",
    ) -> str:
        from_iso, to_iso = _resolve_period(period)
        try:
            q = (
                self.sb.table("meetings")
                .select("title,status,started_at", count="exact")
                .eq("host_user_id", self.user_id)
                .gte("created_at", from_iso)
                .lte("created_at", to_iso)
                .order("started_at", desc=True)
            )
            if status != "all":
                q = q.eq("status", status)
            res = q.limit(10).execute()
            rows = res.data or []
            total = res.count or 0
            if total == 0:
                return f"Nenhuma reunião no período {period}."
            titles = ", ".join(r.get("title") or "sem título" for r in rows[:5])
            return f"{total} reuniões. Últimas: {titles}."
        except Exception as e:
            logger.exception(f"[tool list_meetings] {e}")
            return "Erro consultando reuniões."

    # ─────────────────────────────────────────────────────────────
    # Gestão — Financeiro (client_template_subscriptions + invoices)
    # ─────────────────────────────────────────────────────────────

    @llm.ai_callable(description="Soma a receita mensal recorrente (MRR) — assinaturas ativas dos clientes da agência")
    async def query_mrr(self) -> str:
        if not self.agency_id:
            return "Você ainda não tem agência configurada — sem receita recorrente."
        try:
            res = (
                self.sb.table("client_template_subscriptions")
                .select("agency_price_monthly,status")
                .eq("agency_id", self.agency_id)
                .eq("status", "active")
                .execute()
            )
            rows = res.data or []
            mrr = sum(float(r.get("agency_price_monthly") or 0) for r in rows)
            count = len(rows)
            if count == 0:
                return "Nenhuma assinatura ativa de cliente ainda."
            return f"MRR atual: R$ {mrr:,.2f} com {count} assinaturas ativas."
        except Exception as e:
            logger.exception(f"[tool query_mrr] {e}")
            return "Erro consultando receita."

    @llm.ai_callable(description="Lista faturas (invoices) por status: pendentes, pagas ou atrasadas")
    async def query_invoices(
        self,
        status: Annotated[str, llm.TypeInfo(description="'pending', 'paid', 'overdue', 'all'")] = "pending",
    ) -> str:
        try:
            q = (
                self.sb.table("invoices")
                .select("amount,status,due_date", count="exact")
                .eq("user_id", self.user_id)
                .order("due_date", desc=True)
            )
            if status != "all":
                q = q.eq("status", status)
            res = q.limit(20).execute()
            rows = res.data or []
            total_count = res.count or 0
            if total_count == 0:
                return f"Nenhuma fatura com status {status}."
            total_amount = sum(float(r.get("amount") or 0) for r in rows)
            return f"{total_count} faturas {status}, totalizando R$ {total_amount:,.2f}."
        except Exception as e:
            logger.exception(f"[tool query_invoices] {e}")
            return "Erro consultando faturas."

    # ─────────────────────────────────────────────────────────────
    # Ação — abrir criador de agentes (manda evento pro frontend)
    # ─────────────────────────────────────────────────────────────

    @llm.ai_callable(description="Abre o criador de agentes na interface — use quando o user pedir pra criar/cadastrar um novo agente. Passa a descrição do agente como prompt inicial.")
    async def open_agent_creator(
        self,
        initial_prompt: Annotated[str, llm.TypeInfo(description="O que o user quer no agente — ex: 'SDR pra clínica odontológica' ou 'SAC pra e-commerce de roupas'")],
    ) -> str:
        if not self.room or not self.room.local_participant:
            return "Sem conexão pra abrir o criador agora — tente novamente em alguns segundos."
        try:
            payload = json.dumps({
                "type": "open_agent_creator",
                "initial_prompt": initial_prompt,
            }).encode("utf-8")
            await self.room.local_participant.publish_data(payload, reliable=True)
            return "Abri o criador de agentes pra você."
        except Exception as e:
            logger.exception(f"[tool open_agent_creator] {e}")
            return "Não consegui abrir o criador agora."
