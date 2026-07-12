"""Stark tools — function calling do livekit-agents.

Cada tool é um método anotado com ``@llm.ai_callable`` direto na classe
(que extende ``llm.FunctionContext``). Esse e' o padrao exigido pelo
livekit-agents SDK — registro em runtime com ``fctx.ai_callable(...)(method)``
nao funciona porque metodos bound em Python sao read-only e o decorator
precisa setar metadata na funcao.

SEGURANCA (tenant isolation): o client supabase e' service-role (bypass
RLS), entao TODA query DEVE filtrar explicitamente por tenant:
  - conversations       → agency_id
  - call_logs           → user_id
  - cadence_executions  → agent_id IN (agentes do user)
  - user_agents         → user_id
  - agency_clients / client_template_subscriptions → agency_id
  - meetings            → host_user_id
  - invoices            → user_id
Nunca adicionar query sem filtro de tenant.

PERF: supabase-py e' sincrono — toda query roda via asyncio.to_thread
pra nao bloquear o event loop de audio do agente (senao o TTS/STT
engasga a cada tool call).

Tools enviam comandos pro frontend (ex: open_agent_creator) via
``room.local_participant.publish_data`` — o useStarkLiveKit captura.
"""

from __future__ import annotations

import asyncio
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
        # Cache dos agent ids do user — usado pra filtrar cadence_executions
        # (que nao tem user_id direto, so agent_id FK).
        self._agent_ids: Optional[list[str]] = None

        # livekit-agents guarda funcoes em self._fncs (confirmado no v0.12.20).
        # Property 'ai_functions' e' so um view read-only sobre _fncs.
        registry: dict = getattr(self, "_fncs", None) or {}

        if tools_enabled:
            disabled = [name for name, on in tools_enabled.items() if on is False]
            for name in disabled:
                registry.pop(name, None)
            if disabled:
                logger.info(f"[tools] desabilitadas pelo user: {disabled}")

        # Log das tools finais que ficaram ativas — util pra debug em Railway.
        active = sorted(registry.keys())
        logger.info(f"[tools] {len(active)} ativas: {active}")

    async def _q(self, build):
        """Roda query sincrona do supabase-py fora do event loop."""
        return await asyncio.to_thread(lambda: build().execute())

    async def _get_agent_ids(self) -> list[str]:
        """Agent ids do user (cacheado) — filtro de tenant pra cadences."""
        if self._agent_ids is not None:
            return self._agent_ids
        try:
            res = await self._q(
                lambda: self.sb.table("user_agents")
                .select("id")
                .eq("user_id", self.user_id)
            )
            self._agent_ids = [r["id"] for r in (res.data or [])]
        except Exception as e:
            logger.exception(f"[tools] erro buscando agent ids: {e}")
            self._agent_ids = []
        return self._agent_ids

    # ─────────────────────────────────────────────────────────────
    # Aikortex — Agentes, Mensagens, Ligações, Cadências
    # ─────────────────────────────────────────────────────────────

    @llm.ai_callable(description="Lista os agentes cadastrados pelo user")
    async def list_agents(self) -> str:
        try:
            res = await self._q(
                lambda: self.sb.table("user_agents")
                .select("name,agent_type,status")
                .eq("user_id", self.user_id)
                .order("updated_at", desc=True)
                .limit(20)
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
            conv_count = 0
            if self.agency_id:
                conv = await self._q(
                    lambda: self.sb.table("conversations")
                    .select("id", count="exact")
                    .eq("agency_id", self.agency_id)
                    .contains("outcome_tags", [outcome_tag])
                    .gte("created_at", from_iso)
                    .lte("created_at", to_iso)
                )
                conv_count = conv.count or 0
            calls = await self._q(
                lambda: self.sb.table("call_logs")
                .select("id", count="exact")
                .eq("user_id", self.user_id)
                .contains("outcome_tags", [outcome_tag])
                .gte("created_at", from_iso)
                .lte("created_at", to_iso)
            )
            total = conv_count + (calls.count or 0)
            return f"{total} {outcome_tag} no período {period}."
        except Exception as e:
            logger.exception(f"[tool count_outcomes] {e}")
            return "Erro consultando outcomes."

    @llm.ai_callable(description="Conta conversas (mensagens) trocadas no período")
    async def query_messages(
        self,
        period: Annotated[str, llm.TypeInfo(description="today, yesterday, this_week, last_7_days, this_month, last_30_days")] = "today",
    ) -> str:
        if not self.agency_id:
            return "Você ainda não tem agência configurada — sem conversas registradas."
        from_iso, to_iso = _resolve_period(period)
        try:
            res = await self._q(
                lambda: self.sb.table("conversations")
                .select("id", count="exact")
                .eq("agency_id", self.agency_id)
                .gte("created_at", from_iso)
                .lte("created_at", to_iso)
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
            res = await self._q(
                lambda: self.sb.table("call_logs")
                .select("id", count="exact")
                .eq("user_id", self.user_id)
                .gte("created_at", from_iso)
                .lte("created_at", to_iso)
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
            agent_ids = await self._get_agent_ids()
            if not agent_ids:
                return "Nenhuma execução de cadência — você ainda não tem agentes."
            res = await self._q(
                lambda: self.sb.table("cadence_executions")
                .select("id", count="exact")
                .in_("agent_id", agent_ids)
                .gte("created_at", from_iso)
                .lte("created_at", to_iso)
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
            def build():
                q = (
                    self.sb.table("agency_clients")
                    .select("client_name,status", count="exact")
                    .eq("agency_id", self.agency_id)
                    .order("created_at", desc=True)
                )
                if status != "all":
                    q = q.eq("status", status)
                return q.limit(10)
            res = await self._q(build)
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
            res = await self._q(
                lambda: self.sb.table("agency_clients")
                .select("id", count="exact")
                .eq("agency_id", self.agency_id)
                .gte("created_at", from_iso)
                .lte("created_at", to_iso)
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
            def build():
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
                return q.limit(10)
            res = await self._q(build)
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
            res = await self._q(
                lambda: self.sb.table("client_template_subscriptions")
                .select("agency_price_monthly,status")
                .eq("agency_id", self.agency_id)
                .eq("status", "active")
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
            def build():
                q = (
                    self.sb.table("invoices")
                    .select("amount,status,due_date", count="exact")
                    .eq("user_id", self.user_id)
                    .order("due_date", desc=True)
                )
                if status != "all":
                    q = q.eq("status", status)
                return q.limit(20)
            res = await self._q(build)
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
    # Gestão — Detalhe de cliente + CRM (crm_contacts)
    # ─────────────────────────────────────────────────────────────

    @llm.ai_callable(description="Busca os detalhes de um cliente específico pelo nome — contato, status, módulos, assinaturas")
    async def get_client_details(
        self,
        name: Annotated[str, llm.TypeInfo(description="Nome (ou parte do nome) do cliente")],
    ) -> str:
        if not self.agency_id:
            return "Você ainda não tem agência configurada."
        try:
            res = await self._q(
                lambda: self.sb.table("agency_clients")
                .select("id,client_name,client_email,client_phone,status,enabled_modules,created_at")
                .eq("agency_id", self.agency_id)
                .ilike("client_name", f"%{name}%")
                .limit(3)
            )
            rows = res.data or []
            if not rows:
                return f"Não achei cliente com nome parecido com '{name}'."
            if len(rows) > 1:
                names = ", ".join(r["client_name"] for r in rows)
                return f"Achei {len(rows)} clientes: {names}. Qual deles?"
            c = rows[0]
            subs = await self._q(
                lambda: self.sb.table("client_template_subscriptions")
                .select("status,agency_price_monthly")
                .eq("agency_id", self.agency_id)
                .eq("client_id", c["id"])
                .eq("status", "active")
            )
            active_subs = subs.data or []
            sub_total = sum(float(s.get("agency_price_monthly") or 0) for s in active_subs)
            mods = ", ".join(c.get("enabled_modules") or []) or "nenhum"
            parts = [
                f"{c['client_name']}: status {c.get('status')}",
                f"email {c.get('client_email') or 'não cadastrado'}",
                f"telefone {c.get('client_phone') or 'não cadastrado'}",
                f"módulos: {mods}",
            ]
            if active_subs:
                parts.append(f"{len(active_subs)} assinaturas ativas somando R$ {sub_total:,.2f}/mês")
            return ". ".join(parts) + "."
        except Exception as e:
            logger.exception(f"[tool get_client_details] {e}")
            return "Erro consultando o cliente."

    @llm.ai_callable(description="Mostra o funil de vendas (pipeline CRM) — quantos leads em cada etapa e temperatura")
    async def query_pipeline(self) -> str:
        if not self.agency_id:
            return "Você ainda não tem agência configurada."
        try:
            res = await self._q(
                lambda: self.sb.table("crm_contacts")
                .select("stage_slug,temperature")
                .eq("agency_id", self.agency_id)
                .limit(1000)
            )
            rows = res.data or []
            if not rows:
                return "Pipeline vazio — nenhum lead no CRM ainda."
            by_stage: dict[str, int] = {}
            temp = {"hot": 0, "warm": 0, "cold": 0}
            for r in rows:
                s = r.get("stage_slug") or "new"
                by_stage[s] = by_stage.get(s, 0) + 1
                t = r.get("temperature")
                if t in temp:
                    temp[t] += 1
            stages = ", ".join(f"{v} em {k}" for k, v in sorted(by_stage.items(), key=lambda x: -x[1]))
            return (
                f"{len(rows)} leads no pipeline: {stages}. "
                f"Temperatura: {temp['hot']} quentes, {temp['warm']} mornos, {temp['cold']} frios."
            )
        except Exception as e:
            logger.exception(f"[tool query_pipeline] {e}")
            return "Erro consultando o pipeline."

    @llm.ai_callable(description="Lista os leads quentes (hot) do CRM — quem o user deve priorizar hoje")
    async def list_hot_leads(self) -> str:
        if not self.agency_id:
            return "Você ainda não tem agência configurada."
        try:
            res = await self._q(
                lambda: self.sb.table("crm_contacts")
                .select("name,company,stage_slug,need,timeline")
                .eq("agency_id", self.agency_id)
                .eq("temperature", "hot")
                .order("updated_at", desc=True)
                .limit(8)
            )
            rows = res.data or []
            if not rows:
                return "Nenhum lead quente no momento."
            lines = []
            for r in rows[:5]:
                who = r.get("name") or "sem nome"
                comp = f" da {r['company']}" if r.get("company") else ""
                lines.append(f"{who}{comp} ({r.get('stage_slug')})")
            extra = f" e mais {len(rows) - 5}" if len(rows) > 5 else ""
            return f"{len(rows)} leads quentes: {', '.join(lines)}{extra}."
        except Exception as e:
            logger.exception(f"[tool list_hot_leads] {e}")
            return "Erro consultando leads quentes."

    @llm.ai_callable(description="Lista faturas que vencem nos próximos dias ou que já estão atrasadas")
    async def query_invoices_due(
        self,
        days: Annotated[int, llm.TypeInfo(description="Janela de dias à frente pra considerar (default 7)")] = 7,
    ) -> str:
        try:
            now = datetime.now(timezone.utc)
            horizon = (now + timedelta(days=max(1, min(days, 90)))).isoformat()
            due = await self._q(
                lambda: self.sb.table("invoices")
                .select("amount,due_date,status", count="exact")
                .eq("user_id", self.user_id)
                .eq("status", "pending")
                .lte("due_date", horizon)
            )
            overdue = await self._q(
                lambda: self.sb.table("invoices")
                .select("amount", count="exact")
                .eq("user_id", self.user_id)
                .eq("status", "overdue")
            )
            due_rows = due.data or []
            due_total = sum(float(r.get("amount") or 0) for r in due_rows)
            over_rows = overdue.data or []
            over_total = sum(float(r.get("amount") or 0) for r in over_rows)
            if not due_rows and not over_rows:
                return f"Nenhuma fatura vencendo nos próximos {days} dias, nada atrasado."
            parts = []
            if over_rows:
                parts.append(f"{len(over_rows)} atrasadas somando R$ {over_total:,.2f}")
            if due_rows:
                parts.append(f"{len(due_rows)} vencendo em até {days} dias somando R$ {due_total:,.2f}")
            return "Atenção: " + " e ".join(parts) + "."
        except Exception as e:
            logger.exception(f"[tool query_invoices_due] {e}")
            return "Erro consultando vencimentos."

    @llm.ai_callable(description="Briefing executivo — resumo geral do negócio: clientes novos, MRR, reuniões, faturas e leads quentes numa resposta só")
    async def executive_briefing(self) -> str:
        try:
            month_start = datetime.now(timezone.utc).replace(
                day=1, hour=0, minute=0, second=0, microsecond=0
            ).isoformat()
            parts: list[str] = []

            if self.agency_id:
                new_clients = await self._q(
                    lambda: self.sb.table("agency_clients")
                    .select("id", count="exact")
                    .eq("agency_id", self.agency_id)
                    .gte("created_at", month_start)
                )
                mrr_res = await self._q(
                    lambda: self.sb.table("client_template_subscriptions")
                    .select("agency_price_monthly")
                    .eq("agency_id", self.agency_id)
                    .eq("status", "active")
                )
                hot = await self._q(
                    lambda: self.sb.table("crm_contacts")
                    .select("id", count="exact")
                    .eq("agency_id", self.agency_id)
                    .eq("temperature", "hot")
                )
                mrr = sum(float(r.get("agency_price_monthly") or 0) for r in (mrr_res.data or []))
                parts.append(f"{new_clients.count or 0} clientes novos no mês")
                parts.append(f"MRR de R$ {mrr:,.2f}")
                if hot.count:
                    parts.append(f"{hot.count} leads quentes esperando contato")

            week_start = (datetime.now(timezone.utc) - timedelta(days=7)).isoformat()
            meetings = await self._q(
                lambda: self.sb.table("meetings")
                .select("id", count="exact")
                .eq("host_user_id", self.user_id)
                .gte("created_at", week_start)
            )
            overdue = await self._q(
                lambda: self.sb.table("invoices")
                .select("id", count="exact")
                .eq("user_id", self.user_id)
                .eq("status", "overdue")
            )
            parts.append(f"{meetings.count or 0} reuniões na última semana")
            if overdue.count:
                parts.append(f"ALERTA: {overdue.count} faturas atrasadas")

            return "Resumo: " + ". ".join(parts) + "."
        except Exception as e:
            logger.exception(f"[tool executive_briefing] {e}")
            return "Erro montando o briefing."

    # ─────────────────────────────────────────────────────────────
    # Ações de gestão — criar/editar (SEMPRE confirmar antes por voz)
    # ─────────────────────────────────────────────────────────────

    @llm.ai_callable(description="Cadastra um cliente novo na agência. SEMPRE confirme nome/email/telefone com o user antes de chamar.")
    async def create_client(
        self,
        name: Annotated[str, llm.TypeInfo(description="Nome do cliente ou empresa")],
        email: Annotated[str, llm.TypeInfo(description="Email de contato (vazio se não informado)")] = "",
        phone: Annotated[str, llm.TypeInfo(description="Telefone (vazio se não informado)")] = "",
    ) -> str:
        if not self.agency_id:
            return "Você ainda não tem agência configurada — não dá pra cadastrar."
        if not name.strip():
            return "Preciso do nome do cliente."
        try:
            await self._q(
                lambda: self.sb.table("agency_clients").insert({
                    "agency_id": self.agency_id,
                    "client_name": name.strip()[:120],
                    "client_email": email.strip()[:160] or None,
                    "client_phone": phone.strip()[:40] or None,
                    "status": "active",
                })
            )
            return f"Cliente {name.strip()} cadastrado."
        except Exception as e:
            logger.exception(f"[tool create_client] {e}")
            return "Não consegui cadastrar o cliente."

    @llm.ai_callable(description="Muda o status de um cliente (active/inactive). Confirme com o user antes.")
    async def update_client_status(
        self,
        name: Annotated[str, llm.TypeInfo(description="Nome do cliente")],
        status: Annotated[str, llm.TypeInfo(description="'active' ou 'inactive'")],
    ) -> str:
        if not self.agency_id:
            return "Você ainda não tem agência configurada."
        if status not in ("active", "inactive"):
            return "Status precisa ser active ou inactive."
        try:
            found = await self._q(
                lambda: self.sb.table("agency_clients")
                .select("id,client_name")
                .eq("agency_id", self.agency_id)
                .ilike("client_name", f"%{name}%")
                .limit(3)
            )
            rows = found.data or []
            if not rows:
                return f"Não achei cliente '{name}'."
            if len(rows) > 1:
                return f"Achei {len(rows)} clientes parecidos: {', '.join(r['client_name'] for r in rows)}. Qual?"
            cid = rows[0]["id"]
            await self._q(
                lambda: self.sb.table("agency_clients")
                .update({"status": status})
                .eq("id", cid)
                .eq("agency_id", self.agency_id)
            )
            return f"{rows[0]['client_name']} agora está {status}."
        except Exception as e:
            logger.exception(f"[tool update_client_status] {e}")
            return "Não consegui atualizar o cliente."

    @llm.ai_callable(description="Adiciona um lead novo no CRM. Confirme nome/empresa/temperatura com o user antes.")
    async def create_crm_lead(
        self,
        name: Annotated[str, llm.TypeInfo(description="Nome do lead")],
        company: Annotated[str, llm.TypeInfo(description="Empresa (vazio se não informado)")] = "",
        temperature: Annotated[str, llm.TypeInfo(description="'hot', 'warm' ou 'cold' (default warm)")] = "warm",
    ) -> str:
        if not self.agency_id:
            return "Você ainda não tem agência configurada."
        if not name.strip():
            return "Preciso do nome do lead."
        if temperature not in ("hot", "warm", "cold"):
            temperature = "warm"
        try:
            await self._q(
                lambda: self.sb.table("crm_contacts").insert({
                    "agency_id": self.agency_id,
                    "name": name.strip()[:120],
                    "company": company.strip()[:120] or None,
                    "temperature": temperature,
                    "stage_slug": "new",
                })
            )
            return f"Lead {name.strip()} adicionado no pipeline como {temperature}."
        except Exception as e:
            logger.exception(f"[tool create_crm_lead] {e}")
            return "Não consegui adicionar o lead."

    @llm.ai_callable(description="Move um lead do CRM pra outra etapa do funil. Confirme com o user antes.")
    async def move_lead_stage(
        self,
        name: Annotated[str, llm.TypeInfo(description="Nome do lead")],
        stage: Annotated[str, llm.TypeInfo(description="Nome ou slug da etapa destino (ex: 'negociação', 'qualified')")],
    ) -> str:
        if not self.agency_id:
            return "Você ainda não tem agência configurada."
        try:
            stages_res = await self._q(
                lambda: self.sb.table("crm_pipeline_stages")
                .select("slug,name")
                .eq("agency_id", self.agency_id)
            )
            stages = stages_res.data or []
            target = None
            s_lower = stage.strip().lower()
            for s in stages:
                if s_lower in ((s.get("slug") or "").lower(), (s.get("name") or "").lower()):
                    target = s["slug"]
                    break
            if target is None:
                for s in stages:
                    if s_lower in (s.get("name") or "").lower() or s_lower in (s.get("slug") or "").lower():
                        target = s["slug"]
                        break
            if target is None:
                names = ", ".join(s.get("name") or s.get("slug") for s in stages) or "nenhuma etapa configurada"
                return f"Etapa '{stage}' não existe. Etapas disponíveis: {names}."

            found = await self._q(
                lambda: self.sb.table("crm_contacts")
                .select("id,name")
                .eq("agency_id", self.agency_id)
                .ilike("name", f"%{name}%")
                .limit(3)
            )
            rows = found.data or []
            if not rows:
                return f"Não achei lead '{name}'."
            if len(rows) > 1:
                return f"Achei {len(rows)} leads parecidos: {', '.join(r['name'] for r in rows)}. Qual?"
            await self._q(
                lambda: self.sb.table("crm_contacts")
                .update({"stage_slug": target})
                .eq("id", rows[0]["id"])
                .eq("agency_id", self.agency_id)
            )
            return f"{rows[0]['name']} movido pra etapa {target}."
        except Exception as e:
            logger.exception(f"[tool move_lead_stage] {e}")
            return "Não consegui mover o lead."

    @llm.ai_callable(description="Cria uma sala de reunião. Confirme o título com o user antes.")
    async def create_meeting(
        self,
        title: Annotated[str, llm.TypeInfo(description="Título da reunião")],
    ) -> str:
        if not title.strip():
            return "Preciso do título da reunião."
        try:
            await self._q(
                lambda: self.sb.table("meetings").insert({
                    "host_user_id": self.user_id,
                    "title": title.strip()[:120],
                    "status": "waiting",
                })
            )
            return f"Reunião '{title.strip()}' criada — está na página de reuniões, aguardando início."
        except Exception as e:
            logger.exception(f"[tool create_meeting] {e}")
            return "Não consegui criar a reunião."

    @llm.ai_callable(description="Encerra/cancela uma reunião pelo título. Confirme com o user antes.")
    async def cancel_meeting(
        self,
        title: Annotated[str, llm.TypeInfo(description="Título (ou parte) da reunião a encerrar")],
    ) -> str:
        try:
            found = await self._q(
                lambda: self.sb.table("meetings")
                .select("id,title,status")
                .eq("host_user_id", self.user_id)
                .neq("status", "ended")
                .ilike("title", f"%{title}%")
                .order("created_at", desc=True)
                .limit(3)
            )
            rows = found.data or []
            if not rows:
                return f"Não achei reunião ativa com título '{title}'."
            if len(rows) > 1:
                return f"Achei {len(rows)}: {', '.join(r['title'] for r in rows)}. Qual delas?"
            await self._q(
                lambda: self.sb.table("meetings")
                .update({"status": "ended", "ended_at": datetime.now(timezone.utc).isoformat()})
                .eq("id", rows[0]["id"])
                .eq("host_user_id", self.user_id)
            )
            return f"Reunião '{rows[0]['title']}' encerrada."
        except Exception as e:
            logger.exception(f"[tool cancel_meeting] {e}")
            return "Não consegui encerrar a reunião."

    # ─────────────────────────────────────────────────────────────
    # Navegação — manda o frontend pra outra página
    # ─────────────────────────────────────────────────────────────

    # Whitelist de páginas — nome falado → rota. Nunca navegar pra rota fora
    # dessa lista (o LLM não decide rotas livres).
    NAV_PAGES = {
        "clientes": "/clients",
        "crm": "/aikortex/crm",
        "pipeline": "/aikortex/crm",
        "financeiro": "/financial",
        "reunioes": "/dashboard",
        "dashboard": "/dashboard",
        "relatorios": "/reports",
        "agentes": "/aikortex",
        "apps": "/apps",
        "templates": "/templates",
        "configuracoes": "/settings",
        "home": "/home",
    }

    async def _publish(self, payload: dict) -> bool:
        if not self.room or not self.room.local_participant:
            return False
        try:
            await self.room.local_participant.publish_data(
                json.dumps(payload).encode("utf-8"), reliable=True
            )
            return True
        except Exception as e:
            logger.exception(f"[tools] publish_data falhou: {e}")
            return False

    @llm.ai_callable(description="Navega a interface pra uma página: clientes, crm, pipeline, financeiro, dashboard, relatorios, agentes, apps, templates, configuracoes, home")
    async def navigate_to(
        self,
        page: Annotated[str, llm.TypeInfo(description="Nome da página destino (ex: 'financeiro', 'clientes', 'crm')")],
    ) -> str:
        key = (
            page.strip().lower()
            .replace("ç", "c").replace("õ", "o").replace("ã", "a")
            .replace("é", "e").replace("í", "i").replace("ó", "o")
        )
        path = self.NAV_PAGES.get(key)
        if not path:
            return f"Não conheço a página '{page}'. Opções: {', '.join(sorted(self.NAV_PAGES))}."
        ok = await self._publish({"type": "navigate", "path": path})
        return f"Abrindo {page}." if ok else "Sem conexão pra navegar agora."

    @llm.ai_callable(description="Abre o perfil de um cliente específico na interface")
    async def open_client(
        self,
        name: Annotated[str, llm.TypeInfo(description="Nome do cliente")],
    ) -> str:
        if not self.agency_id:
            return "Você ainda não tem agência configurada."
        try:
            found = await self._q(
                lambda: self.sb.table("agency_clients")
                .select("id,client_name")
                .eq("agency_id", self.agency_id)
                .ilike("client_name", f"%{name}%")
                .limit(3)
            )
            rows = found.data or []
            if not rows:
                return f"Não achei cliente '{name}'."
            if len(rows) > 1:
                return f"Achei {len(rows)}: {', '.join(r['client_name'] for r in rows)}. Qual?"
            ok = await self._publish({"type": "navigate", "path": f"/clients/{rows[0]['id']}"})
            return f"Abrindo o perfil de {rows[0]['client_name']}." if ok else "Sem conexão pra navegar."
        except Exception as e:
            logger.exception(f"[tool open_client] {e}")
            return "Erro abrindo o cliente."

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
