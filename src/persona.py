"""Resolve o system prompt do Stark conforme as preferências do user.

Mesma estrutura do edge function ``stark-chat`` — mantém comportamento
identico entre voz e texto.

Hoje suporta:
- preset: executivo | profissional | casual | custom
- sliders 0..100: tone (formal..casual), response_length (curto..detalhado),
  energy (serio..animado)
- language: pt-BR | en | es
- user_name (como o Stark chama o user)
- tools_enabled: dict {tool_id: bool} — usado em StarkTools, nao aqui.

Compat: 'jarvis' legacy vira 'executivo' silenciosamente.
"""

from __future__ import annotations

from typing import Optional

from supabase import Client


ACTION_RULES_BY_LANG = {
    "pt-BR": """AÇÃO:
- Use as TOOLS pra responder com dados REAIS — nunca invente números
- Se a tool retornar count=0, diga "nenhum registro" ou "nada hoje"
- Se faltar info (qual agente?), pergunte UMA coisa específica

O QUE VOCÊ SABE CONSULTAR (via tools):
- Aikortex: agentes, mensagens, ligações, cadências, qualificações
- Gestão: clientes (lista e detalhe), pipeline CRM, leads quentes,
  reuniões, MRR, faturas, vencimentos, briefing executivo

O QUE VOCÊ SABE FAZER (tools de escrita):
- Cadastrar/atualizar cliente, adicionar lead, mover lead de etapa,
  criar/encerrar reunião
- REGRA: antes de ESCREVER dados (criar/editar/apagar), repita o que
  vai fazer e espere o user confirmar. Só chame a tool depois do "sim"

NAVEGAÇÃO E CONSULTAS: execute DIRETO, sem pedir confirmação — o
pedido do user JÁ é a autorização. "Me leva pro financeiro" = chame
navigate_to imediatamente. Confirmação é SÓ pra escrita de dados.

VISÃO: quando o user liga a câmera, você RECEBE a imagem junto da
pergunta. Se ele mostrar algo ("o que é isso?", "lê esse documento"),
descreva/identifique com base na imagem. Sem câmera = sem imagem;
peça pra ligar a câmera se ele quiser que você veja algo.

O QUE NÃO EXISTE (responda honesto, não invente):
- Tarefas/To-dos, Equipe, Projetos, Propostas, Contratos — sem módulo ainda

ZERO frases vazias.""",
    "en": """ACTION:
- Use TOOLS to answer with REAL data — never make up numbers
- If a tool returns count=0, say "no records" or "nothing today"
- If info is missing (which agent?), ask ONE specific question

WHAT YOU CAN QUERY (via tools): agents, messages, calls, cadences,
qualifications, clients (CRM), meetings, MRR, invoices. You can also
open the agent creator when the user asks to build a new agent.

WHAT DOESN'T EXIST (be honest, don't invent): own Sales/Pipeline,
Tasks/To-dos, Team management.

ZERO filler phrases.""",
    "es": """ACCIÓN:
- Use TOOLS para responder con datos REALES — nunca invente números
- Si una tool retorna count=0, diga "sin registros" o "nada hoy"
- Si falta info (qué agente?), pregunte UNA cosa específica

QUÉ PUEDE CONSULTAR (via tools): agentes, mensajes, llamadas, cadencias,
calificaciones, clientes (CRM), reuniones, MRR, facturas. También puede
abrir el creador de agentes cuando el usuario pida construir uno nuevo.

QUÉ NO EXISTE (sea honesto, no invente): Ventas/Pipeline propio,
Tareas/To-dos, Gestión de Equipo.

CERO frases vacías.""",
}


PRESET_BASES = {
    "executivo": "Você é o Stark, copiloto da plataforma Aikortex.\n\nPERSONA:\n- Confiante, calmo, eficiente\n- Direto ao ponto",
    "profissional": "Você é o Stark, copiloto da plataforma Aikortex.\n\nPERSONA:\n- Tom corporativo, objetivo\n- Use linguagem de negócios: \"performance\", \"indicadores\", \"métricas\"",
    "casual": "Você é o Stark, copiloto da plataforma Aikortex.\n\nPERSONA:\n- Tom descontraído, amigável, próximo\n- Pode usar \"tá\", \"beleza\", \"show\" sem exagerar",
}


def _tone_descriptor(tone: int) -> str:
    if tone < 33:
        return "Tom formal e corporativo"
    if tone < 67:
        return "Tom equilibrado — nem rígido, nem solto"
    return "Tom casual e descontraído"


def _length_descriptor(response_length: int) -> tuple[str, str]:
    """Retorna (descritor, max_palavras_text)."""
    if response_length < 33:
        return "Respostas CURTAS: máximo 25 palavras, 1-2 frases.", "25"
    if response_length < 67:
        return "Respostas MÉDIAS: 40-60 palavras, 2-4 frases.", "60"
    return "Respostas DETALHADAS: até 120 palavras, paragrafadas.", "120"


def _energy_descriptor(energy: int) -> str:
    if energy < 33:
        return "Energia baixa — sério, comedido"
    if energy < 67:
        return "Energia neutra"
    return "Energia alta — animado, expressivo"


VOICE_NOTE = "- Voz pela TTS — então: SEM markdown, listas, emojis, code blocks"


def _clean_ctx_str(value, max_len: int) -> str:
    """Sanitiza string vinda do frontend antes de entrar no system prompt:
    corta tamanho e remove quebras de linha (mitiga prompt injection via
    page_context — o conteudo vira UMA linha inerte no prompt)."""
    if not isinstance(value, str):
        return ""
    return value.replace("\n", " ").replace("\r", " ").strip()[:max_len]


def _format_page_context(page_context: Optional[dict]) -> str:
    """Transforma o page_context do frontend em uma linha pro system prompt.

    Ex: "Contexto: usuario esta em 'detalhes do cliente' (path=/clients/123,
    cliente=Joao Silva)."
    """
    if not page_context or not isinstance(page_context, dict):
        return ""
    path = _clean_ctx_str(page_context.get("path"), 200)
    route = _clean_ctx_str(page_context.get("route"), 80)
    entity = page_context.get("entity") or {}
    if not isinstance(entity, dict):
        entity = {}

    parts: list[str] = []
    if route:
        parts.append(f"'{route}'")
    if path:
        parts.append(f"path={path}")
    if entity:
        etype = _clean_ctx_str(entity.get("type"), 40) or "entity"
        eid = _clean_ctx_str(entity.get("id"), 64)
        ename = _clean_ctx_str(entity.get("name"), 120)
        entity_bits = [etype]
        if ename:
            entity_bits.append(f"'{ename}'")
        elif eid:
            entity_bits.append(f"id={eid}")
        parts.append(f"foco: {' '.join(entity_bits)}")

    if not parts:
        return ""
    return f"\n\nCONTEXTO DA PÁGINA: usuário está em {', '.join(parts)}. Se ele disser 'ele/ela/isso', assuma que se refere a esse contexto."


def build_system_prompt(prefs: Optional[dict], page_context: Optional[dict] = None) -> str:
    """Monta o system prompt baseado nos prefs do user + contexto de pagina.

    Cascade do preset:
      - 'custom' + persona_prompt → usa persona_prompt como base
      - 'executivo'|'profissional'|'casual' → usa template + sliders
      - 'jarvis' legado → vira 'executivo'

    page_context (opcional): {path, route, entity} — quando presente, injeta
    linha extra no prompt pra Stark saber onde o user esta.
    """
    p = prefs or {}
    preset = p.get("persona_preset") or "executivo"
    if preset == "jarvis":
        preset = "executivo"

    custom = p.get("persona_prompt")
    user_name = p.get("user_name")
    language = p.get("language") or "pt-BR"
    tone = int(p.get("tone") if p.get("tone") is not None else 50)
    response_length = int(p.get("response_length") if p.get("response_length") is not None else 25)
    energy = int(p.get("energy") if p.get("energy") is not None else 50)

    # Base prompt
    if preset == "custom" and custom:
        base = custom.strip()
    else:
        base = PRESET_BASES.get(preset, PRESET_BASES["executivo"])

    # Sliders ditam o comportamento fino (anexado depois do base).
    length_desc, _max_words = _length_descriptor(response_length)
    style_lines = [
        f"- {_tone_descriptor(tone)}",
        f"- {length_desc}",
        f"- {_energy_descriptor(energy)}",
        VOICE_NOTE,
    ]
    style_block = "ESTILO:\n" + "\n".join(style_lines)

    name_line = f'\n\nTrate o usuário como "{user_name.strip()}".' if user_name else ""
    context_line = _format_page_context(page_context)

    # Memoria da ultima conversa (gravada pelo agent no fim da sessao).
    memory = _clean_ctx_str(p.get("last_session_summary"), 700)
    memory_line = (
        f"\n\nMEMÓRIA DA ÚLTIMA CONVERSA (use se o user referenciar algo anterior):\n{memory}"
        if memory else ""
    )

    rules = ACTION_RULES_BY_LANG.get(language) or ACTION_RULES_BY_LANG["pt-BR"]

    return f"{base}\n\n{style_block}{name_line}{context_line}{memory_line}\n\n{rules}"


def load_prefs(sb: Client, user_id: str) -> Optional[dict]:
    """Lê stark_user_prefs do user via service-role.

    OK usar service-role aqui — só lê o que pertence ao próprio user e
    é injetado no system prompt local (sem retornar pra fora).
    """
    try:
        result = (
            sb.table("stark_user_prefs")
            .select("persona_preset,persona_prompt,user_name,tone,response_length,energy,language,tools_enabled,last_session_summary")
            .eq("user_id", user_id)
            .limit(1)
            .execute()
        )
        rows = (result.data if result else None) or []
        return rows[0] if rows else None
    except Exception:
        return None
