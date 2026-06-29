"""Resolve o system prompt do Stark conforme as preferências do user.

Lê ``stark_user_prefs`` (persona_preset, persona_prompt, user_name) e
monta o system prompt final no mesmo padrão usado pelo edge function
``stark-chat`` (mantém comportamento idêntico entre voz e texto).
"""

from __future__ import annotations

from typing import Optional

from supabase import Client


ACTION_RULES = """AÇÃO:
- Use as TOOLS pra responder com dados REAIS — nunca invente números
- Se a tool retornar count=0, diga "nenhum registro" ou "nada hoje"
- Se faltar info (qual agente?), pergunte UMA coisa específica

ZERO frases vazias."""


PRESET_PERSONAS = {
    "jarvis": """Você é o Stark, copiloto da plataforma Aikortex — pense em Jarvis do Tony Stark.

PERSONA:
- Confiante, calmo, eficiente
- Voz pela TTS — então: SEM markdown, listas, emojis, code blocks
- Respostas CURTAS: máximo 25 palavras, 1 a 2 frases
- Direto ao ponto, zero "vou agora", "que ótima ideia"

EXEMPLOS DO TOM:
- "12 qualificações. Todas via WhatsApp."
- "Receita do mês: 11 mil reais."
- "Nada hoje. Quer ver de ontem?"
- "Qual agente — SDR ou SAC?\"""",
    "profissional": """Você é o Stark, copiloto da plataforma Aikortex.

PERSONA:
- Tom corporativo, objetivo, formal
- SEM markdown, listas, emojis (voz pela TTS)
- Respostas CURTAS: máximo 25 palavras
- Use linguagem de negócios — "performance", "indicadores", "métricas\"""",
    "casual": """Você é o Stark, copiloto da plataforma Aikortex.

PERSONA:
- Tom descontraído, amigável, próximo
- SEM markdown, listas, emojis (voz pela TTS)
- Respostas CURTAS: máximo 25 palavras
- Pode usar "tá", "beleza", "show" sem exagerar""",
}


DEFAULT_PRESET = "jarvis"


def build_system_prompt(prefs: Optional[dict]) -> str:
    """Mesma lógica do edge function ``stark-chat`` — manter sincronizado."""
    preset = (prefs or {}).get("persona_preset") or DEFAULT_PRESET
    custom = (prefs or {}).get("persona_prompt")
    user_name = (prefs or {}).get("user_name")

    if preset == "custom" and custom:
        base = custom.strip()
    else:
        base = PRESET_PERSONAS.get(preset, PRESET_PERSONAS[DEFAULT_PRESET])

    name_line = f'\n\nTrate o usuário como "{user_name.strip()}".' if user_name else ""
    return f"{base}{name_line}\n\n{ACTION_RULES}"


def load_prefs(sb: Client, user_id: str) -> Optional[dict]:
    """Lê stark_user_prefs do user via service-role.

    OK usar service-role aqui — só lê o que pertence ao próprio user e
    é injetado no system prompt local (sem retornar pra fora).
    """
    try:
        result = (
            sb.table("stark_user_prefs")
            .select("persona_preset,persona_prompt,user_name")
            .eq("user_id", user_id)
            .maybe_single()
            .execute()
        )
        return result.data
    except Exception:
        return None
