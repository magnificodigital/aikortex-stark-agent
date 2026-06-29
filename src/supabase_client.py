"""Wrappers em torno do supabase-py com cache simples por escopo.

Há 2 modos de uso:

- ``supabase_admin()`` — service-role, bypass RLS. Pra escrever telemetria
  (stark_voice_sessions) e chamar funções como ``consume_stark_voice_minutes``.
- ``supabase_user(jwt)`` — autenticado com JWT do user. Pra ler dados via
  RLS (user_agents, conversations, call_logs etc) com isolamento correto.
"""

from __future__ import annotations

import os
from functools import lru_cache

from supabase import Client, create_client


@lru_cache(maxsize=1)
def supabase_admin() -> Client:
    """Cliente service-role — uma instância só, reutilizada."""
    url = os.environ["SUPABASE_URL"]
    key = os.environ["SUPABASE_SERVICE_ROLE_KEY"]
    return create_client(url, key)


def supabase_user(jwt: str) -> Client:
    """Cliente anon-key com Authorization header do user — RLS ativo.

    Cria nova instância por chamada (JWT é per-session, não cacheável).
    """
    url = os.environ["SUPABASE_URL"]
    anon = os.environ["SUPABASE_ANON_KEY"]
    client = create_client(url, anon)
    # supabase-py expõe o session via postgrest options
    client.postgrest.auth(jwt)
    return client
