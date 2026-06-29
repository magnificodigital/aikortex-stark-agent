# Aikortex Stark Agent

Voice agent do **Stark** (copiloto da Aikortex) usando LiveKit Agents SDK.
Pipeline: Deepgram STT → OpenRouter LLM (Claude Haiku) → ElevenLabs TTS.
Tools acessam o Supabase com JWT do user (RLS-aware).

## Stack

- **Linguagem:** Python 3.11+
- **Framework:** [livekit-agents](https://github.com/livekit/agents)
- **STT:** Deepgram Nova-2 streaming
- **LLM:** Claude Haiku 4.5 via OpenRouter
- **TTS:** ElevenLabs Turbo v2.5 streaming
- **DB:** Supabase (Postgres + RLS)
- **Deploy:** Railway (Dockerfile)

## Setup local

```bash
poetry install
cp .env.example .env  # preenche com tuas chaves
poetry run python -m src.main start  # connecta no LiveKit Cloud como worker
```

## Deploy Railway

1. **Conectar repo GitHub** — Railway → New Project → Deploy from GitHub repo
2. **Adicionar Variables** (Settings → Variables) — copia tudo de `.env.example`
3. **Trigger primeiro build** — `git push origin main`
4. **Healthcheck**: o worker conecta OUTBOUND no LiveKit Cloud, então não precisa
   de porta HTTP exposta. Railway vai mostrar `Running` quando o registro com
   LiveKit acontecer.

## Fluxo de conexão

```
Browser (LiveKit React SDK)
    ↓ requesta token
Supabase edge function `stark-token`
    ↓ verifica créditos + gera JWT
Browser conecta → LiveKit Cloud (room stark-{userId})
    ↓ participant.joined
Worker (este serviço) → run_stark_session(ctx)
    ├─ Deepgram STT (streaming)
    ├─ OpenRouter LLM (com tools)
    └─ ElevenLabs TTS (streaming)
```

## Tools disponíveis

Mantidas em sincronia com `supabase/functions/_shared/stark-tools.ts` (edge
function que ainda existe pro modo texto). Cada tool roda com o JWT do user
→ RLS filtra dados.

- `list_agents` — lista agentes do user
- `count_outcomes(tag, period)` — conta resultados (qualified, resolved, etc)
- `query_revenue(period)` — receita das assinaturas Asaas
- `query_messages(period)` — conversas no período
- `query_calls(period)` — ligações no período
- `query_cadences(period)` — execuções de cadência

Períodos válidos: `today`, `yesterday`, `this_week`, `last_7_days`,
`this_month`, `last_30_days`.

## Créditos

Toda sessão começa com check via RPC `consume_stark_voice_minutes(agency_id, 0)`.
Se zero créditos (tier + packs), agent desconecta com evento `no_credits`.

Durante a sessão, debita a cada 30s (tempo desde último débito convertido
em minutos decimais). Quando acabar mid-sessão, fecha graciosamente.

## Telemetria

Cada sessão gera 1 row em `stark_voice_sessions` com:
- `duration_seconds`
- `tools_called` (array)
- `llm_provider`, `llm_model`
- `credit_source` (tier ou pack)

## Observações

- O worker do livekit-agents conecta OUTBOUND no LiveKit Cloud — não precisa
  de inbound port. Railway só precisa que o processo fique rodando.
- Cada job (sessão de um user) roda em task asyncio dentro do worker.
  Escalabilidade horizontal = mais réplicas do serviço no Railway.
