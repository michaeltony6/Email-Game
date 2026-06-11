# Email Game Agent

Competition agent for The Email Game.

The tournament entry point is `my_agent.py`. The rest of this repo is intentionally small: the full game harness can live next to this repo locally, but it is not part of the submitted agent.

## Strategy

This agent is a hybrid strategy agent:

- Deterministic core for safety-critical work:
  - parses moderator instructions
  - requests required signatures
  - signs only when authorized
  - proactively sends authorized signatures when it has current-round message ownership evidence
  - submits valid signatures
  - rejects stale or unsafe fuzzy requests
- OpenAI strategy advisor for adaptive play:
  - classifies opponents when the round is late or high-value enough to justify the API call
  - chooses attack style after the baseline exchange is mostly protected
  - estimates risk and expected value from stored opponent behavior
  - adapts pressure to perceived opponent type and live score posture
  - writes custom per-opponent attack copy when enabled
  - resolves ambiguous fuzzy identity clues with GPT when deterministic transcript matching is not confident
  - helps decide when to exploit, probe, play sterile, counter-poison, or disengage
- Persistent memory:
  - remembers opponent behavior between games
  - marks easy signers, late responders, stale-request senders, hostile identity poisoners, and high scorers
  - adapts farming appetite after ties/losses/wins
  - stores display-name mappings and estimated opponent strength
- Attack portfolio:
  - submission-coach reminders after we sign for another agent
  - confusion-capture packets for agents that mix rounds or placeholders
  - reciprocal ledgers for cooperative/easy signers
  - zero-cost late checks once baseline is protected
- Fuzzy identity resolver:
  - maps "agent who mentioned X" descriptions to transcript evidence from prior rounds
  - cross-checks exact assigned-message aliases, request text, and opponent email history
  - uses GPT only for ambiguous identity cases and requires higher confidence before signing
  - resets per-game identity cache so old ladder games cannot poison new authorizations

The LLM layer is advisory only. It cannot override the deterministic signing checks. The agent also has per-round and per-game API budgets so GPT is used tactically rather than constantly.

## Recommended Competition Setup

Use `gpt-4.1` for the strategy layer.

```bash
export OPENAI_API_KEY="sk-..."
export OPENAI_MODEL="gpt-4.1"
export OPENAI_STRATEGY_MODEL="gpt-4.1"
export EMAIL_GAME_USE_LLM_STRATEGY=1
export EMAIL_GAME_USE_CUSTOM_COPY=1
export EMAIL_GAME_USE_LLM_FUZZY=1
export EMAIL_GAME_LLM_BUDGET_PER_ROUND=4
export EMAIL_GAME_LLM_BUDGET_PER_GAME=8
```

Optional known-opponent tags:

```bash
export EMAIL_GAME_HIGH_ELO_TARGETS="agent1,agent2"
export EMAIL_GAME_AGGRESSIVE_TARGETS="agent3"
export EMAIL_GAME_DEFENSIVE_TARGETS="agent4"
export EMAIL_GAME_ELO_TARGETS="agent1=1200,agent2=1150"
```

Optional persistent memory path:

```bash
export EMAIL_GAME_MEMORY_PATH="../.email_game_memory.json"
```

## Run In The Tournament

From the full game harness directory:

```bash
python scripts/run_custom_agent.py michaeltony6 "Michael Tony" --module ../my_agent.py --server https://COMPETITION-SERVER-URL
```

Replace `https://COMPETITION-SERVER-URL` with the actual tournament server URL.

Leave the process running. The server will queue the agent, run games, and requeue it after completed games.

## Local Validation

From this repo:

```bash
python3 -m py_compile my_agent.py
```

From the local game harness directory:

```bash
python scripts/playtest.py ../my_agent.py
```

To run deterministic fallback mode without the LLM strategy advisor:

```bash
EMAIL_GAME_USE_LLM_STRATEGY=0 python scripts/playtest.py ../my_agent.py
```

## Files

- `my_agent.py`: full competition agent
- `README.md`: setup and run instructions

The local `theemailgame/` folder, if present, is only the test harness clone and is not part of this repo.
