# Projeto integrado: mtcli + codex-mt5-agent

Este diretório foi gerado automaticamente a partir de:

- `mtcli/` (estrutura e código do CLI original)
- `codex-mt5-agent/` (prompts, agent_config, docs MQL5/MT5)

## Estrutura principal

- `agent/`
  - Configuração do agente (agent_config.yaml, tools_index.yaml, knowledge_index.yaml).

- `prompts/`
  - `system.yaml` – system prompt principal.
  - `behavior_contract.yaml` – contrato de comportamento (modo auto-solve, multi-estratégia).
  - `domain/` – conhecimento de domínio (MQL5, MT5, mtcli).
  - `workflows/` – fluxos de solução (problem_solving, bug_diagnosis, indicator_creation).
  - `tools/` – definição de ferramentas (mtcli, docs_offline, web_search, recipes, etc.).

- `docs/`
  - `mql5_reference/` – espaço para documentação offline da linguagem MQL5.
  - `mt5_userguide/` – espaço para documentação offline da plataforma MT5.

- Código do CLI mtcli:
  - `src/`, `dist/`, `bin/`, `tools/`, `scripts/`, `mt_agent_pack/`, `mt_agent_recipes/`, etc.

- `legacy_docs/`
  - Documentação antiga herdada do repositório mtcli original (AGENTS antigo, CHANGELOG, README antigo, docs antigos).
  - Use apenas como referência histórica. Se nada mais for útil, você pode apagá-la.

## O que fazer depois

1. Ajustar `prompts/tools/mtcli.yaml` para apontar para o CLI desta estrutura
   (por exemplo, para `bin/mtcli` ou o entrypoint real do CLI).

2. Ajustar `prompts/tools/docs_offline.yaml` e `agent/knowledge_index.yaml`
   para incluir este repositório integrado como fonte de conhecimento sobre o mtcli
   (por exemplo, AGENTS.md, docs novas do CLI, etc.).

3. Reescrever/atualizar `legacy_docs/README_mtcli_legacy.md` e `legacy_docs/docs_cli_legacy/`
   em novos arquivos dentro de `docs/` (por exemplo, `docs/cli/overview.md`,
   `docs/cli/cli_reference.md`, `docs/cli/recipes.md`).

