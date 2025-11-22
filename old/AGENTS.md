# Agents in this repository

This integrated repository contains:

- A MetaTrader 5 CLI (the code and binaries in this root directory) used to
  interact with the platform and MT5 projects.
- A code-oriented, agentic LLM configuration (`agent/` + `prompts/`) called
  **codex-mt5-agent**, which uses:
  - the CLI in this repository as its primary execution tool,
  - offline documentation in `docs/`,
  - and web search when needed.

This document describes how the agents are expected to behave, which tools
they use, and how all the pieces fit together in this **integrated** project.

---

## 1. Overall architecture

At a high level, the repository is organized into three main layers:

1. **CLI layer – core mtcli implementation (root)**
   - The actual MetaTrader 5 CLI lives directly in this repository root:
     - `src/`, `dist/`, `bin/`, `cli/`
     - `tools/`, `scripts/`
     - `mt_agent_pack/`, `mt_agent_recipes/` (if present)
   - Exposes commands to:
     - manage projects and configuration,
     - run tests/backtests,
     - attach/handle indicators and EAs,
     - inspect logs and environment status,
     - execute helper scripts and recipes.

2. **Agent layer – `agent/` + `prompts/`**
   - Configuration and prompts for a code-centric LLM (“codex-mt5-agent”).
   - The agent:
     - receives high-level user goals in natural language,
     - plans and executes steps autonomously,
     - uses the CLI and documentation to produce concrete results
       (MQL5 code, configs, tests, diagnostics).

3. **Documentation / knowledge layer – `docs/` + `legacy_docs/` + CLI docs**
   - Offline documentation for:
     - the MQL5 language and standard library,
     - MetaTrader 5 platform usage,
     - the mtcli CLI (intention, commands, recipes, examples).
   - Older documentation and notes are preserved under `legacy_docs/`.
   - These documents are exposed to the agent via:
     - `prompts/tools/docs_offline.yaml`
     - `agent/knowledge_index.yaml`.

Language conventions:

- **Agent–user communication:** default in Brazilian Portuguese (`pt-BR`).
- **Code, APIs, commands, identifiers:** English.
- **Documentation files:** mostly English (some Portuguese allowed if useful),
  but technical names remain in English.

---

## 2. The codex-mt5-agent

### 2.1. Location and configuration

The agent’s configuration lives primarily in:

- **`agent/`** – top-level agent config:
  - `agent_config.yaml`
    - Entry point for the orchestrator/runtime.
    - Points to:
      - `prompts/system.yaml` (system prompt),
      - `agent/tools_index.yaml` (available tools),
      - `agent/knowledge_index.yaml` (knowledge sources).
  - `tools_index.yaml`
    - Declares tools such as:
      - `mtcli`        – this repository’s CLI (e.g. `bin/mtcli`).
      - `docs_offline` – local docs and PDFs.
      - `web_search`   – online search when local docs are not enough.
      - (optional) `mt_agent_recipes` – shell/CLI recipes if configured.
  - `knowledge_index.yaml`
    - Declares knowledge sources, for example:
      - `docs/mql5_reference/mql5.pdf` – MQL5 language reference (English).
      - `docs/mt5_userguide/`         – MT5 platform guides (if present).
      - `docs/` and/or `legacy_docs/` – CLI documentation and older notes.

- **`prompts/`** – behavior, tools, domain knowledge and workflows:
  - `system.yaml`
    - System prompt entry point.
    - Includes behavior, tools, domain knowledge and workflows.
  - `behavior_contract.yaml`
    - High-level behavioral rules:
      - Proactive and autonomous (“auto-solve” mode).
      - Takes the lead after the user states a goal.
      - Explains a short plan, then executes with minimal narration.
      - Uses multiple strategies (does not repeat the same failing approach).
      - Asks the user only when strictly necessary.
  - `domain/`
    - Domain-specific knowledge, e.g.:
      - `mql5_bar_indexing.yaml`
      - `mql5_time_series_arrays.yaml`
      - `mql5_indicators_init.yaml`
      - `mql5_consistency_checks.yaml`
      - `mtcli_maintenance.yaml`
    - These encode conventions about bar indexing, time series arrays,
      indicator initialization, buffer consistency, and mtcli evolution.
  - `tools/`
    - Tool definitions for the agent:
      - `mtcli.yaml`
        - How to call and reason about the CLI in this repository
          (paths, capabilities, help commands).
      - `docs_offline.yaml`
        - Where offline docs live (MQL5 PDFs, MT5 guides, mtcli docs).
      - `web_search.yaml`
        - Guidelines for web search (sources, when to prefer offline docs).
      - `mt_agent_recipes.yaml` (optional)
        - Mapping to high-level scripts/recipes (e.g. under `mt_agent_pack/`,
          `mt_agent_recipes/` or `scripts/`).
  - `workflows/`
    - Structured flows for typical tasks:
      - `problem_solving.yaml`
        - Generic MQL5/MT5 problem-solving using the CLI and docs.
      - `bug_diagnosis.yaml`
        - Workflow for reproducing, isolating and fixing bugs in MQL5 code.
      - `indicator_creation.yaml`
        - Workflow for designing, implementing and validating custom indicators.

### 2.2. Behavior summary

The codex-mt5-agent is designed to behave as follows:

- **Takes the lead once the user states a goal.**
  - Briefly restates the goal in its own words.
  - Proposes a short, high-level plan (typically 3–5 steps).
  - Then proceeds to execute autonomously using the CLI and docs.

- **Avoids narrating every small step.**
  - Does *not* describe each internal command or file read.
  - Reports at meaningful checkpoints:
    - after compilation attempts (success/failure),
    - when changing strategy,
    - when a major subgoal is complete,
    - when the overall task is complete.

- **Uses multiple strategies.**
  - If one approach fails, the agent:
    - adjusts indexing logic,
    - revisits array/buffer configuration and initialization,
    - simplifies the problem to a minimal reproducible example,
    - consults offline docs and, if necessary, web search.
  - It avoids blindly repeating the same fix or pattern.

- **Asks the user only when necessary.**
  - Missing critical input (symbol, timeframe, risk preferences, etc.).
  - Decisions that affect trading risk or environment changes.
  - Ambiguities that cannot be resolved by reading code and docs.

- **Communicates in Brazilian Portuguese (`pt-BR`) by default.**
  - Explanations, summaries and questions in Portuguese.
  - Code, function names, parameters, CLI commands and file names in English.

---

## 3. The mtcli CLI (integrated)

### 3.1. Purpose

In this integrated repository, the CLI is the primary tool used by the agent (and humans) to:

- Compile MQL5 indicators, Expert Advisors (EAs) and scripts.
- Manage MetaTrader 5 project structure and configuration.
- Run tests and backtests through the MT5 environment.
- Orchestrate recipes and helper scripts, for example:
  - `mt_agent_pack/` templates,
  - `scripts/` or `mt_agent_recipes/` flows (if present).

Depending on the implementation and version, it may also:

- Attach EAs/indicators to charts,
- Control or query listeners (using `listener.ini`),
- Manage DLL-related tasks,
- Integrate with external editors/IDEs.

### 3.2. Location and code structure

Within this integrated project, the CLI resides directly in the root:

- `bin/`
  - Executable entrypoints (e.g. `bin/mtcli`, `bin/mtcli.cmd`).
- `cli/`, `src/`, `dist/`
  - CLI implementation (TypeScript/JavaScript or Python).
- `tools/`
  - JSON/YAML configuration (e.g. `mtcli_projects.json`, other configs).
- `mt_agent_pack/`
  - Templates and helper scripts for common MT5 tasks.
- `scripts/`, `mt_agent_recipes/` (if present)
  - Shell/PowerShell recipes and quick tasks.
- Config and environment files:
  - `listener.ini`, `script_run.ini`, `mt_boot.ini`, `verify_datapath.ini`.

### 3.3. CLI documentation

Authoritative CLI docs should live under `docs/` (and optionally
`legacy_docs/` for old material).

Examples of expected documentation:

- `docs/cli/overview.md`
  - High-level description: what mtcli is and when to use it.
- `docs/cli/commands.md` or similar
  - Reference of commands, flags and subcommands.
- `docs/agent/agent_guide.md`
  - How LLM-based agents should use mtcli (best practices, pitfalls).
- `docs/cli/recipes.md`
  - Practical examples and recipes for common flows.

The agent is encouraged to:

- Use `mtcli --help` and `mtcli <command> --help` as the primary source
  of truth for syntax.
- Compare documentation with real behavior and suggest updates when
  they diverge.
- Propose new commands or flags when they simplify recurring workflows.

### 3.4. Agent bootstrap (`mtcli init` / `mtcli up`)

- The repository defines a single MetaTrader 5 installation dedicated to automation, referenced by:
  - `DEFAULT_TERMINAL` (`terminal64.exe` path),
  - `DEFAULT_DATA_DIR` (Terminal data folder),
  - `DEFAULT_LIBS` (`MQL5\\Libraries` under that data dir),
  - `DEFAULT_PROFILE`, default symbol/period/subwindow.
- The command `mtcli init` (alias `mtcli up`) is the “power button”:
  - saves/updates the default project (e.g. `agent-terminal`) with those paths,
  - performs a controlled restart of that exact terminal (does **not** close other MT5 installs),
  - launches the terminal with `listener.ini`, `/datapath`, `/profile`,
  - prints a concise summary plus the latest listener log tail.
- `mtcli init` must run before any other chart/indicator/expert/tester command. It guarantees the EA listener is attached so the agent can immediately call higher-level commands without manually poking the terminal.
- Não existe mais subcomando dedicado `mtcli listener ...` no binário; a orquestração do listener é feita implicitamente por `mtcli init`/`mtcli up` e pelos comandos de alto nível.

### 3.5. Listener protocol v2 (cmd/resp)

- Every interaction with the MT5 listener now follows a structured file protocol:
  1. The CLI writes `MQL5/Files/cmd_<ID>.txt`, where the first line is `ID|TYPE|PARAM1|PARAM2|...`.
  2. The EA performs the action and writes `resp_<ID>.txt` containing:
     - line 1: `OK` or `ERROR`,
     - line 2: human-friendly message,
     - optional extra lines with structured data (paths, chart info, etc.).
  3. The CLI waits (polling every ~200 ms) until the matching `resp_<ID>.txt` appears or a timeout occurs, then surfaces the status/message directly to the user/agent.
- Typical high-level commands routed through this protocol:
  - `ATTACH_IND_FULL`, `DETACH_IND_FULL` – attach/remove indicators without restarting the terminal.
  - `ATTACH_EA_FULL`, `DETACH_EA_FULL` – attach/remove Expert Advisors using templates already copied to the terminal.
  - `SCREENSHOT`, `SCREENSHOT_SWEEP`, `WINDOW_FIND`, `CLOSE_ALL`, etc.
- Because every command has a unique ID + response, the agent no longer scrapes logs to guess results; it receives deterministic `OK/ERROR + message` feedback for each CLI call.

---

## 4. Offline documentation

The agent prefers offline documentation whenever possible.

### 4.1. MQL5 reference

- `docs/mql5_reference/mql5.pdf`
  - MQL5 language reference (English).
  - Typically registered as `mql5_reference_en` in:
    - `prompts/tools/docs_offline.yaml`
    - `agent/knowledge_index.yaml`
  - Covers:
    - language syntax,
    - standard functions,
    - built-in indicators and constants,
    - event model (OnInit, OnDeinit, OnCalculate, etc.).

### 4.2. MT5 platform guides

- `docs/mt5_userguide/`
  - Folder for MetaTrader 5 platform documentation:
    - terminal usage,
    - tester configuration,
    - general platform behavior.

### 4.3. CLI documentation and legacy docs

- `docs/`
  - New or curated CLI docs (overview, commands, agent guide, recipes).
- `legacy_docs/`
  - Older docs and notes inherited from prior versions of this project.
  - Kept only for historical reference; new agents should rely primarily
    on `docs/` and up-to-date files.

The typical preference order for information is:

1. **Offline docs** (MQL5 reference, MT5 guides, curated CLI docs),
2. **Source code** (`src/`, `dist/`, `tools/`) when reasoning about implementation,
3. **Web search** (as a complement when offline information is insufficient
   or potentially outdated).

---

## 5. How to use this integrated project

A typical setup for running the codex-mt5-agent on top of this repository:

1. The orchestrator / runtime:
   - Loads `agent/agent_config.yaml`.
   - Uses `prompts/system.yaml` as the main system/root prompt.
   - Exposes tools according to `agent/tools_index.yaml`:
     - `mtcli`        → resolves to the CLI in this repository
                       (e.g. `bin/mtcli` or a Python entrypoint).
     - `docs_offline` → indexing/search over `docs/` (and possibly `legacy_docs/`).
     - `web_search`   → external web lookup tool.

2. The user expresses high-level goals in Portuguese, for example:
   - “Crie um indicador com duas médias móveis e marque o cruzamento.”
   - “Esse EA não está respeitando o stop loss, descubra o motivo e corrija.”
   - “Monte um teste rápido em EURUSD H1 para validar essa estratégia.”

3. The agent:
   - Runs `mtcli init` (alias `mtcli up`) to bring the dedicated MT5 terminal online before touching charts, indicators, experts or tests.
   - Understands the goal.
   - Proposes a short plan.
   - Uses the CLI, code, and documentation to execute that plan.
   - Adapts strategies if the first attempt fails.
   - Reports progress at key checkpoints and delivers final artifacts
     (MQL5 code, configuration changes, logs, etc.).
   - Interacts with MT5 through high-level commands such as `mtcli indicator add`, `mtcli expert add`, `mtcli chart screenshot`, relying on the cmd/resp protocol instead of invoking `mtcli listener ...`.

---

## 6. Extending this setup

To extend or customize this integrated setup:

- **Add new workflows:**
  - Create new YAMLs under `prompts/workflows/`,
    and reference them from `prompts/system.yaml`.

- **Add or refine domain knowledge:**
  - Extend `prompts/domain/*.yaml` for new MT5, MQL5 or CLI concepts.

- **Add tools:**
  - Create new definitions under `prompts/tools/` and list them in
    `agent/tools_index.yaml`.

- **Adjust documentation:**
  - Keep this `AGENTS.md` and the docs under `docs/` aligned with the actual
    behavior of the CLI and the agent.
  - Move obsolete or conflicting material to `legacy_docs/`.
  - Add new PDFs, HTML or Markdown docs under `docs/` and register them in
    `prompts/tools/docs_offline.yaml` and `agent/knowledge_index.yaml`.

Whenever you significantly change how the agent or the CLI behaves, update
this `AGENTS.md` so it remains the high-level, human-readable description
of the integrated system.
