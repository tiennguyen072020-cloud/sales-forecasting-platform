---
description: Analyze verified source/build changes and refresh the AI agent workspace handoff
mode: agent
---

Refresh the project agent workspace after a source change or completed build.

Start with the project handoff and request history:

- `AI agent Workspace/project-context.md`
- `AI agent Workspace/prompts/used-prompts.md`
- `.github/copilot-instructions.md`

Run the verified context scanner from the project root:

```powershell
.\.venv\Scripts\python.exe .\scripts\update_agent_context.py
```

Then inspect the generated results and apply only verified findings. Focus on:

- `pyproject.toml`: Python requirement and dependency changes
- `README.md`: documented commands, requirements, and workflow changes
- `ml/`: package structure and forecasting implementation
- `api/`: API entry point, routes, schemas, dependencies, and services
- `build/lib/`: whether generated Python output matches the current source
- Tests and validation results: record actual results only

Update these files when their content is outdated:

1. Add a generated-source summary to `AI agent Workspace/project-context.md` without duplicating the generated marker section.
2. Preserve new project requests in `AI agent Workspace/prompts/used-prompts.md` when the request is explicitly intended for future sessions.
3. Mirror workflow, rule, or instruction changes from `.github/` into the corresponding files under `AI agent Workspace/`.
4. Keep `.github/` as the runtime source of truth.

Do not overwrite raw data, generated model artifacts, notebooks, build output, or test outputs unless the user explicitly requested it. Do not invent missing dependencies, build status, test results, package structure, or validation evidence.

Finish with a concise summary containing:

- files inspected
- scanner result
- handoff files updated
- dependency/build/source changes detected
- validation performed and any blockers
