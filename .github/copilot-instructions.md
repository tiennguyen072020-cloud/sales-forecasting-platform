# Sales Forecasting Project Instructions

- This is a Python 3.12+ sales forecasting and business intelligence platform. Follow existing patterns under `ml/`.
- Inspect related implementations, callers, tests, and data contracts before editing; make focused changes and preserve existing API behavior.
- For forecasting work, preserve chronological ordering, avoid target leakage, and use time-aware evaluation rather than random splits.
- Preserve the supported 30/60/90-day forecast horizons unless explicitly asked to change them. Keep forecast dates future-only and retain the established output schema.
- Do not overwrite raw data or generated artifacts unless requested. Do not introduce dependencies without updating the manifest.
- Add focused tests for behavior changes, run the smallest relevant test set, and report actual validation results and blockers.
- Keep assumptions and failures explicit; never substitute silent defaults for missing required data or artifacts.
- Never include credentials or private customer data in source control.
- At the start of a new chat in this workspace, consult [the project context handoff](../AI%20agent%20Workspace/project-context.md) and [project request history](../AI%20agent%20Workspace/prompts/used-prompts.md). Use them to continue from known facts instead of repeating broad repository discovery; inspect only files directly relevant to the current task, and refresh the handoff when verified context changes.
- After each completed task, update [the project context handoff](../AI%20agent%20Workspace/project-context.md) when verified project facts or structure changed. Record the user's request and outcome in [the project request history](../AI%20agent%20Workspace/prompts/used-prompts.md) when the request is intended to be preserved. Do not invent unavailable facts or validation results.
- When rules, instructions, or workflows change under `.github/`, mirror the corresponding content into `AI agent Workspace/`. Treat `.github/` as the runtime source of truth and the workspace copy as the human-readable handoff.
- Before finishing documentation or workflow updates, check that `.github/copilot-instructions.md` and the corresponding workspace rules/skills are consistent.
- When asked to preserve new project requests, append them to the request history without inventing unavailable history.

See [the detailed project rules](../AI%20agent%20Workspace/rules/project-rules.md) and [the workflow skill](./skills/sales-forecasting-workflow/SKILL.md).
