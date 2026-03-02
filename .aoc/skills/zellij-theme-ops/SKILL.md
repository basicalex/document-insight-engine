---
name: zellij-theme-ops
description: Create and manage global and project-scoped Zellij themes for AOC.
---

## When to use
- You need a new custom Zellij theme for all projects (global scope).
- You need a project-specific theme that should not collide with other repos.
- You need to apply or persist a theme change quickly during an active session.

## Commands
- `aoc-theme tui`
- `aoc-theme presets list`
- `aoc-theme presets install --all`
- `aoc-theme init --name <theme> --scope global`
- `aoc-theme init --name <theme> --scope project`
- `aoc-theme list --scope all`
- `aoc-theme apply --name <theme> --scope auto`
- `aoc-theme set-default --name <theme> --scope auto`

## Scope model
- Global themes live at `~/.config/zellij/themes/<name>.kdl`.
- Project themes live at `.aoc/themes/<name>.kdl` and are registered as runtime names in `~/.config/zellij/themes/aoc-<project>-<name>.kdl`.
- `--scope auto` prefers a matching project theme in the current repo, then falls back to global.

## Recommended workflow
1. Run `aoc-theme list --scope all` to check existing names.
2. Create the theme in the intended scope with `aoc-theme init`.
3. Edit palette values in the generated file.
4. If inside Zellij, apply live with `aoc-theme apply --name <theme> --scope auto`.
5. Persist for future launches with `aoc-theme set-default --name <theme> --scope auto`.

## Guardrails
- Theme names must match `^[a-z0-9]+(-[a-z0-9]+)*$`.
- Prefer project scope for repo-specific branding and experiments.
- For project themes, keep the generated runtime theme block name unchanged.
- Do not overwrite existing theme files unless replacement was explicitly requested.
- Live apply requires an active Zellij pane (`zellij options --theme ...`).
