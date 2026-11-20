# FURY biweekly demos (2026-11-20)

Local demo branch `demo/biweekly-2026-11-20`. It combines the open fork work so
every demo runs from one checkout. It is for showing, not for a PR.

## Setup

```bash
git checkout demo/biweekly-2026-11-20
pip install -e ".[optional]"      # or PYTHONPATH=$PWD with an existing FURY env
python demos/run.py               # numbered menu
python demos/run.py 1             # one demo; close the window to finish
python demos/run.py physics       # a section, one window after another
python demos/run.py all           # everything in meeting order
```

`--record` on `physics_demos.py`, `layout_demos.py` and `widget_demos.py`
writes GIFs instead of opening a window (offscreen, works on the headless VM):

```bash
cd demos/out && python ../physics_demos.py --record && python ../layout_demos.py --record && python ../widget_demos.py --record
```

## What is in this branch

| Part | Came from | Fork PRs |
|---|---|---|
| UI layout engine (solver, containers, text layout, widget ports, tutorials, e2e tests) | `ui-layout/integration` | #15-#31 (merged into the fork stack) |
| Layout fixes: solver, Tree2D in layouts, docs renames, e2e via real events | `ui-layout/fix-solver`, `fix-tree`, `fix-docs-renames`, `fix-e2e-tests` | #44, #34, #32 (fix-solver has no PR yet) |
| UI themes: theme core, box style, Graphite, default Graphite, Switch/Segmented, built-in themes, theme files | `ui-theme/01` ... `07` (04 and 05 branch off 03, so they are merged separately; their Switch tests are pinned to the classic theme here) | #47-#53 |
| Physics: core, boxes, containers, actors, render loop, forces | `physics/01-core` ... `06-forces` | #35, #36, #40-#43 |
| GPU physics: `GPUWorld`, spheres stepped by WGSL compute shaders | `physics/11-gpu` | new PR |
| `actor.line` offset fix | `bf/line-offset` | #37 |
| Example files only: city drone shot, jungle sim, update-buffers tutorial, Lorenz WIP | `demo/city-drone-shot`, `fix/jungle-city-sims`, `tutorial/update-buffers`, `showcase/lorenz-ensemble` | #45, #38 (city-drone-shot and Lorenz have no PR) |
| Kenney CC0 packs for city and jungle (`docs/examples/kenny_assets`) | copied from the `fury-sims` worktree | - |

The city demo reads `kenny_assets` first, because the polyxios-data catalog does
not publish the Kenney models yet (`px.fetch("kenney-...")` fails with
polyxios 0.5.1).

## Not in this branch (show from their own branch)

These conflict with the theme rework or sit on a much older upstream, so a
combined merge would be an untested rewrite:

| Demo | Branch | How to show |
|---|---|---|
| UI Workbench: live hierarchy, inspector, create widgets (ImGui) | `ui-playground` | `git checkout ui-playground && python docs/examples/viz_ui_playground.py` |
| Tree2D takeover with scroll, clipping, collapse (upstream #1283 superseded) | `treeui-supersede` (fork #54) | `python docs/examples/viz_tree_ui.py`, `viz_tree_actor_modifier.py` |
| Hidden containers hide new children; scene membership sync | `ui-layout/fix-scene-membership` (#39) | GIFs: `fury-ui-layout/demos/gifs/fix_scene_membership_{before,after}.gif` |
| ButtonGroup joins the layout tree | `ui-layout/fix-button-group` (#33) | GIFs: `fury-ui-layout/demos/gifs/fix_button_group_{before,after}.gif` |

## Recordings already on the VM (backup if a live run fails)

- Physics: `~/Repositories/fury-physics/gifs/physics-0{1..6}-*.gif`
- Layout and widgets: `~/Repositories/fury-ui-layout/demos/gifs/`
- Themes: `~/Repositories/fury-ui-theme/recordings/` and `fury-ui-theme/variants.html`
- Jungle and city before/after: `~/Repositories/fury-recordings/jungle-city-sims/`, `fury-recordings/city-drone-path/`
- Fresh GIFs from this branch: `demos/out/` (gitignored)

## Talking points per section

**Physics (`fury.physics`, no new dependency).** NumPy rigid bodies: spheres and
boxes, soft-contact solver, static and kinematic bodies, containers you can
shake, tilt and spin, FURY actors used directly as bodies,
`ShowManager.add_physics` steps worlds with the render loop at a fixed step,
`record_physics` makes frame-rate-independent GIFs, force fields and
`explode`. Planned next: collision events, sleeping, 2D, mouse drag, GPU
spheres, gallery demos (PLAN in `~/Repositories/fury-physics/PLAN.md`).

**GPU physics (`fury.physics.GPUWorld`).** The same idea on the GPU, spheres
only: every sphere's state lives in GPU buffers, WGSL compute kernels step it
(position-based dynamics, hashed grid, Jacobi contacts with friction and
bounce), and the renderer draws from the same buffer, so nothing comes back to
the CPU. Containers (shake, rock, spin, tilt), force fields and explode work
as on the CPU world. `python demos/gpu_stress.py` runs 10k to 1M spheres and
checks they stay finite, inside the box and non-overlapping. On a laptop GPU
raise `SPHERES` in `viz_physics_gpu.py` to 1,000,000. Not ported: boxes,
collision events, the jungle and city agents (branchy per-agent logic).

**Jungle and city sims.** Jungle: Kenney elephant, lion and deer replace
placeholder primitives, UI no longer rebuilt every tick, picking, sliders,
needs, population cap, batching. City: drone path baked once and lifted clear
of buildings, textures no longer upside down, roads and buildings batched
into chunks, procedural traffic. Tests: `~/Repositories/fury-jungle-city-sims/tests`.

**UI layout engine.** Clay-style per-axis sizing (fixed, fit, grow, percent with
min/max), padding/gap/border box model, row/column containers with align and
wrap, text measured by the solver (no more frame-2 relayout, closes #1090 and
#1403), dirty tracking so layout solves only when something changes. Every
composite widget ported. Design: `~/Repositories/fury-ui-layout/DESIGN.md`.

**UI themes.** `Theme`, `set_theme`, `theme_scope`; widgets read defaults from the
theme and explicit arguments still win. Rounded boxes with border, per-corner
radius and shadow. Graphite (default) and Graphite Light with hover, pressed,
disabled and focus states. Built-in Hairline, Soft Pill, Frost, Ember, compact
density, and `load_theme` for theme files. Plan: `~/Repositories/fury-ui-theme/PLAN.md`.
