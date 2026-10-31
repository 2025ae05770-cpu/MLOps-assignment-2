# Required MLflow Evidence

Capture actual MLflow screenshots after the real experiment and reviewed promotion.
Do not substitute synthetic test runs or images for evidence. Use PNG files with
these exact names so the submission packager can validate the checklist.

| Filename | What must be visible |
| --- | --- |
| `experiment.png` | Experiment `eduassist_llmops_assignment` and the three runs from your current batch |
| `prompt_registry.png` | Prompts tab, `eduassist_academic_qa`, all three actual versions, templates and commit messages |
| `metrics.png` | Run comparison showing avg_latency_seconds, avg_input_tokens, avg_output_tokens, avg_cost_per_request, valid_json; also include schema/quality metrics |
| `prompt_diff.png` | Prompt Registry comparison/diff showing changes between assignment v1-v2 or v2-v3 |
| `production_alias.png` | Selected prompt version and its `production` alias |

Start with `python -m eduassist.ui` from `assignment2`; open the printed local URL.
Use the batch ID and registry-version mapping in `outputs/manifest.json` to avoid
mixing earlier runs with the final experiment. On a new database assignment v1-v3
correspond to registry versions 1-3; later runs create later immutable versions.

Prefer enough browser zoom to keep names and metrics legible. Capture only relevant
UI content; exclude API keys, billing/account screens, unrelated personal data,
and terminal environments. The ZIP builder checks PNG headers, not authenticity:
you are responsible for ensuring the screenshots show your actual experiment.