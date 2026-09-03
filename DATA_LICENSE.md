# Data license — CC-BY-4.0

The **data** in this repository is released under the Creative Commons
Attribution 4.0 International License (CC-BY-4.0):
https://creativecommons.org/licenses/by/4.0/

You are free to share and adapt the data for any purpose, provided you give
appropriate credit (cite the paper / this repository) and indicate any changes.

## What this covers

- `datasets/umass_data.entity.chatgpt.{50,400}.jsonl` — the entity-annotated input questions
  and reference documents used to seed the collapse loop.
- Source-level **citation labels** and evaluation artifacts under `evaluation_outputs/`,
  including the direct-elicitation ("explicit") citation outputs in
  `evaluation_outputs/Qwen/citations/`.
- Experiment / entity-extraction outputs committed under `experiment_outputs/`,
  `entity_extraction_output/`, and `camera_ready_outputs/`.

(Source **code** — everything else — is under the MIT License; see `LICENSE`.)

## Upstream attribution — ACTION REQUIRED before public release

The entity dataset is derived from an internal `umass_data` source. **Confirm that the
upstream terms permit redistribution under CC-BY-4.0** and add the correct upstream
citation/attribution here before the repo is made public. If redistribution is not
permitted, ship a script that regenerates the dataset from the original source instead of
the JSONL files.
