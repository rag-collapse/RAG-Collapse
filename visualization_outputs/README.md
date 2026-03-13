## Visualization outputs

This directory stores plots generated from:
- `evaluation_outputs/<MODEL_SUBDIR>/local_*_eval.json`
- `entity_extraction_output/<MODEL_SUBDIR>/local_*_entity_results.json`

### Directory layout

For each model we visualize, figures are written under:

- `visualization_outputs/<MODEL_SUBDIR>/`

where `MODEL_SUBDIR` matches the experiment/eval layout, e.g.:
- `Qwen/Qwen2.5-1.5B-Instruct`
- `Qwen/Qwen2.5-7B-Instruct`
- `Qwen/Qwen2.5-14B-Instruct`

Within each model subdirectory you will see PNG files (only Qwen2.5-14B-Instruct are for 400 questions dataset, old visualizations are for 50 questions dataset):

- `collapse_by_simulation.png`
- `unique_words_per_round.png`
- `unique_entities_per_round.png`
- `entity_similarity_per_round.png`
- `same_answer_percentage_per_round.png` (only meaningful if the judge is enabled)
- `avg_pairwise_tes_per_round.png`
- `avg_pairwise_rouge1_per_round.png`
- `avg_pairwise_rouge2_per_round.png`
- `avg_pairwise_rougeL_per_round.png`

Each plot contains three lines/bars, one per document-setting variant:
- **Replace All**
- **Replace One**
- **Search**

### How to regenerate plots

1. Open `visualization.ipynb`.
2. At the top of the notebook, set:
   - `MODEL_SUBDIR = "<model subdir>"`  
     e.g. `Qwen/Qwen2.5-7B-Instruct`.
3. Ensure the corresponding JSONs exist:
   - `evaluation_outputs/<MODEL_SUBDIR>/local_search_eval.json`
   - `evaluation_outputs/<MODEL_SUBDIR>/local_replace_one_eval.json`
   - `evaluation_outputs/<MODEL_SUBDIR>/local_replace_all_eval.json`
   - `entity_extraction_output/<MODEL_SUBDIR>/local_search_entity_results.json`
   - `entity_extraction_output/<MODEL_SUBDIR>/local_replace_one_entity_results.json`
   - `entity_extraction_output/<MODEL_SUBDIR>/local_replace_all_entity_results.json`
4. Run all cells in the notebook.

New (or updated) PNGs will be written to:

- `visualization_outputs/<MODEL_SUBDIR>/`

### Models and images

Below we list all expected figures for the Qwen/Qwen2.5-14B-Instruct model (for the 400 questions dataset) so results can be browsed directly. 

#### Qwen/Qwen2.5-14B-Instruct for 400 questions dataset

- Collapse by simulation  
  ![14B – collapse_by_simulation](Qwen/Qwen2.5-14B-Instruct/collapse_by_simulation.png)
- Unique words per round  
  ![14B – unique_words_per_round](Qwen/Qwen2.5-14B-Instruct/unique_words_per_round.png)
- Unique entities per round  
  ![14B – unique_entities_per_round](Qwen/Qwen2.5-14B-Instruct/unique_entities_per_round.png)
- Entity similarity per round  
  ![14B – entity_similarity_per_round](Qwen/Qwen2.5-14B-Instruct/entity_similarity_per_round.png)
- Same-answer percentage per round  
  ![14B – same_answer_percentage_per_round](Qwen/Qwen2.5-14B-Instruct/same_answer_percentage_per_round.png)
- Avg TES per round  
  ![14B – avg_pairwise_tes_per_round](Qwen/Qwen2.5-14B-Instruct/avg_pairwise_tes_per_round.png)
- Avg ROUGE-1 per round  
  ![14B – avg_pairwise_rouge1_per_round](Qwen/Qwen2.5-14B-Instruct/avg_pairwise_rouge1_per_round.png)
- Avg ROUGE-2 per round  
  ![14B – avg_pairwise_rouge2_per_round](Qwen/Qwen2.5-14B-Instruct/avg_pairwise_rouge2_per_round.png)
- Avg ROUGE-L per round  
  ![14B – avg_pairwise_rougeL_per_round](Qwen/Qwen2.5-14B-Instruct/avg_pairwise_rougeL_per_round.png)

