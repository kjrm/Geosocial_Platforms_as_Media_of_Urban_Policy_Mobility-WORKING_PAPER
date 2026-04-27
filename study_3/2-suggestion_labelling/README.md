# suggestion labelling

This repository contains the code (`stage1_llama.py`, `stage2_gpt.py`, `stage3_combine.py`, `stage4_combine.py`) and input data (`PA_brewpub_reviews.csv`) necessary to complete the second part of study 3.

Each of the files that requires LLM API calls includes setup instructions. The LLM-ensemled labelling process proceeds as follows:

1. `stage1_llama.py` builds the sample, uses llama3 to label each record as either YES, MAYBE, or NO; and creates output data file for the next labelling stage.
2. `stage2_gpt.py` takes the output file from the previous stage; uses gpt-4o-mini to independently label the same records as either YES, MAYBE, or NO; and creates an updated data file for the next stage.
3. `stage3_combine.py` takes the updated data file; labels which records are labelled the same by the two models (agreement between the two models) and which one aren't (disagreement between the two models); generates some statistics; and creates an updated data file for the next stage.
4. `stage4_claude.py` takes the updated data file; finds which records for which there was disagreement; and asks claude-sonnet-4-5 to indepently label these as either YES, MAYBE, or NO; and creates an update data file.
5. The final output of these stages was then manually completed by taking those records for which there was completed disagreement across the three models (some permuation including YES, MAYBE, NO), and labelling these records the medial value of MAYBE.

The repository `3-sugggestion_map` includes the final output of this process (`labelled_reviews.csv`), as it is used as input for that part of the study.

