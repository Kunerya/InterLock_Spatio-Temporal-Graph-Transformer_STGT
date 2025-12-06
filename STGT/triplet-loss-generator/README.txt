Triplet-Loss Generator for Scenario Similarity
==============================================

This folder contains a set of scripts used to construct human-annotated
triplet data (anchor–positive–negative) for training the STGT scenario
embedding model.

The scripts implement a multi-stage pipeline:
1) generate pairwise similarity tasks,
2) collect human judgments via Streamlit-based web UIs,
3) derive high-confidence positive candidates and negative candidates,
4) build triplets, and
5) obtain final fine-grained similarity scores for training.

Prerequisites
-------------

- Python 3.10+
- Streamlit
- Pandas, NumPy, etc. (see the main project requirements)

The CSV files used in this pipeline (pair_tasks.csv, pair_answers.csv,
pos_candidate_answers.csv, etc.) are NOT included in this public
repository because they are derived from privacy-sensitive driving
scenarios. You can adapt the scripts to your own data by following the
same formats.

Script Overview
---------------

1_gen_pair_tasks.py
    Generate the first-round questionnaire `pair_tasks.csv`, which
    contains scenario pairs (anchor, candidate) to be judged as
    "similar" or "not similar".

2_label_app.py
    Streamlit app for interactive labeling of pairwise similarity
    tasks. It reads a CSV file (e.g., `pair_tasks.csv` or a derived
    questionnaire) and writes the human answers to
    `pair_answers.csv` (first round) or other answer files.

3_parse_pair_results.py
    Parse the first-round answers and build the second-round
    questionnaire for positive-candidate selection. The resulting CSV
    is used to construct 3-choice questions (one anchor with several
    candidate scenarios) for "pick the most similar" labeling.

4_scores.py
    From the second-round answers (e.g., `pos_candidate_answers.csv`),
    compute reference similarity scores for candidate pairs. This
    produces `pair_scores.csv`, which is used as weak supervision for
    similarity regression.

5_build_neg_from_answers.py
    Use the first-round "negative" answers together with the
    second-round results to construct a list of negative candidates
    for each anchor.

6_neg_picker_app.py
    Streamlit app for manually checking and filtering negative
    candidates to ensure they are genuinely dissimilar to the anchor.
    The output is a cleaned negative list suitable for triplet
    construction.

7_combine_and_reduce_pairs.py
    Combine high-confidence positive pairs (from the scoring step) and
    manually confirmed negative pairs into a single table. This table
    is used to define, for each anchor, 1–2 positive scenarios and at
    least one negative scenario, while reducing redundant pairs.

8_survey_app.py
    Streamlit app for the final fine-grained scoring survey. It
    presents candidate scenarios for each anchor and collects
    numeric similarity scores. The resulting CSV (about ~1200
    human-judged pairs in our experiments) is used to build the final
    triplet set and supervised similarity labels for STGT training.

Usage Notes
-----------

- The exact CSV schemas (column names, ID formats, etc.) follow the
  internal data pipeline of the InterLock project. Please refer to
  the code and adapt them to your own scenario IDs and metadata.

- A typical usage pattern is:

    1. Run `1_gen_pair_tasks.py` to generate `pair_tasks.csv`.
    2. Run `streamlit run 2_label_app.py` to label the first-round
       pairwise tasks and obtain `pair_answers.csv`.
    3. Run `3_parse_pair_results.py` to build the second-round
       questionnaire and relabel it with `2_label_app.py`, yielding
       `pos_candidate_answers.csv`.
    4. Run `4_scores.py` to compute `pair_scores.csv`.
    5. Run `5_build_neg_from_answers.py` and then
       `streamlit run 6_neg_picker_app.py` to construct and clean
       negative candidates.
    6. Run `7_combine_and_reduce_pairs.py` to obtain the final list of
       anchor–positive–negative candidates.
    7. Run `streamlit run 8_survey_app.py` to collect fine-grained
       similarity scores and export the final triplet/similarity
       dataset for training STGT.

- The resulting triplets and scores are consumed by the main training
  script (e.g., `train_triplet.py`) to optimize the STGT embedding
  space under a triplet-loss and regression-based objective.
