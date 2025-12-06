This directory is intentionally left empty in the public release.

In our internal experiments, it stores:
- interaction-extracted graph-tensor files for each scenario (e.g., *.pkl);
- the corresponding scene_index_mapping.csv that maps raw scenario IDs / timestamps to tensor indices.

These files are generated from real-world autonomous-driving scenarios and contain privacy-sensitive information, so they are not included in this repository.

To run the code, please generate your own interaction graph tensors and scene_index_mapping.csv in the same format (see configs/ and dataset.py for reference), or create small toy examples for testing.
