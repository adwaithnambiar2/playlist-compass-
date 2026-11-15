# Playlist Compass

Build an explainable discovery queue from several favorite tracks. A weighted audio profile captures the seeds; a greedy relevance/diversity objective reduces near-duplicate recommendations.

## Run
```bash
python -m pip install -r requirements.txt
python compass.py --seeds track-000 track-004 --diversity 0.35 -k 10
python -m unittest -v
```
Python 3.10+ recommended. No API keys, account, or GPU required.

## How it works
1. Standardize each feature across the supplied catalog.
2. Apply nonnegative feature weights and average the seed vectors.
3. Score candidates by `1 / (1 + distance_to_profile)`.
4. Greedily select tracks using `(1-diversity)*relevance - diversity*maximum_similarity_to_selected`.
5. Explain each choice using the two smallest weighted squared feature differences.

At diversity zero, ranking follows profile similarity. Higher diversity trades similarity for variety; it does not guarantee better listening quality. Retrieval scans the catalog; greedy selection costs roughly O(k*n*k*d) with the simple implementation.

## Bring your own catalog
Use CSV columns `id,title,artist,energy,danceability,acousticness,valence,tempo`, with unique IDs and finite numeric features. Run `python compass.py --catalog your_tracks.csv --seeds YOUR_ID`. Use consistent units. Optional `--weights 2 1 1 1 0.5` follows the feature order above. Only upload data you have permission to redistribute.

## Data and validation
The bundled 120 tracks are synthetic, fictional examples generated for this project. They are not Spotify data and cannot demonstrate real listener satisfaction. Tests check seed exclusion, unique outputs, unit invariance from scaling, relevance ordering, and invalid inputs.

For real evaluation, collect held-out listener preferences and compare precision@k, catalog coverage, artist diversity, and listening feedback against a relevance-only baseline. No real-world recommendation accuracy is claimed.

## Provenance
Fresh implementation inspired by general nearest-neighbor recommendation concepts. No course notebook, assignment text, starter code, or submitted solution is included. Created with AI coding assistance; review and understand it before presenting your individual contribution. See LICENSE for code permissions; synthetic demo data is covered by the same license.
