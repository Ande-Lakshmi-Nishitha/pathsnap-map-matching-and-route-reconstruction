# PathSnap — Distributed Map-Matching & Route Reconstruction

UE24CS352A Machine Learning hackathon, PES University.

Team: Ananya Raghavendra (PES1UG24AM037), Ananya S (PES1UG24AM038), Anarghyaa Kashyap (PES1UG24AM039), Ande Lakshmi Nishitha (PES1UG24AM040)

PathSnap snaps noisy GPS trajectories onto a road network:

1. **Candidate generation** – for every GPS observation, the 8 nearest directed road arcs within 50 m (UTM-projected, KD-tree over densified arcs; nearest-8 fallback if nothing is within 50 m).
2. **Candidate classifier** – a binary model scores each (observation, candidate arc) pair: "is this the arc the vehicle was on?". 20 engineered features (distance, heading agreement, window fit, connectivity, …).
3. **HMM + Viterbi** – hidden states = candidate arcs per observation; emissions = normalised classifier probabilities; transitions = exp(−|network distance − straight-line distance| / β) using bounded Dijkstra on the directed road graph. Viterbi gives a coherent arc sequence, which is stitched into a continuous route.
4. **FastAPI backend (Laptop 1)** + **web UI (Laptop 2)**.

## Repository layout

```
configs/            config.yaml (all pipeline settings), hmm_tuned.json (decoder params chosen on train OOF)
src/pathsnap/       io, labels, candidates, features, models, metrics, graph, hmm, pipeline
scripts/            eda.py, build_dataset.py, train_models.py, training_curves.py, bayes_opt.py, hmm_eval.py,
                    make_plots.py, make_maps.py, fit_diagnostics.py, error_analysis.py
api/main.py         FastAPI server
ui/                 index.html + app.js (Leaflet evaluator UI)
models/             saved models (*.joblib), final_model.joblib = tuned LightGBM
results/            CSV/JSON experiment outputs (eda/, models/, hpo/, hmm/, model_comparison.csv)
figures/            all report plots
maps/               interactive Folium HTML maps
data/               raw data (not committed): data/train, data/student_test, data/evaluator_test
```

## Setup

```bash
python -m venv .venv
source .venv/bin/activate            # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

Put the raw data in `data/` so that `data/train/00000001/00000001.track` etc. exist (same for `data/student_test/`).

## Reproduce everything (in order)

```bash
python scripts/build_dataset.py       # labels + candidates + features  -> results/data/*.pkl
python scripts/eda.py                 # inventory, quality checks, EDA figures, overview map
python scripts/train_models.py        # 8 models: 5-fold GroupKFold CV + student-test evaluation
python scripts/training_curves.py     # epoch-wise curves (LightGBM, MLP only)
python scripts/bayes_opt.py --trials 30   # Optuna TPE on LightGBM -> models/final_model.joblib
python scripts/hmm_eval.py            # decoder tuning on train OOF, classifier vs classifier+HMM on student test
python scripts/make_plots.py          # comparison plots + results/model_comparison.csv
python scripts/fit_diagnostics.py     # train vs CV vs test scores (over/underfitting evidence)
python scripts/error_analysis.py      # error taxonomy, classifier-only vs HMM
python scripts/make_maps.py           # Folium maps -> maps/*.html
```

Seeds are fixed in `configs/config.yaml` (`seed: 42`). The evaluator-testing trips are never read by any training or tuning script.

## Deployment (two laptops, same Wi-Fi / hotspot)

**Laptop 1 – model + API**

```bash
# put the 10 evaluator trips in data/evaluator_test/<trip_id>/<trip_id>.{track,nodes,arcs,route}
# (or point PATHSNAP_EVAL_DIR at the folder)
uvicorn api.main:app --host 0.0.0.0 --port 8000
```

Find Laptop 1's IP (`ipconfig` on Windows, `ifconfig`/`ip a` on Mac/Linux). Allow port 8000 through the firewall. Every request is logged to the terminal and to `logs/api.log` — show this terminal to the evaluator.

Endpoints: `GET /health`, `GET /trips`, `POST /predict {"trip_id": "..."}`. Interactive docs at `http://<ip>:8000/docs`.

**Laptop 2 – UI**

```bash
cd ui
python -m http.server 8501
```

Open `http://localhost:8501`, type `http://<laptop1-ip>:8000` in the API box, click **Connect & load trips**, then click any trip. The UI only receives the predicted route, GPS points and metrics; reference routes stay on Laptop 1 and are only read *after* prediction, to compute accuracy.

**Quick local test (one machine):** `mkdir -p data/evaluator_test && cp -r data/student_test/* data/evaluator_test/`, then run both commands above and use `http://127.0.0.1:8000`.

## Results (student test, 10 trips)

| | Top-1 accuracy | Route F1 |
| --- | --- | --- |
| Tuned LightGBM, classifier only | 0.8956 | 0.852 |
| Tuned LightGBM + HMM/Viterbi (deployed) | 0.9445 | 0.973 |

Full tables: `results/model_comparison.csv`, `results/hpo/tuned_vs_untuned.json`, `results/hmm/student_test_summary.csv`.

## Metrics

- **Observation Top-1 accuracy** (primary): share of GPS observations whose chosen arc equals the reference arc. The reference arc per observation comes from a monotone dynamic-programming alignment of the GPS sequence to the ordered reference route. Observations > 50 m from their aligned arc are flagged as unlabelled (e.g. parked off-network) and excluded from training and scoring.
- **Candidate macro F1 / precision / recall / log loss**: binary candidate classification at threshold 0.5.
- **Candidate miss rate**: share of labelled observations whose reference arc is not among the 8 candidates.
- **Route continuity**: share of consecutive (distinct) predicted arcs that are directly connected in the road graph.
- **Route precision / recall / F1**: length-weighted overlap between predicted route arcs and reference route arcs.


## Model training & optimisation (Anarghyaa Kashyap, PES1UG24AM039)
- scripts/train_models.py: 5-fold GroupKFold CV + student-test evaluation for all models
- scripts/bayes_opt.py: Optuna TPE search (30 trials) on LightGBM -> models/final_model.joblib
- api/main.py: FastAPI backend (GET /health, GET /trips, POST /predict)
- Random Forest and Extra Trees weights are not committed (over 25 MB); regenerate with:
  python scripts/train_models.py --models random_forest extra_trees
