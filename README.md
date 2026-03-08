Lizard Kernels 
---

#### Setup

- on ml-institute machines, run `SETUP` once per clone.
    - it setups git user with dedicated token
- on each login, `conda activate jku-ai-practical-project` should be used to activate conda environment.

#### Run Jupyter

```bash
gcloud compute ssh whisper-l4-worker \
  --zone asia-southeast1-a \
  --project jku-practical-project \
  -- -L 8888:localhost:8888
```

```bash
jupyter lab --no-browser --port=8888 --ip=127.0.0.1
```

