Sliding Window Attention Kernels 
---

### Setup

- on ml-institute machines, run `SETUP` once per clone.
    - it setups git user with dedicated token
- on each login, `conda activate jku-ai-practical-project` should be used to activate conda environment.

### Run Jupyter

```bash
gcloud compute ssh whisper-l4-worker \
  --zone asia-southeast1-a \
  --project jku-practical-project \
  -- -L 8888:localhost:8888
```

##### Jupyter in background

```bash
nohup jupyter lab --no-browser --port=8888 --ip=127.0.0.1 > jupyter.log 2>&1 &
```

```bash
(jku-ai-practical-project) mahdikhashan@whisper-l4-worker:~$ jupyter server list
Currently running servers:
http://127.0.0.1:8888/?token=25fdaae1232d9bafd3879e8e7ce3eb12c6b95e18462c46c9 :: /home/mahdikhashan
```
