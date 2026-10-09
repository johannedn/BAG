

- GPU: Tesla P100-PCIE-12GB (12 GB), driver 580.178.04, max CUDA 13.0 (login banner says 11.4, outdated).
- TGN on wiki: ~3 min/epoch on the P100, ~3.1 GB GPU memory, GPU utilization ~17% (CPU-bound neighbor sampling).


How to continue running when you lock your pc (tmux):
- Start a named session: `tmux new -s organic`
- Run the job inside it: `bash scripts/run_organic.sh`
- Detach (job keeps running): `Ctrl+b`, release, then `d`
- List sessions: `tmux ls`
- Reattach: `tmux attach -t organic` (or `tmux a` for the most recent)
- Scroll output: `Ctrl+b` then `[`, arrows/PgUp/PgDn, `q` to exit
- Close a session (kills the job): `exit` from inside, or `tmux kill-session -t organic` from outside
- Kill all sessions: `tmux kill-server`


Bugs found in the upstream code:
1. `detectors/edge/CTDG.py` (`CTDGDetector.__init__`): after the if/elif chain builds the right backbone (TGN, DyGFormer, FreeDyG, ...), two leftover lines (`gnn = globals()[model_config['model']]`, `backbone = gnn(**model_config)`) overwrote it, passing the model config dict as kwargs. Fixed in commit 814a17e by removing those lines.
2. `prepare_data.py`: mooc has organic anomaly labels like wiki and reddit, but it was missing from the labeled list (`['wiki', 'reddit', 'yelp']`). Its labels were zeroed and replaced with injected synthetic anomalies. Fixed by adding `'mooc'` to the list.
3. `prepare_data.py`: for the labeled datasets (wiki/reddit/mooc), `save_dtdg_data` and `save_ctdg_data` were commented out, so only the static data was written and the discrete/continuous models had no input. Fixed by uncommenting them.



