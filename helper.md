# Helper: running the simulation tests on a PC and saving videos

This page explains how to start a sorting simulation with any object seed, where the videos go, and what was measured when it was run on the development PC.

## Where the videos are
Everything is under `runs/pc_sim_videos/`. That folder is git-ignored, so it is not part of the repository. Videos are H.264 (640 x 480), so they play in any browser or player.

| Set | Folder | Result |
|---|---|---|
| Normal sorting, seeds 1000-1004 (4 objects each) | `sorting_seeds_1000-1004/` | 17 of 20 objects in the right bin |
| Unseen object in the scene, seeds 5000-5002 | `unseen_object_seeds_5000-5002/` | e-stop on the unknown object in 3 of 3 trials |

Details of the normal set:
- Seeds 1001, 1002 and 1003 sorted 4/4.
- Seed 1000 ended 3/4: one cylinder ended outside its tray.
- Seed 1004 stopped early with 2/4: the vision scored a known cylinder as unknown (u = 1.00) and the arm e-stopped.

Details of the unseen-object set: the e-stop fires before the arm moves, so each video is about 5.5 s (the scene plus a caption such as "E-STOP: unknown object (u=1.00)").

## Commands to run a test with different seeds
```bash
scripts/run_pc_sim.sh                                      # 3 trials, seeds 1000-1002
scripts/run_pc_sim.sh --seed 2000 --trials 5               # seeds 2000-2004
scripts/run_pc_sim.sh --seed 5000 --trials 3 --unknown 1   # unseen object: the arm should e-stop
scripts/run_pc_sim.sh --seed 42 --objects 6 --name my_test # 6 objects, custom folder name
scripts/run_pc_sim.sh --help                               # all options
```

Options: `--seed N` (first seed), `--trials N`, `--objects N` (1 to 6), `--unknown N` (0 to 2), `--model PATH` (`.onnx`, `.pt` or `.engine`), `--name LABEL` (output folder name).

- Trial k uses seed `--seed + k`, so the same seed always gives the same layout on the same machine and backend.
- Output: `runs/pc_sim_videos/<name>/trial_XX_seedNNNN.mp4` and `runs/pc_sim_videos/<name>.json` (results).
- Without the wrapper: `.venv/bin/python scripts/sort_demo.py --trials 5 --seed 2000 --video-dir runs/my_videos`.
- Add `--gui` to `sort_demo.py` to watch live in PyBullet's own window. `--gui` does not record video.

## Code changes made for this
- `scripts/sort_demo.py`: videos used to start only when the arm moved, so runs that e-stop before any motion (an unseen object) produced no file. Videos now start with 1.5 s of the starting layout and end with 2 s of the final layout, or a labelled hold when an e-stop happens.
- `scripts/run_pc_sim.sh` (new): the wrapper above. It runs the demo, converts the videos to H.264 with `ffmpeg` and lists the output.

## Results differ slightly between backends
The reports state 74 / 80 = 92.5% for the 20-trial run (seeds 1000-1019). That run used the PyTorch backend. The one-line PC reproduce command and the wrapper use the committed ONNX model instead.

The same 20-trial run on ONNX gave **75 / 80 = 93.8%**, with trials 0, 4, 13 and 19 not perfect (16 of 20 trials sorted 4/4). The earlier PyTorch run had different trials failing. Individual trials depend on the backend (tiny numeric differences change how objects roll), while the aggregate rate is about the same. For example, the Jetson's TensorRT run sorted seed 1000 4/4, while this PC sorts it 3/4.

When quoting a result, state the backend and the seeds.
