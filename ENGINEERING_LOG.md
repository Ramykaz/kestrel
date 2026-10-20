# Engineering log

## 2026-10-20: PyBullet will not install on Windows

What broke: `pybullet` 3.2.7 publishes no Windows wheel, only a source archive, and this laptop has no MSVC build tools. `gym-pybullet-drones` is not on PyPI in a usable version; its current GitHub head requires Python 3.12 and declares torch, stable-baselines3 and matplotlib as hard dependencies.

How I found it: `pip download pybullet --only-binary=:all: --platform win_amd64` returned no match, and reading the upstream `pyproject.toml` showed the Python pin and the dependency list.

What fixed it: development runs in WSL Ubuntu 24.04, which gets the manylinux wheel, the same one Kaggle uses. `gym-pybullet-drones` is pinned to commit `f83fbdd` and its dependency metadata is overridden in `pyproject.toml` to the four packages that `DSLPIDControl` and the CF2X assets actually import. That keeps Python 3.11 working and keeps torch out of the simulator install.

## 2026-10-20: Stepping physics ourselves instead of using CtrlAviary

What broke: `CtrlAviary.step` runs all 5 physics substeps of a 48 Hz control step inside one call. Gate crossing and frame collisions must be checked between individual 240 Hz physics steps, and a contact that starts and ends within one control step would be missed.

What fixed it: `kestrel/sim/world.py` owns the PyBullet client, loads the same CF2X URDF with the same flags, and applies the same per-motor thrust and yaw torque formula as upstream `Physics.PYB` before every `stepSimulation`. `DSLPIDControl` is used unchanged.

## 2026-10-20: Onboard camera rendered from inside the drone mesh

What broke: the first smoke test rendered at 25 frames/s, while the render benchmark got about 130 frames/s on the same machine. The saved frame also showed pieces of the drone's own airframe floating in view.

How I found it: timing renders at different camera positions. From inside the drone at 1 m: 41 ms per frame. From 5 m away at the same height: 9.5 ms. TinyRenderer spends most of the frame on the CF2X mesh triangles that cross the near plane.

What fixed it: `World` sets every drone link's visual alpha to 0. No PyBullet camera ever needs to show the drone (the website draws it in three.js). Render time from inside the drone dropped to 6.7 ms, and the 5 s smoke test went from 6.19 s to 0.95 s wall time.

## 2026-10-20: Local render throughput

Laptop numbers only (Intel i7-1195G7, 8 logical CPUs, WSL Ubuntu 24.04). Kaggle numbers still to be measured with `notebooks/kaggle_round.ipynb`.

Scene: ground plane plus 12 box distractors, 320x240 onboard camera, random poses. No gates yet, so this will be measured again in phase 1.

| Renderer | Workers | Frames/s |
| :- | :- | :- |
| TinyRenderer | 1 | 153.6 |
| TinyRenderer | 8 | 439.0 |
| EGL | 1 | 78.0 |
| EGL | 8 | 122.3 |

EGL loads in WSL but runs slower than TinyRenderer, which points to a software OpenGL fallback rather than the GPU. The EGL number that matters is the one from a Kaggle T4.

## 2026-10-20: TinyRenderer draws half a pixel off

What broke: the first version of the label projection test failed. Projected gate corners and rendered corners disagreed by up to 1.09 px, more than the 1 px the spec allows.

How I found it: averaging the signed corner offsets over 200 corners gave (+0.48, -0.48) px with a spread of only 0.13 px, so a constant offset, not noise. The same test with the EGL renderer gave (-0.008, -0.004) px. TinyRenderer samples pixel i at viewport coordinate i instead of i + 0.5, then flips rows, which shifts its whole image by (+0.5, -0.5) px against an ideal pinhole camera.

What fixed it: `onboard_intrinsics` takes the renderer. For TinyRenderer the principal point is (160.0, 119.0); for EGL it is (159.5, 119.5). Datasets will store the intrinsics of the renderer that drew them, so PnP uses the right ones.

## 2026-10-20: Measuring rendered corners by line fitting was the wrong test

What broke: after the intrinsics fix, one corner in 200 still measured 1.02 px off. The projection was fine. The top edge of that gate rose only 0.9 px over 18 px, so it rendered as a staircase with a single step, and a least-squares line through one step overestimates the slope and misplaces the extrapolated corner. A max-margin line fit made it worse (1.23 px).

What fixed it: the test no longer fits lines. The quad through the projected keypoints predicts, for every pixel center, whether it is in the gate opening; the farthest pixel that disagrees with that prediction bounds the projection error directly. Worst case over 50 frames: 0.136 px for TinyRenderer, 0.131 px for EGL. Injecting a known offset reads back correctly (0.5 px measures 0.5 to 0.6 px), which also showed that the spec's 1 px limit would have let the half pixel TinyRenderer bug through. The test threshold is 0.25 px.

## 2026-10-20: Course spacing broke the 7 m limit on climbing courses

What broke: `tech` courses produced consecutive gates 7.03 m apart. The generator sampled the horizontal distance and then added the height change on top.

How I found it: the course constraint test checks 3D spacing on 40 seeds per preset.

What fixed it: the generator samples the 3D spacing first and derives the horizontal distance from it and the climb.

## 2026-10-20: EGL in WSL is software rendering

The pybullet EGL plugin in WSL reports `GL_RENDERER=llvmpipe`, Mesa's CPU rasterizer. That explains why EGL measured slower than TinyRenderer on this laptop. EGL numbers only mean something on a machine with a real GPU behind EGL, such as a Kaggle T4.
