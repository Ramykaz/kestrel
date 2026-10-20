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

Laptop numbers only (Intel i7-1195G7, 8 logical CPUs, WSL Ubuntu 24.04). Kaggle numbers are in the next entry.

Scene: ground plane plus 12 box distractors, 320x240 onboard camera, random poses. No gates yet, so this will be measured again in phase 1.

| Renderer | Workers | Frames/s |
| :- | :- | :- |
| TinyRenderer | 1 | 153.6 |
| TinyRenderer | 8 | 439.0 |
| EGL | 1 | 78.0 |
| EGL | 8 | 122.3 |

EGL loads in WSL but runs slower than TinyRenderer, which points to a software OpenGL fallback rather than the GPU. The EGL number that matters is the one from a Kaggle T4.

## 2026-10-20: Kaggle render throughput

Measured with `notebooks/kaggle_round.ipynb` on a Kaggle notebook with 4 CPUs, Python 3.11.17, same scene as the local table. Clone to smoke pass took 26 s, and the smoke hover error was 0.005 m.

| Renderer | Workers | Frames/s |
| :- | :- | :- |
| TinyRenderer | 1 | 98.3 |
| TinyRenderer | 4 | 213.7 |
| EGL | 1 | 50.3 |
| EGL | 4 | 95.9 |

EGL on Kaggle reported `GL_RENDERER=llvmpipe` and one EGL device, so it drew on the CPU, as in WSL, and the T4 did no rendering. TinyRenderer stays the default, as the spec already says. At 213.7 frames/s, the 8k frame first dataset is under a minute of rendering before label ray tests.

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

## 2026-10-20: The reference autopilot fell out of the sky

What broke: the first ground-truth autopilot, built to make development flights for the website, crashed into the ground on almost every course above 1.5 m/s.

How I found it: working through the forces. It chased a point 0.5 s ahead of the drone, so at 2.5 m/s DSLPIDControl saw a 1.25 m position error and asked for about 0.5 N of horizontal force. The CF2X weighs 0.26 N and tops out at 0.6 N of thrust, so it tilted about 60 degrees and lost altitude.

What fixed it: the drone now tracks a reference point whose acceleration is limited to 2.5 m/s², and which slows into each approach point so it can line up with the gate's normal. At 1.5 m/s it finishes all 8 gates on every tech course tried. At higher speeds and with gate position error it still crashes into frames, which is the spread of outcomes the replay needs for testing.

## 2026-10-20: The spec's own alert red fails its own contrast rule

What broke: the WCAG check added to CI failed on its first run. The day theme's `--alert` (`#c6362b`) measures 4.46:1 on `--paper`, under the 4.5:1 that SPEC section 13.2 requires for all text.

What fixed it: `--alert` is `#c3352a` in the day theme, the nearest shade that passes (4.58:1). Every other pair in both themes passed as specified.

## 2026-10-20: A frame rate problem that was not mine

What broke: in the automated Chrome window, the flight replay ran at about 1.5 frames per second and playback crawled at a fifth of real time.

How I found it: the static Notes page measured the same 1.5 frames per second, and the top-down view, which draws no WebGL at all, measured 2. The browser was throttling animation frames for that window, not the page doing too much work. Lighthouse in a separate headless Chrome measured the real Flight view at 91 for mobile performance.

What fixed it: nothing in the renderer needed fixing. The player did get one real fix out of it: it capped each frame's time step at 0.1 s, which made playback run slow on any device that cannot keep up. It now advances by real elapsed time and only ignores gaps longer than 0.5 s, such as returning to a hidden tab.

## 2026-10-20: EGL plugin killed pytest in CI

The first GitHub Actions run stopped after 26 tests with exit code 1 and no pytest summary. Test 27 is the EGL projection test. On a runner with no EGL display, PyBullet's EGL plugin prints "failed to EGL with glad." and calls exit() inside C, so the fallback to TinyRenderer and the test's skip never ran. I reproduced it in WSL by pointing `__EGL_VENDOR_LIBRARY_FILENAMES` at a missing file. `World` now loads the plugin only after a child process has loaded it and printed a success line. The exit status alone was not enough, because one local run of the crashing load returned 0. CI also installs Mesa's software EGL, so the EGL test runs there instead of skipping.

## 2026-10-20: vLLM on a Kaggle T4

vLLM 0.10.1.1 serves `SmolVLM-256M-Instruct` in fp16 on a Kaggle T4, after three fixes found in `notebooks/kaggle_vllm_check.ipynb`.

- The pip install found no matching version. The Kaggle kernel runs Python 3.13 and vLLM 0.10.1.1 requires 3.12 or older, so vLLM now gets its own Python 3.12 environment made with `uv`.
- The server exited with code 1. Its log showed `GPT2Tokenizer has no attribute all_special_tokens_extended`: vLLM does not cap transformers, the install pulled transformers 5.20.2, and transformers 5 removed that attribute. Pinning `transformers<5` installed 4.57.6.
- The first chat request got a body that was not JSON. It used an image URL on Wikimedia that vLLM fetches itself; I did not capture that response, so the cause is not confirmed. A 320x240 PNG sent inline as a data URL worked.

On the T4, vLLM fell back to its V0 engine ("Compute Capability < 8.0 is not supported by the V1 Engine") and to the XFormers attention backend, since FlashAttention-2 does not support Turing. The model loaded in 3.8 s using 0.49 GiB, and the server was healthy 46 s after launch, including 19 s of CUDA graph capture. One request with the inline image (877 prompt tokens, 9 completion tokens) took 2.10 s and answered "In the image there is a square." vLLM stays optional; llama.cpp remains the serving path the spec plans around.

## 2026-10-20: TinyRenderer remembers the last light

While trying lighting options, a render with only `shadow=1` came out as dark as the previous render, which had set a dim blue light. TinyRenderer keeps the light state from the last `getCameraImage` call for any argument that is left out. `Lighting.render_kwargs` now always passes every light argument, so a frame's look never depends on the frame rendered before it.

## 2026-10-20: A seam every 2 m on the voronoi ground

The ground mesh repeats its texture every 2 m, so every texture must tile. A test that compares the wrap-around step with the largest step inside the tile failed only for the voronoi pattern, with a step of 97.6 against at most 6.4 inside. The tiled seed array is laid out seed by seed over 9 offsets, and I took the seed as `index % 12` instead of `index // 9`, so copies of one cell in neighboring tiles got different shades. The first version of that test compared against the mean step and also failed on checker, tiles, stripes and concrete, whose hard edges sit on the tile border by design.
