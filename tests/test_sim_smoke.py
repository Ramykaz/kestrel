import math

import numpy as np
import pybullet as p

from kestrel.eval.suite import EVAL_LOOKS
from kestrel.sim.camera import HEIGHT, WIDTH, OnboardCamera, onboard_intrinsics
from kestrel.sim.course import generate_course, load_preset
from kestrel.sim.gates import FRONT_MARKER_LINK_INDICES, FRONT_MARKER_RGBA
from kestrel.sim.scene import apply_style
from kestrel.sim.smoke import HOVER_TOLERANCE_M, run_hover
from kestrel.sim.world import World


def test_intrinsics_match_90_degree_horizontal_fov():
    for renderer in (p.ER_TINY_RENDERER, p.ER_BULLET_HARDWARE_OPENGL):
        intrinsics = onboard_intrinsics(renderer)
        assert math.isclose(intrinsics.fx, 160.0)
        assert math.isclose(intrinsics.fy, intrinsics.fx)
    assert onboard_intrinsics(p.ER_BULLET_HARDWARE_OPENGL).cx == 159.5
    assert onboard_intrinsics(p.ER_TINY_RENDERER).cx == 160.0


def test_projection_matrix_agrees_with_intrinsics():
    world = World()
    try:
        camera = OnboardCamera(world.client, world.renderer)
        projection = np.array(camera.projection_matrix).reshape(4, 4, order="F")
        # OpenGL clip x scale is 2 fx / width.
        assert math.isclose(projection[0, 0], 2 * camera.intrinsics.fx / WIDTH, rel_tol=1e-5)
        assert math.isclose(projection[1, 1], 2 * camera.intrinsics.fy / HEIGHT, rel_tol=1e-5)
    finally:
        world.close()


def test_camera_is_pitched_up_15_degrees_in_level_flight():
    world = World()
    try:
        camera = OnboardCamera(world.client, world.renderer)
        view = np.array(camera.view_matrix(np.zeros(3), np.array([0, 0, 0, 1.0])))
        view = view.reshape(4, 4, order="F")
        # The camera looks down its own -z axis; row 2 of the view rotation is camera +z in world.
        look_world = -view[2, :3]
        assert math.isclose(math.degrees(math.asin(look_world[2])), 15.0, abs_tol=1e-4)
        assert look_world[0] > 0.9
    finally:
        world.close()


def test_render_returns_rgb_frame():
    world = World()
    try:
        camera = OnboardCamera(world.client, world.renderer)
        frame = camera.render(
            np.array([0.0, 0.0, 1.0]), np.array(p.getQuaternionFromEuler([0, 0, 0]))
        )
        assert frame.rgb.shape == (HEIGHT, WIDTH, 3)
        assert frame.rgb.dtype == np.uint8
        assert frame.rgb.std() > 0
    finally:
        world.close()


def test_drone_hovers_at_target():
    result = run_hover(sim_seconds=4.0)
    assert result.final_error_m < HOVER_TOLERANCE_M
    assert result.frames_rendered == 120


def test_gate_front_markers_are_visual_only_and_untextured():
    world = World()
    try:
        gate = world.load_course(generate_course(1000, load_preset("straight")))[0]
        apply_style(world.client, world.ground_id, [gate], EVAL_LOOKS["day"].style)
        for link in FRONT_MARKER_LINK_INDICES:
            assert p.getCollisionShapeData(gate, link, physicsClientId=world.client) == ()
        visuals = p.getVisualShapeData(gate, physicsClientId=world.client)
        marker_colors = {v[1]: v[7] for v in visuals if v[1] in FRONT_MARKER_LINK_INDICES}
        assert set(marker_colors) == set(FRONT_MARKER_LINK_INDICES)
        assert all(np.allclose(c, FRONT_MARKER_RGBA, atol=1e-3) for c in marker_colors.values())
    finally:
        world.close()
