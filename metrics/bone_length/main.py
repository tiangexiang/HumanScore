"""Variable-scale skeleton fit for metric (II) bone length (paper implementation).

The same MeTRAbs detection and biomechanical skeleton fit as ``baseline/main.py``, except
that the segment scales may change over time: the fitted trajectory also outputs the
body-scale parameters at every frame, and a smoothness penalty (``1e3``) keeps them from
jumping.  A real person's bone lengths then stay constant, while a body whose limbs grow or
shrink shows varying lengths, which ``metrics/bone_length/bone_length.py`` scores.  The
fitting code lives in this folder's copy of ``monocular_demos`` (the variable-scale
version of ``baseline/monocular_demos``).

Where it sits in HumanScore
    HumanScore runs this fit through ``hmb/fit_engine.py`` (same objective and settings),
    only when ``--paper-scores`` is requested.  This CLI is the standalone equivalent.

Run from the repository root::

    python -m metrics.bone_length.main --video clip.mp4 --output_dir results/bone_fit --skip_rendering --skip_opensim

Steps (each can be skipped): MeTRAbs keypoints (``<stem>_keypoints.npz``), variable-scale
fit (``<stem>_bone_length.npz``), skeleton video (``<stem>_mjx.mp4``), OpenSim motion
(``<stem>_opensim_skeleton.mot``), all in ``--output_dir``.
"""
import os
from pathlib import Path
from typing import List
import argparse
import json
import time

HUMANOID_DIR = Path(__file__).resolve().parent / "monocular_demos/biomechanics_mjx/data/humanoid"


def configure_runtime(use_tensorflow=True, use_jax=True):
    """Set up TensorFlow GPU memory growth and JAX (x64, persistent compilation cache).

    Called after argument parsing so ``--help`` does not import either library.
    """
    os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")
    if use_tensorflow:
        from .monocular_demos.utils import tensorflow_memory_limit
        tensorflow_memory_limit()
    if use_jax:
        import jax
        jax.config.update("jax_compilation_cache_dir", os.environ.get("HMB_JAX_CACHE_DIR", os.environ.get("JAX_COMPILATION_CACHE_DIR", str(Path(__file__).resolve().parents[2] / ".jax_cache"))))
        jax.config.update("jax_persistent_cache_min_entry_size_bytes", -1)
        jax.config.update("jax_persistent_cache_min_compile_time_secs", 0)
        jax.config.update("jax_enable_x64", True)


def save_metrabs_data(accumulated, video_name, output_dir, camera_calibration=None, calibration_source=None):
    """Write the first detected person of every frame to ``<stem>_keypoints.npz``.

    Frames without a detection get zero keypoints and zero confidence.
    """
    import numpy as np
    if accumulated is None:
        raise ValueError(f"No decodable video frames: {video_name}")
    fname = Path(video_name).stem
    boxes, pose3d, pose2d, confs = [], [], [], []
    for i, (box, p3d, p2d) in enumerate(
        zip(accumulated["boxes"], accumulated["poses3d"], accumulated["poses2d"])
    ):
        # first detection of the frame (no tracking across frames)
        if len(box) == 0:
            boxes.append(np.zeros((5)))
            pose3d.append(np.zeros((87, 3)))
            pose2d.append(np.zeros((87, 2)))
            confs.append(np.zeros((87)))
            print("no boxes")
            continue
        boxes.append(box[0].numpy())
        pose3d.append(p3d[0].numpy())
        pose2d.append(p2d[0].numpy())
        confs.append(np.ones((87)))

    output_file = os.path.join(output_dir, f"{fname}_keypoints.npz")
    metadata = {}
    if camera_calibration is not None:
        metadata["camera_intrinsics_px"] = np.asarray(camera_calibration["mtx"]) * 1000.0
        metadata["camera_distortion"] = np.asarray(camera_calibration["dist"])
        metadata["camera_calibration_source"] = np.array(calibration_source or "unknown")
    with open(output_file, "wb") as f:
        np.savez(f, keypoints3d=pose3d, keypoints2d=pose2d, boxes=boxes, confs=confs, **metadata)
    print(f"Saved keypoints to {output_file}")

def render_mjx(selected_file, output_dir):
    """Render the fitted skeleton of ``selected_file`` to ``<stem>_mjx.mp4`` in ``output_dir``."""
    import numpy as np
    from .monocular_demos.biomechanics_mjx.visualize import render_trajectory
    if not selected_file or not os.path.exists(selected_file):
        print(f"Error: Model file not found: {selected_file}")
        return False
    
    fname = os.path.basename(selected_file).replace('_bone_length.npz', '').replace('_fitted_model.npz', '')
    
    print(f"Rendering video for {fname}...")
    
    if os.path.exists(selected_file):
        with open(selected_file, "rb") as f:
            data = np.load(f, allow_pickle=True)
            print(f"Loaded biomechanics data: {selected_file}")
            qpos = data['qpos']
    
    video_filename = os.path.join(output_dir, f"{fname}_mjx.mp4")
    print("Rendering video (this may take a while)...")
    render_trajectory(
        qpos,
        filename = video_filename,
        xml_path=str(HUMANOID_DIR / "humanoid_torque_vis.xml"),
        height=800,
        width=800,
        azimuth=0,
    )
    print(f"Rendered visualization saved to: {video_filename}")
    return True



def get_framerate(video_path):
    """Frame rate of a video file (``ValueError`` if it cannot be read)."""

    import cv2
    import math
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise ValueError(f"Could not open video file: {video_path}")
    fps = cap.get(cv2.CAP_PROP_FPS)
    cap.release()
    if not math.isfinite(fps) or fps <= 0:
        raise ValueError(f"Invalid frame rate {fps!r} for {video_path}")
    return fps


def load_metrabs_data(video_name, output_dir):
    """``(boxes, keypoints2d, keypoints3d, confs)`` from ``<stem>_keypoints.npz``, or four ``None``."""
    import numpy as np
    fname = Path(video_name).stem
    keypoints_file = os.path.join(output_dir, f"{fname}_keypoints.npz")
    try:
        with open(keypoints_file, "rb") as f:
            data = np.load(f, allow_pickle=True)
            boxes = data["boxes"]
            keypoints2d = data["keypoints2d"]
            keypoints3d = data["keypoints3d"]
            confs = data["confs"]

        return boxes, keypoints2d, keypoints3d, confs
    except FileNotFoundError:
        print(f"No saved data found for this video at {keypoints_file}")
        return None, None, None, None


def process_videos_with_metrabs(
    video_files: List[str],
    output_dir: str,
    model_path: str | None = None,
    camera_calibration: str | None = None,
) -> str:
    """Detect MeTRAbs ``bml_movi_87`` keypoints for every video and save ``<stem>_keypoints.npz``."""
    import tensorflow as tf
    import numpy as np
    from baseline.main import load_camera_calibration, record_runtime_profile
    from .monocular_demos.utils import load_metrabs, video_reader

    if not video_files:
        return "No videos uploaded."

    print("Loading Metrabs detection model...")
    load_started = time.perf_counter()
    model = load_metrabs(model_path, detection_only=True)
    record_runtime_profile(output_dir, detector_model_load_seconds=time.perf_counter() - load_started,
                           detector_load_details=load_metrabs.last_timing)
    print("Model loaded successfully.")
    skeleton = "bml_movi_87"

    video_count = 0
    for video_idx, video_path in enumerate(video_files):
        if video_path is not None:
            inference_started = time.perf_counter()
            print(f"\nProcessing video {video_idx+1}/{len(video_files)}: {video_path}")
            vid, n_frames = video_reader(video_path)
            calibration = load_camera_calibration(camera_calibration, video_path=video_path)
            fx, fy, cx, cy = calibration["mtx"][0] * 1000.0
            intrinsic_matrix = np.array([[[fx, 0, cx], [0, fy, cy], [0, 0, 1]]], dtype=np.float32)
            accumulated = None
            for frame_idx, frame_batch in enumerate(vid):
                if frame_idx % 10 == 0:
                    print(f"Processing frame batch {frame_idx * 8}/{n_frames}")

                pred = model.detect_poses_batched(
                    frame_batch, skeleton=skeleton, intrinsic_matrix=intrinsic_matrix,
                    distortion_coeffs=np.asarray(calibration["dist"], dtype=np.float32),
                )

                if accumulated is None:
                    accumulated = pred

                else:
                    # concatenate the ragged tensor along the batch for each element in the dictionary
                    for key in accumulated.keys():
                        accumulated[key] = tf.concat(
                            [accumulated[key], pred[key]], axis=0
                        )
            save_metrabs_data(accumulated, video_path, output_dir, calibration,
                              camera_calibration or "assumed_fov55")
            record_runtime_profile(output_dir, detector_video_seconds=time.perf_counter() - inference_started,
                                   detector_frames=n_frames, video=str(video_path))
            video_count += 1

    return f"Successfully processed {video_count} videos with Metrabs."


def process_videos_with_biomechanics(
    video_files: List[str], output_dir: str, max_iters: int = 10000,
    unconstrained_joints: bool = False,
    camera_calibration: str | None = None,
    export_backend: str | None = None,
) -> str:
    """Fit the variable-scale skeleton to the saved keypoints and write ``<stem>_bone_length.npz``.

    All videos are fitted together (one trajectory each) and must share camera intrinsics.
    The archive holds the trajectory (``qpos``, ``qvel``), body and site positions
    (``joints``, ``sites``), the per-frame scale parameters (``scale``,
    ``body_scale_factors``), the keypoint confidences and the fit settings.
    """
    import equinox as eqx
    import jax
    import jax.numpy as jnp
    import numpy as np
    from .monocular_demos.biomechanics_mjx.monocular_trajectory import fit_model, get_model, export_trajectory
    from .monocular_demos.dataset import MonocularDataset
    from .monocular_demos.utils import joint_names
    from baseline.main import load_camera_calibration, record_runtime_profile

    if max_iters < 1:
        raise ValueError("max_iters must be positive")
    export_backend = export_backend or os.environ.get("HMB_FIT_EXPORT_BACKEND", "native")
    if export_backend not in ("native", "mjx"):
        raise ValueError("export_backend must be native or mjx")
    setup_started = time.perf_counter()
    def step_callback(step, model, dataset, metrics_dict, **kwargs):
        if step % 500 == 0:
            print(f"Fitting model: Step {step}/{max_iters}")

    if not video_files:
        return "No videos uploaded."

    timestamps_list = []
    keypoints2d_list = []
    keypoints3d_list = []
    confs_list = []
    fitted_videos = []
    calibrations = []
    keypoint_calibration_sources = []
    for i, video_path in enumerate(video_files):
        if video_path is not None:
            boxes, keypoints2d, keypoints3d, confs = load_metrabs_data(video_path, output_dir)
            if boxes is None:
                raise FileNotFoundError(f"Video {video_path}: no saved MeTRAbs keypoints found")
            if len(keypoints2d) == 0 or not np.any(confs > 0):
                raise ValueError(f"Video {video_path}: no valid person detections found")

            calibration = load_camera_calibration(camera_calibration, video_path=video_path)
            if calibrations and not np.allclose(calibrations[0]["mtx"], calibration["mtx"]):
                raise ValueError("Batch fitting needs shared camera intrinsics; process different frame sizes separately")
            calibrations.append(calibration)
            keypoint_file = Path(output_dir) / f"{Path(video_path).stem}_keypoints.npz"
            with np.load(keypoint_file, allow_pickle=False) as keypoint_data:
                source = (str(keypoint_data["camera_calibration_source"].item())
                          if "camera_calibration_source" in keypoint_data else "unknown_cached")
                if "camera_intrinsics_px" in keypoint_data and not np.allclose(
                    keypoint_data["camera_intrinsics_px"], calibration["mtx"] * 1000.0
                ):
                    raise ValueError("Keypoint camera intrinsics differ from fitting calibration; regenerate keypoints or use matching calibration")
                stored_timestamps = (np.asarray(keypoint_data["timestamps"], dtype=float)
                                     if "timestamps" in keypoint_data else None)
            keypoint_calibration_sources.append(source)

            # Detection may have sampled frames (input/frames.json); use its exact timestamps.
            if stored_timestamps is not None and stored_timestamps.shape == (len(keypoints2d),):
                timestamps = stored_timestamps - stored_timestamps[0]
            else:
                fps = get_framerate(video_path)
                timestamps = np.arange(0, len(keypoints2d)) / fps
            timestamps_list.append(timestamps)
            keypoints2d_list.append(keypoints2d[jnp.newaxis])  # Add camera dimension
            keypoints3d_list.append(keypoints3d[jnp.newaxis])  # Add camera dimension
            confs_list.append(confs[jnp.newaxis])
            fitted_videos.append(video_path)

    dataset = MonocularDataset(
        timestamps=timestamps_list,
        keypoints_2d=keypoints2d_list,
        keypoints_3d=keypoints3d_list,
        keypoint_confidence=confs_list,
        camera_params=calibrations[0],
        phone_attitude=None,
    )
    jax.block_until_ready(dataset)
    record_runtime_profile(output_dir, fit_data_setup_seconds=time.perf_counter() - setup_started)

    build_started = time.perf_counter()
    model = get_model(
        dataset, xml_path=str(HUMANOID_DIR / "humanoid_torque.xml"), joint_names=joint_names,
        enforce_joint_limits=not unconstrained_joints,
    )  # might need to change the site names
    jax.block_until_ready(model)
    record_runtime_profile(output_dir, fit_model_build_seconds=time.perf_counter() - build_started)
    fit_started = time.perf_counter()
    model, metrics = fit_model(
        model,
        dataset,
        lr_init_value=1e-3,
        max_iters=max_iters,
        step_callback=step_callback,
        callback_interval=500,
        scale_smoothness_regularization=1e3,  # add scale smoothness regularization
    )
    jax.block_until_ready((model, metrics))
    record_runtime_profile(output_dir, fit_optimizer_seconds=time.perf_counter() - fit_started,
                           fitting_iterations=max_iters)
    print("Biomechanics model fit successfully.")

    export_started = time.perf_counter()
    for i, video_path in enumerate(fitted_videos):
        print(f"Saving biomechanics for video {i+1}/{len(fitted_videos)}: {video_path}")

        timestamps = dataset.get_all_timestamps(i)

        inference_options = dict(trajectory_selection=i, steps=0, skip_action=True,
                                 fast_inference=True, check_constraints=False)
        if export_backend == "mjx":
            (state, _, _), (qpos, qvel, _), rnc = model(timestamps, **inference_options)
            joints, sites = np.asarray(state.xpos), np.asarray(state.site_xpos)
        else:
            qpos, qvel, _, rnc, geometry_scale_params = export_trajectory(model, timestamps, trajectory_selection=i)
        
        # qpos contains joint angles only; scale outputs live in the trajectory.
        scale_start_idx = model.njoints  # skip joint-angle part
        if hasattr(model, 'nactions'):
            scale_start_idx += model.nactions  # skip action dims (if present)
        if hasattr(model, 'nextra'):
            scale_start_idx += model.nextra  # skip extra dims

        # The scale parameters are extra outputs of the trajectory network after the joint
        # angles: evaluate it again and apply the same bounded (tanh) limits as the model.
        trajectory = jax.tree_util.tree_map(lambda x: jnp.take(x, i, axis=0), model.trajectory)
        
        def get_scale_params(t):
            output = trajectory(t)[..., 0]
            # Apply limits
            lower_limit = jnp.broadcast_to(model.lower_limit, output.shape)
            upper_limit = jnp.broadcast_to(model.upper_limit, output.shape)
            range_val = (upper_limit - lower_limit) / 2.0
            range_val = jnp.clip(range_val, 0.01, None)
            middle = (upper_limit + lower_limit) / 2.0
            scaled = jnp.tanh(output) * range_val + middle
            output = jnp.where(model.apply_limits, scaled, output)
            # Extract scale parameters
            scale_params = output[scale_start_idx:]
            return scale_params
        # Compute scale parameters for all timestamps
        scale_params_all = jax.vmap(get_scale_params)(timestamps)
        scale_params_all = np.array(scale_params_all)  # shape: (n_frames, n_scale_params)
        if export_backend == "native":
            from baseline.native_export import export_kinematic_coordinates
            joints, sites = export_kinematic_coordinates(
                model.forward.xml_path, qpos,
                1.0 + np.asarray(geometry_scale_params) @ np.asarray(model.scale_mixer).T,
                model.site_offsets, model.forward.site_reorder,
            )

        # save zip archive
        fname = Path(video_path).stem
        output_file = os.path.join(output_dir, f"{fname}_bone_length.npz")
        with open(output_file, "wb") as f:
            np.savez(
                f,
                timestamps=np.array(timestamps),
                qpos=np.array(qpos),
                qvel=np.array(qvel),
                rnc=np.array(rnc),
                sites=sites,
                joints=joints,
                scale=scale_params_all,  # time-varying scale, shape: (n_frames, n_scale_params)
                scale_mixer=np.array(model.scale_mixer),  # save scale_mixer for later use
                body_scale_factors=1.0 + scale_params_all @ np.array(model.scale_mixer).T,
                scale_parameterization=np.array("additive"),
                confidence=np.array(confs_list[i][0]),
                scale_time_varying=np.array(True),
                joint_limits_enforced=np.array(not unconstrained_joints),
                fitting_iterations=np.array(max_iters),
                site_offsets=np.asarray(model.site_offsets),
                export_backend=np.array(export_backend),
                camera_intrinsics_px=np.array(dataset.camera_params["mtx"]) * 1000.0,
                camera_distortion=np.array(dataset.camera_params["dist"]),
                camera_calibration_source=np.array(str(camera_calibration or "assumed_fov55")),
                keypoint_camera_calibration_source=np.array(keypoint_calibration_sources[i]),
            )

    record_runtime_profile(output_dir, fit_export_seconds=time.perf_counter() - export_started,
                           export_backend=export_backend)
    return f"Successfully processed {len(dataset)} videos with biomechanics fitting."



def convert_to_opensim(fitted_model_path, output_dir):
    """Write ``<stem>_opensim_skeleton.mot`` (OpenSim coordinates) for a fitted ``.npz``."""
    from baseline.metrab2opensim import convert_fitted_model_to_mot
    
    fname = os.path.basename(fitted_model_path).replace('_bone_length.npz', '').replace('_fitted_model.npz', '')
    output_file = os.path.join(output_dir, f"{fname}_opensim_skeleton.mot")
    
    print(f"Converting {fname} to OpenSim format...")
    if not convert_fitted_model_to_mot(fitted_model_path, output_file):
        raise RuntimeError(f"OpenSim conversion failed for {fitted_model_path}")
    print(f"OpenSim skeleton saved to {output_file}")
    return output_file


def main():
    """Command-line entry point; see the module docstring."""
    parser = argparse.ArgumentParser(description='Variable-scale skeleton fit for the bone-length metric (paper implementation)')
    parser.add_argument('--config', type=str, help='Path to config JSON file')
    parser.add_argument('--video', type=str, help='Path to single video file')
    parser.add_argument('--video_dir', type=str, default='data',
                        help='Directory containing video files')
    parser.add_argument('--output_dir', type=str, default='results/baseline',
                        help='Output directory for results')
    parser.add_argument('--skip_metrabs', action='store_true', help='Skip Metrabs keypoint detection')
    parser.add_argument('--skip_fitting', action='store_true', help='Skip biomechanics fitting')
    parser.add_argument('--skip_rendering', action='store_true', help='Skip video rendering')
    parser.add_argument('--skip_opensim', action='store_true', help='Skip OpenSim conversion')
    parser.add_argument('--metrabs-model', default=None,
                        help='Local MeTRAbs SavedModel directory or TF Hub handle; also settable with HMB_METRABS_MODEL.')
    parser.add_argument('--max-iters', type=int, default=10000,
                        help='Biomechanics fitting iterations (default: 10000). Small values are for smoke tests only.')
    parser.add_argument('--unconstrained-joints', action='store_true',
                        help='Fit without anatomical joint bounds so range-of-motion violations remain observable.')
    parser.add_argument('--export-backend', choices=['native', 'mjx'], default=None,
                        help='How body and site positions are exported. Default: native MuJoCo; mjx evaluates them with MJX.')
    parser.add_argument('--camera-calibration', default=None,
                        help='JSON with fx,fy,cx,cy in pixels and optional distortion (five coefficients), or legacy-samsung '
                             '(fixed phone intrinsics of the upstream demo). '
                             'Default: 55-degree field of view from the frame size, as MeTRAbs assumes.')
    
    args = parser.parse_args()
    if args.max_iters < 1:
        parser.error('--max-iters must be positive')
    
    # Ensure output directory exists
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Determine which videos to process
    video_files = []
    if args.config:
        with open(args.config, 'r') as f:
            config = json.load(f)
            video_files = config.get('videos', [])
    elif args.video:
        video_files = [args.video]
    else:
        # Process all videos in video_dir
        video_extensions = ['.mp4', '.avi', '.mov', '.mkv']
        for file in os.listdir(args.video_dir):
            if any(file.lower().endswith(ext) for ext in video_extensions):
                video_files.append(os.path.join(args.video_dir, file))
    
    if not video_files:
        print("No videos found to process!")
        return
    
    print(f"Processing {len(video_files)} video(s):")
    for video in video_files:
        print(f"  - {video}")
    print(f"Output directory: {args.output_dir}\n")
    if not (args.skip_fitting and args.skip_rendering and args.skip_opensim):
        # Reproducible GPU kernels for the JAX steps (baseline/main.py::fit_xla_flags, the rule of
        # hmb.fit_engine); set before configure_runtime imports JAX. Detection-only runs are left alone.
        from baseline.main import fit_xla_flags

        os.environ["XLA_FLAGS"] = fit_xla_flags()
        print(f"XLA_FLAGS={os.environ['XLA_FLAGS']!r}")
    configure_runtime(use_tensorflow=not args.skip_metrabs, use_jax=not args.skip_fitting)
    
    # Step 1: Metrabs keypoint detection
    if not args.skip_metrabs:
        print("=" * 50)
        print("STEP 1: Metrabs Keypoint Detection")
        print("=" * 50)
        result = process_videos_with_metrabs(
            video_files, args.output_dir, model_path=args.metrabs_model,
            camera_calibration=args.camera_calibration,
        )
        print(result)
    else:
        print("Skipping Metrabs keypoint detection")
    
    # Step 2: Biomechanics fitting
    if not args.skip_fitting:
        print("\n" + "=" * 50)
        print("STEP 2: Biomechanics Fitting")
        print("=" * 50)
        result = process_videos_with_biomechanics(
            video_files, args.output_dir, max_iters=args.max_iters,
            unconstrained_joints=args.unconstrained_joints,
            camera_calibration=args.camera_calibration,
            export_backend=args.export_backend,
        )
        print(result)
    else:
        print("Skipping biomechanics fitting")
    
    # Step 3: Render visualization videos
    if not args.skip_rendering:
        print("\n" + "=" * 50)
        print("STEP 3: Rendering Visualization Videos")
        print("=" * 50)
        for video_path in video_files:
            fname = Path(video_path).stem
            fitted_model_path = os.path.join(args.output_dir, f"{fname}_bone_length.npz")
            if os.path.exists(fitted_model_path):
                render_mjx(fitted_model_path, args.output_dir)
            else:
                print(f"Warning: Fitted model not found for {fname}")
    else:
        print("Skipping video rendering")
    
    # Step 4: Convert to OpenSim format
    if not args.skip_opensim:
        print("\n" + "=" * 50)
        print("STEP 4: Converting to OpenSim Format")
        print("=" * 50)
        for video_path in video_files:
            fname = Path(video_path).stem
            fitted_model_path = os.path.join(args.output_dir, f"{fname}_bone_length.npz")
            if os.path.exists(fitted_model_path):
                convert_to_opensim(fitted_model_path, args.output_dir)
            else:
                print(f"Warning: Fitted model not found for {fname}")
    else:
        print("Skipping OpenSim conversion")
    
    print("\n" + "=" * 50)
    print("PROCESSING COMPLETE!")
    print("=" * 50)
    print(f"All results saved to: {args.output_dir}")


if __name__ == "__main__":
    main()
