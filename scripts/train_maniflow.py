"""Launch ManiFlow multi-modal training on the local machine.

Builds the ``train_hvla.py`` command line from high-level flags (cameras, depth,
target / language conditioning, action space, data filters, ...) and runs it with
``accelerate launch`` (one node, all visible GPUs by default) or ``torchrun``
(several nodes, see ``--num_nodes``).

Examples:
    # all visible GPUs on this machine
    python scripts/train_maniflow.py --data_root_dir /path/to/dataset --rgb_224

    # RGB + depth (front + gripper), 2 GPUs, print the command only
    python scripts/train_maniflow.py --data_root_dir /path/to/dataset --rgb_224 \
        --camera_config front_gripper_rgb_depth --depth_g12 g10_224 --num_gpus_per_node 2 --dry_run

    # resume a run (architecture flags are read from <run_root_dir>/<run_id>/config.yaml)
    python scripts/train_maniflow.py --data_root_dir /path/to/dataset --resume <run_id>
"""
import json
import os
import shlex
import shutil
import subprocess
import sys
import argparse
from datetime import datetime
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent


def visible_gpu_count() -> int:
    """Number of GPUs this process may use (CUDA_VISIBLE_DEVICES aware)."""
    visible = os.environ.get("CUDA_VISIBLE_DEVICES")
    if visible is not None:
        return len([d for d in visible.split(",") if d.strip() != ""])
    try:
        import torch

        return torch.cuda.device_count()
    except ImportError:
        return 0


def default_wandb_mode() -> str:
    """'online' when W&B credentials are available, otherwise 'offline' (logs stay in the run dir)."""
    if os.environ.get("WANDB_MODE"):
        return os.environ["WANDB_MODE"]
    if os.environ.get("WANDB_API_KEY"):
        return "online"
    netrc = Path(os.environ.get("NETRC", Path.home() / ".netrc"))
    try:
        if "api.wandb.ai" in netrc.read_text():
            return "online"
    except OSError:
        pass
    return "offline"


def main():
    parser = argparse.ArgumentParser(description="Launch ManiFlow multi-modal training on this machine")
    parser.add_argument("--user_prefix", type=str, default="", help="User prefix for experiment name")
    parser.add_argument("--num_gpus_per_node", type=int, default=None,
                        help="GPUs per node (default: all visible GPUs)")
    parser.add_argument("--num_nodes", type=int, default=1,
                        help="Number of nodes. >1 launches with torchrun; run this script once per node "
                             "with the same flags and its own --node_rank")
    parser.add_argument("--node_rank", type=int, default=int(os.environ.get("NODE_RANK", 0)),
                        help="Rank of this node (multi-node only; default: $NODE_RANK or 0)")
    parser.add_argument("--master_addr", type=str, default=os.environ.get("MASTER_ADDR", "127.0.0.1"),
                        help="Address of node 0 (multi-node only; default: $MASTER_ADDR)")
    parser.add_argument("--master_port", type=int, default=int(os.environ.get("MASTER_PORT", 29500)),
                        help="Rendezvous port on node 0 (multi-node only; default: $MASTER_PORT or 29500)")
    parser.add_argument("--wandb_project", type=str, default=os.environ.get("WANDB_PROJECT", "maniflow-tb"),
                        help="W&B project (default: $WANDB_PROJECT or 'maniflow-tb')")
    parser.add_argument("--wandb_entity", type=str, default=os.environ.get("WANDB_ENTITY"),
                        help="W&B entity (default: $WANDB_ENTITY or your default entity)")
    parser.add_argument("--wandb_mode", type=str, default=None, choices=["online", "offline", "disabled"],
                        help="W&B mode (default: $WANDB_MODE, else online if credentials exist, else offline)")
    parser.add_argument("--run_id", type=str, default=None,
                        help="Run name (default: derived from the flags + a timestamp). Multi-node runs must "
                             "pass the same --run_id on every node.")
    parser.add_argument("--dry_run", action="store_true", help="Print the training command and exit")
    parser.add_argument("--dbs", type=int, default=128, help="device batch size")
    parser.add_argument("--data_mix", type=str, default="apr13_test")
    parser.add_argument("--grad_accum", type=int, default=1)
    parser.add_argument("--optimizer", type=str, default="adamw", choices=["adamw", "muon"],
                        help="Optimizer backend: 'adamw' (DeepSpeed ZeRO-2, default) or "
                             "'muon' (distributed Muon + AdamW). Muon disables DeepSpeed and "
                             "runs on plain DDP, since distributed Muon's own parameter "
                             "all_gather is incompatible with ZeRO partitioning.")
    parser.add_argument("--muon_lr", type=float, default=2e-2,
                        help="Muon learning rate (only used with --optimizer muon). Lower it "
                             "(e.g. 0.01 / 0.005) if training diverges or NaNs after warmup.")
    parser.add_argument("--num_workers", type=int, default=12)
    parser.add_argument("--data_root_dir", type=str, nargs="+", default=None,
                        help="One or more dataset root dirs. Multiple dirs are co-trained (merged at load time). "
                             "Required for a new run; a --resume run reads it from its saved config.")
    parser.add_argument("--run_root_dir", type=str, default="runs",
                        help="Runs are written to <run_root_dir>/<run_id> (default: ./runs)")
    parser.add_argument("--duration", type=int, default=200000)
    parser.add_argument("--action_mode", type=str, default="abs", choices=["abs", "relative", "ee", "ee_target_7d", "rel_ee_target_7d"],
                        help="Action space: abs/relative (9D joints) or ee (8D end-effector)")
    parser.add_argument("--action_keys", type=str, nargs="+", default=None,
                        help="Custom action column names (overrides action_mode for dim). "
                             "E.g.: action.cube_position.block_r action.cube_distance.block_r")
    parser.add_argument("--state_estimation", action="store_true",
                        help="State estimation mode: predict cube_position.block_red1 (3) + "
                             "cube_distance.block_red1 (1) = 4D action, horizon=1/1")
    parser.add_argument("--state_dim", type=int, default=9,
                        help="Proprioceptive state dim (default: joint_pos(7) + gripper(1) + pd_mode(1) = 9)")
    parser.add_argument("--horizon", type=int, default=50, help="Action chunk size (prediction horizon)")
    parser.add_argument("--n_action_steps", type=int, default=25, help="Number of action steps to execute")
    parser.add_argument("--num_inference_steps", type=int, default=12, help="Number of ODE solver steps at inference")
    parser.add_argument("--resume", type=str, default=None,
                        help="Run ID to resume from (e.g. may1_rotate_cw_nopurtub_50_25_abs-05-02-19-30-00)")
    parser.add_argument("--pretrained_checkpoint", type=str, default=None,
                        help="Checkpoint file whose model weights initialize a new run at step 0. "
                             "Does not restore optimizer, scheduler, EMA, or W&B state.")
    target_group = parser.add_mutually_exclusive_group()
    target_group.add_argument("--token", action="store_true", default=True,
                              help="Target: heatmap as cross-attention context tokens (default)")
    target_group.add_argument("--timestep_concat", action="store_true",
                              help="Target: heatmap fused into AdaLN time conditioning")
    target_group.add_argument("--target_embedding", action="store_true",
                              help="Target: Gaussian bias on RGB tokens (no heatmap encoder)")
    target_group.add_argument("--target_pe", action="store_true",
                              help="Target: per-patch learned encoding on RGB tokens (no heatmap encoder)")
    target_group.add_argument("--no_target", action="store_true",
                              help="Disable target conditioning entirely")
    parser.add_argument("--language_conditioned", action="store_true", default=True,
                        help="Enable language conditioning (default: on)")
    parser.add_argument("--language_model", type=str, default="t5-small",
                        choices=["t5-small", "t5-base", "t5-large"],
                        help="T5 encoder model used for language conditioning")
    parser.add_argument("--no_language", action="store_true",
                        help="Disable language conditioning")
    parser.add_argument("--language_mode", type=str, default="action_only",
                        choices=["action_goal", "action_only"],
                        help="Language string mode: action_goal keeps coords, action_only drops them (default: action_only)")
    parser.add_argument("--lang_injection_mode", type=str, default="token",
                        choices=["token", "timestep_concat"],
                        help="Language injection mode: token (cross-attention) or timestep_concat (AdaLN fused)")
    parser.add_argument("--heatmap_encoder_type", type=str, default="shared_rgb",
                        choices=["shared_rgb", "convnextv2", "lightweight", "cnn"],
                        help="Backbone for heatmap (only for --token/--timestep_concat): shared_rgb "
                             "(reuse CLIP ViT), convnextv2 (separate pretrained), or cnn (from-scratch "
                             "CNN on the same 14x14 grid, sharing the RGB positional embedding)")
    parser.add_argument("--rgb_backbone", type=str, default=None,
                        help="timm RGB backbone (default: vit_base_patch16_clip_224.openai). "
                             "SigLIP options: vit_base_patch16_siglip_224.webli, "
                             "vit_base_patch16_siglip_224.v2_webli (SigLIP2), "
                             "vit_so400m_patch14_siglip_224.webli (400M, 256 tokens). "
                             "Normalization mean/std is read from the checkpoint automatically.")
    parser.add_argument("--siglip", action="store_true",
                        help="Shorthand for --rgb_backbone vit_base_patch16_siglip_224.v2_webli")
    parser.add_argument("--freeze_rgb_backbone", action="store_true",
                        help="Freeze the pretrained RGB backbone (feature extractor only). "
                             "Default is to fine-tune it.")
    parser.add_argument("--target_mask_prob", type=float, default=0.1,
                        help="Dropout probability for target conditioning during training")
    parser.add_argument("--target_sigma", type=float, default=0.8,
                        help="Gaussian sigma for target heatmap (in patch units)")
    parser.add_argument("--filter_config", type=str, default=None,
                        help="Episode filter preset name or YAML path (e.g. spatial_filter). "
                             "See hvla/config/episode_filtering/ for available presets.")
    parser.add_argument("--filter_override", type=str, nargs="+", default=None,
                        help="Dotlist overrides for filter config "
                             "(e.g. spatial_region=inside bounds='[0.4,0.7,-0.1,0.1]')")
    parser.add_argument("--filter_joint_jumps", type=float, nargs="?", const=0.5, default=None, metavar="RAD",
                        help="Drop episodes where any joint changes by more than RAD between consecutive frames "
                             "(teleport resets). Bare flag uses 0.5. Applied to train and val.")
    parser.add_argument("--filter_small_rotates", type=float, nargs="?", const=10.0, default=None, metavar="DEG",
                        help="Drop episodes with any rotate cycle that turns the block less than DEG about z "
                             "(failed rotate attempts). Bare flag uses 10. Applied to train and val.")
    parser.add_argument("--no_gripper", action="store_true",
                        help="(legacy) Same as --camera_config front_rgb")
    parser.add_argument("--camera_config", type=str, default=None,
                        choices=["front_rgb", "front_gripper_rgb",
                                 "front_rgb_depth", "front_gripper_rgb_front_depth",
                                 "front_gripper_rgb_depth",
                                 "front_gripper_rgb_gripper_depth",
                                 "front_gripper_depth_only",
                                 "gripper_rgb_depth", "gripper_rgb"],
                        help="Camera input preset: "
                             "front_rgb (front RGB only), "
                             "front_gripper_rgb (front+gripper RGB), "
                             "front_rgb_depth (front RGB + front depth), "
                             "front_gripper_rgb_front_depth (front+gripper RGB + front depth), "
                             "front_gripper_rgb_depth (front+gripper RGB + front+gripper depth), "
                             "front_gripper_rgb_gripper_depth (front+gripper RGB + gripper depth only), "
                             "front_gripper_depth_only (front+gripper depth, NO RGB), "
                             "gripper_rgb_depth (gripper RGB + gripper depth, no front camera), "
                             "gripper_rgb (gripper RGB only, no front camera, no depth)")
    parser.add_argument("--depth", action="store_true",
                        help="Use depth (raw depth + GPU back-projection at training time)")
    parser.add_argument("--sparse_depth", action="store_true",
                        help="Use sparse PointNet encoder for depth instead of dense ConvNeXtV2")
    parser.add_argument("--depth_encoder", type=str, default="shared_rgb",
                        choices=["dense", "sparse", "shared_rgb", "defm"],
                        help="Depth encoder type (overrides --sparse_depth)")
    parser.add_argument("--depth_rg8", type=str, default=None,
                        help="Decode depth videos as RG8-split instead of PyAV gray12le, "
                             "reading from the <camera>_<variant> video dir. "
                             "'rg8' / 'rg8_224': official pack Depth=R+G*256 "
                             "(native or pre-downscaled 224). "
                             "'rg8_55' / 'rg8_55_224': 10-bit 5+5 pack Depth=(R<<5)|G, "
                             "clipped to 1023 mm. Only for datasets re-encoded with "
                             "scripts/depth_conversion/reencode_to_rg8_batch.py")
    parser.add_argument("--rgb_224", action="store_true",
                        help="Read pre-resized RGB videos from <camera>_224 "
                             "Default is native 640x360. Missing 224 videos fail the run.")
    parser.add_argument("--depth_g12", type=str, default=None,
                        help="Read depth from gray HEVC videos <depth_key>_<variant> (PyAV). "
                             "'g12_224': lossless gray12le. "
                             "'g10_224': lossless gray10le (clip 1023 mm). "
                             "'g10_224_crf035': gray10le x265 CRF 0.35. "
                             "Mutually exclusive with --depth_rg8.")
    parser.add_argument("--add_depth_noise", action="store_true",
                        help="Enable RGB-guided depth noise augmentation at train time")
    parser.add_argument("--depth_noise_ideal_mix", type=float, default=0.6,
                        help="Blend factor for clean vs corrupted depth (1.0=clean, 0.0=fully corrupted)")
    parser.add_argument("--logging_frequency", type=int, default=100,
                        help="Log metrics every N steps")
    parser.add_argument("--profile_steps", type=int, default=0,
                        help="cProfile first N steps on rank 0, dump to <run_dir>/profiles/")
    parser.add_argument("--faulthandler", action="store_true",
                        help="Set PYTHONFAULTHANDLER=1 so a native crash (SIGSEGV/SIGABRT) dumps a "
                             "Python traceback per thread instead of just an exit code")
    parser.add_argument("--save_interval", type=int, default=10000,
                        help="Save checkpoint every N steps")
    parser.add_argument("--val_ratio", type=float, default=0.05,
                        help="Fraction of episodes for validation (default: 0.05)")
    parser.add_argument("--max_episodes", type=str, default=None,
                        help="Use only N episodes, then split val/train from that pool "
                             "(e.g. 10000 and val_ratio 0.05 → 500 val + 9500 train). "
                             "Per-dataset hard caps: 10000,5000,2000 (one int per --data_root_dir).")
    parser.add_argument("--dataset_ratio", type=str, default=None,
                        help="Sampling ratio across datasets, comma-separated (e.g. 1,1 or 10,5,2). "
                             "Subsamples larger datasets to match the ratio.")
    parser.add_argument("--phase_sample_weights", type=str, default=None,
                        help="Enable sampling weights by action.low_level_action phase. provide default string or a dictionary")
    parser.add_argument("--skip_actions", type=str, default=None,
                        help="Comma-separated low-level actions to skip")
    parser.add_argument("--drop_redundant_table_moves", action="store_true",
                        help="Drop redundant back-to-back table-to-table retries; keep the last attempt")
    parser.add_argument("--goal_action_filter", type=str, default=None,
                        help="goal.action allow-list(s) for sample starts. Mutually exclusive "
                             "with --phase_sample_weights. Formats: "
                             "(1) 'pick,place'  -> single list applied to every --data_root_dir; "
                             "(2) 'pick,place;stack,done;lift'  -> one ';'-separated group per "
                             "--data_root_dir, in the same order (comma-separated within a group); "
                             "(3) 'pick,place;;lift'  -> empty middle group leaves that dataset "
                             "unfiltered. Group count must equal 1 or len(--data_root_dir).")
    parser.add_argument("--max_frames_per_episode", type=int, default=None,
                        help="Truncate each episode to first N frames (e.g. 100 for approach-only)")
    parser.add_argument("--skip_first_n_steps_per_episode", type=int, default=None,
                        help="Skip first N timesteps from every episode (e.g. skip approach phase)")
    parser.add_argument("--include_jvel_ee_pos", action="store_true",
                        help="Add joint velocity (7) and end-effector XYZ (3) to the default joint-only state (state_dim=19)")
    parser.add_argument("--action_norm_band", type=str, default="q01_q99",
                        choices=["q01_q99", "valid_min_max"],
                        help="Arm-action normalization band: q01_q99 (stats.json quantiles; clips the "
                             "outer 1%% of labels) or valid_min_max (min/max over action.is_valid rows; no clipping)")
    parser.add_argument("--zero_state", action="store_true",
                        help="Zero out the state vector (no proprioception info, keeps state_dim for architecture)")
    parser.add_argument("--ft", action="store_true",
                        help="Include 6D wrench (F/T sensor) in the state vector (state_dim += 6)")
    parser.add_argument("--per_cam_norm", action="store_true",
                        help="Use per-camera pointmap normalization bounds instead of global")
    parser.add_argument("--use_previous_front_frame", action="store_true",
                        help="Include previous front camera frame as a 3rd RGB view")
    parser.add_argument("--previous_front_frame_lookback_weights", type=str, default=None,
                        help="Weighted lookback ranges for the previous front frame, e.g. '0-1_1'")
    parser.add_argument("--freeze_on_goal_change", action="store_true",
                        help="When the lookback crosses a goal change, relabel the action chunk as "
                             "'hold position'. Requires --use_previous_front_frame.")
    parser.add_argument("--pause_on_attempt", action="store_true",
                        help="When a pick, place, flip, or rotate switch falls inside the action "
                             "chunk, repeat that frame's action through the rest of the chunk and "
                             "set done=1 from the switch. Incompatible with --use_previous_front_frame.")
    parser.add_argument("--target_crop", type=int, default=-1,
                        help="Target-centered square front crop. -1=off; 0=100%% above the "
                             "target; 50=centered; 33=67/33 top/bottom; 25=75/25. "
                             "Incompatible with --use_previous_front_frame.")
    parser.add_argument("--ptmap_backbone", type=str, default=None,
                        help="Pointmap encoder backbone (default: convnextv2_tiny.fcmae_ft_in22k_in1k). "
                             "E.g. vit_small_patch14_dinov2 for DINOv2")
    parser.add_argument("--save_first_batch", action="store_true",
                        help="Save preprocessed first batch data to <run_dir>/first_batch_data/ for diagnostics")
    parser.add_argument("--save_arch_diagram", action="store_true",
                        help="Save ASCII architecture diagram to <run_dir>/architecture.txt")
    args = parser.parse_args()
    if not args.resume and not args.data_root_dir:
        parser.error("--data_root_dir is required (unless --resume is given)")
    if args.num_nodes > 1 and not (args.run_id or args.resume):
        parser.error("multi-node runs need the same --run_id (or --resume) on every node")
    if args.depth_rg8 and args.depth_g12:
        parser.error("--depth_rg8 and --depth_g12 are mutually exclusive; pass one of them")

    if args.freeze_on_goal_change and not args.use_previous_front_frame:
        parser.error("--freeze_on_goal_change requires --use_previous_front_frame")
    if args.pause_on_attempt and args.use_previous_front_frame:
        parser.error("--pause_on_attempt cannot be combined with --use_previous_front_frame")
    if args.num_gpus_per_node is None:
        args.num_gpus_per_node = visible_gpu_count()
        if args.num_gpus_per_node < 1 and not args.dry_run:
            parser.error("no visible GPU; pass --num_gpus_per_node explicitly")
        args.num_gpus_per_node = max(args.num_gpus_per_node, 1)
    if args.optimizer == "muon" and not args.dry_run:
        try:
            import muon  # noqa: F401
        except ImportError:
            parser.error("--optimizer muon needs the muon-optimizer package, see INSTALL.md")

    # Handle --no_language override
    if args.no_language:
        args.language_conditioned = False

    if args.siglip and args.rgb_backbone is None:
        args.rgb_backbone = "vit_base_patch16_siglip_224.v2_webli"

    # Derive target config from shorthand flags
    if args.no_target:
        args.target_injection_mode = "null"
    elif args.token:
        args.target_injection_mode = "token"
    elif args.timestep_concat:
        args.target_injection_mode = "timestep_concat"
    elif args.target_embedding:
        args.target_injection_mode = "target_embedding"
    elif args.target_pe:
        args.target_injection_mode = "target_pe"
    else:
        args.target_injection_mode = "null"

    if args.state_estimation:
        if not args.action_keys:
            args.action_keys = [
                "action.cube_position.block_red1",
                "action.cube_distance.block_red1",
            ]
        args.horizon = 1
        args.n_action_steps = 1
        args.save_interval = 2000

    ACTION_DIM = {"abs": 9, "relative": 9, "ee": 8, "ee_target_7d": 9, "rel_ee_target_7d": 9}
    if args.action_keys and args.data_root_dir:  # on --resume the action shape comes from the saved config
        info_path = Path(args.data_root_dir[0]) / "meta" / "info.json"
        with open(info_path) as f:
            _info = json.load(f)
        action_dim = 0
        for _k in args.action_keys:
            _shape = _info['features'][_k]['shape']
            d = 1
            for s in _shape:
                d *= s
            action_dim += d
        print(f"Custom action keys: {args.action_keys} → action_dim={action_dim}")
    else:
        action_dim = ACTION_DIM[args.action_mode]
    state_dim = args.state_dim
    if args.include_jvel_ee_pos:
        state_dim = 19  # joint_pos(7) + joint_vel(7) + ee_xyz(3) + gripper(1) + pd_mode(1)
    if args.ft:
        state_dim += 6

    # Resolve camera config: --camera_config takes precedence, then legacy flags
    cam_cfg = args.camera_config
    if cam_cfg is None:
        if args.no_gripper and args.depth:
            cam_cfg = "front_rgb_depth"
        elif args.no_gripper:
            cam_cfg = "front_rgb"
        elif args.depth:
            cam_cfg = None  # default: both cameras, both depth (legacy behavior)
        else:
            cam_cfg = None  # default: both cameras, no depth
    # Derive flags from camera config
    _CAM_CONFIGS = {
        "front_rgb":                       {"gripper_rgb": False, "front_depth": False, "gripper_depth": False},
        "front_gripper_rgb":               {"gripper_rgb": True,  "front_depth": False, "gripper_depth": False},
        "front_rgb_depth":                 {"gripper_rgb": False, "front_depth": True,  "gripper_depth": False},
        "front_gripper_rgb_front_depth":   {"gripper_rgb": True,  "front_depth": True,  "gripper_depth": False},
        "front_gripper_rgb_depth":         {"gripper_rgb": True,  "front_depth": True,  "gripper_depth": True},
        "front_gripper_rgb_gripper_depth": {"gripper_rgb": True,  "front_depth": False, "gripper_depth": True},
        "front_gripper_depth_only":        {"gripper_rgb": False, "front_depth": True,  "gripper_depth": True, "no_rgb": True},
        "gripper_rgb_depth":               {"gripper_rgb": True,  "front_depth": False, "gripper_depth": True, "no_front_rgb": True},
        "gripper_rgb":                     {"gripper_rgb": True,  "front_depth": False, "gripper_depth": False, "no_front_rgb": True},
    }
    if cam_cfg is not None:
        cc = _CAM_CONFIGS[cam_cfg]
        use_depth = cc["front_depth"] or cc["gripper_depth"]
        use_rgb = not cc.get("no_rgb", False)
        if use_depth:
            args.depth = True
    else:
        cc = None
        use_depth = args.depth
        use_rgb = True

    # Auto-enable per-camera pointmap normalization for depth jobs
    if use_depth and not args.per_cam_norm:
        args.per_cam_norm = True

    # No-RGB safety: disable target/heatmap modes that require RGB
    if not use_rgb:
        if args.target_injection_mode in ("token", "timestep_concat", "target_embedding", "target_pe"):
            print(f"[WARN] Forcing --no_target because use_rgb=False (was {args.target_injection_mode})")
            args.target_injection_mode = "null"

    # Training configuration
    data_mix = args.data_mix

    # Short name for camera config in experiment name
    _CAM_SHORT = {
        "front_rgb": "F",
        "front_gripper_rgb": "FG",
        "front_rgb_depth": "Fd",
        "front_gripper_rgb_front_depth": "FGfd",
        "front_gripper_rgb_depth": "FGd",
        "front_gripper_rgb_gripper_depth": "FGgd",
        "front_gripper_depth_only": "donly",
        "gripper_rgb_depth": "Ggd",
        "gripper_rgb": "G",
    }

    resume_config_path = None
    if args.resume:
        exp_name = args.resume
        assert len(exp_name) < 123, f"Experiment name {exp_name} is too long and resume will be problematic"
        # force: fail loudly if the run dir holds no loadable checkpoint, rather than
        # quietly retraining from scratch under a typo'd run id.
        resume_mode = "force"
        resume_config_path = os.path.join(args.run_root_dir, exp_name, "config.yaml")
    else:
        dataset_name = "_".join(Path(r).name for r in args.data_root_dir)
        exp_name = f"{dataset_name}_{args.horizon}_{args.n_action_steps}_{args.action_mode}"
        if args.target_injection_mode != "null":
            exp_name += f"_{args.target_injection_mode}"
            if args.target_injection_mode in ("token", "timestep_concat"):
                exp_name += f"_{args.heatmap_encoder_type}"
            if args.target_mask_prob > 0:
                exp_name += f"_mask{args.target_mask_prob}"
            if args.target_sigma != 1.5:
                exp_name += f"_sigma{args.target_sigma}"
        if args.language_conditioned:
            exp_name += f"_lang_{args.language_model}_{args.language_mode}"
            if args.lang_injection_mode != "token":
                exp_name += f"_{args.lang_injection_mode}"
        if cam_cfg is not None:
            exp_name += f"_{_CAM_SHORT[cam_cfg]}"
        if use_depth:
            _de = args.depth_encoder or ("sparse" if args.sparse_depth else "dense")
            exp_name += f"_depth_{_de}" if _de != "dense" else "_depth"
            if args.depth_rg8:
                exp_name += f"_{args.depth_rg8}"
        if args.freeze_rgb_backbone:
            exp_name += "_frozenbb"
        if args.rgb_backbone:
            exp_name += "_siglip2" if "v2_webli" in args.rgb_backbone else \
                        "_siglip" if "siglip" in args.rgb_backbone else "_bb"
            if args.depth_g12:
                exp_name += f"_{args.depth_g12}"
        if args.rgb_224:
            exp_name = f"rgb224_{exp_name}"
        if args.max_episodes is not None:
            exp_name = f"Ep{str(args.max_episodes).replace(',', '-')}_{exp_name}"
        if args.per_cam_norm:
            exp_name += "_pcn"
        if args.use_previous_front_frame:
            exp_name += "_prev"
        if args.freeze_on_goal_change:
            exp_name += "_freeze"
        if args.pause_on_attempt:
            exp_name += "_pause"
        if args.target_crop >= 0:
            exp_name = f"tcrop{args.target_crop}_{exp_name}"
        if args.ptmap_backbone:
            bb = args.ptmap_backbone
            if "dinov3" in bb:
                short_bb = "dinov3s" if "small" in bb else ("dinov3b" if "base" in bb else "dinov3")
            elif "dinov2" in bb:
                short_bb = "dinov2s" if "small" in bb else ("dinov2b" if "base" in bb else "dinov2")
            else:
                short_bb = bb.replace(".", "_")[:20]
            exp_name += f"_{short_bb}"
        if args.state_estimation:
            exp_name += "_state_est"
        elif args.action_keys:
            short_keys = "_".join(k.split(".")[-1] for k in args.action_keys)
            exp_name += f"_ak_{short_keys}"
        if args.include_jvel_ee_pos:
            exp_name += "_jvel_ee"
        if args.zero_state:
            exp_name += "_zerostate"
        if args.action_norm_band == "valid_min_max":
            exp_name += "_vmm"
        if args.max_frames_per_episode:
            exp_name += f"_{args.max_frames_per_episode}f"
        if args.skip_first_n_steps_per_episode:
            exp_name += f"_skip{args.skip_first_n_steps_per_episode}"
        if args.ft:
            exp_name += "_ft"
        if args.filter_config:
            exp_name += f"_filt_{Path(args.filter_config).stem}"
            if args.filter_override:
                ov_short = "_".join(v for ov in args.filter_override for _, _, v in [ov.partition("=")])
                exp_name += f"_{ov_short}"
        if args.filter_joint_jumps is not None:
            exp_name += f"_nojump{args.filter_joint_jumps:g}"
        if args.filter_small_rotates is not None:
            exp_name += f"_minrot{args.filter_small_rotates:g}"
        if args.dataset_ratio:
            exp_name += f"_ratio{args.dataset_ratio.replace(':', '-').replace(',', '-')}"
        if args.phase_sample_weights:
            exp_name += "_phaseweights"
        if args.optimizer == "muon":
            exp_name += f"_muon_lr{args.muon_lr}"
        if args.user_prefix:
            exp_name = f"{args.user_prefix}-{exp_name}"
        date_time = datetime.now().strftime("%m-%d-%H-%M-%S")
        exp_name = f"{exp_name}-{date_time}"
        # Must stay under the --resume length limit above, or the run can never be resumed.
        max_len = 122
        if len(exp_name) > max_len:
            exp_name = exp_name[:max_len - len(date_time) - 1] + "-" + date_time
        if args.run_id:
            exp_name = args.run_id
            assert len(exp_name) < 123, f"--run_id {exp_name} is too long and resume will be problematic"
        # auto: the timestamped run dir is empty on the first attempt, so this is a fresh start;
        # rerunning the same command with --resume <run_id> continues from its checkpoints.
        resume_mode = "auto"

    total_procs = args.num_gpus_per_node * args.num_nodes
    python = shlex.quote(sys.executable)
    accelerate_cli = shutil.which("accelerate") or f"{python} -m accelerate.commands.launch"
    if accelerate_cli != f"{python} -m accelerate.commands.launch":
        accelerate_cli = f"{shlex.quote(accelerate_cli)} launch"

    # For single-node: use accelerate launch (DeepSpeed's launcher is fine).
    # For multi-node: use torchrun, which sets up the distributed env across nodes
    # without the SSH access DeepSpeed's multi-node launcher needs.
    # Accelerator() auto-detects the torchrun env.
    if args.num_nodes > 1:
        launcher = (
            f"{python} -m torch.distributed.run"
            f" --nnodes {args.num_nodes}"
            f" --nproc_per_node {args.num_gpus_per_node}"
            f" --node_rank {args.node_rank}"
            f" --master_addr {shlex.quote(args.master_addr)}"
            f" --master_port {args.master_port}"
        )
    elif args.optimizer == "muon":
        # DDP (no DeepSpeed): distributed Muon manages its own cross-rank param sync,
        # which conflicts with ZeRO. --multi_gpu selects DDP; omit it for a single process.
        multi_gpu = "--multi_gpu " if total_procs > 1 else ""
        launcher = (
            f"{accelerate_cli} {multi_gpu}"
            f"--mixed_precision bf16"
            f" --num_processes {total_procs}"
            f" --num_machines 1"
            f" --main_process_port $MASTER_PORT"
        )
    else:
        launcher = (
            f"{accelerate_cli}"
            f" --config_file hvla/config/deepseeds/deepspeed_zero2.yaml"
            f" --num_processes {total_procs}"
            f" --num_machines 1"
            f" --main_process_port $MASTER_PORT"
        )

    data_roots = [str(Path(r).resolve()) for r in (args.data_root_dir or [])]
    run_root_dir = str(Path(args.run_root_dir).resolve())
    if resume_config_path is not None:
        resume_config_path = os.path.join(run_root_dir, exp_name, "config.yaml")
        if not os.path.exists(resume_config_path) and not args.dry_run:
            parser.error(f"--resume: {resume_config_path} does not exist")

    faulthandler_export = "export PYTHONFAULTHANDLER=1" if args.faulthandler else "true"
    wandb_mode = args.wandb_mode or default_wandb_mode()
    wandb_entity_export = f"export WANDB_ENTITY={shlex.quote(args.wandb_entity)}" if args.wandb_entity else "true"

    command = f"""
    export WANDB_PROJECT={shlex.quote(args.wandb_project)} &&
    {wandb_entity_export} &&
    export WANDB_MODE={wandb_mode} &&
    {faulthandler_export} &&
    export ACCELERATE_GRADIENT_ACCUMULATION_STEPS={args.grad_accum} &&
    export NCCL_BLOCKING_WAIT=1 &&
    export NCCL_ASYNC_ERROR_HANDLING=1 &&
    export NCCL_TIMEOUT=1000 &&
    export MASTER_PORT=$({python} -c 'import socket; s=socket.socket(); s.bind(("",0)); print(s.getsockname()[1]); s.close()') &&
    export PROFILE_STEPS={args.profile_steps} &&
    export NODE_RANK={args.node_rank} &&
    {launcher} \\
      hvla/training/train_hvla.py \\
      --config_yaml ./examples/Franka/train_maniflow_multimodal.yaml \\
      --framework.name ManiFlowMultiModal \\
      --datasets.vla_data.num_workers {args.num_workers} \\
      --datasets.vla_data.per_device_batch_size {args.dbs} \\
      --run_root_dir {run_root_dir} \\
      --run_id {exp_name} \\
      --trainer.gradient_accumulation_steps {args.grad_accum} \\
      --trainer.save_interval {args.save_interval} \\
      --trainer.logging_frequency {getattr(args, 'logging_frequency', 100)} \\
      --trainer.max_train_steps {args.duration} \\
      --trainer.resume_mode {resume_mode} """

    # Optimizer selection drives both the optimizer and the distributed backend
    # (DDP vs DeepSpeed) in train_hvla.py, so emit it on fresh runs and resumes alike.
    if args.optimizer == "muon":
        command += f""" \\
      --trainer.optimizer.name muon \\
      --trainer.optimizer.muon_lr {args.muon_lr}"""

    if args.pretrained_checkpoint:
        command += f""" \\
      --trainer.pretrained_checkpoint {shlex.quote(args.pretrained_checkpoint)}"""

    # Adding this for now because hvla historically does not save this
    command += f""" \\
        --framework.shape_meta.obs.agent_pos.shape '[{state_dim}]' """

    if resume_config_path is not None:
        command += f""" \\
      --resume_config_yaml {resume_config_path}"""

    # Architecture-defining flags (action/state space, cameras, target/language
    # conditioning, depth encoder, ...) are only meaningful for a fresh run. On
    # resume, these MUST come from the resumed run's saved config so the model
    # always matches the checkpoint being loaded — emitting them here too (even
    # with argparse defaults) would misleadingly suggest they're active when
    # `--resume_config_yaml` (merged in train_hvla.py) actually overrides them.
    if not args.resume:
        command += f""" \\
      --datasets.vla_data.val_ratio {args.val_ratio} \\
      --datasets.vla_data.data_root_dir '{",".join(data_roots)}' \\
      --datasets.vla_data.data_mix {data_mix} \\
      --datasets.vla_data.action_mode {args.action_mode} \\
      --datasets.vla_data.use_ft {str(args.ft).lower()} \\
      --datasets.vla_data.include_ptmap {str(use_depth).lower()} \\
      --datasets.vla_data.use_depth {str(use_depth).lower()} \\
      --datasets.vla_data.depth_rg8 {args.depth_rg8 or "null"} \\
      --datasets.vla_data.depth_g12 {args.depth_g12 or "null"} \\
      --datasets.vla_data.use_rgb_224 {str(args.rgb_224).lower()} \\
      --framework.use_ptmap {str(use_depth).lower()} \\
      --framework.use_rgb {str(use_rgb).lower()} \\
      --framework.horizon {args.horizon} \\
      --framework.n_action_steps {args.n_action_steps} \\
      --framework.num_inference_steps {args.num_inference_steps} \\
      --framework.target_injection_mode {args.target_injection_mode} \\
      --framework.heatmap_encoder_type {args.heatmap_encoder_type} \\
      --framework.target_mask_prob {args.target_mask_prob} \\
      --framework.target_sigma {args.target_sigma} \\
      --framework.use_previous_front_frame {str(args.use_previous_front_frame).lower()} \\
      --framework.use_target_front_crop {args.target_crop} \\
      --framework.language_conditioned {str(args.language_conditioned).lower()} \\
      --framework.language_model {args.language_model} \\
      --framework.language_mode {args.language_mode} \\
      --framework.lang_injection_mode {args.lang_injection_mode} \\
      --framework.rgb_backbone {args.rgb_backbone or 'vit_base_patch16_clip_224.openai'} \\
      --framework.freeze_rgb_backbone {str(args.freeze_rgb_backbone).lower()} \\
      --framework.ptmap_encoder_type {args.depth_encoder or ('sparse' if args.sparse_depth else 'dense')} \\
      --framework.ptmap_backbone {((args.rgb_backbone or 'vit_base_patch16_clip_224.openai') if (args.depth_encoder == 'shared_rgb') else (args.ptmap_backbone or 'convnextv2_tiny.fcmae_ft_in22k_in1k'))} \\
      --framework.shape_meta.action.shape '[{action_dim}]' \\
      --framework.shape_meta.obs.agent_pos.shape '[{state_dim}]' \\
      --framework.shape_meta.obs.front_ptmap null \\
      --framework.shape_meta.obs.gripper_ptmap null"""

        if cc is not None:
            if cc.get("no_rgb", False):
                command += """ \\
            --framework.shape_meta.obs.front_rgb null \\
            --framework.shape_meta.obs.gripper_rgb null"""
            elif cc.get("no_front_rgb", False):
                command += """ \\
            --framework.shape_meta.obs.front_rgb null"""
            elif not cc["gripper_rgb"]:
                command += """ \\
            --framework.shape_meta.obs.gripper_rgb null"""

            if cc["front_depth"]:
                command += """ \\
            --framework.shape_meta.obs.front_depth.shape '[3,224,224]'"""
            else:
                command += """ \\
            --framework.shape_meta.obs.front_depth null"""

            if cc["gripper_depth"]:
                command += """ \\
            --framework.shape_meta.obs.gripper_depth.shape '[3,224,224]'"""
            else:
                command += """ \\
            --framework.shape_meta.obs.gripper_depth null"""
            n_depth_cams = int(cc["front_depth"]) + int(cc["gripper_depth"])

            if n_depth_cams > 0:
                command += f""" \\
            --framework.shape_meta.obs.camera_intrinsics.shape '[{n_depth_cams},4]'"""
            else:
                command += """ \\
            --framework.shape_meta.obs.camera_intrinsics null"""

        if args.per_cam_norm and use_depth:
            command += """ \\
          --framework.per_camera_ptmap_bounds.front_depth.min '[-0.6,-0.4,0.4]' \\
          --framework.per_camera_ptmap_bounds.front_depth.max '[0.6,0.3,1.0]' \\
          --framework.per_camera_ptmap_bounds.gripper_depth.min '[-0.5,-0.4,0.1]' \\
          --framework.per_camera_ptmap_bounds.gripper_depth.max '[0.5,0.4,0.8]'"""

        if args.use_previous_front_frame and args.previous_front_frame_lookback_weights is not None:
            command += f""" \\
          --datasets.vla_data.previous_front_frame_lookback_weights '{args.previous_front_frame_lookback_weights}'"""

        if args.freeze_on_goal_change:
            command += """ \\
          --datasets.vla_data.freeze_on_goal_change true"""

        if args.pause_on_attempt:
            command += """ \\
          --datasets.vla_data.pause_on_attempt true"""

        if args.add_depth_noise:
            command += f""" \\
          --datasets.vla_data.add_depth_noise true \\
          --datasets.vla_data.depth_noise_ideal_mix {args.depth_noise_ideal_mix}"""

        if args.max_episodes is not None:
            command += f""" \\
          --datasets.vla_data.max_episodes '{args.max_episodes}'"""

        if args.max_frames_per_episode is not None:
            command += f""" \\
          --datasets.vla_data.max_frames_per_episode {args.max_frames_per_episode}"""

        if args.skip_first_n_steps_per_episode is not None:
            command += f""" \\
          --datasets.vla_data.skip_first_n_steps_per_episode {args.skip_first_n_steps_per_episode}"""

        if args.include_jvel_ee_pos:
            command += """ \\
          --datasets.vla_data.joint_only_state false"""

        if args.zero_state:
            command += """ \\
          --datasets.vla_data.zero_state true"""

        if args.action_norm_band != "q01_q99":
            command += f""" \\
          --datasets.vla_data.action_norm_band {args.action_norm_band}"""

        if args.action_keys:
            ak_csv = ",".join(args.action_keys)
            command += f""" \\
          --datasets.vla_data.action_keys '{ak_csv}'"""

        if args.filter_config:
            command += f""" \\
          --datasets.vla_data.filter_config {args.filter_config}"""
            if args.filter_override:
                for ov in args.filter_override:
                    key, _, val = ov.partition("=")
                    command += f""" \\
          --datasets.vla_data.filter_override.{key} {val}"""

        if args.filter_joint_jumps is not None:
            command += f""" \\
          --datasets.vla_data.filter_joint_jumps {args.filter_joint_jumps}"""

        if args.filter_small_rotates is not None:
            command += f""" \\
          --datasets.vla_data.filter_small_rotates {args.filter_small_rotates}"""

        if args.dataset_ratio:
            command += f""" \\
          --datasets.vla_data.dataset_ratio '{args.dataset_ratio}'"""

        if args.phase_sample_weights:
            command += f""" \\
          --datasets.vla_data.phase_sample_weights {shlex.quote(args.phase_sample_weights)}"""

        if args.skip_actions is not None:
            command += f""" \\
          --datasets.vla_data.skip_actions {shlex.quote(args.skip_actions)}"""

        if args.drop_redundant_table_moves:
            command += """ \\
          --datasets.vla_data.drop_redundant_table_moves true"""

        if args.goal_action_filter is not None:
            command += f""" \\
          --datasets.vla_data.goal_action_filter {shlex.quote(args.goal_action_filter)}"""

    if args.save_first_batch:
        command += """ \\
      --save_first_batch true"""

    if args.save_arch_diagram:
        command += """ \\
      --save_arch_diagram true"""

    print(f"Run: {exp_name}  ({args.num_nodes} node(s) x {args.num_gpus_per_node} GPU(s), W&B {wandb_mode})")
    print(f"Run dir: {os.path.join(run_root_dir, exp_name)}")
    print(f"Command:\n{command}\n")
    if args.dry_run:
        return
    env = os.environ.copy()
    for k in ("WANDB_API_KEY", "HF_TOKEN", "HF_ACCESS_TOKEN"):
        if k in env:
            env[k] = env[k].strip()
    result = subprocess.run(["/bin/bash", "-c", command], cwd=str(REPO_ROOT), env=env)
    sys.exit(result.returncode)


if __name__ == "__main__":
    main()
