#!/usr/bin/env bash
# Checks the pinned third-party sources (git submodules) and, with --assets, the model files;
# it never downloads model files or installs packages. ./install.sh runs it in its preflight and
# in step "sources" (with --init).
#
#   bash install/check_sources.sh            # GVHMR, both HADM checkouts, torch-mesh-isect
#   bash install/check_sources.sh --init     # first restore missing ones (git submodule update)
#   bash install/check_sources.sh --assets   # also the model files where the pipeline reads them
#
# Exit status 1 when a source is missing or at another commit, or (with --assets) a required model
# file that is not downloaded on first use is missing (HADM-G, SMPL-X).
set -euo pipefail

repo_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
initialize=0
check_assets=0
for argument in "$@"; do
    case "$argument" in
        --init) initialize=1 ;;
        --assets) check_assets=1 ;;
        --help|-h)
            echo 'Usage: bash install/check_sources.sh [--init] [--assets]'
            echo 'Default: verify pinned source revisions. --init restores missing submodules.'
            echo '--assets also checks model files where the pipeline looks for them; it never downloads.'
            echo 'Exit status 1 when a source is missing or at another revision, or (--assets) HADM-G or'
            echo 'SMPL-X is missing; files downloaded on first use and the optional multi-person files never fail.'
            exit 0 ;;
        *) echo "Unknown argument: $argument" >&2; exit 2 ;;
    esac
done

# The expected commit of each submodule is the one recorded in this repository (git ls-files --stage).
sources=(GVHMR HADM metrics/multi-limbs/HADM torch-mesh-isect)
if (( initialize )); then
    git -C "$repo_dir" submodule update --init -- "${sources[@]}"
fi

failed=0
for source_dir in "${sources[@]}"; do
    expected="$(git -C "$repo_dir" ls-files --stage -- "$source_dir" | awk '$1 == "160000" {print $2}')"
    if [[ ! -e "$repo_dir/$source_dir/.git" ]]; then
        echo "MISSING source: $source_dir (run with --init)" >&2
        failed=1
        continue
    fi
    actual="$(git -C "$repo_dir/$source_dir" rev-parse HEAD)"
    if [[ -z "$expected" || "$actual" != "$expected" ]]; then
        echo "REVISION MISMATCH: $source_dir expected=$expected actual=$actual" >&2
        failed=1
    else
        echo "OK source: $source_dir $actual"
    fi
done

file_size() { stat -L -c %s -- "$1" 2>/dev/null || stat -L -f %z -- "$1"; }   # GNU or BSD stat

if (( check_assets )); then
    # GVHMR checkpoints (path, size in bytes): the same locations and sizes as hmb/gvhmr_prepare.py
    # (CHECKPOINTS, prepare_assets) and hmb/doctor.py.
    gvhmr_checkpoints=(
        "gvhmr/gvhmr_siga24_release.ckpt 163508011"
        "hmr2/epoch=10-step=25000.ckpt 2709494041"
        "vitpose/vitpose-h-multi-coco.pth 2549075546"
        "yolo/yolov8x.pt 136867539"
    )
    for entry in "${gvhmr_checkpoints[@]}"; do
        relative="${entry% *}"
        size="${entry##* }"
        found=""
        for candidate in "$repo_dir/GVHMR/inputs/checkpoints/$relative" "$repo_dir/outputs/assets/gvhmr/checkpoints/$relative"; do
            if [[ -f "$candidate" && "$(file_size "$candidate")" == "$size" ]]; then
                found="$candidate"
                break
            fi
        done
        if [[ -n "$found" ]]; then
            echo "OK asset: $found"
        else
            echo "AUTO asset: $relative (bash install.sh downloads it, else it is downloaded on first use into outputs/assets/gvhmr/checkpoints/)"
        fi
    done

    hadm_weights="$repo_dir/metrics/multi-limbs/HADM/pretrained_models/HADM-G_0249999.pth"
    if [[ -s "$hadm_weights" ]]; then
        echo "OK asset: $hadm_weights"
    else
        echo "MISSING asset: $hadm_weights (bash install.sh downloads HADM-G_0249999.pth; see README)" >&2
        failed=1
    fi

    smplx=""
    if [[ -n "${HMB_SMPLX_MODEL:-}" ]]; then
        smplx="$HMB_SMPLX_MODEL"
    else
        for candidate in "$repo_dir/outputs/assets/smplx/SMPLX_NEUTRAL.npz" \
                         "$repo_dir/GVHMR/inputs/checkpoints/body_models/smplx/SMPLX_NEUTRAL.npz"; do
            if [[ -s "$candidate" ]]; then smplx="$candidate"; break; fi
        done
    fi
    if [[ -n "$smplx" && -s "$smplx" ]]; then
        echo "OK model: $smplx"
    else
        echo "MISSING model: SMPLX_NEUTRAL.npz (licensed; bash install.sh --smplx-dir PATH places it in outputs/assets/smplx/, or set HMB_SMPLX_MODEL)" >&2
        failed=1
    fi

    metrabs="${HMB_METRABS_MODEL:-}"
    # Unpacked by install.sh (install/assets.json, hmb/model_paths.py).
    local_metrabs="$repo_dir/outputs/assets/metrabs/metrabs_eff2xl_y4_384px_800k_28ds"
    if [[ -z "$metrabs" && -f "$local_metrabs/saved_model.pb" ]]; then
        echo "OK model: MeTRAbs $local_metrabs"
    elif [[ -z "$metrabs" || "$metrabs" == *://* ]]; then
        echo "AUTO model: MeTRAbs (downloaded on first use into the TensorFlow Hub cache ${TFHUB_CACHE_DIR:-${TMPDIR:-/tmp}/tfhub_modules})"
    elif [[ -f "$metrabs/saved_model.pb" ]]; then
        echo "OK model: MeTRAbs $metrabs"
    else
        echo "MISSING model: HMB_METRABS_MODEL=$metrabs has no saved_model.pb" >&2
        failed=1
    fi

    # Optional: multi-person tracking (scripts/humanscore_multi.py).
    yolo_seg="${HMB_YOLO_SEG_WEIGHTS:-yolov8x-seg.pt}"
    [[ "$yolo_seg" == */* ]] || yolo_seg="$repo_dir/outputs/assets/yolo/$yolo_seg"
    if [[ -e "$yolo_seg" ]]; then
        echo "OK optional: $yolo_seg"
    else
        echo "AUTO optional: $yolo_seg (YOLO segmentation weights, downloaded on first use of --tracker yolo)"
    fi
    if [[ -n "${HMB_SAM3_CHECKPOINT:-}" ]]; then
        if [[ -s "$HMB_SAM3_CHECKPOINT" ]]; then
            echo "OK optional: SAM 3.1 checkpoint $HMB_SAM3_CHECKPOINT"
        else
            echo "ABSENT optional: HMB_SAM3_CHECKPOINT=$HMB_SAM3_CHECKPOINT does not exist"
        fi
    elif [[ -s "$repo_dir/outputs/assets/sam3/sam3.1_multiplex.pt" ]]; then
        echo "OK optional: SAM 3.1 checkpoint $repo_dir/outputs/assets/sam3/sam3.1_multiplex.pt"
    else
        echo "INFO optional: SAM 3.1 (gated facebook/sam3.1; bash install.sh --with-sam3 downloads it); check it with: python -m hmb.tracking --check-sam3"
    fi
fi

exit "$failed"
