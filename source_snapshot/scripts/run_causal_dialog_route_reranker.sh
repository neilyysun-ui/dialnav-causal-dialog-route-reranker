#!/usr/bin/env bash
set -euo pipefail

if [[ $# -ne 4 ]]; then
  echo "Usage: $0 RUN_ID GPU_ID SPLIT ANNOTATION_PATH" >&2
  exit 2
fi

RELEASE_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
RAINBOW_ROOT="${DIALNAV_RAINBOW_ROOT:-${RELEASE_ROOT}/external/RAINbow}"
MATTERSIM_BUILD="${DIALNAV_MATTERSIM_BUILD:-${RELEASE_ROOT}/external/Matterport3DSimulator/build}"
PYTHON_BIN="${DIALNAV_PYTHON:-python}"
RUN_ID="$1"
GPU_ID="$2"
SPLIT="$3"
ANNOTATION_PATH="$(realpath "$4")"
RUN_DIR="${RELEASE_ROOT}/runs/${RUN_ID}"

if [[ "${SPLIT}" != "val_seen" && "${SPLIT}" != "val_unseen" && "${SPLIT}" != "test" ]]; then
  echo "SPLIT must be val_seen, val_unseen, or test" >&2
  exit 2
fi
if [[ -e "${RUN_DIR}" ]]; then
  echo "Refusing to overwrite existing run: ${RUN_DIR}" >&2
  exit 3
fi

mkdir -p "${RUN_DIR}"
export CUDA_VISIBLE_DEVICES="${GPU_ID}"
export OMP_NUM_THREADS="${DIALNAV_THREADS:-20}"
export MKL_NUM_THREADS="${DIALNAV_THREADS:-20}"
export PYTHONPATH="${MATTERSIM_BUILD}${PYTHONPATH:+:${PYTHONPATH}}"

annotation_args=()
case "${SPLIT}" in
  val_seen) annotation_args+=(--val_seen_anno_paths "${ANNOTATION_PATH}") ;;
  val_unseen) annotation_args+=(--val_unseen_anno_paths "${ANNOTATION_PATH}") ;;
  test) annotation_args+=(--test_anno_paths "${ANNOTATION_PATH}") ;;
esac

cd "${RAINBOW_ROOT}/holistic"
"${PYTHON_BIN}" -u main.py \
  --id "${RUN_ID}" \
  --seed 0 \
  --output_path "${RUN_DIR}" \
  --basepath "${RAINBOW_ROOT}" \
  --connectivity_dir "${RAINBOW_ROOT}/dataset/connectivity" \
  "${annotation_args[@]}" \
  --env_names "${SPLIT}" \
  --batch_size 8 \
  --max_action_len 50 \
  --nav_resume_file "${RAINBOW_ROOT}/dataset/checkpoints/nav_rainbow" \
  --nav_act_visited_nodes \
  --qg_resume_file "${RAINBOW_ROOT}/dataset/checkpoints/q_rainbow" \
  --wta_mode ctc_0.7_2 \
  --ag_resume_file "${RAINBOW_ROOT}/dataset/checkpoints/a_rainbow" \
  --ag_max_answer_seen_path 20 \
  --loc_resume_file "${RAINBOW_ROOT}/dataset/checkpoints/loc_rainbow.pth" \
  --qa_clip_tokenizer_path "${RAINBOW_ROOT}/dataset/modules/clip_tokenizer/bpe_simple_vocab_16e6.txt.gz" \
  --temporal_weight 0.5 \
  --temporal_top_k 5
