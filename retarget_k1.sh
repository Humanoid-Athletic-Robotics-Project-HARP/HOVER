#!/bin/bash
#
# SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
GRAD_FIT_SCRIPT="$SCRIPT_DIR/scripts/data_process/grad_fit_k1.py"
H2H_DIR="$SCRIPT_DIR/third_party/human2humanoid"
AMASS_DIR="$H2H_DIR/data/AMASS/AMASS_Complete"

print_usage() {
    echo "Usage: $0 --amass-file <path.npz> [OPTIONS]"
    echo "       $0 --motions-file <file.yaml> [OPTIONS]"
    echo "Retarget AMASS motion clips to the Booster K1 robot."
    echo ""
    echo "AMASS and SMPL data must already be set up by install_deps.sh."
    echo ""
    echo "Options:"
    echo "  --amass-file FILE    Clip path relative to AMASS_Complete (e.g. CMU/13/13_17_poses.npz)"
    echo "  --motions-file FILE  YAML listing clips to retarget in batch (e.g. cmu_punch.yaml)"
    echo "  --save-dir DIR       Output directory for .pkl files (default: data/k1/)"
    echo ""
    echo "Examples:"
    echo "  $0 --amass-file CMU/13/13_17_poses.npz"
    echo "  $0 --motions-file cmu_punch.yaml --save-dir data/k1/"
}

AMASS_FILE=""
MOTIONS_YAML=""
SAVE_DIR="$SCRIPT_DIR/data/k1"

while [[ $# -gt 0 ]]; do
    case $1 in
        --amass-file)   AMASS_FILE="$2";   shift 2 ;;
        --motions-file) MOTIONS_YAML="$2"; shift 2 ;;
        --save-dir)     SAVE_DIR="$2";     shift 2 ;;
        -h|--help)      print_usage; exit 0 ;;
        *)              echo "Unknown option: $1"; print_usage; exit 1 ;;
    esac
done

if [ -z "$AMASS_FILE" ] && [ -z "$MOTIONS_YAML" ]; then
    echo "Error: provide --amass-file or --motions-file."
    print_usage
    exit 1
fi

mkdir -p "$SAVE_DIR"

if [ -n "$AMASS_FILE" ]; then
    _stem=$(echo "$AMASS_FILE" | tr '/' '_' | sed 's/\.npz$//')
    _out="$SAVE_DIR/${_stem}.pkl"
    echo ">>> $AMASS_FILE → $_out"
    python3 "$GRAD_FIT_SCRIPT" \
        --amass_file  "$AMASS_FILE" \
        --amass_root  "$AMASS_DIR" \
        --save_path   "$_out"
    exit 0
fi

if [ ! -f "$MOTIONS_YAML" ]; then
    echo "Motions file not found: $MOTIONS_YAML"
    exit 1
fi

_yaml_stem=$(basename "$MOTIONS_YAML" .yaml)
_combined_pkl="$SAVE_DIR/${_yaml_stem}.pkl"

python3 "$GRAD_FIT_SCRIPT" \
    --motions_file "$MOTIONS_YAML" \
    --amass_root   "$AMASS_DIR" \
    --save_path    "$_combined_pkl"

echo "All done. Output: $_combined_pkl"
