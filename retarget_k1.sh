#!/bin/bash
#
# SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Retarget AMASS dataset to Booster K1 robot.
# Analogous to retarget_h1.sh but for the K1 skeleton.

HUMAN2HUMANOID_DIR="third_party/human2humanoid"
AMASS_DIR="$HUMAN2HUMANOID_DIR/data/AMASS/AMASS_Complete"
SMPL_DIR="$HUMAN2HUMANOID_DIR/data/smpl"
SMPL_MODEL_DIR="$SMPL_DIR/SMPL_python_v.1.1.0/smpl/models"
AMASS_FILTERED_DIR="$HUMAN2HUMANOID_DIR/data/AMASS/AMASS_Filtered"

check_amass() {
    if [ -d "$AMASS_DIR" ]; then
        if find "$AMASS_DIR" -type f -name "*.npz" | grep -q . ; then
            echo "AMASS dataset is already extracted and ready."
        elif compgen -G "$AMASS_DIR/*.tar.bz2" > /dev/null || compgen -G "$AMASS_DIR/*.zip" > /dev/null; then
            echo "Extracting compressed files..."
            find "$AMASS_DIR" -name "*.tar.bz2" -exec tar -xvjf {} -C "$AMASS_DIR" \;
            find "$AMASS_DIR" -name "*.zip" -exec unzip -o {} -d "$AMASS_DIR" \;
        else
            echo "Please download the AMASS dataset in the 'SMPL + H G' format from https://amass.is.tue.mpg.de/index.html and place it under $AMASS_DIR"
            exit 1
        fi
    else
        echo "$AMASS_DIR folder does not exist. Please create it and download the AMASS dataset."
        exit 1
    fi
}

check_files_exist() {
    for file in "$@"; do
        if [ ! -f "$file" ]; then
            return 1
        fi
    done
    return 0
}

check_smpl() {
    FEMALE_MODEL="$SMPL_MODEL_DIR/basicmodel_f_lbs_10_207_0_v1.1.0.pkl"
    MALE_MODEL="$SMPL_MODEL_DIR/basicmodel_m_lbs_10_207_0_v1.1.0.pkl"
    NEUTRAL_MODEL="$SMPL_MODEL_DIR/basicmodel_neutral_lbs_10_207_0_v1.1.0.pkl"
    MODELS=($FEMALE_MODEL $MALE_MODEL $NEUTRAL_MODEL)

    RENAMED_FEMALE_MODEL="$SMPL_DIR/SMPL_FEMALE.pkl"
    RENAMED_MALE_MODEL="$SMPL_DIR/SMPL_MALE.pkl"
    RENAMED_NEUTRAL_MODEL="$SMPL_DIR/SMPL_NEUTRAL.pkl"
    RENAMED_MODELS=($RENAMED_FEMALE_MODEL $RENAMED_MALE_MODEL $RENAMED_NEUTRAL_MODEL)

    if [ -d "$SMPL_DIR" ]; then
        if check_files_exist "${RENAMED_MODELS[@]}"; then
            echo "SMPL files are already available."
        else
            if [ -f "$SMPL_DIR/SMPL_python_v.1.1.0.zip" ]; then
                echo "Extracting SMPL_python_v.1.1.0.zip..."
                mkdir -p "$SMPL_MODEL_DIR"
                unzip -o "$SMPL_DIR/SMPL_python_v.1.1.0.zip" -d "$SMPL_DIR"

                if check_files_exist "${MODELS[@]}"; then
                    echo "Renaming SMPL model files..."
                    cp "$FEMALE_MODEL" "$RENAMED_FEMALE_MODEL"
                    cp "$MALE_MODEL" "$RENAMED_MALE_MODEL"
                    cp "$NEUTRAL_MODEL" "$RENAMED_NEUTRAL_MODEL"
                else
                    echo "Error: Required SMPL model files not found after extraction"
                    exit 1
                fi
            else
                echo "Please download SMPL files from https://download.is.tue.mpg.de/download.php?domain=smpl&sfile=SMPL_python_v.1.1.0.zip and place under $SMPL_DIR"
                exit 1
            fi
        fi
    else
        echo "$SMPL_DIR folder does not exist."
        exit 1
    fi
}

prepare_filtered_motions() {
    local yaml_file=$1
    if [ -f "$yaml_file" ]; then
        if [ -d "$AMASS_FILTERED_DIR" ] && [ "$(ls -A $AMASS_FILTERED_DIR)" ]; then
            read -p "Filtered motion files already exist. Keep them? [y/N] " response
            if [[ ! $response =~ ^[Yy]$ ]]; then
                rm -rf "$AMASS_FILTERED_DIR"/*
            else
                return
            fi
        fi

        echo "Preparing filtered motions from $yaml_file..."
        mkdir -p "$AMASS_FILTERED_DIR"

        while IFS= read -r line; do
            if [[ $line =~ \"(.*)\" ]]; then
                pattern="${BASH_REMATCH[1]}"
                if [[ $pattern == *"*"* ]]; then
                    while IFS= read -r source_file; do
                        relative_path="${source_file#$AMASS_DIR/}"
                        target_dir="$AMASS_FILTERED_DIR/$(dirname "$relative_path")"
                        mkdir -p "$target_dir"
                        if [ -f "$source_file" ]; then
                            ln -s "$(realpath "$source_file")" "$target_dir/$(basename "$source_file")"
                        fi
                    done < <(find "$AMASS_DIR" -path "$AMASS_DIR/$pattern")
                else
                    source_file="$AMASS_DIR/$pattern"
                    relative_path="${pattern}"
                    target_dir="$AMASS_FILTERED_DIR/$(dirname "$relative_path")"
                    mkdir -p "$target_dir"
                    if [ -f "$source_file" ]; then
                        ln -s "$(realpath "$source_file")" "$target_dir/$(basename "$source_file")"
                    fi
                fi
            fi
        done < <(grep -o '"[^"]*"' "$yaml_file")
    fi
}

retarget() {
    echo "Installing retargeting dependencies (offline, no GitHub)..."
    pip3 install smplx easydict --no-deps --quiet || true
    pip3 install -e smpl_sim_shim --no-deps --quiet || true
    pip3 install -e phc --no-deps --quiet || true
    python3 -c "import smpl_sim, smplx, phc, easydict" || { echo "Critical retargeting packages missing. Aborting."; return 1; }

    # Step 1: Fit SMPL shape to K1 proportions
    echo "Running grad_fit_k1_shape.py..."
    python3 scripts/data_process/grad_fit_k1_shape.py || return 1

    # Step 2: Retarget AMASS to K1
    local amass_dir="data/AMASS/AMASS_Complete"
    local filtered_amass_dir="data/AMASS/AMASS_Filtered"
    if [ -d "$filtered_amass_dir" ] && [ "$(ls -A $filtered_amass_dir)" ]; then
        amass_dir="$filtered_amass_dir"
        echo "Using filtered motion files from $filtered_amass_dir"
    fi

    echo "Running grad_fit_k1.py on $amass_dir..."
    python3 scripts/data_process/grad_fit_k1.py --amass_root "$amass_dir" || return 1

    return 0
}

print_usage() {
    echo "Usage: $0 [OPTIONS]"
    echo "Retarget AMASS dataset to Booster K1 robot."
    echo ""
    echo "Options:"
    echo "  --motions-file FILE    Specify a YAML file with motion files to process"
    echo ""
    echo "Example:"
    echo "  $0 --motions-file motions.yaml"
}

YAML_FILE=""
while [[ $# -gt 0 ]]; do
    case $1 in
        --motions-file)
            YAML_FILE="$2"
            shift 2
            ;;
        -h|--help)
            print_usage
            exit 0
            ;;
        *)
            echo "Unknown option: $1"
            print_usage
            exit 1
            ;;
    esac
done

check_amass
check_smpl

if [ -n "$YAML_FILE" ]; then
    prepare_filtered_motions "$YAML_FILE"
fi

echo "Move to $HUMAN2HUMANOID_DIR"
pushd $HUMAN2HUMANOID_DIR

retarget
if [ $? -ne 0 ]; then
    echo "Motion retargeting failed."
else
    echo "Motion retargeting finished. Find the retargeted dataset at $HUMAN2HUMANOID_DIR/data/k1/amass_all.pkl."
fi

popd
