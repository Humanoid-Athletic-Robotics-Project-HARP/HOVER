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
set -e

# We allow this to fail since in docker we only have the files without the git
# info.
git submodule update --init --recursive || true

echo "Resetting changes in third_party/human2humanoid..."
pushd third_party/human2humanoid
git reset --hard || true
popd

# Apply patch to files
if patch --dry-run --silent -f third_party/human2humanoid/phc/phc/utils/torch_utils.py < third_party/phc_torch_utils.patch; then
    echo "Dry run succeeded. Applying the patch..."
    patch third_party/human2humanoid/phc/phc/utils/torch_utils.py < third_party/phc_torch_utils.patch
    echo "Patch applied successfully."
else
    echo "Dry run failed. Patch was not applied. Check if the patch file contains errors."
    exit 1
fi

# Accept Isaac Sim EULA non-interactively (pip-based install, no isaaclab.sh).
EULA_FILE="$(python3 -c "import os,isaacsim.kit.kit_app as k; print(os.path.join(os.path.dirname(k.__file__), 'EULA_ACCEPTED'))" 2>/dev/null || true)"
if [ -n "$EULA_FILE" ] && [ ! -f "$EULA_FILE" ]; then
    echo "Accepting Isaac Sim EULA..."
    mkdir -p "$(dirname "$EULA_FILE")" && echo "Yes" > "$EULA_FILE"
fi

# Install libraries (pip-based IsaacLab — no isaaclab.sh wrapper needed).
pip3 install --upgrade pip wheel
pip3 install -e .
# chumpy uses a legacy build system that calls pip internally; --no-build-isolation avoids that.
pip3 install chumpy --no-build-isolation --root-user-action=ignore
# Patch chumpy for Python 3.11+ compatibility:
#   ch.py:      inspect.getargspec removed → getfullargspec
#   __init__.py: numpy removed bool/int/float/... aliases
CHUMPY_DIR=$(pip3 show chumpy 2>/dev/null | awk '/^Location:/{print $2"/chumpy"}')
if [ -n "$CHUMPY_DIR" ] && [ -d "$CHUMPY_DIR" ]; then
    sed -i 's/inspect\.getargspec/inspect.getfullargspec/g' "$CHUMPY_DIR/ch.py"
    sed -i 's/from numpy import bool, int, float, complex, object, unicode, str, nan, inf/from numpy import nan, inf/g' "$CHUMPY_DIR/__init__.py"
fi
pip3 install -r requirements.txt --root-user-action=ignore

# Sync AMASS and SMPL data from shared /data if not already in place.
command -v rsync &>/dev/null || apt-get install -y rsync
AMASS_DEST="third_party/human2humanoid/data/AMASS/AMASS_Complete"
SMPL_DEST="third_party/human2humanoid/data/smpl"

if [ -f "/data/CMU.tar.bz2" ]; then
    if ! find "$AMASS_DEST" -name "*.npz" 2>/dev/null | grep -q . && \
       [ ! -f "$AMASS_DEST/CMU.tar.bz2" ]; then
        echo "Syncing CMU.tar.bz2 from /data..."
        mkdir -p "$AMASS_DEST"
        rsync -ah --progress /data/CMU.tar.bz2 "$AMASS_DEST/"
    fi
fi

if [ -f "/data/SMPL_python_v.1.1.0.zip" ]; then
    if [ ! -f "$SMPL_DEST/SMPL_FEMALE.pkl" ] && \
       [ ! -f "$SMPL_DEST/SMPL_python_v.1.1.0.zip" ]; then
        echo "Syncing SMPL_python_v.1.1.0.zip from /data..."
        mkdir -p "$SMPL_DEST"
        rsync -ah --progress /data/SMPL_python_v.1.1.0.zip "$SMPL_DEST/"
    fi
fi

# Install git hooks so post-push auto-bumps the submodule pointer in testing-grounds.
GIT_HOOKS_DIR="$(git rev-parse --git-dir 2>/dev/null)/hooks" || true
if [ -n "$GIT_HOOKS_DIR" ]; then
    ln -sf "$(pwd)/.githooks/post-push" "${GIT_HOOKS_DIR}/post-push"
    echo "Installed post-push hook."
fi
