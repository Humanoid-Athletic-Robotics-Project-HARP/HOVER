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
    echo "Yes" > "$EULA_FILE"
fi

# Install libraries (pip-based IsaacLab — no isaaclab.sh wrapper needed).
pip3 install --upgrade pip wheel
pip3 install -e .
pip3 install -r requirements.txt --root-user-action=ignore
