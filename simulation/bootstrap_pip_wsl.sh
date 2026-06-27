#!/usr/bin/env bash
set -euo pipefail
cd /tmp
curl -fsSL https://bootstrap.pypa.io/get-pip.py -o get-pip.py
python3 get-pip.py --user
export PATH="${HOME}/.local/bin:${PATH}"
python3 -m pip install --user "coppeliasim-zmqremoteapi-client>=2.0.4" numpy pyyaml
python3 -c "import coppeliasim_zmqremoteapi_client, numpy, yaml; print('pip ok')"
