# setup_env.sh
#!/bin/bash
# Usage: 
# $ source ./setup_env.sh

python3 -m venv venv
source venv/bin/activate
pip install tensorflow
pip install matplotlib
pip install librosa
pip install numpy
pip install tqdm
pip install lyon