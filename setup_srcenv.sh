#source setup_srcenv.sh 

echo "Setting up general"


cd /scr


CONDA_DIR=/scr/anaconda3_ruiqi
if [ ! -f "Anaconda3-2024.10-1-Linux-x86_64.sh" ]; then
   wget https://repo.anaconda.com/archive/Anaconda3-2025.06-0-Linux-x86_64.sh
fi
if [ ! -d "$CONDA_DIR" ]; then
   bash Anaconda3-2025.06-0-Linux-x86_64.sh -b -p ${CONDA_DIR}
   export PATH=${CONDA_DIR}/bin:${PATH}
fi


# Initialize conda
$CONDA_DIR/bin/conda init
source ~/.bashrc


echo "Creating general environment"


conda tos accept --override-channels --channel https://repo.anaconda.com/pkgs/main
conda tos accept --override-channels --channel https://repo.anaconda.com/pkgs/r
if [ ! -d "$CONDA_DIR/envs/general" ]; then
   conda create -n general python=3.10 -y
   conda activate general
else
   conda activate general 
fi


pip install transformers
#pip install -U transformers==4.45.2

#when use KeyeVL
#pip install transformers==4.47.1 -U


pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu130
pip install einops timm
pip install decord
pip install accelerate
pip install av
pip install tiktoken
#pip install --upgrade keye-vl-utils==1.5.2 -i https://pypi.org/simple





mkdir /scr/ruiqi
mkdir /scr/ruiqi/hf_hub_cache
mkdir /scr/ruiqi/hf_home
mkdir /scr/ruiqi/hf_datasets
mkdir /scr/ruiqi/hf_cache


export TRANSFORMERS_CACHE=/scr/ruiqi/hf_cache
export HF_HOME=/gscratch/makelab/ruiqi/hf_home
export HF_DATASETS_CACHE=/scr/ruiqi/hf_datasets
export HF_HUB_CACHE=/scr/ruiqi/hf_hub_cache



echo "Setup done"