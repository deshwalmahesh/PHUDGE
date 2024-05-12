source activate python3
conda create -n train_env python=3.10 -q -y
conda activate train_env
export CUDA_HOME=/usr/local/cuda

pip3 install "torch==2.1.2" -qq
pip3 uninstall -y transformers && pip install git+https://github.com/huggingface/transformers
pip3 install "accelerate==0.27.2" -qq
pip3 install "peft==0.9.0" -qq

pip3 install -U deepspeed huggingface_hub trl bitsandbytes scikit-learn einops datasets evaluate ipykernel pyarrow pandas shortuuid s3fs pandarallel wandb seaborn matplotlib sentencepiece  -qq
pip3 install flash-attn --no-build-isolation -qq

pip3 install -U bs4 lancedb einops -qq

python -m ipykernel install --user --name=train_env --display-name "train_env"

mkdir ~/.cache/huggingface/
mkdir ~/.cache/huggingface/accelerate/
cp ./accelerate_default_config.yaml ~/.cache/huggingface/accelerate/default_config.yaml
