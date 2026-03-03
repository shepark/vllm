// docker
nvcr.io/nvidia/pytorch:25.12-py3

docker run -it --rm \
   --gpus all --runtime=nvidia \
   --name peter_ops \
   --net=host --ipc=host --cap-add=sys_nice \
   -e https_proxy=http://proxy-us.intel.com:912 -e http_proxy=http://proxy-us.intel.com:911 -e ftp_proxy=http://proxy-us.intel.com:911 \
   -e socks_proxy=http://proxy-us.intel.com:1080 -e no_proxy=localhost,127.0.0.1,0.0.0.0,artifactory.habana-labs.com \
   -v /home/separk:/work -v /software:/software --entrypoint /bin/bash nvcr.io/nvidia/pytorch:25.12-py3

// setup

// vllm
   1a014a0a9327ed64a1bdec8e1afa43b9ea70a3c1

   pip install pyparsing==3.2.0
   VLLM_USE_PRECOMPILED=1 pip install -e . -v


// server

   export VMODEL=/software/data/pytorch/huggingface/hub/models--Qwen--Qwen3.5-35B-A3B/snapshots/ec2d4ece1ffb563322cbee9a48fe0e3fcbce0307/

// for original triton op
   vllm serve $VMODEL --tensor-parallel-size 2 --dtype bfloat16 --max-model-len 65535 --port 12345

// for torch compile op
   VLLM_Q35_TC_OP=1 vllm serve $VMODEL --tensor-parallel-size 2 --dtype bfloat16 --max-model-len 65535 --port 12345


// client
   python test_video.py

