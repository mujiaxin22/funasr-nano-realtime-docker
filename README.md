# funasr-nano-realtime-docker
funasr-nano-realtime-docker,funasr的实时流版本
先build镜像，以及内置了国内modelscope下载
cd /opt && docker build -t funasr-nano-vllm:local .
1、原版初始版本，官方原版
 docker run -d --name funasr-nano --gpus all --ipc=host   -p 10095:10095   -v /opt/model:/opt/model   funasr-nano-vllm:local   funasr-realtime-server     --port 10095     --gpu-memory-utilization 0.15     --max-model-len 1024



# --max-model-len 1024 默认就是 2048，最大4096，最小1024
# --gpu-memory-utilization 0.15  总显存15%可以改更大改的越大并发就越高，看现场显存剩多少
# --vad-silence-ms 断句静音阈值，固定值（默认 0 = 官方动态表）
# --vad-max-segment-ms 15000 单句说满就强制切（默认 0 = 官方 60 秒）
# --history-max-sentences 1	每条响应最多带几句（默认 0 = 全部）

2、改良版，修改了输出格式，一句话一条输出，不带之前的结果，增加了lang字段，自动识别语言

docker run -d --name funasr-nano --gpus all --ipc=host \
  -p 10095:10095 \
  -v /opt/model:/opt/model \
  funasr-nano-vllm:local \
  funasr-realtime-server --port 10095 --gpu-memory-utilization 0.38 --max-model-len 2048 --partial-window-sec 5 --decode-interval 0.6 \
    --vad-silence-ms 800 --vad-max-segment-ms 15000 --history-max-sentences 1
