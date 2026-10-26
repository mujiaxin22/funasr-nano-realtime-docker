FROM vllm/vllm-openai:v0.19.1

ARG PIP_INDEX=https://pypi.tuna.tsinghua.edu.cn/simple

RUN sed -i 's|http://archive.ubuntu.com|https://mirrors.tuna.tsinghua.edu.cn|g; s|http://security.ubuntu.com|https://mirrors.tuna.tsinghua.edu.cn|g' /etc/apt/sources.list \
&& apt-get update && apt-get install -y --no-install-recommends \
ffmpeg libsndfile1 && rm -rf /var/lib/apt/lists/*

ENV PIP_INDEX_URL=${PIP_INDEX} \
PIP_TRUSTED_HOST=pypi.tuna.tsinghua.edu.cn \
PIP_TIMEOUT=120 \
PIP_RETRIES=5

RUN python3 -m pip freeze | grep -iE "^(torch|nvidia-|triton)" > /tmp/c.txt
RUN pip install --no-cache-dir -c /tmp/c.txt \
"funasr>=1.3.26" \
scipy threadpoolctl librosa soundfile \
PyYAML tqdm requests regex "websockets>=10.4" \
"omegaconf>=2.0" "hydra-core>=1.3.2" modelscope huggingface_hub safetensors \
transformers tiktoken sentencepiece "kaldiio>=2.17.0" \
jieba jamo jaconv "rapidfuzz>=3.0.0" torch_complex
RUN pip install --no-cache-dir -c /tmp/c.txt umap_learn oss2 tensorboardX \
|| echo "[warn] 可选依赖安装失败，可忽略"

ENV MODELSCOPE_CACHE=/opt/model \
VLLM_USE_MODELSCOPE=true

EXPOSE 10095

# 必须：清掉基础镜像的 ENTRYPOINT ["vllm","serve"]
ENTRYPOINT []

# ---------------------------------------------------------------------------
# 语种标签 / 2pass 式断句 / 响应瘦身：单文件补丁，不改动 funasr 源码
# ---------------------------------------------------------------------------
COPY lid_server.py /opt/lid_server.py
# 自检脚本，进容器就能验（不需要 GPU）。不想要就删掉这一行
COPY test_lid_server.py /opt/test_lid_server.py

# 顶替官方的 funasr-realtime-server（它等于 funasr.bin.realtime_ws:cli_main），
# 这样 docker run 的命令行和官方写法保持一模一样
RUN cat /opt/lid_server.py > /usr/local/bin/funasr-realtime-server \
 && chmod 0755 /usr/local/bin/funasr-realtime-server

# 下面三个开关都可单独开关，去掉就回到官方行为：
#   --vad-silence-ms 800       断句静音阈值固定成 800ms（官方动态表短句要 2 秒）
#   --vad-max-segment-ms 15000 单句说满 15 秒强制切（官方是 60 秒）
#   --history-max-sentences 1  每条响应只发最近一句（官方是发全部历史）
CMD ["funasr-realtime-server", "--port", "10095", \
     "--vad-silence-ms", "800", \
     "--vad-max-segment-ms", "15000", \
     "--history-max-sentences", "1"]
