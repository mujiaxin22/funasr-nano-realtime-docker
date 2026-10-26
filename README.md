# funasr-nano-realtime-docker

FunASR 的实时流式版本 Docker 部署。

## 构建镜像

镜像内置了国内 ModelScope 下载源：

```bash
cd /opt && docker build -t funasr-nano-vllm:local .
```

## 运行

### 1. 原版（官方原版）

```bash
docker run -d --name funasr-nano --gpus all --ipc=host \
  -p 10095:10095 \
  -v /opt/model:/opt/model \
  funasr-nano-vllm:local \
  funasr-realtime-server \
    --port 10095 \
    --gpu-memory-utilization 0.15 \
    --max-model-len 1024
```

### 2. 改良版

修改了输出格式：一句话一条输出、不携带之前的结果，新增 `lang` 字段自动识别语言。

```bash
docker run -d --name funasr-nano --gpus all --ipc=host \
  -p 10095:10095 \
  -v /opt/model:/opt/model \
  funasr-nano-vllm:local \
  funasr-realtime-server \
    --port 10095 \
    --gpu-memory-utilization 0.38 \
    --max-model-len 2048 \
    --partial-window-sec 5 \
    --decode-interval 0.6 \
    --vad-silence-ms 800 \
    --vad-max-segment-ms 15000 \
    --history-max-sentences 1
```

## 参数说明（其余的参数看funasr官方https://github.com/modelscope/FunASR/blob/main/docs/vllm_guide.md）

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `--max-model-len` | 2048 | 模型最大上下文长度。最小 1024，最大 4096 |
| `--gpu-memory-utilization` | - | 占用总显存的比例，如 `0.15` 表示 15%。调得越大并发越高，根据现场剩余显存决定 |
| `--vad-silence-ms` | 0 | 断句静音阈值（毫秒），固定值。`0` 表示使用官方动态表 |
| `--vad-max-segment-ms` | 0 | 单句说满即强制切分（毫秒）。`0` 表示使用官方的 60 秒 |
| `--history-max-sentences` | 0 | 每条响应最多携带几句结果。`0` 表示携带全部 |
