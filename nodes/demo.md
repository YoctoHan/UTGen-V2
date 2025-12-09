# 🚀 Run ByteDance Seed-Coder-8B-Base on Ascend via vLLM

本教程展示了如何在 **Ascend NPU** 环境中使用 **vLLM-Ascend** 部署并调用 **ByteDance-Seed/Seed-Coder-8B-Base** 模型。

---

## 🧩 Step 1：拉取 Ascend vLLM 镜像

从 **quay.io** 获取最新版本：

```bash
docker pull quay.io/ascend/vllm-ascend:latest
```

该镜像内置了 vLLM 相关依赖，适配 Huawei Ascend NPU。

---

## 🧭 Step 2：通过 HF-Mirror 下载模型

> 使用 Hugging Face 镜像源，利用 **hfd** 工具快速下载模型。

HF-Mirror 说明引用自 [hf-mirror.com](https://hf-mirror.com)：  
设置 `HF_ENDPOINT=https://hf-mirror.com` 可使 `hfd` 与 `huggingface-cli` 工具在国内稳定连接。

执行命令：

```bash
HF_ENDPOINT="https://hf-mirror.com" \
hfd ByteDance-Seed/Seed-Coder-8B-Base \
--hf_token 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 \
--local-dir /data1/YoctoHan/models/ByteDance-Seed/Seed-Coder-8B-Base
```

### 说明

| 参数 | 含义 |
|------|------|
| `HF_ENDPOINT` | 通过 **HF Mirror** 代理 Hugging Face 域名，加速模型下载 |
| `--hf_token` | 你的个人 Access Token，用于访问 gated 模型 |
| `--local-dir` | 指定保存模型的本地路径 |
| `hfd` | HF-Mirror 提供的高速下载脚本，支持断点续传与高速并发 |

下载完成后模型文件位于：/data1/YoctoHan/models/ByteDance-Seed/Seed-Coder-8B-Base/

---

## 🐳 Step 3：启动 Docker 容器

设定设备环境变量并挂载对应目录：

```bash
export DEVICE=/dev/davinci0
export IMAGE=quay.io/ascend/vllm-ascend:latest

docker run --interactive --detach \
--name vllm-ascend \
--shm-size=1g \
--device $DEVICE \
--device /dev/davinci_manager \
--device /dev/devmm_svm \
--device /dev/hisi_hdc \
-v /usr/local/dcmi:/usr/local/dcmi \
-v /usr/local/bin/npu-smi:/usr/local/bin/npu-smi \
-v /usr/local/Ascend/driver/lib64/:/usr/local/Ascend/driver/lib64/ \
-v /usr/local/Ascend/driver/version.info:/usr/local/Ascend/driver/version.info \
-v /etc/ascend_install.info:/etc/ascend_install.info \
-v /root/.cache:/root/.cache \
-v /data1/YoctoHan/models/ByteDance-Seed/Seed-Coder-8B-Base:/ByteDance-Seed/Seed-Coder-8B-Base \
-p 8000:8000 \
-it $IMAGE bash
```

容器启动成功后会进入交互式终端。

---

## ⚙️ Step 4：启动 vLLM 模型服务

在容器内执行：

```bash
vllm serve /ByteDance-Seed/Seed-Coder-8B-Base/ &
```

该命令会启动 OpenAI API 兼容的 HTTP 服务（默认端口 `8000`）：

http://localhost:8000/v1/completions

---

## 🔮 Step 5：发起推理请求

使用 `curl` 发送 JSON 请求：

```bash
curl http://localhost:8000/v1/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "/ByteDance-Seed/Seed-Coder-8B-Base/",
    "prompt": "<[fim-suffix]>\n# use the function for testing\nif __name__ == \"__main__\":\n    assert add_numbers(2, 3) == 5\n    assert add_numbers(10, -1) == 9\n    print(\"All tests passed!\")\n<[fim-prefix]>\ndef add_numbers(a, b):\n    result = <[fim-middle]>",
    "max_tokens": 64
  }'
```

---

## 🧠 Step 6：理解输入与输出

### 🔹 模型输入 (`prompt`)
```text
<[fim-suffix]>
# use the function for testing
if __name__ == "__main__":
    assert add_numbers(2, 3) == 5
    assert add_numbers(10, -1) == 9
    print("All tests passed!")
<[fim-prefix]>
def add_numbers(a, b):
    result = <[fim-middle]>
```

含义：
- **prefix（前缀）**：函数定义部分。
- **suffix（后缀）**：单元测试断言，清楚表明期望输出行为（加法）。
- **middle（中间段）**：模型需要补全的逻辑区域。


---

### 🔹 模型输出 (`response`)
示例输出结果：

```json
{
  "choices": [
    {
      "text": "a + b\n    return result\n"
    }
  ]
}
```

最终生成的完整 Python 代码：

```python
def add_numbers(a, b):
    result = a + b
    return result

if __name__ == "__main__":
    assert add_numbers(2, 3) == 5
    assert add_numbers(10, -1) == 9
    print("All tests passed!")
```

---

## 📊 Step 7：结果分析

| 指标 | 值 | 意义 |
|------|----|------|
| `finish_reason` | `"stop"` | 模型自然停止输出 |
| `completion_tokens` | 9 | 输出简洁精确 |
| 输出逻辑 | ✅ | 成功补全正确的加法逻辑 |
| 模型状态 | ✅ | 响应正常，HTTP 200 返回 |

生成的代码通过了输入断言测试，证明模型在上下文推理下成功理解 “a + b” 的语义。

---

## ✅ 总结

本流程实现内容：

1. 使用 `HF-Mirror` 加速下载 Hugging Face 模型。  
2. 在 Ascend NPU 上通过 `vLLM` 快速部署。  
3. 成功发起推理，验证模型生成逻辑正确。  

👉 适用于在本地或算力平台上离线推理大型开源模型的场景。

---

**作者**：YoctoHan  
**联系方式**: YoctoInch@gmail.com  
**环境**：Python 3.11 + Ascend NPU + vLLM-Ascend   