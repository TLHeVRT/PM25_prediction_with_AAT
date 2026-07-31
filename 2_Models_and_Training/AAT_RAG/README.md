# AAT-RAG 实验运行与复现

## 环境

- Ubuntu 22.04，Python 3.12
- PyTorch `2.8.0+cu128`，CUDA Runtime `12.8`，cuDNN `9.10.2.21`
- NumPy `2.3.2`，pandas `3.0.3`，tslearn `0.9.0`

## 统一运行命令

```bash
python run_parallel_pipeline.py --seeds 1 2 3 4 --max-parallel 2 --cpu-threads 8
```

## 复现说明

流程全程固定 Python、NumPy、PyTorch CPU/CUDA 随机种子，并为每个 seed 使用独立的数据缓存、RAG bank、模型和日志目录。时间数据划分完全固定，随机抽样和测试子集由 seed 控制，满足本研究的可复现需求。

本项目不追求逐 bit 一致，因为强制确定性 kernel 会显著降低训练和检索速度。只有主模型训练和验证使用 BF16；RAG、bank 生成、检索和最终测试均使用 FP32，其他可用的 FP32 运算使用 TF32 加速，RAG 检索点积单独保持 FP32。当前同时启用 cuDNN benchmark 和原生 `torch.topk`。即使 seed 相同，不同 GPU、CUDA、cuDNN 或并行 kernel 也可能产生末位浮点差异；这类差异不影响基于四个固定 seed 的研究结果统计与比较。
