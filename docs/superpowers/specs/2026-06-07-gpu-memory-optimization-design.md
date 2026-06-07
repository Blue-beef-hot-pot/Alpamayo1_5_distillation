# GPU 显存优化设计文档

## 概述

为 Alpamayo 1.5 蒸馏训练添加轻量级显存管理，防止显存溢出，留出 5-10GB 缓冲。

## 目标

1. **显存监控**：每 100 步输出显存使用，超过 85% 告警
2. **自动清理**：超过 90% 自动调用 `torch.cuda.empty_cache()`
3. **生命周期优化**：loss 计算后释放 teacher_out
4. **关键节点清理**：checkpoint 保存后、阶段切换时清理缓存

## 设计

### 1. 显存监控器

**文件**：`src/alpamayo1_5_distill/memory_monitor.py`

**类**：`GPUMemoryMonitor`

**功能**：
- `get_stats()`: 获取显存统计（allocated, reserved, total, free）
- `check_and_cleanup(step)`: 检查显存，必要时清理
- 阈值配置：warning_threshold=0.85, cleanup_threshold=0.90

**集成位置**：`train_stage` 函数中，每 100 步调用一次

### 2. Teacher_out 生命周期优化

**文件**：`scripts/train_distill_staged.py`

**修改位置**：`train_stage` 函数的训练循环

**优化逻辑**：
```python
# teacher forward 后
teacher_out = teacher_forward(...)

# student forward + loss 计算
losses = distill_loss(...)

# 提取需要的数据
teacher_sequences = teacher_out.sequences

# 释放 teacher_out
del teacher_out
torch.cuda.empty_cache()

# 继续计算 loss
loss = losses["total"]
loss.backward()
```

**预计节省**：2-6 GB 峰值显存

### 3. 关键节点 empty_cache()

**文件**：`scripts/train_distill_staged.py`

**添加位置**：
1. Checkpoint 保存后（line ~313）
2. Best checkpoint 保存后（line ~327）
3. 阶段切换时（line ~484）

**注意事项**：
- 只在低频操作后调用，不影响训练速度
- 不在每个 batch 调用，避免性能开销

## 预期效果

| GPU | 当前显存 | 优化后显存 | 节省 |
|-----|---------|-----------|------|
| GPU 1 | 75 GB | 65-70 GB | 5-10 GB |
| GPU 2 | 68 GB | 60-65 GB | 3-8 GB |
| GPU 3 | 81 GB | 70-75 GB | 6-11 GB |

## 实现计划

1. 创建 `memory_monitor.py` 文件
2. 修改 `train_distill_staged.py`，集成显存监控器
3. 优化 teacher_out 生命周期
4. 添加关键节点 empty_cache()
5. 测试验证

## 风险评估

- **风险**：`empty_cache()` 有轻微性能开销（~1ms）
- **缓解**：只在低频操作后调用，不影响训练速度
- **验证**：监控训练速度和显存使用
