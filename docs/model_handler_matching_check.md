# Qlib 模型与因子匹配逻辑检查

## 1. MODEL_REGISTRY (模型注册表)

| 模型名称 | 模块路径 | 类型 | 说明 |
|---------|---------|------|------|
| LGBModel | qlib.contrib.model.gbdt | 传统机器学习 | LightGBM |
| XGBModel | qlib.contrib.model.xgboost | 传统机器学习 | XGBoost |
| CatBoostModel | qlib.contrib.model.catboost_model | 传统机器学习 | CatBoost |
| Linear | qlib.contrib.model.linear | 传统机器学习 | 线性回归 |
| DEnsembleModel | qlib.contrib.model.double_ensemble | 传统机器学习 | 双重集成 |
| LSTM | qlib.contrib.model.pytorch_lstm | 深度学习 | 长短期记忆网络 |
| GRU | qlib.contrib.model.pytorch_gru | 深度学习 | 门控循环单元 |
| ALSTM | qlib.contrib.model.pytorch_alstm | 深度学习 | 注意力LSTM |
| DNN | qlib.contrib.model.pytorch_nn | 深度学习 | 全连接神经网络 |
| GATs | qlib.contrib.model.pytorch_gats | 深度学习 | 图注意力网络 |
| TCN | qlib.contrib.model.pytorch_tcn | 深度学习 | 时间卷积网络 |
| SFM | qlib.contrib.model.pytorch_sfm | 深度学习 | 状态频率模型 |
| TabnetModel | qlib.contrib.model.pytorch_tabnet | 深度学习 | TabNet |

## 2. HANDLER_REGISTRY (因子注册表)

| 因子名称 | 模块路径 | 基类 | 说明 |
|---------|---------|------|------|
| Alpha158 | qlib.contrib.data.handler | DataHandlerLP | 158个经典因子 |
| Alpha360 | qlib.contrib.data.handler | DataHandlerLP | 360个扩展因子 |
| Alpha158DL | qlib.contrib.data.loader | QlibDataLoader | 158因子DL版本 |
| Alpha360DL | qlib.contrib.data.loader | QlibDataLoader | 360因子DL版本 |
| Alpha158vwap | qlib.contrib.data.handler | DataHandlerLP | 158因子VWAP版本 |
| Alpha360vwap | qlib.contrib.data.handler | DataHandlerLP | 360因子VWAP版本 |

## 3. 关键区别

### DataHandlerLP vs QlibDataLoader

| 特性 | DataHandlerLP | QlibDataLoader |
|------|---------------|----------------|
| 参数 | 接受 start_time, end_time, fit_start_time, fit_end_time, instruments | 只接受 config |
| 使用场景 | 传统机器学习模型 | 深度学习模型 |
| DatasetH 兼容性 | ✅ 兼容 | ❌ 不兼容 |

## 4. 当前匹配逻辑

### 4.1 模型配置 (_get_model_config)

```python
# 传统机器学习模型 (LGBModel, XGBModel, CatBoostModel, Linear, DEnsembleModel)
- 使用树模型参数: max_depth, num_leaves, subsample, colsample_bytree
- 或使用线性模型参数

# 深度学习模型 (LSTM, GRU, ALSTM, DNN, GATs, TCN, SFM)
- d_feat: 158 (固定，对应 Alpha158)
- hidden_size: 64
- num_layers: 2
- n_epochs: 200
- batch_size: 800

# TabnetModel (特殊)
- 单独配置，没有 d_feat
```

### 4.2 因子转换逻辑 (train_model_with_progress)

```python
dl_to_normal_map = {
    "Alpha158DL": "Alpha158",
    "Alpha360DL": "Alpha360",
}
```

**问题**: 如果用户选择 Alpha360，但模型配置中 d_feat 固定为 158，这会导致不匹配！

## 5. 发现的问题

### 问题 1: d_feat 固定为 158
**位置**: `_get_model_config` 中的 pytorch_models

**当前代码**:
```python
"d_feat": 158,  # 固定值
```

**问题**: 如果 handler_type 是 Alpha360，应该有 360 个特征，但 d_feat 固定为 158

**修复方案**:
```python
# 需要根据 handler_type 动态设置 d_feat
d_feat = 158 if "Alpha158" in handler_type else 360 if "Alpha360" in handler_type else 158
```

### 问题 2: 缺少 vwap 因子转换
**位置**: `dl_to_normal_map`

**当前**:
```python
dl_to_normal_map = {
    "Alpha158DL": "Alpha158",
    "Alpha360DL": "Alpha360",
}
```

**问题**: 如果用户选择 Alpha158vwap 或 Alpha360vwap，这些是正常的 DataHandlerLP，不需要转换

**状态**: ✅ 正确，vwap 版本已经是 DataHandlerLP

### 问题 3: 模块路径不匹配
**位置**: HANDLER_REGISTRY

| 因子 | 当前模块路径 | 正确模块路径 |
|------|-------------|-------------|
| Alpha158DL | qlib.contrib.data.loader | ✅ 正确 |
| Alpha360DL | qlib.contrib.data.loader | ✅ 正确 |

**状态**: ✅ 正确

## 6. 修复建议

### 修复 1: 动态设置 d_feat

修改 `_get_model_config` 方法，添加 handler_type 参数：

```python
def _get_model_config(self, model_type, lr, max_depth, num_leaves, subsample, colsample_bytree, seed, num_threads, handler_type="Alpha158"):
    # ...
    pytorch_models = ["LSTM", "GRU", "ALSTM", "DNN", "GATs", "TCN", "SFM"]
    if model_type in pytorch_models:
        # 根据因子类型设置 d_feat
        if "Alpha360" in handler_type:
            d_feat = 360
        elif "Alpha158" in handler_type:
            d_feat = 158
        else:
            d_feat = 158  # 默认值
        
        return {
            "class": model_type,
            "module_path": module_path,
            "kwargs": {
                "d_feat": d_feat,
                # ...
            },
        }
```

然后在 `train_model_with_progress` 中调用时传入 handler_type：

```python
model_config = self._get_model_config(
    model_type, lr, max_depth, num_leaves, subsample, colsample_bytree, 
    seed, num_threads, actual_handler_type  # 传入 handler_type
)
```

## 7. 测试建议

1. 测试 Alpha158 + LightGBM
2. 测试 Alpha158 + ALSTM
3. 测试 Alpha360 + LightGBM
4. 测试 Alpha360 + ALSTM
5. 测试 Alpha158vwap + LightGBM
6. 测试 Alpha158DL (应该被转换为 Alpha158)

## 8. 总结

当前代码的主要问题是：
- ✅ 因子转换逻辑正确
- ✅ 模块路径配置正确
- ❌ **d_feat 固定为 158，不适用于 Alpha360**
- ❌ **_get_model_config 没有接收 handler_type 参数**

需要修复 d_feat 的动态设置。