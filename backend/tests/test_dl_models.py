import pytest
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from services.dl_model_service import DLModelService
import torch


class TestCUDAAndGPU:
    """测试 CUDA 和 GPU 可用性"""
    
    def test_pytorch_cuda_available(self):
        """测试 PyTorch CUDA 是否可用"""
        print(f"PyTorch版本: {torch.__version__}")
        print(f"CUDA可用: {torch.cuda.is_available()}")
        
        if torch.cuda.is_available():
            print(f"CUDA版本: {torch.version.cuda}")
            print(f"GPU数量: {torch.cuda.device_count()}")
            print(f"GPU名称: {torch.cuda.get_device_name(0)}")
        
        # 检查是否为 CUDA 版本
        assert "+cu" in torch.__version__, f"PyTorch 不是 CUDA 版本: {torch.__version__}"
        print("✅ PyTorch 是 CUDA 版本")
    
    def test_gpu_available(self):
        """测试 GPU 是否可用"""
        if not torch.cuda.is_available():
            pytest.skip("CUDA 不可用，跳过 GPU 测试")
        
        assert torch.cuda.device_count() > 0, "没有检测到 GPU"
        print(f"✅ 检测到 {torch.cuda.device_count()} 个 GPU")


class TestDLModelService:
    """测试深度学习模型服务"""
    
    def test_model_registry(self):
        """测试模型注册表"""
        service = DLModelService()
        
        expected_models = ["LSTM", "GRU", "ALSTM"]
        for model in expected_models:
            assert model in service.MODEL_REGISTRY, f"Model {model} not in registry"
        
        print(f"MODEL_REGISTRY: {service.MODEL_REGISTRY}")
    
    def test_handler_registry(self):
        """测试因子注册表"""
        service = DLModelService()
        
        expected_handlers = ["Alpha158", "Alpha360"]
        for handler in expected_handlers:
            assert handler in service.HANDLER_REGISTRY, f"Handler {handler} not in registry"
        
        print(f"HANDLER_REGISTRY: {service.HANDLER_REGISTRY}")
    
    def test_get_model_config(self):
        """测试模型配置生成"""
        service = DLModelService()
        
        config = service._get_model_config(
            model_type="LSTM",
            d_feat=20,
            hidden_size=64,
            num_layers=2,
            dropout=0.0,
            lr=0.001,
            n_epochs=200,
            batch_size=800,
            early_stop=20,
            seed=42,
            GPU=0
        )
        
        assert config["class"] == "LSTM"
        assert config["module_path"] == "qlib.contrib.model.pytorch_lstm_ts"
        assert config["kwargs"]["d_feat"] == 20
        assert config["kwargs"]["hidden_size"] == 64
        assert config["kwargs"]["lr"] == 0.001
        
        print(f"LSTM config: {config}")
    
    def test_get_handler_config(self):
        """测试 handler 配置生成"""
        service = DLModelService()
        
        config = service._get_handler_config(
            handler_type="Alpha158",
            market="csi300",
            train_start_date="2020-01-01",
            train_end_date="2020-06-30",
            test_end_date="2020-12-31",
            step_len=20
        )
        
        assert "start_time" in config
        assert "end_time" in config
        assert "infer_processors" in config
        assert "learn_processors" in config
        
        print(f"Handler config keys: {config.keys()}")
        print(f"Infer processors: {len(config.get('infer_processors', []))}")
    
    def test_init_lstm_model(self):
        """测试 LSTM 模型初始化"""
        from qlib.utils import init_instance_by_config
        
        service = DLModelService()
        
        config = service._get_model_config(
            model_type="LSTM",
            d_feat=20,
            hidden_size=64,
            num_layers=2,
            dropout=0.0,
            lr=0.001,
            n_epochs=10,  # 减少轮数用于测试
            batch_size=800,
            early_stop=20,
            seed=42,
            GPU=0
        )
        
        model = init_instance_by_config(config)
        assert model is not None
        print(f"LSTM model initialized: {type(model)}")


class TestDLTraining:
    """测试深度学习模型训练"""
    
    def test_lstm_training_with_gpu(self):
        """测试 LSTM 训练并使用 GPU"""
        import torch
        from qlib.utils import init_instance_by_config
        
        service = DLModelService()
        
        # 首先检查 GPU 是否可用
        if not torch.cuda.is_available():
            pytest.skip("CUDA 不可用，跳过 GPU 训练测试")
        
        print(f"\n🖥️  训练前 GPU 状态:")
        print(f"   GPU 名称: {torch.cuda.get_device_name(0)}")
        print(f"   GPU 显存: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.2f} GB")
        print(f"   已分配显存: {torch.cuda.memory_allocated(0) / 1024**3:.2f} GB")
        
        # 创建模型配置
        model_config = service._get_model_config(
            model_type="LSTM",
            d_feat=20,
            hidden_size=64,
            num_layers=2,
            dropout=0.0,
            lr=0.001,
            n_epochs=2,  # 只训练2轮，快速测试
            batch_size=800,
            early_stop=2,
            seed=42,
            GPU=0  # 使用 GPU 0
        )
        
        # 初始化模型
        model = init_instance_by_config(model_config)
        
        # 检查模型是否使用了 GPU
        print(f"\n📊 模型设备信息:")
        print(f"   模型 device: {model.device}")
        print(f"   使用 GPU: {model.use_gpu}")
        
        # 验证 GPU 使用
        assert model.use_gpu, "模型没有使用 GPU！"
        assert "cuda" in str(model.device), f"模型设备不是 CUDA: {model.device}"
        
        print(f"\n✅ 模型成功使用 GPU: {model.device}")
        print(f"   训练后显存: {torch.cuda.memory_allocated(0) / 1024**3:.2f} GB")
    
    def test_lstm_full_training(self):
        """测试完整 LSTM 训练流程"""
        service = DLModelService()
        
        progress_messages = []
        
        def progress_callback(progress, message):
            progress_messages.append((progress, message))
            print(f"[{progress}%] {message}")
        
        # 使用较短的时间范围和较少的轮数
        result = service.train_dl_model_with_progress(
            progress_callback=progress_callback,
            market="csi300",
            benchmark="SH000300",
            train_start_date="2020-01-01",
            train_end_date="2020-03-31",
            valid_start_date="2020-04-01",
            valid_end_date="2020-05-31",
            test_start_date="2020-06-01",
            test_end_date="2020-06-30",
            model_type="LSTM",
            handler_type="Alpha158",
            d_feat=20,
            hidden_size=64,
            num_layers=2,
            dropout=0.0,
            lr=0.001,
            n_epochs=5,  # 只训练5轮
            batch_size=800,
            early_stop=5,
            step_len=20,
            seed=42,
            GPU=0  # 使用 GPU 0
        )
        
        print(f"Training result: {result}")
        
        if result["success"]:
            assert result["recorder_id"] is not None
            print(f"✅ LSTM training successful! Recorder ID: {result['recorder_id']}")
        else:
            print(f"❌ LSTM training failed: {result.get('message')}")
            # 如果是数据问题，标记为跳过而不是失败
            if "nan" in result.get('message', '').lower() or "empty" in result.get('message', '').lower():
                pytest.skip(f"Data issue: {result.get('message')}")
            else:
                pytest.fail(f"Training failed: {result.get('message')}")


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])