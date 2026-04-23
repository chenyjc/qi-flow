"""
所有模型的集成测试
确保从页面进行训练操作不会报错
"""

import pytest
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from services.qlib_service import QlibService
from services.dl_model_service import DLModelService
import torch


# ============== 传统机器学习模型测试 ==============

class TestTraditionalModels:
    """测试传统机器学习模型"""
    
    @pytest.fixture(autouse=True)
    def setup(self):
        self.qs = QlibService()
        if not QlibService._qlib_initialized:
            self.qs.init_qlib()
    
    def test_lightgbm_alpha158(self):
        """测试 LightGBM + Alpha158"""
        result = self._train_model("LGBModel", "Alpha158")
        assert result["success"], f"LightGBM 训练失败: {result.get('message')}"
        print(f"✅ LightGBM + Alpha158 成功，Recorder ID: {result['recorder_id']}")
    
    def test_xgboost_alpha158(self):
        """测试 XGBoost + Alpha158"""
        result = self._train_model("XGBModel", "Alpha158")
        assert result["success"], f"XGBoost 训练失败: {result.get('message')}"
        print(f"✅ XGBoost + Alpha158 成功，Recorder ID: {result['recorder_id']}")
    
    def test_catboost_alpha158(self):
        """测试 CatBoost + Alpha158"""
        result = self._train_model("CatBoostModel", "Alpha158")
        assert result["success"], f"CatBoost 训练失败: {result.get('message')}"
        print(f"✅ CatBoost + Alpha158 成功，Recorder ID: {result['recorder_id']}")
    
    def test_linear_alpha158(self):
        """测试 Linear + Alpha158"""
        result = self._train_model("Linear", "Alpha158")
        assert result["success"], f"Linear 训练失败: {result.get('message')}"
        print(f"✅ Linear + Alpha158 成功，Recorder ID: {result['recorder_id']}")
    
    def test_densemble_alpha158(self):
        """测试 DEnsembleModel + Alpha158"""
        result = self._train_model("DEnsembleModel", "Alpha158")
        assert result["success"], f"DEnsembleModel 训练失败: {result.get('message')}"
        print(f"✅ DEnsembleModel + Alpha158 成功，Recorder ID: {result['recorder_id']}")
    
    def test_lightgbm_alpha360(self):
        """测试 LightGBM + Alpha360"""
        result = self._train_model("LGBModel", "Alpha360")
        assert result["success"], f"LightGBM + Alpha360 训练失败: {result.get('message')}"
        print(f"✅ LightGBM + Alpha360 成功，Recorder ID: {result['recorder_id']}")
    
    def test_lightgbm_alpha158vwap(self):
        """测试 LightGBM + Alpha158vwap"""
        result = self._train_model("LGBModel", "Alpha158vwap")
        assert result["success"], f"LightGBM + Alpha158vwap 训练失败: {result.get('message')}"
        print(f"✅ LightGBM + Alpha158vwap 成功，Recorder ID: {result['recorder_id']}")
    
    def _train_model(self, model_type, handler_type):
        """辅助方法：训练模型"""
        def progress_callback(progress, message):
            if progress % 20 == 0:  # 只打印关键进度
                print(f"[{progress}%] {message}")
        
        return self.qs.train_model_with_progress(
            progress_callback=progress_callback,
            market="csi300",
            benchmark="SH000300",
            train_start_date="2020-01-01",
            train_end_date="2020-03-31",
            valid_start_date="2020-04-01",
            valid_end_date="2020-05-31",
            test_start_date="2020-06-01",
            test_end_date="2020-06-30",
            model_type=model_type,
            lr=0.05,
            max_depth=6,
            num_leaves=100,
            subsample=0.8,
            colsample_bytree=0.8,
            seed=42,
            num_threads=1,
            handler_type=handler_type
        )


# ============== 深度学习模型测试 ==============

class TestDeepLearningModels:
    """测试深度学习模型"""
    
    @pytest.fixture(autouse=True)
    def setup(self):
        self.service = DLModelService()
        if not DLModelService._qlib_initialized:
            self.service.init_qlib()
    
    def test_lstm_alpha158(self):
        """测试 LSTM + Alpha158"""
        result = self._train_dl_model("LSTM", "Alpha158")
        assert result["success"], f"LSTM 训练失败: {result.get('message')}"
        print(f"✅ LSTM + Alpha158 成功，Recorder ID: {result['recorder_id']}")
    
    def test_gru_alpha158(self):
        """测试 GRU + Alpha158"""
        result = self._train_dl_model("GRU", "Alpha158")
        assert result["success"], f"GRU 训练失败: {result.get('message')}"
        print(f"✅ GRU + Alpha158 成功，Recorder ID: {result['recorder_id']}")
    
    def test_alstm_alpha158(self):
        """测试 ALSTM + Alpha158"""
        result = self._train_dl_model("ALSTM", "Alpha158")
        assert result["success"], f"ALSTM 训练失败: {result.get('message')}"
        print(f"✅ ALSTM + Alpha158 成功，Recorder ID: {result['recorder_id']}")
    
    def test_lstm_alpha360(self):
        """测试 LSTM + Alpha360"""
        result = self._train_dl_model("LSTM", "Alpha360")
        assert result["success"], f"LSTM + Alpha360 训练失败: {result.get('message')}"
        print(f"✅ LSTM + Alpha360 成功，Recorder ID: {result['recorder_id']}")
    
    def test_lstm_alpha158vwap(self):
        """测试 LSTM + Alpha158vwap"""
        result = self._train_dl_model("LSTM", "Alpha158vwap")
        assert result["success"], f"LSTM + Alpha158vwap 训练失败: {result.get('message')}"
        print(f"✅ LSTM + Alpha158vwap 成功，Recorder ID: {result['recorder_id']}")
    
    def _train_dl_model(self, model_type, handler_type):
        """辅助方法：训练深度学习模型"""
        def progress_callback(progress, message):
            if progress % 20 == 0:  # 只打印关键进度
                print(f"[{progress}%] {message}")
        
        return self.service.train_dl_model_with_progress(
            progress_callback=progress_callback,
            market="csi300",
            benchmark="SH000300",
            train_start_date="2020-01-01",
            train_end_date="2020-03-31",
            valid_start_date="2020-04-01",
            valid_end_date="2020-05-31",
            test_start_date="2020-06-01",
            test_end_date="2020-06-30",
            model_type=model_type,
            handler_type=handler_type,
            d_feat=20,
            hidden_size=64,
            num_layers=2,
            dropout=0.0,
            lr=0.001,
            n_epochs=3,  # 减少轮数加快测试
            batch_size=800,
            early_stop=3,
            step_len=20,
            seed=42,
            GPU=0 if torch.cuda.is_available() else -1
        )


# ============== GPU 可用性测试 ==============

class TestGPUAvailability:
    """测试 GPU 可用性"""
    
    def test_cuda_available(self):
        """测试 CUDA 是否可用"""
        print(f"PyTorch 版本: {torch.__version__}")
        print(f"CUDA 可用: {torch.cuda.is_available()}")
        
        if torch.cuda.is_available():
            print(f"CUDA 版本: {torch.version.cuda}")
            print(f"GPU 数量: {torch.cuda.device_count()}")
            print(f"GPU 名称: {torch.cuda.get_device_name(0)}")
            print("✅ GPU 可用")
        else:
            print("⚠️  GPU 不可用，将使用 CPU 训练")


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])