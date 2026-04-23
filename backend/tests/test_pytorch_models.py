import pytest
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from services.qlib_service import QlibService


class TestPytorchModels:
    """测试 PyTorch 深度学习模型"""
    
    def test_lstm_alpha158(self):
        """测试 LSTM 模型"""
        qs = QlibService()
        
        def progress_callback(progress, message):
            print(f"[{progress}%] {message}")
        
        result = qs.train_model_with_progress(
            progress_callback=progress_callback,
            market="csi300",
            benchmark="SH000300",
            train_start_date="2018-01-01",
            train_end_date="2019-12-31",
            valid_start_date="2020-01-01",
            valid_end_date="2020-06-30",
            test_start_date="2020-07-01",
            test_end_date="2020-12-31",
            model_type="LSTM",
            lr=0.001,
            max_depth=6,
            num_leaves=100,
            subsample=0.8,
            colsample_bytree=0.8,
            seed=42,
            num_threads=1,
            handler_type="Alpha158"
        )
        
        print(f"LSTM result: {result}")
        assert result["success"] == True, f"LSTM failed: {result.get('message')}"
    
    def test_gru_alpha158(self):
        """测试 GRU 模型"""
        qs = QlibService()
        
        def progress_callback(progress, message):
            print(f"[{progress}%] {message}")
        
        result = qs.train_model_with_progress(
            progress_callback=progress_callback,
            market="csi300",
            benchmark="SH000300",
            train_start_date="2018-01-01",
            train_end_date="2019-12-31",
            valid_start_date="2020-01-01",
            valid_end_date="2020-06-30",
            test_start_date="2020-07-01",
            test_end_date="2020-12-31",
            model_type="GRU",
            lr=0.001,
            max_depth=6,
            num_leaves=100,
            subsample=0.8,
            colsample_bytree=0.8,
            seed=42,
            num_threads=1,
            handler_type="Alpha158"
        )
        
        print(f"GRU result: {result}")
        assert result["success"] == True, f"GRU failed: {result.get('message')}"


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])