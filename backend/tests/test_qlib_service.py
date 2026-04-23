import pytest
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from services.qlib_service import QlibService
from qlib.utils import init_instance_by_config
import qlib


class TestQlibService:
    
    def test_model_registry(self):
        qs = QlibService()
        
        expected_models = [
            "LGBModel", "XGBModel", "CatBoostModel", "Linear", "DEnsembleModel",
            "LSTM", "GRU", "ALSTM", "DNN", "GATs", "TCN", "SFM", "TabnetModel"
        ]
        for model in expected_models:
            assert model in qs.MODEL_REGISTRY, f"Model {model} not in registry"
        
        print(f"MODEL_REGISTRY: {qs.MODEL_REGISTRY}")
    
    def test_handler_registry(self):
        qs = QlibService()
        
        expected_handlers = [
            "Alpha158", "Alpha360", "Alpha158DL", "Alpha360DL", 
            "Alpha158vwap", "Alpha360vwap"
        ]
        for handler in expected_handlers:
            assert handler in qs.HANDLER_REGISTRY, f"Handler {handler} not in registry"
        
        print(f"HANDLER_REGISTRY: {qs.HANDLER_REGISTRY}")
    
    def test_get_model_config_lightgbm(self):
        qs = QlibService()
        
        config = qs._get_model_config(
            model_type="LGBModel",
            lr=0.05,
            max_depth=8,
            num_leaves=200,
            subsample=0.8,
            colsample_bytree=0.8,
            seed=42,
            num_threads=4
        )
        
        assert config["class"] == "LGBModel"
        assert config["module_path"] == "qlib.contrib.model.gbdt"
        assert config["kwargs"]["learning_rate"] == 0.05
        assert config["kwargs"]["max_depth"] == 8
        assert config["kwargs"]["num_leaves"] == 200
        
        print(f"LGBModel config: {config}")
    
    def test_get_model_config_linear(self):
        qs = QlibService()
        
        config = qs._get_model_config(
            model_type="Linear",
            lr=0.05,
            max_depth=8,
            num_leaves=200,
            subsample=0.8,
            colsample_bytree=0.8,
            seed=42,
            num_threads=4
        )
        
        assert config["class"] == "LinearModel"
        assert config["module_path"] == "qlib.contrib.model.linear"
        assert "estimator" in config["kwargs"]
        
        print(f"Linear config: {config}")
    
    def test_get_model_config_catboost(self):
        qs = QlibService()
        
        config = qs._get_model_config(
            model_type="CatBoostModel",
            lr=0.05,
            max_depth=8,
            num_leaves=200,
            subsample=0.8,
            colsample_bytree=0.8,
            seed=42,
            num_threads=4
        )
        
        assert config["class"] == "CatBoostModel"
        assert config["module_path"] == "qlib.contrib.model.catboost_model"
        
        print(f"CatBoost config: {config}")
    
    def test_get_model_config_pytorch_models(self):
        qs = QlibService()
        
        pytorch_models = ["LSTM", "GRU", "ALSTM", "DNN", "GATs", "TCN", "SFM"]
        
        for model_type in pytorch_models:
            config = qs._get_model_config(
                model_type=model_type,
                lr=0.05,
                max_depth=8,
                num_leaves=200,
                subsample=0.8,
                colsample_bytree=0.8,
                seed=42,
                num_threads=4
            )
            
            assert config["class"] == model_type
            assert "d_feat" in config["kwargs"]
            assert "hidden_size" in config["kwargs"]
            assert "n_epochs" in config["kwargs"]
            
            print(f"{model_type} config: {config}")
    
    def test_get_model_config_tabnet(self):
        qs = QlibService()
        
        config = qs._get_model_config(
            model_type="TabnetModel",
            lr=0.05,
            max_depth=8,
            num_leaves=200,
            subsample=0.8,
            colsample_bytree=0.8,
            seed=42,
            num_threads=4
        )
        
        assert config["class"] == "TabnetModel"
        assert config["module_path"] == "qlib.contrib.model.pytorch_tabnet"
        
        print(f"TabnetModel config: {config}")
    
    def test_handler_type_mapping_for_dl_models(self):
        qs = QlibService()
        
        pytorch_models = ["LSTM", "GRU", "ALSTM", "DNN", "GATs", "TCN", "SFM", "TabnetModel"]
        
        test_cases = [
            ("Alpha158", "Alpha158DL"),
            ("Alpha360", "Alpha360DL"),
        ]
        
        for original, expected in test_cases:
            for model_type in pytorch_models:
                is_dl_model = model_type in pytorch_models
                
                if is_dl_model:
                    if original in ["Alpha158", "Alpha360"]:
                        actual_handler_type = original + "DL"
                    else:
                        actual_handler_type = original
                else:
                    actual_handler_type = original
                
                assert actual_handler_type == expected, \
                    f"For model {model_type} with handler {original}, expected {expected} but got {actual_handler_type}"
        
        print("Handler type mapping for DL models: OK")
    
    def test_handler_module_path(self):
        qs = QlibService()
        
        assert qs.HANDLER_REGISTRY["Alpha158"] == "qlib.contrib.data.handler"
        assert qs.HANDLER_REGISTRY["Alpha360"] == "qlib.contrib.data.handler"
        assert qs.HANDLER_REGISTRY["Alpha158DL"] == "qlib.contrib.data.loader"
        assert qs.HANDLER_REGISTRY["Alpha360DL"] == "qlib.contrib.data.loader"
        
        print("Handler module paths: OK")


class TestModelInitialization:
    
    @pytest.fixture(autouse=True)
    def setup_qlib(self):
        qs = QlibService()
        if not QlibService._qlib_initialized:
            qs.init_qlib()
    
    def test_init_lightgbm_model(self):
        qs = QlibService()
        
        config = qs._get_model_config(
            model_type="LGBModel",
            lr=0.05,
            max_depth=6,
            num_leaves=100,
            subsample=0.8,
            colsample_bytree=0.8,
            seed=42,
            num_threads=1
        )
        
        model = init_instance_by_config(config)
        assert model is not None
        print(f"LGBModel initialized: {type(model)}")
    
    def test_init_linear_model(self):
        qs = QlibService()
        
        config = qs._get_model_config(
            model_type="Linear",
            lr=0.05,
            max_depth=6,
            num_leaves=100,
            subsample=0.8,
            colsample_bytree=0.8,
            seed=42,
            num_threads=1
        )
        
        model = init_instance_by_config(config)
        assert model is not None
        print(f"LinearModel initialized: {type(model)}")
    
    def test_init_alstm_model(self):
        qs = QlibService()
        
        config = qs._get_model_config(
            model_type="ALSTM",
            lr=0.001,
            max_depth=6,
            num_leaves=100,
            subsample=0.8,
            colsample_bytree=0.8,
            seed=42,
            num_threads=1
        )
        
        model = init_instance_by_config(config)
        assert model is not None
        print(f"ALSTM initialized: {type(model)}")
        print(f"ALSTM device: {model.device}")
    
    def test_init_gru_model(self):
        qs = QlibService()
        
        config = qs._get_model_config(
            model_type="GRU",
            lr=0.001,
            max_depth=6,
            num_leaves=100,
            subsample=0.8,
            colsample_bytree=0.8,
            seed=42,
            num_threads=1
        )
        
        model = init_instance_by_config(config)
        assert model is not None
        print(f"GRU initialized: {type(model)}")


class TestHandlerInitialization:
    
    @pytest.fixture(autouse=True)
    def setup_qlib(self):
        qs = QlibService()
        if not QlibService._qlib_initialized:
            qs.init_qlib()
    
    def test_init_alpha158_handler(self):
        from qlib.contrib.data.handler import Alpha158
        
        config = {
            "start_time": "2020-01-01",
            "end_time": "2020-12-31",
            "fit_start_time": "2020-01-01",
            "fit_end_time": "2020-06-30",
            "instruments": "csi300",
        }
        
        handler = Alpha158(**config)
        assert handler is not None
        print(f"Alpha158 initialized: {type(handler)}")
    
    def test_init_alpha158dl_loader(self):
        from qlib.contrib.data.loader import Alpha158DL
        
        loader = Alpha158DL(config=None)
        assert loader is not None
        print(f"Alpha158DL initialized: {type(loader)}")
    
    def test_init_alpha360dl_loader(self):
        from qlib.contrib.data.loader import Alpha360DL
        
        loader = Alpha360DL(config=None)
        assert loader is not None
        print(f"Alpha360DL initialized: {type(loader)}")


class TestTrainingFlow:
    
    @pytest.fixture(autouse=True)
    def setup_qlib(self):
        qs = QlibService()
        if not QlibService._qlib_initialized:
            qs.init_qlib()
    
    def test_train_lightgbm_alpha158(self):
        qs = QlibService()
        
        progress_messages = []
        
        def progress_callback(progress, message):
            progress_messages.append((progress, message))
            print(f"[{progress}%] {message}")
        
        result = qs.train_model_with_progress(
            progress_callback=progress_callback,
            market="csi300",
            benchmark="SH000300",
            train_start_date="2020-01-01",
            train_end_date="2020-06-30",
            valid_start_date="2020-07-01",
            valid_end_date="2020-09-30",
            test_start_date="2020-10-01",
            test_end_date="2020-12-31",
            model_type="LGBModel",
            lr=0.05,
            max_depth=6,
            num_leaves=100,
            subsample=0.8,
            colsample_bytree=0.8,
            seed=42,
            num_threads=1,
            handler_type="Alpha158"
        )
        
        print(f"Training result: {result}")
        assert result["success"] == True
        assert result["recorder_id"] != "unknown"
        assert result["recorder_id"] is not None
    
    def test_train_alstm_alpha158(self):
        qs = QlibService()
        
        progress_messages = []
        
        def progress_callback(progress, message):
            progress_messages.append((progress, message))
            print(f"[{progress}%] {message}")
        
        # 使用更长的训练时间，确保有足够数据
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
            model_type="ALSTM",
            lr=0.001,
            max_depth=6,
            num_leaves=100,
            subsample=0.8,
            colsample_bytree=0.8,
            seed=42,
            num_threads=1,
            handler_type="Alpha158"
        )
        
        print(f"Training result: {result}")
        assert result["success"] == True, f"Training failed: {result.get('message', 'Unknown error')}"
        assert result["recorder_id"] is not None
        print(f"Recorder ID: {result['recorder_id']}")


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])