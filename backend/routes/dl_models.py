"""
深度学习模型训练 API 路由

使用 TSDatasetH + pytorch_lstm_ts 架构，支持时间序列深度学习模型
"""

from fastapi import APIRouter, HTTPException
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
from ..services.dl_model_service import DLModelService
import json
import asyncio

router = APIRouter()
dl_service = DLModelService()


class DLTrainRequest(BaseModel):
    market: str = "csi300"
    benchmark: str = "SH000300"
    train_start_date: str
    train_end_date: str
    valid_start_date: str
    valid_end_date: str
    test_start_date: str
    test_end_date: str
    model_type: str = "LSTM"  # LSTM, GRU, ALSTM
    handler_type: str = "Alpha158"  # Alpha158, Alpha360
    # 深度学习模型参数
    d_feat: int = 20  # 选择的特征数量
    hidden_size: int = 64
    num_layers: int = 2
    dropout: float = 0.0
    lr: float = 0.001
    n_epochs: int = 200
    batch_size: int = 800
    early_stop: int = 20
    step_len: int = 20  # 时间序列步长
    seed: int = 42
    GPU: int = 0
    use_all_features: bool = False  # 是否使用全部特征（158/360）


@router.post("/train_dl_stream")
def train_dl_stream(request: DLTrainRequest):
    """
    训练深度学习模型（SSE流式返回进度）
    
    使用 TSDatasetH + pytorch_lstm_ts 架构
    """
    async def event_generator():
        queue = asyncio.Queue()
        
        def sync_progress_callback(progress, message):
            asyncio.create_task(queue.put((progress, message)))
        
        async def run_train():
            try:
                result = dl_service.train_dl_model_with_progress(
                    progress_callback=sync_progress_callback,
                    market=request.market,
                    benchmark=request.benchmark,
                    train_start_date=request.train_start_date,
                    train_end_date=request.train_end_date,
                    valid_start_date=request.valid_start_date,
                    valid_end_date=request.valid_end_date,
                    test_start_date=request.test_start_date,
                    test_end_date=request.test_end_date,
                    model_type=request.model_type,
                    handler_type=request.handler_type,
                    d_feat=request.d_feat,
                    hidden_size=request.hidden_size,
                    num_layers=request.num_layers,
                    dropout=request.dropout,
                    lr=request.lr,
                    n_epochs=request.n_epochs,
                    batch_size=request.batch_size,
                    early_stop=request.early_stop,
                    step_len=request.step_len,
                    seed=request.seed,
                    GPU=request.GPU,
                    use_all_features=request.use_all_features
                )
                await queue.put(("complete", result))
            except Exception as e:
                await queue.put(("error", str(e)))
        
        train_task = asyncio.create_task(run_train())
        
        while True:
            try:
                item = await asyncio.wait_for(queue.get(), timeout=1.0)
                if item[0] == "complete":
                    result = item[1]
                    if result.get("success"):
                        yield f"data: {json.dumps({'progress': 100, 'message': '训练完成！', 'recorder_id': result.get('recorder_id')})}\n\n"
                    else:
                        yield f"data: {json.dumps({'progress': -1, 'message': result.get('message', '训练失败')})}\n\n"
                    break
                elif item[0] == "error":
                    yield f"data: {json.dumps({'progress': -1, 'message': f'训练失败: {item[1]}'})}\n\n"
                    break
                else:
                    progress, message = item
                    yield f"data: {json.dumps({'progress': progress, 'message': message})}\n\n"
            except asyncio.TimeoutError:
                if train_task.done():
                    break
                continue
    
    return StreamingResponse(
        event_generator(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no"
        }
    )


@router.post("/train_dl")
def train_dl(request: DLTrainRequest):
    """
    训练深度学习模型（非流式）
    """
    def dummy_callback(progress, message):
        pass
    
    result = dl_service.train_dl_model_with_progress(
        progress_callback=dummy_callback,
        market=request.market,
        benchmark=request.benchmark,
        train_start_date=request.train_start_date,
        train_end_date=request.train_end_date,
        valid_start_date=request.valid_start_date,
        valid_end_date=request.valid_end_date,
        test_start_date=request.test_start_date,
        test_end_date=request.test_end_date,
        model_type=request.model_type,
        handler_type=request.handler_type,
        d_feat=request.d_feat,
        hidden_size=request.hidden_size,
        num_layers=request.num_layers,
        dropout=request.dropout,
        lr=request.lr,
        n_epochs=request.n_epochs,
        batch_size=request.batch_size,
        early_stop=request.early_stop,
        step_len=request.step_len,
        seed=request.seed,
        GPU=request.GPU,
        use_all_features=request.use_all_features
    )
    
    if not result.get("success"):
        raise HTTPException(status_code=500, detail=result.get("message", "训练失败"))
    
    return result