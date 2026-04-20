"""
Qlib 相关 API 路由

API 端点清单:
=============

数据管理:
  GET    /check_data_release      - 检查 Qlib 数据发布日期
  GET    /download_data_stream      - 下载 Qlib 数据（SSE流式）
  POST   /download_data            - 下载 Qlib 数据（非流式）[Deprecated]
  POST   /update_stock_db          - 更新股票信息数据库
  POST   /preview_data             - 预览数据

模型训练:
  POST   /train_stream             - 训练模型（SSE流式）
  POST   /train                    - 训练模型（非流式）[Deprecated]

回测:
  POST   /backtest                 - 执行回测

记录查询:
  GET    /recorders                - 获取训练记录
  GET    /train_result/{id}        - 获取训练结果（模型评估）
  GET    /backtest_recorders       - 获取回测记录
  GET    /backtest_result/{id}     - 获取回测结果

删除操作:
  DELETE /recorders/{id}           - 删除训练记录
  DELETE /recorders                - 删除所有训练记录
  DELETE /backtest_recorders/{id}  - 删除回测记录

Deprecated API:
===============
- POST /download_data  -> 请使用 GET /download_data_stream
- POST /train          -> 请使用 POST /train_stream
"""

from fastapi import APIRouter, HTTPException
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
from ..services.qlib_service import QlibService
import json
import asyncio

router = APIRouter()
qlib_service = QlibService()

class TrainRequest(BaseModel):
    market: str = "csi300"
    benchmark: str = "SH000300"
    train_start_date: str
    train_end_date: str
    valid_start_date: str
    valid_end_date: str
    test_start_date: str
    test_end_date: str
    model_type: str = "LGBModel"
    lr: float = 0.0421
    max_depth: int = 8
    num_leaves: int = 210
    subsample: float = 0.8789
    colsample_bytree: float = 0.8879
    seed: int = 42
    num_threads: int = 1

class BacktestRequest(BaseModel):
    recorder_id: str
    market: str = "csi300"
    benchmark: str = "SH000300"
    start_date: str
    end_date: str
    initial_account: int = 1000000
    topk: int = 10
    n_drop: int = 1
    hold_days: int = 3
    stop_loss: float = 5.0
    strategy_type: str = "TopkDropoutStrategy"

@router.get("/check_data_release")
async def check_data_release():
    """检查 Qlib 数据发布日期"""
    return qlib_service.check_data_release()

@router.get("/download_data_stream")
async def download_qlib_data_stream():
    """下载 Qlib 数据（SSE流式返回进度）"""
    async def event_generator():
        progress_queue = asyncio.Queue()
        loop = asyncio.get_running_loop()
        last_progress = 0
        
        def sync_progress_callback(progress, message):
            # 使用主线程的事件循环
            asyncio.run_coroutine_threadsafe(
                progress_queue.put({"progress": progress, "message": message}),
                loop
            )
        
        # 在后台线程中执行下载
        import threading
        error = None
        
        def run_download():
            try:
                qlib_service.download_data_with_progress(sync_progress_callback)
            except Exception as e:
                error = e
        
        thread = threading.Thread(target=run_download)
        thread.start()
        
        # 发送进度事件
        while thread.is_alive() or not progress_queue.empty():
            try:
                item = await asyncio.wait_for(progress_queue.get(), timeout=0.5)
                last_progress = item.get("progress", last_progress)
                yield f"data: {json.dumps(item)}\n\n"
            except asyncio.TimeoutError:
                # 超时时保持上一次的进度，不发送新消息
                pass
        
        thread.join()
        
        # 发送最终结果
        if error:
            yield f"data: {json.dumps({'progress': -1, 'message': f'下载失败: {str(error)}', 'success': False})}\n\n"
        else:
            yield f"data: {json.dumps({'progress': 100, 'message': 'Qlib数据已更新完成！旧数据已备份。', 'success': True})}\n\n"
    
    return StreamingResponse(event_generator(), media_type="text/event-stream")

@router.post("/download_data")
async def download_qlib_data():
    """下载 Qlib 数据 (Deprecated: 请使用 /download_data_stream)

    .. deprecated::
        该接口已被弃用，请使用 SSE 流式接口 /download_data_stream 代替
    """
    try:
        result = qlib_service.download_data()
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/update_stock_db")
async def update_stock_database():
    """更新股票信息数据库"""
    try:
        result = qlib_service.update_stock_db()
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/train_stream")
async def train_model_stream(request: TrainRequest):
    """训练模型（SSE流式返回进度）"""
    async def event_generator():
        progress_queue = asyncio.Queue()
        loop = asyncio.get_running_loop()
        last_progress = 0

        def sync_progress_callback(progress, message):
            asyncio.run_coroutine_threadsafe(
                progress_queue.put({"progress": progress, "message": message}),
                loop
            )

        import threading
        # 使用列表以便在嵌套函数中修改（列表是可变对象）
        error_holder = [None]
        result_holder = [None]

        def run_train():
            try:
                result_holder[0] = qlib_service.train_model_with_progress(
                    sync_progress_callback,
                    market=request.market,
                    benchmark=request.benchmark,
                    train_start_date=request.train_start_date,
                    train_end_date=request.train_end_date,
                    valid_start_date=request.valid_start_date,
                    valid_end_date=request.valid_end_date,
                    test_start_date=request.test_start_date,
                    test_end_date=request.test_end_date,
                    model_type=request.model_type,
                    lr=request.lr,
                    max_depth=request.max_depth,
                    num_leaves=request.num_leaves,
                    subsample=request.subsample,
                    colsample_bytree=request.colsample_bytree,
                    seed=request.seed,
                    num_threads=request.num_threads
                )
            except Exception as e:
                error_holder[0] = e

        thread = threading.Thread(target=run_train)
        thread.start()

        while thread.is_alive() or not progress_queue.empty():
            try:
                item = await asyncio.wait_for(progress_queue.get(), timeout=0.5)
                last_progress = item.get("progress", last_progress)
                yield f"data: {json.dumps(item)}\n\n"
            except asyncio.TimeoutError:
                pass

        thread.join()

        if error_holder[0]:
            yield f"data: {json.dumps({'progress': -1, 'message': f'训练失败: {str(error_holder[0])}', 'success': False})}\n\n"
        elif result_holder[0]:
            recorder_id = result_holder[0].get('recorder_id', 'unknown')
            yield f"data: {json.dumps({'progress': 100, 'message': '模型训练完成！', 'success': True, 'recorder_id': recorder_id})}\n\n"
        else:
            yield f"data: {json.dumps({'progress': -1, 'message': '训练失败: 无返回结果', 'success': False})}\n\n"

    return StreamingResponse(event_generator(), media_type="text/event-stream")

@router.post("/train")
async def train_model(request: TrainRequest):
    """训练模型 (Deprecated: 请使用 /train_stream)

    .. deprecated::
        该接口已被弃用，请使用 SSE 流式接口 /train_stream 代替以获得实时进度反馈
    """
    try:
        result = qlib_service.train_model(
            market=request.market,
            benchmark=request.benchmark,
            train_start_date=request.train_start_date,
            train_end_date=request.train_end_date,
            valid_start_date=request.valid_start_date,
            valid_end_date=request.valid_end_date,
            test_start_date=request.test_start_date,
            test_end_date=request.test_end_date,
            model_type=request.model_type,
            lr=request.lr,
            max_depth=request.max_depth,
            num_leaves=request.num_leaves,
            subsample=request.subsample,
            colsample_bytree=request.colsample_bytree,
            seed=request.seed,
            num_threads=request.num_threads
        )
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/backtest")
async def backtest_model(request: BacktestRequest):
    """执行回测"""
    try:
        result = qlib_service.backtest_model(
            recorder_id=request.recorder_id,
            market=request.market,
            benchmark=request.benchmark,
            start_date=request.start_date,
            end_date=request.end_date,
            initial_account=request.initial_account,
            topk=request.topk,
            n_drop=request.n_drop,
            hold_days=request.hold_days,
            stop_loss=request.stop_loss,
            strategy_type=request.strategy_type
        )
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/train_result/{recorder_id}")
async def get_train_result(recorder_id: str):
    """获取训练结果（模型评估可视化数据）"""
    try:
        result = qlib_service.get_train_result(recorder_id)
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/recorders")
async def get_recorders(experiment_name: str = "train_model"):
    """获取训练记录"""
    try:
        result = qlib_service.get_recorders(experiment_name)
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/backtest_recorders")
async def get_backtest_recorders():
    """获取回测记录"""
    try:
        result = qlib_service.get_backtest_recorders()
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/backtest_result/{recorder_id}")
async def get_backtest_result(recorder_id: str):
    """获取回测结果"""
    try:
        result = qlib_service.get_backtest_result(recorder_id)
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

class PreviewDataRequest(BaseModel):
    market: str = "csi300"

@router.post("/preview_data")
async def preview_data(request: PreviewDataRequest):
    """预览数据"""
    try:
        result = qlib_service.preview_data(request.market)
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.delete("/recorders/{recorder_id}")
async def delete_train_recorder(recorder_id: str):
    """删除训练记录"""
    try:
        result = qlib_service.delete_train_recorder(recorder_id)
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.delete("/recorders")
async def delete_all_train_recorders():
    """删除所有训练记录"""
    try:
        result = qlib_service.delete_all_train_recorders()
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.delete("/backtest_recorders/{recorder_id}")
async def delete_backtest_recorder(recorder_id: str):
    """删除回测记录"""
    try:
        result = qlib_service.delete_backtest_recorder(recorder_id)
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
