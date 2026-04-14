from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from ..services.stock_service import StockService

router = APIRouter()
stock_service = StockService()

class StockRequest(BaseModel):
    symbol: str
    start_date: str
    end_date: str

class BatchRequest(BaseModel):
    symbols: list[str]
    start_date: str
    end_date: str

@router.get("/csi300")
async def get_csi300_stocks():
    """获取沪深300成分股"""
    try:
        result = stock_service.get_csi300_stocks()
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/history")
async def get_stock_history(request: StockRequest):
    """获取股票历史行情数据"""
    try:
        result = stock_service.get_stock_history(
            request.symbol,
            request.start_date,
            request.end_date
        )
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/factors")
async def get_stock_factors(request: StockRequest):
    """获取股票因子数据"""
    try:
        result = stock_service.get_stock_factors(
            request.symbol,
            request.start_date,
            request.end_date
        )
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/backtest")
async def backtest_strategy(request: StockRequest):
    """策略回测"""
    try:
        result = stock_service.backtest_strategy(
            request.symbol,
            request.start_date,
            request.end_date
        )
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/batch")
async def batch_analysis(request: BatchRequest):
    """批量分析股票"""
    try:
        result = stock_service.batch_analysis(
            request.symbols,
            request.start_date,
            request.end_date
        )
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
