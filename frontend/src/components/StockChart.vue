<template>
  <div class="stock-chart-page">
    <!-- 搜索栏 -->
    <div class="search-section">
      <div class="search-box">
        <el-input
          v-model="stockCodeInput"
          placeholder="输入股票代码，如：000001.SZ"
          class="stock-input"
          @keyup.enter="searchStock"
        >
          <template #prefix>
            <span class="search-icon">🔍</span>
          </template>
        </el-input>
        <el-button type="primary" class="search-btn" @click="searchStock">
          查询
        </el-button>
      </div>
      <div class="quick-tags">
        <el-tag
          v-for="stock in quickStocks"
          :key="stock.code"
          class="quick-tag"
          @click="selectStock(stock.code)"
        >
          {{ stock.name }} {{ stock.code }}
        </el-tag>
      </div>
    </div>

    <!-- 股票信息卡片 -->
    <div v-if="currentStock.code" class="stock-info-card">
      <div class="stock-header">
        <div class="stock-title">
          <h2>{{ currentStock.name }}</h2>
          <span class="stock-code">{{ currentStock.code }}</span>
        </div>
        <div class="stock-price" :class="priceChangeClass">
          <span class="price">{{ currentStock.price?.toFixed(2) }}</span>
          <span class="change">{{ currentStock.change?.toFixed(2) }}%</span>
        </div>
      </div>
      <div class="stock-metrics">
        <div class="metric-item">
          <span class="metric-label">今开</span>
          <span class="metric-value">{{ currentStock.open?.toFixed(2) }}</span>
        </div>
        <div class="metric-item">
          <span class="metric-label">最高</span>
          <span class="metric-value high">{{ currentStock.high?.toFixed(2) }}</span>
        </div>
        <div class="metric-item">
          <span class="metric-label">最低</span>
          <span class="metric-value low">{{ currentStock.low?.toFixed(2) }}</span>
        </div>
        <div class="metric-item">
          <span class="metric-label">昨收</span>
          <span class="metric-value">{{ currentStock.preClose?.toFixed(2) }}</span>
        </div>
        <div class="metric-item">
          <span class="metric-label">成交量</span>
          <span class="metric-value">{{ formatVolume(currentStock.volume) }}</span>
        </div>
        <div class="metric-item">
          <span class="metric-label">成交额</span>
          <span class="metric-value">{{ formatAmount(currentStock.amount) }}</span>
        </div>
      </div>
    </div>

    <!-- K 线图区域 -->
    <div class="chart-section">
      <!-- 时间周期选择 -->
      <div class="time-period-selector">
        <el-button-group>
          <el-button size="small" :class="{ active: timePeriod === '3M' }" @click="setTimePeriod('3M')">3 个月</el-button>
          <el-button size="small" :class="{ active: timePeriod === '6M' }" @click="setTimePeriod('6M')">半年</el-button>
          <el-button size="small" :class="{ active: timePeriod === '1Y' }" @click="setTimePeriod('1Y')">1 年</el-button>
          <el-button size="small" :class="{ active: timePeriod === '3Y' }" @click="setTimePeriod('3Y')">3 年</el-button>
        </el-button-group>
      </div>
      
      <!-- 技术指标显示 -->
      <div class="technical-indicators" v-if="currentStock.code">
        <span class="indicator-item" style="color: #f0c040">MA5: {{ maValues.ma5?.toFixed(2) }}</span>
        <span class="indicator-item" style="color: #409eff">MA10: {{ maValues.ma10?.toFixed(2) }}</span>
        <span class="indicator-item" style="color: #67c23a">MA20: {{ maValues.ma20?.toFixed(2) }}</span>
        <span class="indicator-item" style="color: #9c27b0">MA30: {{ maValues.ma30?.toFixed(2) }}</span>
        <span class="indicator-item" style="color: #ff9800">MA60: {{ maValues.ma60?.toFixed(2) }}</span>
        <span class="indicator-item" style="color: #795548">MA120: {{ maValues.ma120?.toFixed(2) }}</span>
      </div>
      
      <div ref="chartContainer" class="chart-container"></div>
      
      <!-- 十字线数据悬浮窗 -->
      <div class="crosshair-tooltip" v-if="tooltipVisible" :style="tooltipStyle">
        <div class="tooltip-header">
          <span class="tooltip-date">{{ crosshairData.time }}</span>
        </div>
        <div class="tooltip-body">
          <div class="tooltip-row">
            <span class="tooltip-label">开盘</span>
            <span class="tooltip-value">{{ crosshairData.open?.toFixed(2) }}</span>
          </div>
          <div class="tooltip-row">
            <span class="tooltip-label">最高</span>
            <span class="tooltip-value high">{{ crosshairData.high?.toFixed(2) }}</span>
          </div>
          <div class="tooltip-row">
            <span class="tooltip-label">最低</span>
            <span class="tooltip-value low">{{ crosshairData.low?.toFixed(2) }}</span>
          </div>
          <div class="tooltip-row">
            <span class="tooltip-label">收盘</span>
            <span class="tooltip-value">{{ crosshairData.close?.toFixed(2) }}</span>
          </div>
          <div class="tooltip-row">
            <span class="tooltip-label">涨跌</span>
            <span :class="['tooltip-value', crosshairData.change >= 0 ? 'up' : 'down']">
              {{ crosshairData.change?.toFixed(2) }} / {{ crosshairData.changePercent?.toFixed(2) }}%
            </span>
          </div>
        </div>
      </div>
      
      <!-- 成交量指标 -->
      <div class="volume-indicators" v-if="currentStock.code">
        <span class="indicator-item">VOL: {{ formatVolume(currentStock.volume) }}</span>
        <span class="indicator-item" style="color: #f0c040">VOLMA5: {{ formatVolume(volumeMaValues.ma5) }}</span>
        <span class="indicator-item" style="color: #409eff">VOLMA10: {{ formatVolume(volumeMaValues.ma10) }}</span>
        <span class="indicator-item" style="color: #67c23a">VOLMA20: {{ formatVolume(volumeMaValues.ma20) }}</span>
      </div>
    </div>

    <!-- 加载状态 -->
    <div v-if="loading" class="loading-overlay">
      <div class="loading-spinner"></div>
      <span>加载中...</span>
    </div>
  </div>
</template>

<script setup>
import { ref, onMounted, onUnmounted, computed, nextTick } from 'vue'
import { createChart, CandlestickSeries, HistogramSeries, LineSeries } from 'lightweight-charts'
import axios from 'axios'

const API_BASE_URL = '/api'

// 响应式数据
const stockCodeInput = ref('')
const currentStock = ref({
  code: '',
  name: '',
  price: 0,
  change: 0,
  open: 0,
  high: 0,
  low: 0,
  preClose: 0,
  volume: 0,
  amount: 0
})
const loading = ref(false)
const isLoadingMore = ref(false) // 是否正在加载更多数据
const timePeriod = ref('3M') // 默认 3 个月
const chartContainer = ref(null)
let chart = null
let candlestickSeries = null
let volumeSeries = null
let ma5Series = null
let ma10Series = null
let ma20Series = null
let ma30Series = null
let ma60Series = null
let ma120Series = null
let volumeMa5Series = null
let volumeMa10Series = null
let volumeMa20Series = null

// 技术指标值
const maValues = ref({
  ma5: 0,
  ma10: 0,
  ma20: 0,
  ma30: 0,
  ma60: 0,
  ma120: 0
})

// 成交量均线值
const volumeMaValues = ref({
  ma5: 0,
  ma10: 0,
  ma20: 0
})

// 十字线数据
const crosshairData = ref({
  time: '',
  open: 0,
  high: 0,
  low: 0,
  close: 0,
  change: 0,
  changePercent: 0
})

// 悬浮窗状态
const tooltipVisible = ref(false)
const tooltipStyle = ref({
  left: '0px',
  top: '0px'
})
const tooltipRef = ref(null)

// 2026 年热门股票
const quickStocks = [
  { code: '002594.SZ', name: '比亚迪' },
  { code: '300750.SZ', name: '宁德时代' },
  { code: '603288.SH', name: '海天味业' },
  { code: '600276.SH', name: '恒瑞医药' },
  { code: '002475.SZ', name: '立讯精密' },
  { code: '300760.SZ', name: '迈瑞医疗' },
  { code: '002714.SZ', name: '牧原股份' },
  { code: '600900.SH', name: '长江电力' }
]

// 涨跌幅样式
const priceChangeClass = computed(() => {
  const change = currentStock.value.change || 0
  if (change > 0) return 'up'
  if (change < 0) return 'down'
  return 'flat'
})

// 格式化成交量
const formatVolume = (vol) => {
  if (!vol) return '-'
  if (vol >= 1e8) return (vol / 1e8).toFixed(2) + '亿'
  if (vol >= 1e4) return (vol / 1e4).toFixed(2) + '万'
  return vol.toString()
}

// 格式化成交额
const formatAmount = (amt) => {
  if (!amt) return '-'
  if (amt >= 1e8) return (amt / 1e8).toFixed(2) + '亿'
  if (amt >= 1e4) return (amt / 1e4).toFixed(2) + '万'
  return amt.toString()
}

// 格式化时间标签
const formatTimeLabel = (time) => {
  if (!time) return ''
  // Lightweight Charts 的时间可能是 timestamp 或字符串
  if (typeof time === 'number') {
    const date = new Date(time * 1000)
    return date.toISOString().split('T')[0]
  }
  return time
}

// 获取前收盘价（从已加载的数据中查找）
let cachedStockData = []
const getPreClose = (time) => {
  const currentTime = formatTimeLabel(time)
  const currentIndex = cachedStockData.findIndex(item => item.date === currentTime)
  if (currentIndex > 0) {
    return cachedStockData[currentIndex - 1].close
  }
  return null
}

// 初始化图表 - 使用 Lightweight Charts v5 最新 API
const initChart = () => {
  if (!chartContainer.value) return

  // 创建图表实例
  chart = createChart(chartContainer.value, {
    layout: {
      background: { type: 'solid', color: '#ffffff' },
      textColor: '#333',
    },
    grid: {
      vertLines: { color: '#f0f0f0' },
      horzLines: { color: '#f0f0f0' },
    },
    crosshair: {
      mode: 1,
      vertLine: {
        color: '#667eea',
        labelBackgroundColor: '#667eea',
      },
      horzLine: {
        color: '#667eea',
        labelBackgroundColor: '#667eea',
      },
    },
    rightPriceScale: {
      borderColor: '#e0e0e0',
    },
    timeScale: {
      borderColor: '#e0e0e0',
      timeVisible: true,
      secondsVisible: false,
      fixLeftEdge: false,
      fixRightEdge: false,
      lockVisibleTimeRangeOnResize: false,
      rightBarStaysOnScroll: true,
      handleScroll: {
        mouseWheel: true,
        pressedMouseMove: true,
        horzTouchDrag: true,
        vertTouchDrag: false,
      },
      handleScale: {
        axisPressedMouseMove: {
          time: false,
          price: true,
        },
        axisDoubleClickReset: {
          time: true,
          price: true,
        },
        mouseWheel: true,
        pinch: true,
      },
    },
    crosshair: {
      mode: 1,
      vertLine: {
        color: '#667eea',
        labelBackgroundColor: '#667eea',
        crosshairMarkerVisible: true,
        crosshairMarkerRadius: 6,
      },
      horzLine: {
        color: '#667eea',
        labelBackgroundColor: '#667eea',
        crosshairMarkerVisible: true,
        crosshairMarkerRadius: 6,
      },
    },
  })

  // 添加 K 线系列到第一个面板（paneIndex: 0）
  candlestickSeries = chart.addSeries(CandlestickSeries, {
    upColor: '#ef232a',
    downColor: '#14b143',
    borderUpColor: '#ef232a',
    borderDownColor: '#14b143',
    wickUpColor: '#ef232a',
    wickDownColor: '#14b143',
  }, 0)

  // 自定义 K 线 tooltip 显示
  candlestickSeries.priceLineSource = 1 // 使用最新数据

  // 添加均线到第一个面板（不显示 title，避免在右侧坐标轴显示）
  ma5Series = chart.addSeries(LineSeries, {
    color: '#f0c040',
    lineWidth: 1,
  }, 0)
  
  ma10Series = chart.addSeries(LineSeries, {
    color: '#409eff',
    lineWidth: 1,
  }, 0)
  
  ma20Series = chart.addSeries(LineSeries, {
    color: '#67c23a',
    lineWidth: 1,
  }, 0)
  
  ma30Series = chart.addSeries(LineSeries, {
    color: '#9c27b0',
    lineWidth: 1,
  }, 0)
  
  ma60Series = chart.addSeries(LineSeries, {
    color: '#ff9800',
    lineWidth: 1,
  }, 0)
  
  ma120Series = chart.addSeries(LineSeries, {
    color: '#795548',
    lineWidth: 1,
  }, 0)

  // 添加成交量系列到第二个面板（paneIndex: 1）
  volumeSeries = chart.addSeries(HistogramSeries, {
    color: '#26a69a',
    priceFormat: {
      type: 'volume',
    },
  }, 1)

  // 添加成交量均线到第二个面板（不显示 title）
  volumeMa5Series = chart.addSeries(LineSeries, {
    color: '#f0c040',
    lineWidth: 1,
  }, 1)
  
  volumeMa10Series = chart.addSeries(LineSeries, {
    color: '#409eff',
    lineWidth: 1,
  }, 1)
  
  volumeMa20Series = chart.addSeries(LineSeries, {
    color: '#67c23a',
    lineWidth: 1,
  }, 1)

  // 响应窗口大小变化
  const resizeObserver = new ResizeObserver(entries => {
    if (entries.length === 0 || entries[0].contentRect.width === 0) return
    const { width, height } = entries[0].contentRect
    chart.applyOptions({ width, height })
  })
  resizeObserver.observe(chartContainer.value)

  // 监听十字线移动，显示详细数据
  chart.subscribeCrosshairMove(param => {
    if (!param.time || !param.point) {
      // 隐藏悬浮窗
      tooltipVisible.value = false
      return
    }

    // 获取当前时间点的 K 线数据
    const data = param.seriesData.get(candlestickSeries)
    if (data) {
      const open = data.open
      const high = data.high
      const low = data.low
      const close = data.close
      
      // 计算涨跌幅（需要获取前一天的收盘价）
      const preClose = getPreClose(data.time) || open
      const change = close - preClose
      const changePercent = (change / preClose) * 100
      
      // 更新时间轴标签格式
      const timeStr = formatTimeLabel(data.time)
      
      // 更新状态栏数据
      crosshairData.value = {
        time: timeStr,
        open,
        high,
        low,
        close,
        change,
        changePercent
      }
      
      // 计算悬浮窗位置
      const chartRect = chartContainer.value.getBoundingClientRect()
      const tooltipWidth = 180
      const tooltipHeight = 160
      
      // 获取图表在视口中的位置
      const chartX = chartRect.left
      const chartY = chartRect.top
      
      // 计算鼠标在图表内的相对位置
      const mouseX = param.point.x
      const mouseY = param.point.y
      
      // 默认显示在十字线右侧
      let left = mouseX + 15
      let top = mouseY - tooltipHeight / 2
      
      // 如果超出右边界，显示在左侧
      if (left + tooltipWidth > chartRect.width) {
        left = mouseX - tooltipWidth - 15
      }
      
      // 如果超出上边界
      if (top < 0) {
        top = 10
      }
      
      // 如果超出下边界
      if (top + tooltipHeight > chartRect.height) {
        top = chartRect.height - tooltipHeight - 10
      }
      
      tooltipStyle.value = {
        position: 'fixed',
        left: `${chartX + left}px`,
        top: `${chartY + top}px`,
        zIndex: 9999
      }
      
      // 显示悬浮窗
      tooltipVisible.value = true
    }
  })

  // 监听可见范围变化，实现动态加载
  let visibleRangeTimeout = null
  chart.timeScale().subscribeVisibleLogicalRangeChange(() => {
    // 防抖处理，避免频繁触发
    if (visibleRangeTimeout) {
      clearTimeout(visibleRangeTimeout)
    }
    visibleRangeTimeout = setTimeout(() => {
      handleVisibleRangeChange()
    }, 300)
  })
}

// 处理可见范围变化（缩放/滚动时触发）
let loadMoreTimeout = null
let lastVisibleFrom = null
const handleVisibleRangeChange = () => {
  const visibleRange = chart.timeScale().getVisibleLogicalRange()
  if (!visibleRange) return

  // 计算当前可见的 K 线数量
  const visibleBarsCount = Math.round(visibleRange.to - visibleRange.from)
  
  // 记录当前可见范围起始位置
  lastVisibleFrom = visibleRange.from
  
  // 根据可见的 K 线数量判断缩放状态
  if (visibleBarsCount < 20) {
    console.log('放大：当前可见', visibleBarsCount, '根 K 线')
  } else if (visibleBarsCount > 100) {
    console.log('缩小：当前可见', visibleBarsCount, '根 K 线')
  }
  
  // 自动加载功能已禁用，如需启用请确保有严格的触发条件
}

// 计算移动平均线（用于价格数据）
const calculatePriceMA = (dayCount, data) => {
  const result = []
  for (let i = 0; i < data.length; i++) {
    if (i < dayCount - 1) {
      continue
    }
    let sum = 0
    for (let j = 0; j < dayCount; j++) {
      sum += data[i - j].close
    }
    result.push({
      time: data[i].time,
      value: sum / dayCount
    })
  }
  return result
}

// 计算成交量移动平均线
const calculateVolumeMA = (dayCount, data) => {
  const result = []
  for (let i = 0; i < data.length; i++) {
    if (i < dayCount - 1) {
      continue
    }
    let sum = 0
    for (let j = 0; j < dayCount; j++) {
      sum += data[i - j].value
    }
    result.push({
      time: data[i].time,
      value: sum / dayCount
    })
  }
  return result
}

// 加载股票数据
const loadStockData = async () => {
  if (!currentStock.value.code) return

  loading.value = true
  try {
    // 转换股票代码格式：从 002594.SZ 转为 SZ002594
    const formatStockCode = (code) => {
      const parts = code.split('.')
      if (parts.length === 2) {
        const exchange = parts[1].toUpperCase()
        const stockCode = parts[0]
        return `${exchange}${stockCode}`
      }
      return code
    }
    
    const formattedCode = formatStockCode(currentStock.value.code)
    
    // 计算日期范围：直接加载3年的数据
    const endDate = new Date()
    const startDate = new Date()
    startDate.setFullYear(startDate.getFullYear() - 3)

    // 调用后端 API 获取数据
    const res = await axios.get(`${API_BASE_URL}/qlib/stock/quote`, {
      params: {
        code: formattedCode,
        start_date: startDate.toISOString().split('T')[0],
        end_date: endDate.toISOString().split('T')[0]
      }
    })

    if (res.data.success && res.data.data && res.data.data.length > 0) {
      const data = res.data.data
      updateStockInfo(data)
      updateChart(data)
      // 根据当前选择的时间周期设置显示范围
      setTimeRange()
    } else {
      console.warn('未获取到股票数据，请检查：1. 后端服务是否启动 2. Qlib 数据是否已下载 3. 股票代码格式是否正确')
      clearChart()
    }
  } catch (e) {
    console.error('加载股票数据失败:', e)
    console.error('请确保：1. 后端服务已启动 (python backend/main.py) 2. Qlib 数据已下载')
    clearChart()
  } finally {
    loading.value = false
  }
}

// 根据时间周期设置显示范围
const setTimeRange = () => {
  if (!chart || cachedStockData.length === 0) return
  
  const endDate = new Date()
  const startDate = new Date()
  
  switch (timePeriod.value) {
    case '3M':
      startDate.setMonth(startDate.getMonth() - 3)
      break
    case '6M':
      startDate.setMonth(startDate.getMonth() - 6)
      break
    case '1Y':
      startDate.setFullYear(startDate.getFullYear() - 1)
      break
    case '3Y':
      startDate.setFullYear(startDate.getFullYear() - 3)
      break
  }
  
  // 设置时间范围
  chart.timeScale().setVisibleRange({
    from: startDate.getTime() / 1000,
    to: endDate.getTime() / 1000
  })
}

// 更新股票信息
const updateStockInfo = (data) => {
  if (data && data.length > 0) {
    const latest = data[data.length - 1]
    currentStock.value = {
      ...currentStock.value,
      price: latest.close,
      change: ((latest.close - latest.pre_close) / latest.pre_close * 100),
      open: latest.open,
      high: latest.high,
      low: latest.low,
      preClose: latest.pre_close,
      volume: latest.volume,
      amount: latest.amount
    }
  }
}

// 更新图表
const updateChart = (data) => {
  if (!candlestickSeries || !data || data.length === 0) return

  // 缓存股票数据用于计算涨跌幅
  cachedStockData = data

  // 转换为 lightweight-charts 格式
  const candleData = data.map(item => ({
    time: item.date,
    open: item.open,
    high: item.high,
    low: item.low,
    close: item.close
  }))

  // 成交量数据
  const volumeData = data.map(item => ({
    time: item.date,
    value: item.volume,
    color: item.close >= item.open ? '#ef232a' : '#14b143'
  }))

  // 使用 setData 更新所有数据
  candlestickSeries.setData(candleData)
  volumeSeries.setData(volumeData)

  // 计算并设置均线
  const ma5Data = calculatePriceMA(5, candleData)
  const ma10Data = calculatePriceMA(10, candleData)
  const ma20Data = calculatePriceMA(20, candleData)
  const ma30Data = calculatePriceMA(30, candleData)
  const ma60Data = calculatePriceMA(60, candleData)
  const ma120Data = calculatePriceMA(120, candleData)

  ma5Series.setData(ma5Data)
  ma10Series.setData(ma10Data)
  ma20Series.setData(ma20Data)
  ma30Series.setData(ma30Data)
  ma60Series.setData(ma60Data)
  ma120Series.setData(ma120Data)

  // 计算并设置成交量均线
  const volumeMa5Data = calculateVolumeMA(5, volumeData)
  const volumeMa10Data = calculateVolumeMA(10, volumeData)
  const volumeMa20Data = calculateVolumeMA(20, volumeData)

  volumeMa5Series.setData(volumeMa5Data)
  volumeMa10Series.setData(volumeMa10Data)
  volumeMa20Series.setData(volumeMa20Data)

  // 更新技术指标值
  updateMaValues(candleData)
  updateVolumeMaValues(volumeData)
}

// 更新技术指标值
const updateMaValues = (data) => {
  if (data.length === 0) return
  
  maValues.value.ma5 = calculateLatestMA(5, data)
  maValues.value.ma10 = calculateLatestMA(10, data)
  maValues.value.ma20 = calculateLatestMA(20, data)
  maValues.value.ma30 = calculateLatestMA(30, data)
  maValues.value.ma60 = calculateLatestMA(60, data)
  maValues.value.ma120 = calculateLatestMA(120, data)
}

// 更新成交量均线值
const updateVolumeMaValues = (data) => {
  if (data.length === 0) return
  
  volumeMaValues.value.ma5 = calculateLatestMA(5, data)
  volumeMaValues.value.ma10 = calculateLatestMA(10, data)
  volumeMaValues.value.ma20 = calculateLatestMA(20, data)
}

// 计算最新的移动平均值
const calculateLatestMA = (dayCount, data) => {
  if (data.length < dayCount) return 0
  
  let sum = 0
  for (let i = 0; i < dayCount; i++) {
    sum += data[data.length - 1 - i].close || data[data.length - 1 - i].value
  }
  return sum / dayCount
}

// 清空图表
const clearChart = () => {
  if (candlestickSeries) candlestickSeries.setData([])
  if (volumeSeries) volumeSeries.setData([])
  if (ma5Series) ma5Series.setData([])
  if (ma10Series) ma10Series.setData([])
  if (ma20Series) ma20Series.setData([])
  if (ma30Series) ma30Series.setData([])
  if (ma60Series) ma60Series.setData([])
  if (ma120Series) ma120Series.setData([])
  if (volumeMa5Series) volumeMa5Series.setData([])
  if (volumeMa10Series) volumeMa10Series.setData([])
  if (volumeMa20Series) volumeMa20Series.setData([])
}

// 设置时间周期
const setTimePeriod = (period) => {
  timePeriod.value = period
  // 只控制显示窗口，不重新加载数据
  setTimeRange()
}

// 搜索股票
const searchStock = () => {
  const code = stockCodeInput.value.trim()
  if (!code) return

  currentStock.value.code = code
  currentStock.value.name = code

  const found = quickStocks.find(s => s.code === code)
  if (found) {
    currentStock.value.name = found.name
  }

  loadStockData()
}

// 选择股票
const selectStock = (code) => {
  stockCodeInput.value = code
  searchStock()
}

// 监听股票选择事件
const handleStockSelect = (event) => {
  const code = event.detail?.code
  if (code) {
    stockCodeInput.value = code
    searchStock()
  }
}

onMounted(() => {
  nextTick(() => {
    initChart()
    if (quickStocks.length > 0) {
      selectStock(quickStocks[0].code)
    }
  })
  window.addEventListener('select-stock', handleStockSelect)
})

onUnmounted(() => {
  if (chart) {
    chart.remove()
    chart = null
  }
  window.removeEventListener('select-stock', handleStockSelect)
})
</script>

<style scoped>
.stock-chart-page {
  padding: 20px;
  max-width: 1200px;
  margin: 0 auto;
}

.search-section {
  background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
  border-radius: 16px;
  padding: 24px;
  margin-bottom: 20px;
  color: white;
}

.search-box {
  display: flex;
  gap: 12px;
  margin-bottom: 16px;
}

.stock-input {
  flex: 1;
}

.stock-input :deep(.el-input__wrapper) {
  background: rgba(255, 255, 255, 0.95);
  border-radius: 10px;
  box-shadow: 0 4px 15px rgba(0, 0, 0, 0.1);
}

.stock-input :deep(.el-input__inner) {
  height: 44px;
  font-size: 16px;
  color: #2c3e50;
}

.search-icon {
  font-size: 18px;
  margin-right: 8px;
}

.search-btn {
  height: 44px;
  padding: 0 32px;
  font-size: 16px;
  font-weight: 600;
  border-radius: 10px;
  background: rgba(255, 255, 255, 0.95) !important;
  color: #667eea !important;
  border: none !important;
  transition: all 0.3s ease;
}

.search-btn:hover {
  transform: translateY(-2px);
  box-shadow: 0 8px 25px rgba(0, 0, 0, 0.2);
}

.quick-tags {
  display: flex;
  align-items: center;
  gap: 10px;
  flex-wrap: wrap;
}

.tag-label {
  font-size: 14px;
  opacity: 0.9;
}

.quick-tag {
  cursor: pointer;
  background: rgba(255, 255, 255, 0.2);
  border: 1px solid rgba(255, 255, 255, 0.3);
  color: white;
  transition: all 0.2s;
}

.quick-tag:hover {
  background: rgba(255, 255, 255, 0.3);
  transform: translateY(-1px);
}

.stock-info-card {
  background: white;
  border-radius: 16px;
  padding: 24px;
  margin-bottom: 20px;
  box-shadow: 0 4px 20px rgba(0, 0, 0, 0.08);
}

.stock-header {
  display: flex;
  justify-content: space-between;
  align-items: center;
  margin-bottom: 20px;
  padding-bottom: 20px;
  border-bottom: 1px solid #f0f0f0;
}

.stock-title h2 {
  font-size: 24px;
  font-weight: 600;
  color: #2c3e50;
  margin: 0;
}

.stock-code {
  font-size: 14px;
  color: #8a94a6;
  margin-left: 8px;
}

.stock-price {
  text-align: right;
}

.stock-price .price {
  font-size: 32px;
  font-weight: 700;
  margin-right: 12px;
}

.stock-price .change {
  font-size: 18px;
  font-weight: 500;
  padding: 4px 12px;
  border-radius: 6px;
}

.stock-price.up .price,
.stock-price.up .change {
  color: #ef232a;
}

.stock-price.up .change {
  background: rgba(239, 35, 42, 0.1);
}

.stock-price.down .price,
.stock-price.down .change {
  color: #14b143;
}

.stock-price.down .change {
  background: rgba(20, 177, 67, 0.1);
}

.stock-price.flat .price,
.stock-price.flat .change {
  color: #8a94a6;
}

.stock-metrics {
  display: grid;
  grid-template-columns: repeat(6, 1fr);
  gap: 16px;
}

.metric-item {
  display: flex;
  flex-direction: column;
  gap: 6px;
}

.metric-label {
  font-size: 13px;
  color: #8a94a6;
}

.metric-value {
  font-size: 16px;
  font-weight: 600;
  color: #2c3e50;
}

.metric-value.high {
  color: #ef232a;
}

.metric-value.low {
  color: #14b143;
}

.chart-section {
  background: white;
  border-radius: 16px;
  padding: 24px;
  box-shadow: 0 4px 20px rgba(0, 0, 0, 0.08);
}

.time-period-selector {
  margin-bottom: 16px;
  overflow-x: auto;
  white-space: nowrap;
  padding-bottom: 8px;
}

.time-period-selector .el-button-group {
  display: inline-flex;
}

.time-period-selector .el-button {
  border-radius: 0;
  border-right: 1px solid #e0e0e0;
}

.time-period-selector .el-button:first-child {
  border-radius: 4px 0 0 4px;
}

.time-period-selector .el-button:last-child {
  border-radius: 0 4px 4px 0;
  border-right: none;
}

.time-period-selector .el-button.active {
  background-color: #667eea;
  color: white;
  border-color: #667eea;
}

.technical-indicators {
  display: flex;
  flex-wrap: wrap;
  gap: 16px;
  margin-bottom: 16px;
  padding: 12px;
  background-color: #f8f9fa;
  border-radius: 8px;
  font-size: 12px;
}

.indicator-item {
  display: inline-block;
  padding: 4px 8px;
  border-radius: 4px;
  background-color: white;
  box-shadow: 0 1px 3px rgba(0, 0, 0, 0.1);
}

.volume-indicators {
  display: flex;
  flex-wrap: wrap;
  gap: 16px;
  margin-top: 16px;
  padding: 12px;
  background-color: #f8f9fa;
  border-radius: 8px;
  font-size: 12px;
}

.chart-container {
  width: 100%;
  height: 650px;
  position: relative;
}

/* 十字线悬浮窗 */
.crosshair-tooltip {
  position: absolute;
  background: rgba(255, 255, 255, 0.98);
  border: 1px solid #e0e0e0;
  border-radius: 8px;
  padding: 12px;
  box-shadow: 0 4px 12px rgba(0, 0, 0, 0.15);
  z-index: 1000;
  min-width: 180px;
  pointer-events: none;
  font-size: 12px;
  transition: opacity 0.2s ease;
}

.tooltip-header {
  padding-bottom: 8px;
  margin-bottom: 8px;
  border-bottom: 1px solid #f0f0f0;
}

.tooltip-date {
  font-size: 13px;
  font-weight: 600;
  color: #2c3e50;
}

.tooltip-body {
  display: flex;
  flex-direction: column;
  gap: 6px;
}

.tooltip-row {
  display: flex;
  justify-content: space-between;
  align-items: center;
}

.tooltip-label {
  color: #8a94a6;
  font-weight: 500;
}

.tooltip-value {
  font-weight: 600;
  color: #2c3e50;
  text-align: right;
}

.tooltip-value.high {
  color: #ef232a;
}

.tooltip-value.low {
  color: #14b143;
}

.tooltip-value.up {
  color: #ef232a;
}

.tooltip-value.down {
  color: #14b143;
}

.loading-overlay {
  position: fixed;
  top: 0;
  left: 0;
  right: 0;
  bottom: 0;
  background: rgba(255, 255, 255, 0.8);
  display: flex;
  flex-direction: column;
  align-items: center;
  justify-content: center;
  gap: 16px;
  z-index: 1000;
}

.loading-spinner {
  width: 48px;
  height: 48px;
  border: 4px solid rgba(102, 126, 234, 0.2);
  border-top-color: #667eea;
  border-radius: 50%;
  animation: spin 1s linear infinite;
}

@keyframes spin {
  to {
    transform: rotate(360deg);
  }
}

@media (max-width: 768px) {
  .stock-metrics {
    grid-template-columns: repeat(3, 1fr);
  }

  .chart-container {
    height: 450px;
  }
}
</style>
