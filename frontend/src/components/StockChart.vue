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
        <span class="tag-label">快速选择：</span>
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

    <!-- K线图区域 -->
    <div class="chart-section">
      <div class="chart-header">
        <h3>K线图</h3>
        <div class="time-range">
          <el-radio-group v-model="timeRange" size="small" @change="loadStockData">
            <el-radio-button label="1M">1月</el-radio-button>
            <el-radio-button label="3M">3月</el-radio-button>
            <el-radio-button label="6M">6月</el-radio-button>
            <el-radio-button label="1Y">1年</el-radio-button>
            <el-radio-button label="YTD">本年</el-radio-button>
            <el-radio-button label="ALL">全部</el-radio-button>
          </el-radio-group>
        </div>
      </div>
      <div ref="chartContainer" class="chart-container"></div>
    </div>

    <!-- 加载状态 -->
    <div v-if="loading" class="loading-overlay">
      <div class="loading-spinner"></div>
      <span>加载中...</span>
    </div>
  </div>
</template>

<script setup>
import { ref, onMounted, onUnmounted, computed, nextTick, watch } from 'vue'
import * as echarts from 'echarts'
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
const timeRange = ref('3M')
const chartContainer = ref(null)
let chartInstance = null

// 常用股票
const quickStocks = [
  { code: '000001.SZ', name: '平安银行' },
  { code: '000002.SZ', name: '万科A' },
  { code: '600000.SH', name: '浦发银行' },
  { code: '600036.SH', name: '招商银行' },
  { code: '000858.SZ', name: '五粮液' },
  { code: '600519.SH', name: '贵州茅台' },
  { code: '000333.SZ', name: '美的集团' },
  { code: '002415.SZ', name: '海康威视' }
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

// 初始化图表
const initChart = () => {
  if (!chartContainer.value) return

  chartInstance = echarts.init(chartContainer.value)

  const option = {
    tooltip: {
      trigger: 'axis',
      axisPointer: {
        type: 'cross'
      },
      formatter: function (params) {
        const data = params[0]
        if (!data) return ''
        const date = data.name
        const values = data.data
        return `
          <div style="font-weight:bold;margin-bottom:5px">${date}</div>
          <div>开盘: ${values[1]}</div>
          <div>收盘: ${values[2]}</div>
          <div>最低: ${values[3]}</div>
          <div>最高: ${values[4]}</div>
          <div>成交量: ${(values[5] / 10000).toFixed(2)}万</div>
        `
      }
    },
    legend: {
      data: ['日K', 'MA5', 'MA10', 'MA20', 'MA30'],
      top: 10
    },
    grid: [
      {
        left: '10%',
        right: '8%',
        height: '50%'
      },
      {
        left: '10%',
        right: '8%',
        top: '68%',
        height: '16%'
      }
    ],
    xAxis: [
      {
        type: 'category',
        data: [],
        scale: true,
        boundaryGap: false,
        axisLine: { onZero: false },
        splitLine: { show: false },
        min: 'dataMin',
        max: 'dataMax'
      },
      {
        type: 'category',
        gridIndex: 1,
        data: [],
        scale: true,
        boundaryGap: false,
        axisLine: { onZero: false },
        axisTick: { show: false },
        splitLine: { show: false },
        axisLabel: { show: false },
        min: 'dataMin',
        max: 'dataMax'
      }
    ],
    yAxis: [
      {
        scale: true,
        splitArea: {
          show: true
        }
      },
      {
        scale: true,
        gridIndex: 1,
        splitNumber: 2,
        axisLabel: { show: false },
        axisLine: { show: false },
        axisTick: { show: false },
        splitLine: { show: false }
      }
    ],
    dataZoom: [
      {
        type: 'inside',
        xAxisIndex: [0, 1],
        start: 50,
        end: 100
      },
      {
        show: true,
        xAxisIndex: [0, 1],
        type: 'slider',
        top: '85%',
        start: 50,
        end: 100
      }
    ],
    series: [
      {
        name: '日K',
        type: 'candlestick',
        data: [],
        itemStyle: {
          color: '#ef232a',
          color0: '#14b143',
          borderColor: '#ef232a',
          borderColor0: '#14b143'
        }
      },
      {
        name: 'MA5',
        type: 'line',
        data: [],
        smooth: true,
        lineStyle: { opacity: 0.5 }
      },
      {
        name: 'MA10',
        type: 'line',
        data: [],
        smooth: true,
        lineStyle: { opacity: 0.5 }
      },
      {
        name: 'MA20',
        type: 'line',
        data: [],
        smooth: true,
        lineStyle: { opacity: 0.5 }
      },
      {
        name: 'MA30',
        type: 'line',
        data: [],
        smooth: true,
        lineStyle: { opacity: 0.5 }
      },
      {
        name: '成交量',
        type: 'bar',
        xAxisIndex: 1,
        yAxisIndex: 1,
        data: [],
        itemStyle: {
          color: (params) => {
            const data = params.data
            return data[2] > data[1] ? '#ef232a' : '#14b143'
          }
        }
      }
    ]
  }

  chartInstance.setOption(option)

  // 响应窗口大小变化
  window.addEventListener('resize', handleResize)
}

// 处理窗口大小变化
const handleResize = () => {
  if (chartInstance) {
    chartInstance.resize()
  }
}

// 计算移动平均线
const calculateMA = (dayCount, data) => {
  const result = []
  for (let i = 0; i < data.length; i++) {
    if (i < dayCount - 1) {
      result.push('-')
      continue
    }
    let sum = 0
    for (let j = 0; j < dayCount; j++) {
      sum += parseFloat(data[i - j][1])
    }
    result.push((sum / dayCount).toFixed(2))
  }
  return result
}

// 加载股票数据
const loadStockData = async () => {
  if (!currentStock.value.code) return

  loading.value = true
  try {
    // 计算日期范围
    const endDate = new Date()
    const startDate = new Date()
    switch (timeRange.value) {
      case '1M':
        startDate.setMonth(startDate.getMonth() - 1)
        break
      case '3M':
        startDate.setMonth(startDate.getMonth() - 3)
        break
      case '6M':
        startDate.setMonth(startDate.getMonth() - 6)
        break
      case '1Y':
        startDate.setFullYear(startDate.getFullYear() - 1)
        break
      case 'YTD':
        startDate.setMonth(0, 1)
        break
      case 'ALL':
        startDate.setFullYear(2020, 0, 1)
        break
    }

    // 调用后端API获取数据
    const res = await axios.get(`${API_BASE_URL}/stock/quote`, {
      params: {
        code: currentStock.value.code,
        start_date: startDate.toISOString().split('T')[0],
        end_date: endDate.toISOString().split('T')[0]
      }
    })

    if (res.data.success) {
      const data = res.data.data
      updateStockInfo(data)
      updateChart(data)
    } else {
      // 使用模拟数据演示
      useMockData()
    }
  } catch (e) {
    console.error('加载股票数据失败:', e)
    useMockData()
  } finally {
    loading.value = false
  }
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
  if (!chartInstance || !data || data.length === 0) return

  const dates = data.map(item => item.date)
  const candleData = data.map(item => [item.open, item.close, item.low, item.high, item.volume])
  const volumeData = data.map((item, index) => {
    const prevClose = index > 0 ? data[index - 1].close : item.open
    return [item.date, item.volume, item.close, prevClose]
  })

  const closes = data.map(item => item.close)

  chartInstance.setOption({
    xAxis: [
      { data: dates },
      { data: dates }
    ],
    series: [
      {
        name: '日K',
        data: candleData
      },
      {
        name: 'MA5',
        data: calculateMA(5, candleData),
        type: 'line',
        smooth: true,
        lineStyle: { opacity: 0.5, color: '#f0c040' }
      },
      {
        name: 'MA10',
        data: calculateMA(10, candleData),
        type: 'line',
        smooth: true,
        lineStyle: { opacity: 0.5, color: '#409eff' }
      },
      {
        name: 'MA20',
        data: calculateMA(20, candleData),
        type: 'line',
        smooth: true,
        lineStyle: { opacity: 0.5, color: '#67c23a' }
      },
      {
        name: 'MA30',
        data: calculateMA(30, candleData),
        type: 'line',
        smooth: true,
        lineStyle: { opacity: 0.5, color: '#e6a23c' }
      },
      {
        name: '成交量',
        data: volumeData,
        type: 'bar',
        xAxisIndex: 1,
        yAxisIndex: 1
      }
    ]
  })
}

// 使用模拟数据
const useMockData = () => {
  const dates = []
  const data = []
  const basePrice = 10 + Math.random() * 50

  for (let i = 60; i >= 0; i--) {
    const date = new Date()
    date.setDate(date.getDate() - i)
    dates.push(date.toISOString().split('T')[0])

    const open = basePrice + (Math.random() - 0.5) * 2
    const close = open + (Math.random() - 0.5) * 1.5
    const low = Math.min(open, close) - Math.random() * 0.5
    const high = Math.max(open, close) + Math.random() * 0.5
    const volume = Math.floor(Math.random() * 1000000) + 500000
    const preClose = open

    data.push({
      date: dates[dates.length - 1],
      open: parseFloat(open.toFixed(2)),
      close: parseFloat(close.toFixed(2)),
      low: parseFloat(low.toFixed(2)),
      high: parseFloat(high.toFixed(2)),
      volume: volume,
      amount: volume * close,
      pre_close: preClose
    })
  }

  // 更新最新价格
  if (data.length > 0) {
    const latest = data[data.length - 1]
    currentStock.value = {
      ...currentStock.value,
      price: latest.close,
      change: ((latest.close - data[data.length - 2]?.open || latest.open) / (data[data.length - 2]?.open || latest.open) * 100),
      open: latest.open,
      high: latest.high,
      low: latest.low,
      preClose: data[data.length - 2]?.close || latest.open,
      volume: latest.volume,
      amount: latest.amount
    }
  }

  updateChart(data)
}

// 搜索股票
const searchStock = () => {
  const code = stockCodeInput.value.trim()
  if (!code) return

  currentStock.value.code = code
  currentStock.value.name = code

  // 从快速选择中查找名称
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
    // 默认显示第一个股票
    if (quickStocks.length > 0) {
      selectStock(quickStocks[0].code)
    }
  })
  // 监听全局股票选择事件
  window.addEventListener('select-stock', handleStockSelect)
})

onUnmounted(() => {
  if (chartInstance) {
    chartInstance.dispose()
    chartInstance = null
  }
  window.removeEventListener('resize', handleResize)
  window.removeEventListener('select-stock', handleStockSelect)
})
</script>

<style scoped>
.stock-chart-page {
  padding: 20px;
  max-width: 1200px;
  margin: 0 auto;
}

/* 搜索区域 */
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

/* 股票信息卡片 */
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

/* 图表区域 */
.chart-section {
  background: white;
  border-radius: 16px;
  padding: 24px;
  box-shadow: 0 4px 20px rgba(0, 0, 0, 0.08);
}

.chart-header {
  display: flex;
  justify-content: space-between;
  align-items: center;
  margin-bottom: 20px;
}

.chart-header h3 {
  font-size: 18px;
  font-weight: 600;
  color: #2c3e50;
  margin: 0;
}

.chart-container {
  width: 100%;
  height: 500px;
}

/* 加载动画 */
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

/* 响应式 */
@media (max-width: 768px) {
  .stock-metrics {
    grid-template-columns: repeat(3, 1fr);
  }

  .chart-container {
    height: 400px;
  }
}
</style>
