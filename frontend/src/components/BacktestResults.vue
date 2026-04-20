<template>
  <div class="results-page">
    <!-- 记录选择 -->
    <div class="selector-section">
      <div class="selector-header">
        <span class="selector-icon">📋</span>
        <span>回测记录选择</span>
      </div>
      <div class="selector-row">
        <el-select v-model="selectedId" @change="loadResult" class="selector-box" placeholder="选择回测记录">
          <el-option v-for="r in recorders" :key="r.id" :label="r.name || `${r.start_time} - ${r.id}`" :value="r.id" />
        </el-select>
        <button class="refresh-btn" @click="loadRecorders" :disabled="loadingRecorders">
          <span v-if="!loadingRecorders">🔄</span>
          <span v-else class="btn-spinner-small"></span>
          <span>刷新</span>
        </button>
        <button class="delete-btn" :disabled="!selectedId" @click="deleteRecorder">
          <span>🗑️</span> 删除
        </button>
        <button class="delete-all-btn" :disabled="!recorders.length" @click="deleteAllRecorders">
          <span>🗑️🗑️</span> 全部删除
        </button>
      </div>
    </div>

    <!-- 加载状态 -->
    <div v-if="loading" class="loading-section">
      <div class="spinner"></div>
      <span>加载中...</span>
    </div>

    <!-- 回测配置信息 -->
    <div v-if="config" class="config-section">
      <div class="config-header">
        <span>⚙️</span>
        <span>回测配置参数</span>
      </div>
      <div class="config-grid">
        <div class="config-item">
          <span class="config-label">回测市场</span>
          <span class="config-value">{{ config.market }}</span>
        </div>
        <div class="config-item">
          <span class="config-label">基准指数</span>
          <span class="config-value">{{ config.benchmark }}</span>
        </div>
        <div class="config-item">
          <span class="config-label">开始日期</span>
          <span class="config-value">{{ config.start_date }}</span>
        </div>
        <div class="config-item">
          <span class="config-label">结束日期</span>
          <span class="config-value">{{ config.end_date }}</span>
        </div>
        <div class="config-item">
          <span class="config-label">初始资金</span>
          <span class="config-value">{{ (config.initial_account / 10000).toFixed(0) }}万</span>
        </div>
        <div class="config-item">
          <span class="config-label">持仓数量</span>
          <span class="config-value">{{ config.topk }}</span>
        </div>
        <div class="config-item">
          <span class="config-label">调仓卖出数</span>
          <span class="config-value">{{ config.n_drop }}</span>
        </div>
        <div class="config-item">
          <span class="config-label">持仓周期</span>
          <span class="config-value">{{ config.hold_days }}天</span>
        </div>
        <div class="config-item">
          <span class="config-label">止损比例</span>
          <span class="config-value">{{ config.stop_loss }}%</span>
        </div>
        <div class="config-item">
          <span class="config-label">策略类型</span>
          <span class="config-value">{{ config.strategy_type }}</span>
        </div>
      </div>
    </div>

    <!-- 关键指标 -->
    <div v-if="metrics" class="metrics-section">
      <div class="metrics-header">
        <span>📊</span>
        <span>关键绩效指标</span>
      </div>
      <div class="metrics-grid">
        <div class="metric-card strategy">
          <div class="metric-icon">📈</div>
          <div class="metric-content">
            <span class="metric-label">策略总收益</span>
            <span class="metric-value">{{ metrics.total_return }}%</span>
          </div>
        </div>
        <div class="metric-card benchmark">
          <div class="metric-icon">📉</div>
          <div class="metric-content">
            <span class="metric-label">基准总收益</span>
            <span class="metric-value">{{ metrics.bench_return }}%</span>
          </div>
        </div>
        <div class="metric-card excess">
          <div class="metric-icon">🎯</div>
          <div class="metric-content">
            <span class="metric-label">超额收益</span>
            <span class="metric-value">{{ metrics.excess_return }}%</span>
          </div>
        </div>
        <div class="metric-card annual">
          <div class="metric-icon">📅</div>
          <div class="metric-content">
            <span class="metric-label">年化收益率</span>
            <span class="metric-value">{{ (metrics.annualized_return * 100).toFixed(2) }}%</span>
          </div>
        </div>
        <div class="metric-card drawdown">
          <div class="metric-icon">⚠️</div>
          <div class="metric-content">
            <span class="metric-label">最大回撤</span>
            <span class="metric-value">{{ (metrics.max_drawdown * 100).toFixed(2) }}%</span>
          </div>
        </div>
        <div class="metric-card ratio">
          <div class="metric-icon">⚖️</div>
          <div class="metric-content">
            <span class="metric-label">信息比率</span>
            <span class="metric-value">{{ metrics.information_ratio }}</span>
          </div>
        </div>
      </div>
    </div>

    <!-- 图表区域 -->
    <div v-if="cumulativeData" class="chart-section">
      <div class="chart-header">
        <span>📈</span>
        <span>累计收益曲线对比</span>
        <span class="chart-hint">点击曲线查看当日持仓</span>
      </div>
      <div class="chart-box">
        <canvas ref="cumulativeRef"></canvas>
      </div>
    </div>

    <div v-if="dailyData" class="chart-section small">
      <div class="chart-header">
        <span>📊</span>
        <span>日收益率分布</span>
        <span class="chart-hint">点击曲线查看当日持仓</span>
      </div>
      <div class="chart-box">
        <canvas ref="dailyRef"></canvas>
      </div>
    </div>

    <!-- 持仓浮动弹窗 -->
    <div v-if="showPositionPopup" class="position-popup-overlay" @click="closePopup">
      <div class="position-popup" @click.stop>
        <div class="popup-header">
          <span>💼 {{ selectedDate }} 持仓明细</span>
          <button class="popup-close" @click="closePopup">✕</button>
        </div>
        <div class="popup-table">
          <el-table :data="positions" style="width: 100%" size="small">
            <el-table-column prop="stock_code" label="股票代码" align="center" width="90" />
            <el-table-column prop="stock_name" label="股票名称" align="center" width="100" />
            <el-table-column prop="weight" label="权重" align="center" width="80">
              <template #default="{ row }">
                <span>{{ (row.weight * 100).toFixed(2) }}%</span>
              </template>
            </el-table-column>
            <el-table-column prop="hold_days" label="持仓天数" align="center" width="80" />
            <el-table-column prop="amount" label="数量" align="center" width="90">
              <template #default="{ row }">
                <span>{{ formatNumber(row.amount) }}</span>
              </template>
            </el-table-column>
            <el-table-column prop="cost_price" label="成本价" align="center" width="90">
              <template #default="{ row }">
                <span>¥{{ row.cost_price.toFixed(2) }}</span>
              </template>
            </el-table-column>
            <el-table-column prop="current_price" label="当日价" align="center" width="90">
              <template #default="{ row }">
                <span>¥{{ row.current_price.toFixed(2) }}</span>
              </template>
            </el-table-column>
            <el-table-column prop="hold_value" label="持仓金额" align="center" width="110">
              <template #default="{ row }">
                <span>¥{{ formatCurrency(row.hold_value) }}</span>
              </template>
            </el-table-column>
            <el-table-column prop="profit" label="盈亏" align="center" width="100">
              <template #default="{ row }">
                <span :class="row.profit >= 0 ? 'up-red' : 'down-green'">
                  {{ row.profit >= 0 ? '+' : '' }}{{ formatCurrency(row.profit) }}
                </span>
              </template>
            </el-table-column>
            <el-table-column prop="profit_rate" label="收益率" align="center" width="90">
              <template #default="{ row }">
                <span :class="row.profit_rate >= 0 ? 'up-red' : 'down-green'">
                  {{ row.profit_rate >= 0 ? '+' : '' }}{{ row.profit_rate.toFixed(2) }}%
                </span>
              </template>
            </el-table-column>
          </el-table>
        </div>
        <div class="popup-count">共 {{ positions.length }} 只股票</div>
      </div>
    </div>

    <!-- 持仓列表 -->
    <div v-if="positions.length" class="positions-section">
      <div class="positions-header">
        <span>💼</span>
        <span>最终持仓明细</span>
        <span class="positions-date">日期：{{ lastDate }}</span>
        <span class="positions-count">共 {{ positions.length }} 只股票</span>
      </div>
      <div class="table-wrapper">
        <el-table
          :data="positions"
          style="width: 100%"
        >
          <el-table-column prop="stock_code" label="股票代码" align="center" />
          <el-table-column prop="stock_name" label="股票名称" align="center" />
          <el-table-column prop="weight" label="权重" align="center">
            <template #default="{ row }">
              <span>{{ (row.weight * 100).toFixed(2) }}%</span>
            </template>
          </el-table-column>
          <el-table-column prop="hold_days" label="持仓天数" align="center" />
          <el-table-column prop="amount" label="数量" align="center">
            <template #default="{ row }">
              <span>{{ formatNumber(row.amount) }}</span>
            </template>
          </el-table-column>
          <el-table-column prop="cost_price" label="成本价" align="center">
            <template #default="{ row }">
              <span>¥{{ row.cost_price.toFixed(2) }}</span>
            </template>
          </el-table-column>
          <el-table-column prop="current_price" label="当前价" align="center">
            <template #default="{ row }">
              <span>¥{{ row.current_price.toFixed(2) }}</span>
            </template>
          </el-table-column>
          <el-table-column prop="hold_value" label="持仓金额" align="center">
            <template #default="{ row }">
              <span>¥{{ formatCurrency(row.hold_value) }}</span>
            </template>
          </el-table-column>
          <el-table-column prop="profit" label="盈亏" align="center">
            <template #default="{ row }">
              <span :class="row.profit >= 0 ? 'up-red' : 'down-green'">
                {{ row.profit >= 0 ? '+' : '' }}{{ formatCurrency(row.profit) }}
              </span>
            </template>
          </el-table-column>
          <el-table-column prop="profit_rate" label="收益率" align="center">
            <template #default="{ row }">
              <span :class="row.profit_rate >= 0 ? 'up-red' : 'down-green'">
                {{ row.profit_rate >= 0 ? '+' : '' }}{{ row.profit_rate.toFixed(2) }}%
              </span>
            </template>
          </el-table-column>
        </el-table>
      </div>
    </div>

    <!-- 历史交易 -->
    <div class="trades-section">
      <div class="trades-header">
        <span>📜</span>
        <span>历史交易记录</span>
        <span class="trades-count">共 {{ allPositionsFlat.length }} 条</span>
      </div>
      <div v-if="allPositionsFlat.length">
        <div class="table-wrapper">
          <el-table
            :data="paginatedTrades"
            style="width: 100%"
            :default-sort="{ prop: 'date', order: 'descending' }"
          >
            <el-table-column prop="date" label="日期" sortable align="center" />
            <el-table-column prop="stock_code" label="股票代码" align="center" />
            <el-table-column prop="stock_name" label="股票名称" align="center" />
            <el-table-column prop="weight" label="权重" align="center">
              <template #default="{ row }">
                <span>{{ (row.weight * 100).toFixed(2) }}%</span>
              </template>
            </el-table-column>
            <el-table-column prop="hold_days" label="持仓天数" align="center" />
            <el-table-column prop="amount" label="数量" align="center">
              <template #default="{ row }">
                <span>{{ formatNumber(row.amount) }}</span>
              </template>
            </el-table-column>
            <el-table-column prop="cost_price" label="成本价" align="center">
              <template #default="{ row }">
                <span>¥{{ row.cost_price.toFixed(2) }}</span>
              </template>
            </el-table-column>
            <el-table-column prop="current_price" label="当日价" align="center">
              <template #default="{ row }">
                <span>¥{{ row.current_price.toFixed(2) }}</span>
              </template>
            </el-table-column>
            <el-table-column prop="hold_value" label="持仓金额" align="center">
              <template #default="{ row }">
                <span>¥{{ formatCurrency(row.hold_value) }}</span>
              </template>
            </el-table-column>
            <el-table-column prop="profit" label="盈亏" align="center">
              <template #default="{ row }">
                <span :class="row.profit >= 0 ? 'up-red' : 'down-green'">
                  {{ row.profit >= 0 ? '+' : '' }}{{ formatCurrency(row.profit) }}
                </span>
              </template>
            </el-table-column>
            <el-table-column prop="profit_rate" label="收益率" align="center">
              <template #default="{ row }">
                <span :class="row.profit_rate >= 0 ? 'up-red' : 'down-green'">
                  {{ row.profit_rate >= 0 ? '+' : '' }}{{ row.profit_rate.toFixed(2) }}%
                </span>
              </template>
            </el-table-column>
          </el-table>
        </div>
        <div class="pagination-wrap">
          <el-pagination
            v-model:current-page="currentTradePage"
            v-model:page-size="tradePageSize"
            :page-sizes="[5, 10, 20, 50]"
            :total="allPositionsFlat.length"
            layout="total, sizes, first, prev, pager, next, last"
            :pager-count="5"
            background
          />
        </div>
      </div>
      <div v-else class="empty-trades">
        <span>暂无历史交易数据</span>
      </div>
    </div>

    <!-- 消息 -->
    <div v-if="error" class="error-box">
      <span>❌</span>
      <span>{{ error }}</span>
    </div>
  </div>
</template>

<script setup>
import { ref, onMounted, nextTick, computed } from 'vue'
import { ElMessage, ElMessageBox } from 'element-plus'
import axios from 'axios'
import { Chart, registerables } from 'chart.js'

Chart.register(...registerables)

const API = '/api'

const loading = ref(false)
const loadingRecorders = ref(false)
const selectedId = ref('')
const recorders = ref([])
const metrics = ref(null)
const config = ref(null)
const cumulativeData = ref(null)
const dailyData = ref(null)
const positions = ref([])
const finalPositions = ref([])
const allPositions = ref({})
const selectedDate = ref('')
const showPositionPopup = ref(false)
const lastDate = ref('')
const error = ref('')
const cumulativeRef = ref(null)
const dailyRef = ref(null)
const currentPage = ref(1)
const pageSize = ref(5)
const currentTradePage = ref(1)
const tradePageSize = ref(5)

let cumulativeChart = null
let dailyChart = null

// 格式化数字
const formatNumber = (num) => {
  return num.toLocaleString('zh-CN', { maximumFractionDigits: 0 })
}

// 格式化货币
const formatCurrency = (num) => {
  return num.toLocaleString('zh-CN', { minimumFractionDigits: 2, maximumFractionDigits: 2 })
}

// 将 allPositions 转换为扁平数组
const allPositionsFlat = computed(() => {
  const result = []
  if (allPositions.value) {
    Object.keys(allPositions.value).forEach(date => {
      allPositions.value[date].forEach(pos => {
        result.push({ ...pos, date })
      })
    })
  }
  // 按日期倒序排列
  result.sort((a, b) => new Date(b.date) - new Date(a.date))
  return result
})

// 分页后的历史交易数据
const paginatedTrades = computed(() => {
  const start = (currentTradePage.value - 1) * tradePageSize.value
  const end = start + tradePageSize.value
  return allPositionsFlat.value.slice(start, end)
})

onMounted(() => loadRecorders())

const loadRecorders = async () => {
  loadingRecorders.value = true
  try {
    const res = await axios.get(`${API}/qlib/backtest_recorders`)
    if (res.data.success) {
      recorders.value = res.data.recorders
      if (recorders.value.length) {
        selectedId.value = recorders.value[0].id
        loadResult()
      }
    }
  } catch (e) {
    ElMessage.error('加载回测记录失败')
  } finally {
    loadingRecorders.value = false
  }
}

const loadResult = async () => {
  if (!selectedId.value) return
  loading.value = true
  error.value = ''
  metrics.value = null
  config.value = null
  cumulativeData.value = null
  dailyData.value = null
  positions.value = []

  try {
    const res = await axios.get(`${API}/qlib/backtest_result/${selectedId.value}`)
    if (res.data.success) {
      metrics.value = res.data.key_metrics
      config.value = {
        hold_days: 3,
        stop_loss: 5,
        ...res.data.config
      }
      cumulativeData.value = res.data.cumulative_data
      dailyData.value = res.data.daily_data
      positions.value = res.data.positions || []
      finalPositions.value = res.data.positions || []
      allPositions.value = res.data.all_positions || {}
      lastDate.value = res.data.last_date || ''
      selectedDate.value = res.data.last_date || ''
      console.log('All positions loaded:', Object.keys(allPositions.value).length, 'dates')
      await nextTick()
      renderCharts()
    } else {
      error.value = res.data.message
    }
  } catch (e) {
    error.value = `加载失败：${e.message}`
  } finally {
    loading.value = false
  }
}

const closePopup = () => {
  showPositionPopup.value = false
  positions.value = finalPositions.value
  selectedDate.value = lastDate.value
}

const renderCharts = () => {
  if (cumulativeRef.value && cumulativeData.value) {
    if (cumulativeChart) cumulativeChart.destroy()
    cumulativeChart = new Chart(cumulativeRef.value, {
      type: 'line',
      data: {
        labels: cumulativeData.value.dates,
        datasets: [
          {
            label: '策略',
            data: cumulativeData.value.strategy,
            borderColor: '#667eea',
            backgroundColor: 'rgba(102,126,234,0.1)',
            borderWidth: 2,
            fill: true,
            tension: 0.3,
            pointRadius: 4,
            pointHoverRadius: 8,
            pointBackgroundColor: '#667eea',
            pointBorderColor: '#fff',
            pointBorderWidth: 2
          },
          {
            label: '基准',
            data: cumulativeData.value.benchmark,
            borderColor: '#38ef7d',
            backgroundColor: 'rgba(56,239,125,0.1)',
            borderWidth: 2,
            fill: true,
            tension: 0.3,
            pointRadius: 4,
            pointHoverRadius: 8,
            pointBackgroundColor: '#38ef7d',
            pointBorderColor: '#fff',
            pointBorderWidth: 2
          }
        ]
      },
      options: {
        responsive: true,
        maintainAspectRatio: false,
        interaction: {
          mode: 'nearest',
          axis: 'x',
          intersect: false
        },
        plugins: {
          title: { display: false },
          tooltip: {
            callbacks: {
              title: (items) => `点击查看 ${items[0].label} 持仓`
            }
          }
        },
        onClick: (event, elements) => {
          console.log('Chart clicked, elements:', elements.length)
          if (elements.length > 0) {
            const index = elements[0].index
            const date = cumulativeData.value.dates[index]
            console.log('Clicked date:', date)
            console.log('allPositions keys:', Object.keys(allPositions.value))
            console.log('allPositions sample:', allPositions.value)
            console.log('Has positions:', allPositions.value[date]?.length)
            selectedDate.value = date
            if (allPositions.value && allPositions.value[date]) {
              positions.value = allPositions.value[date]
              showPositionPopup.value = true
            }
          }
        }
      }
    })
  }
  if (dailyRef.value && dailyData.value) {
    if (dailyChart) dailyChart.destroy()
    dailyChart = new Chart(dailyRef.value, {
      type: 'line',
      data: {
        labels: dailyData.value.dates,
        datasets: [
          {
            label: '策略',
            data: dailyData.value.strategy,
            borderColor: '#667eea',
            borderWidth: 1,
            fill: false,
            tension: 0.3,
            pointRadius: 3,
            pointHoverRadius: 6,
            pointBackgroundColor: '#667eea'
          },
          {
            label: '基准',
            data: dailyData.value.benchmark,
            borderColor: '#38ef7d',
            borderWidth: 1,
            fill: false,
            tension: 0.3,
            pointRadius: 3,
            pointHoverRadius: 6,
            pointBackgroundColor: '#38ef7d'
          }
        ]
      },
      options: {
        responsive: true,
        maintainAspectRatio: false,
        interaction: {
          mode: 'nearest',
          axis: 'x',
          intersect: false
        },
        plugins: { title: { display: false } },
        onClick: (event, elements) => {
          console.log('Daily chart clicked, elements:', elements.length)
          if (elements.length > 0) {
            const index = elements[0].index
            const date = dailyData.value.dates[index]
            console.log('Clicked date:', date)
            selectedDate.value = date
            if (allPositions.value && allPositions.value[date]) {
              positions.value = allPositions.value[date]
              showPositionPopup.value = true
            }
          }
        }
      }
    })
  }
}

const deleteRecorder = async () => {
  try {
    await ElMessageBox.confirm(`确定删除 ${selectedId.value}？`, '确认', { type: 'warning' })
    const res = await axios.delete(`${API}/qlib/backtest_recorders/${selectedId.value}`)
    if (res.data.success) {
      ElMessage.success(res.data.message)
      metrics.value = null
      config.value = null
      cumulativeData.value = null
      dailyData.value = null
      positions.value = []
      loadRecorders()
    } else {
      ElMessage.error(res.data.message)
    }
  } catch (e) {
    if (e !== 'cancel') ElMessage.error(`删除失败`)
  }
}

const deleteAllRecorders = async () => {
  try {
    await ElMessageBox.confirm(`确定删除所有 ${recorders.value.length} 条回测记录？此操作不可恢复！`, '确认全部删除', { type: 'warning' })
    let successCount = 0
    let failCount = 0
    for (const rec of recorders.value) {
      try {
        const res = await axios.delete(`${API}/qlib/backtest_recorders/${rec.id}`)
        if (res.data.success) successCount++
        else failCount++
      } catch (e) {
        failCount++
      }
    }
    if (successCount > 0) {
      ElMessage.success(`成功删除 ${successCount} 条记录`)
    }
    if (failCount > 0) {
      ElMessage.error(`${failCount} 条记录删除失败`)
    }
    metrics.value = null
    config.value = null
    cumulativeData.value = null
    dailyData.value = null
    positions.value = []
    loadRecorders()
  } catch (e) {
    if (e !== 'cancel') ElMessage.error(`删除失败`)
  }
}
</script>

<style scoped>
.results-page {
  padding: 8px;
}

.selector-section {
  background: #f8f9fa;
  padding: 16px;
  border-radius: 12px;
  margin-bottom: 20px;
}

.selector-header {
  display: flex;
  align-items: center;
  gap: 8px;
  margin-bottom: 12px;
  font-weight: 600;
  color: #2c3e50;
}

.selector-row {
  display: flex;
  gap: 12px;
}

.selector-box {
  flex: 1;
}

.delete-btn {
  display: flex;
  align-items: center;
  gap: 6px;
  padding: 10px 16px;
  border: 1px solid #dc3545;
  background: white;
  color: #dc3545;
  border-radius: 8px;
  cursor: pointer;
}

.delete-btn:hover:not(:disabled) {
  background: #dc3545;
  color: white;
}

.delete-btn:disabled {
  opacity: 0.5;
}

.refresh-btn {
  display: flex;
  align-items: center;
  gap: 6px;
  padding: 10px 16px;
  border: 1px solid #667eea;
  background: white;
  color: #667eea;
  border-radius: 8px;
  cursor: pointer;
  font-size: 14px;
  transition: all 0.2s;
}

.refresh-btn:hover:not(:disabled) {
  background: #667eea;
  color: white;
}

.refresh-btn:disabled {
  opacity: 0.5;
  cursor: not-allowed;
}

.delete-all-btn {
  display: flex;
  align-items: center;
  gap: 6px;
  padding: 10px 16px;
  border: 1px solid #dc3545;
  background: white;
  color: #dc3545;
  border-radius: 8px;
  cursor: pointer;
  font-size: 14px;
  transition: all 0.2s;
}

.delete-all-btn:hover:not(:disabled) {
  background: #dc3545;
  color: white;
}

.delete-all-btn:disabled {
  opacity: 0.5;
  cursor: not-allowed;
}

.btn-spinner-small {
  width: 14px;
  height: 14px;
  border: 2px solid rgba(102, 126, 234, 0.3);
  border-top-color: #667eea;
  border-radius: 50%;
  animation: spin 0.8s linear infinite;
}

.loading-section {
  display: flex;
  flex-direction: column;
  align-items: center;
  gap: 16px;
  padding: 48px;
}

.spinner {
  width: 40px;
  height: 40px;
  border: 3px solid #e9ecef;
  border-top-color: #667eea;
  border-radius: 50%;
  animation: spin 0.8s linear infinite;
}

@keyframes spin { to { transform: rotate(360deg); } }

.config-section {
  background: #f8f9fa;
  border-radius: 12px;
  padding: 16px;
  margin-bottom: 24px;
}

.config-header {
  display: flex;
  align-items: center;
  gap: 8px;
  margin-bottom: 16px;
  font-weight: 600;
  color: #2c3e50;
}

.config-grid {
  display: grid;
  grid-template-columns: repeat(8, 1fr);
  gap: 12px;
}

.config-item {
  background: white;
  padding: 12px;
  border-radius: 8px;
  display: flex;
  flex-direction: column;
  gap: 4px;
  border: 1px solid #e9ecef;
}

.config-item .config-label {
  font-size: 12px;
  color: #6c757d;
}

.config-item .config-value {
  font-size: 14px;
  color: #2c3e50;
  font-weight: 600;
}

.metrics-section {
  margin-bottom: 24px;
}

.metrics-header {
  display: flex;
  align-items: center;
  gap: 8px;
  margin-bottom: 16px;
  font-weight: 600;
}

.metrics-grid {
  display: grid;
  grid-template-columns: repeat(6, 1fr);
  gap: 12px;
}

.metric-card {
  background: white;
  padding: 16px;
  border-radius: 10px;
  display: flex;
  align-items: center;
  gap: 12px;
  border: 1px solid #e9ecef;
}

.metric-icon {
  font-size: 24px;
}

.metric-content {
  display: flex;
  flex-direction: column;
}

.metric-label {
  font-size: 12px;
  color: #6c757d;
}

.metric-value {
  font-size: 18px;
  font-weight: 700;
}

.metric-card.strategy .metric-value { color: #667eea; }
.metric-card.benchmark .metric-value { color: #38ef7d; }
.metric-card.excess .metric-value { color: #f5576c; }
.metric-card.annual .metric-value { color: #11998e; }
.metric-card.drawdown .metric-value { color: #dc3545; }
.metric-card.ratio .metric-value { color: #6c757d; }

.chart-section {
  background: white;
  padding: 16px;
  border-radius: 12px;
  border: 1px solid #e9ecef;
  margin-bottom: 20px;
}

.chart-section.small {
  padding: 12px;
}

.chart-header {
  display: flex;
  align-items: center;
  gap: 8px;
  margin-bottom: 12px;
  font-weight: 600;
}

.chart-hint {
  margin-left: auto;
  font-size: 12px;
  color: #667eea;
  background: rgba(102, 126, 234, 0.1);
  padding: 4px 12px;
  border-radius: 6px;
}

.chart-box {
  height: 250px;
}

.chart-section.small .chart-box {
  height: 150px;
}

/* 持仓浮动弹窗 */
.position-popup-overlay {
  position: fixed;
  top: 0;
  left: 0;
  right: 0;
  bottom: 0;
  background: rgba(0, 0, 0, 0.5);
  z-index: 999;
  display: flex;
  align-items: center;
  justify-content: center;
}

.position-popup {
  background: white;
  border-radius: 16px;
  box-shadow: 0 8px 32px rgba(0, 0, 0, 0.3);
  max-width: 1000px;
  width: 95%;
  max-height: 80vh;
  overflow: hidden;
  animation: popupIn 0.2s ease;
}

@keyframes popupIn {
  from { opacity: 0; transform: scale(0.9); }
  to { opacity: 1; transform: scale(1); }
}

.popup-header {
  display: flex;
  justify-content: space-between;
  align-items: center;
  padding: 16px;
  background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
  color: white;
  font-weight: 600;
}

.popup-close {
  background: rgba(255, 255, 255, 0.2);
  border: none;
  color: white;
  width: 28px;
  height: 28px;
  border-radius: 8px;
  cursor: pointer;
  font-size: 16px;
  display: flex;
  align-items: center;
  justify-content: center;
}

.popup-close:hover {
  background: rgba(255, 255, 255, 0.3);
}

.popup-table {
  padding: 12px;
  overflow-y: auto;
  max-height: calc(80vh - 100px);
}

.popup-count {
  padding: 12px;
  background: #f8f9fa;
  text-align: center;
  font-size: 13px;
  color: #6c757d;
}

/* 表格样式 */
.positions-section {
  background: white;
  padding: 16px;
  border-radius: 12px;
  border: 1px solid #e9ecef;
  box-shadow: 0 2px 8px rgba(0, 0, 0, 0.05);
}

.positions-header {
  display: flex;
  align-items: center;
  gap: 8px;
  margin-bottom: 16px;
  font-weight: 600;
  color: #2c3e50;
}

.positions-date {
  margin-left: 8px;
  font-size: 13px;
  color: #6c757d;
  font-weight: 400;
}

.positions-count {
  margin-left: auto;
  font-size: 13px;
  color: #6c757d;
  background: #f8f9fa;
  padding: 4px 12px;
  border-radius: 6px;
}

.table-wrapper {
  border-radius: 8px;
  overflow: hidden;
}

.modern-table {
  background: white;
}

.modern-table :deep(.el-table__header th) {
  background: linear-gradient(135deg, #667eea 0%, #764ba2 100%) !important;
  color: #ffffff !important;
  font-weight: 600 !important;
  font-size: 14px !important;
  border: none !important;
}

.modern-table :deep(.el-table__row) {
  transition: all 0.2s ease;
  border-bottom: 1px solid #f0f0f0;
}

.modern-table :deep(.el-table__row:hover) {
  background: linear-gradient(90deg, rgba(102, 126, 234, 0.05) 0%, rgba(118, 75, 162, 0.05) 100%) !important;
  transform: scale(1.002);
}

.modern-table :deep(.el-table__cell) {
  padding: 12px 8px !important;
  font-size: 13px;
}

/* 数值标签样式 */
.value-tag {
  display: inline-block;
  background: linear-gradient(135deg, rgba(102, 126, 234, 0.1) 0%, rgba(118, 75, 162, 0.1) 100%);
  color: #667eea;
  padding: 4px 10px;
  border-radius: 6px;
  font-weight: 600;
  font-size: 13px;
}

.number-value {
  color: #2c3e50;
  font-weight: 500;
  font-family: 'Consolas', 'Monaco', monospace;
}

.number-value.highlight {
  color: #667eea;
  font-weight: 600;
}

/* 盈亏徽章样式 */
.profit-loss-badge {
  display: inline-block;
  padding: 4px 12px;
  border-radius: 6px;
  font-weight: 600;
  font-size: 13px;
  min-width: 70px;
  text-align: center;
}

.profit-badge {
  background: linear-gradient(135deg, rgba(56, 239, 125, 0.15) 0%, rgba(17, 153, 142, 0.15) 100%);
  color: #11998e;
  border: 1px solid rgba(17, 153, 142, 0.3);
}

.loss-badge {
  background: linear-gradient(135deg, rgba(220, 53, 69, 0.15) 0%, rgba(245, 87, 108, 0.15) 100%);
  color: #dc3545;
  border: 1px solid rgba(220, 53, 69, 0.3);
}

/* 分页样式 */
.pagination-wrap {
  margin-top: 20px;
  display: flex;
  justify-content: center;
}

.modern-pagination :deep(.el-pagination) {
  padding: 16px 8px;
}

.modern-pagination :deep(.el-pager li) {
  background: white;
  border: 1px solid #e9ecef;
  border-radius: 6px;
  margin: 0 3px;
  min-width: 32px;
  height: 32px;
  line-height: 32px;
  transition: all 0.2s;
}

.modern-pagination :deep(.el-pager li:hover) {
  background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
  color: white;
  border-color: transparent;
}

.modern-pagination :deep(.el-pager li.is-active) {
  background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
  color: white;
  border-color: transparent;
  font-weight: 600;
}

.modern-pagination :deep(.el-pagination .btn-prev),
.modern-pagination :deep(.el-pagination .btn-next) {
  background: white;
  border: 1px solid #e9ecef;
  border-radius: 6px;
  margin: 0 3px;
  min-width: 32px;
  height: 32px;
  transition: all 0.2s;
}

.modern-pagination :deep(.el-pagination .btn-prev:hover),
.modern-pagination :deep(.el-pagination .btn-next:hover) {
  background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
  color: white;
  border-color: transparent;
}

.modern-pagination :deep(.el-select) {
  --el-select-hover-border: #667eea;
}

/* 历史交易样式 */
.trades-section {
  background: white;
  padding: 16px;
  border-radius: 12px;
  border: 1px solid #e9ecef;
  box-shadow: 0 2px 8px rgba(0, 0, 0, 0.05);
  margin-top: 20px;
}

.trades-header {
  display: flex;
  align-items: center;
  gap: 8px;
  margin-bottom: 16px;
  font-weight: 600;
  color: #2c3e50;
}

.trades-count {
  margin-left: auto;
  font-size: 13px;
  color: #6c757d;
  background: #f8f9fa;
  padding: 4px 12px;
  border-radius: 6px;
}

.empty-trades {
  padding: 48px;
  text-align: center;
  color: #6c757d;
  font-size: 14px;
  background: #f8f9fa;
  border-radius: 8px;
}

/* 涨红跌绿 */
.up-red { color: #dc3545; font-weight: 600; }
.down-green { color: #28a745; font-weight: 600; }

.error-box {
  display: flex;
  align-items: center;
  gap: 12px;
  padding: 20px;
  background: linear-gradient(135deg, #f8d7da 0%, #f5c6cb 100%);
  color: #721c24;
  border-radius: 12px;
  margin-top: 20px;
}
</style>
