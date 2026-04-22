<template>
  <div class="eval-page">
    <!-- 页面标题 -->
    <div class="page-header">
      <h2>📊 模型评估</h2>
      <p class="subtitle">查看训练好的模型在测试集上的表现</p>
    </div>

    <!-- 训练记录选择 -->
    <div class="selector-section">
      <div class="selector-card">
        <label class="selector-label">选择训练记录</label>
        <el-select
          v-model="selectedRecorderId"
          placeholder="请选择训练记录"
          class="recorder-select"
          @change="onRecorderChange"
          :loading="loadingRecorders"
        >
          <el-option
            v-for="recorder in recorders"
            :key="recorder.id"
            :label="formatRecorderLabel(recorder)"
            :value="recorder.id"
          />
        </el-select>
        <button class="action-btn load-btn" :disabled="!selectedRecorderId || loading" @click="loadEvaluation">
          <span v-if="!loading">📊</span>
          <span v-else class="btn-spinner-small"></span>
          <span>加载评估</span>
        </button>
        <div class="btn-divider"></div>
        <button class="action-btn refresh-btn" @click="refreshRecorders" :disabled="loadingRecorders">
          <span v-if="!loadingRecorders">🔄</span>
          <span v-else class="btn-spinner-small"></span>
          <span>刷新</span>
        </button>
        <button class="action-btn delete-btn" :disabled="!selectedRecorderId" @click="deleteSelectedRecorder">
          <span>🗑️</span>
          <span>删除</span>
        </button>
        <button class="action-btn delete-all-btn" :disabled="!recorders.length" @click="deleteAllRecorders">
          <span>🗑️🗑️</span>
          <span>全部删除</span>
        </button>
      </div>
    </div>

    <!-- 加载中 -->
    <div v-if="loading" class="loading-section">
      <el-skeleton :rows="6" animated />
    </div>

    <!-- 评估结果 -->
    <div v-if="evaluationResult && !loading" class="result-section">
      <!-- 基本信息卡片 -->
      <div class="info-cards">
        <div class="info-card">
          <span class="info-label">市场</span>
          <span class="info-value">{{ getMarketLabel(evaluationResult.params?.market) }}</span>
        </div>
        <div class="info-card">
          <span class="info-label">基准指数</span>
          <span class="info-value">{{ evaluationResult.params?.benchmark || '-' }}</span>
        </div>
        <div class="info-card">
          <span class="info-label">模型类型</span>
          <span class="info-value">{{ evaluationResult.params?.model_type || '-' }}</span>
        </div>
        <div class="info-card">
          <span class="info-label">训练日期</span>
          <span class="info-value">{{ evaluationResult.params?.train_start_date }} ~ {{ evaluationResult.params?.train_end_date }}</span>
        </div>
      </div>

      <!-- 关键指标卡片 -->
      <div class="metrics-section">
        <h3>关键评估指标</h3>
        <div class="metrics-grid">
          <div class="metric-card">
            <div class="metric-icon">📈</div>
            <div class="metric-content">
              <div class="metric-header">
                <span class="metric-label">IC</span>
                <el-tooltip placement="top" effect="light">
                  <template #content>
                    <div class="metric-tooltip">
                      <strong>IC (Information Coefficient)</strong><br/>
                      预测值与实际收益的相关系数。<br/><br/>
                      <strong>解读：</strong><br/>
                      • IC > 0.05：预测能力较强<br/>
                      • IC > 0.02：有一定预测能力<br/>
                      • IC < 0：预测方向相反<br/><br/>
                      <strong>计算：</strong> Pearson相关系数
                    </div>
                  </template>
                  <el-icon class="help-icon"><QuestionFilled /></el-icon>
                </el-tooltip>
              </div>
              <span class="metric-value" :class="getMetricClass(evaluationResult.metrics?.IC)">
                {{ formatMetric(evaluationResult.metrics?.IC) }}
              </span>
              <span class="metric-desc">信息系数</span>
            </div>
          </div>
          <div class="metric-card">
            <div class="metric-icon">🎯</div>
            <div class="metric-content">
              <div class="metric-header">
                <span class="metric-label">ICIR</span>
                <el-tooltip placement="top" effect="light">
                  <template #content>
                    <div class="metric-tooltip">
                      <strong>ICIR (Information Ratio)</strong><br/>
                      IC的均值除以IC的标准差，衡量预测稳定性。<br/><br/>
                      <strong>解读：</strong><br/>
                      • ICIR > 0.5：预测非常稳定<br/>
                      • ICIR > 0.2：预测较稳定<br/>
                      • ICIR < 0：预测不稳定<br/><br/>
                      <strong>计算：</strong> IC均值 / IC标准差
                    </div>
                  </template>
                  <el-icon class="help-icon"><QuestionFilled /></el-icon>
                </el-tooltip>
              </div>
              <span class="metric-value" :class="getMetricClass(evaluationResult.metrics?.ICIR)">
                {{ formatMetric(evaluationResult.metrics?.ICIR) }}
              </span>
              <span class="metric-desc">信息比率</span>
            </div>
          </div>
          <div class="metric-card">
            <div class="metric-icon">📊</div>
            <div class="metric-content">
              <div class="metric-header">
                <span class="metric-label">Rank IC</span>
                <el-tooltip placement="top" effect="light">
                  <template #content>
                    <div class="metric-tooltip">
                      <strong>Rank IC</strong><br/>
                      预测排名与实际收益排名的相关系数。<br/><br/>
                      <strong>解读：</strong><br/>
                      • Rank IC > 0.05：排序预测能力强<br/>
                      • Rank IC > 0.02：有一定排序能力<br/>
                      • Rank IC < 0：排序方向相反<br/><br/>
                      <strong>计算：</strong> Spearman秩相关系数<br/>
                      <strong>优势：</strong> 对异常值更稳健
                    </div>
                  </template>
                  <el-icon class="help-icon"><QuestionFilled /></el-icon>
                </el-tooltip>
              </div>
              <span class="metric-value" :class="getMetricClass(evaluationResult.metrics?.Rank_IC)">
                {{ formatMetric(evaluationResult.metrics?.Rank_IC) }}
              </span>
              <span class="metric-desc">排名信息系数</span>
            </div>
          </div>
          <div class="metric-card">
            <div class="metric-icon">🎲</div>
            <div class="metric-content">
              <div class="metric-header">
                <span class="metric-label">Rank ICIR</span>
                <el-tooltip placement="top" effect="light">
                  <template #content>
                    <div class="metric-tooltip">
                      <strong>Rank ICIR</strong><br/>
                      Rank IC的均值除以标准差，衡量排序预测稳定性。<br/><br/>
                      <strong>解读：</strong><br/>
                      • Rank ICIR > 0.5：排序预测非常稳定<br/>
                      • Rank ICIR > 0.2：排序预测较稳定<br/>
                      • Rank ICIR < 0：排序预测不稳定<br/><br/>
                      <strong>计算：</strong> Rank IC均值 / Rank IC标准差
                    </div>
                  </template>
                  <el-icon class="help-icon"><QuestionFilled /></el-icon>
                </el-tooltip>
              </div>
              <span class="metric-value" :class="getMetricClass(evaluationResult.metrics?.Rank_ICIR)">
                {{ formatMetric(evaluationResult.metrics?.Rank_ICIR) }}
              </span>
              <span class="metric-desc">排名信息比率</span>
            </div>
          </div>
          <div class="metric-card">
            <div class="metric-icon">📉</div>
            <div class="metric-content">
              <div class="metric-header">
                <span class="metric-label">Long Precision</span>
                <el-tooltip placement="top" effect="light">
                  <template #content>
                    <div class="metric-tooltip">
                      <strong>Long Precision (多头精度)</strong><br/>
                      预测为正收益的股票中，实际正收益的比例。<br/><br/>
                      <strong>解读：</strong><br/>
                      • > 50%：多头预测有效<br/>
                      • > 55%：多头预测较好<br/>
                      • > 60%：多头预测优秀<br/><br/>
                      <strong>应用：</strong> 用于选股策略，判断买入信号的可靠性
                    </div>
                  </template>
                  <el-icon class="help-icon"><QuestionFilled /></el-icon>
                </el-tooltip>
              </div>
              <span class="metric-value">
                {{ formatPercent(evaluationResult.metrics?.Long_precision) }}
              </span>
              <span class="metric-desc">多头精度</span>
            </div>
          </div>
          <div class="metric-card">
            <div class="metric-icon">📈</div>
            <div class="metric-content">
              <div class="metric-header">
                <span class="metric-label">Short Precision</span>
                <el-tooltip placement="top" effect="light">
                  <template #content>
                    <div class="metric-tooltip">
                      <strong>Short Precision (空头精度)</strong><br/>
                      预测为负收益的股票中，实际负收益的比例。<br/><br/>
                      <strong>解读：</strong><br/>
                      • > 50%：空头预测有效<br/>
                      • > 55%：空头预测较好<br/>
                      • > 60%：空头预测优秀<br/><br/>
                      <strong>应用：</strong> 用于做空策略或规避风险股票
                    </div>
                  </template>
                  <el-icon class="help-icon"><QuestionFilled /></el-icon>
                </el-tooltip>
              </div>
              <span class="metric-value">
                {{ formatPercent(evaluationResult.metrics?.Short_precision) }}
              </span>
              <span class="metric-desc">空头精度</span>
            </div>
          </div>
        </div>
      </div>

      <!-- 图表区域 -->
      <div class="charts-section">
        <h3>可视化分析</h3>
        <div class="charts-grid">
          <!-- 测试集分组收益累计曲线 -->
          <div class="chart-card">
            <div class="chart-header">
              <h4>测试集分组收益</h4>
              <el-tooltip content="测试集上按预测分数分为5组，Group1为最高分，Group5为最低分">
                <el-icon><QuestionFilled /></el-icon>
              </el-tooltip>
            </div>
            <div ref="testReturnChart" class="chart-container"></div>
          </div>

          <!-- IC时间序列 -->
          <div class="chart-card">
            <div class="chart-header">
              <h4>IC时间序列</h4>
              <el-tooltip content="IC：预测值与真实收益的相关系数；Rank IC：预测排名的相关系数">
                <el-icon><QuestionFilled /></el-icon>
              </el-tooltip>
            </div>
            <div ref="icChart" class="chart-container"></div>
          </div>

          <!-- IC分布直方图 -->
          <div class="chart-card">
            <div class="chart-header">
              <h4>IC分布</h4>
              <el-tooltip content="展示IC值的分布情况">
                <el-icon><QuestionFilled /></el-icon>
              </el-tooltip>
            </div>
            <div ref="icDistChart" class="chart-container"></div>
          </div>

          <!-- 月度IC热力图 -->
          <div class="chart-card">
            <div class="chart-header">
              <h4>月度IC热力图</h4>
              <el-tooltip content="按月展示的IC平均值">
                <el-icon><QuestionFilled /></el-icon>
              </el-tooltip>
            </div>
            <div ref="monthlyIcChart" class="chart-container"></div>
          </div>
        </div>
      </div>

      <!-- 详细数据表格 -->
      <div class="detail-section" v-if="evaluationResult.test_returns">
        <h3>测试集分组收益详情</h3>
        <el-table :data="testReturnTableData" style="width: 100%" border>
          <el-table-column prop="group" label="分组" width="120" />
          <el-table-column prop="avg_return" label="平均日收益" width="150">
            <template #default="scope">
              <span :class="scope.row.avg_return >= 0 ? 'positive' : 'negative'">
                {{ formatPercent(scope.row.avg_return) }}
              </span>
            </template>
          </el-table-column>
          <el-table-column prop="cum_return" label="累计收益" width="150">
            <template #default="scope">
              <span :class="scope.row.cum_return >= 0 ? 'positive' : 'negative'">
                {{ formatPercent(scope.row.cum_return) }}
              </span>
            </template>
          </el-table-column>
          <el-table-column prop="description" label="说明" />
        </el-table>
      </div>
    </div>

    <!-- 空状态 -->
    <div v-if="!evaluationResult && !loading && !selectedRecorderId" class="empty-state">
      <el-empty description="请选择训练记录并点击加载评估">
        <template #image>
          <div class="empty-icon">📊</div>
        </template>
      </el-empty>
    </div>
  </div>
</template>

<script setup>
import { ref, onMounted, onUnmounted, nextTick, computed } from 'vue'
import { QuestionFilled } from '@element-plus/icons-vue'
import * as echarts from 'echarts'
import axios from 'axios'
import { ElMessage, ElMessageBox } from 'element-plus'

const API_BASE_URL = '/api'

// 响应式数据
const recorders = ref([])
const selectedRecorderId = ref('')
const loadingRecorders = ref(false)
const loading = ref(false)
const evaluationResult = ref(null)

// 图表实例
const testReturnChart = ref(null)
const icChart = ref(null)
const icDistChart = ref(null)
const monthlyIcChart = ref(null)

let testChartInstance = null
let icChartInstance = null
let icDistChartInstance = null
let monthlyIcChartInstance = null

// 市场标签映射
const marketLabels = {
  csi300: '沪深300',
  csi500: '中证500',
  csi800: '中证800',
  csi1000: '中证1000',
  csiall: 'CSI All',
  all: '全部市场'
}

// 获取市场标签
const getMarketLabel = (market) => marketLabels[market] || market || '-'

// 格式化记录标签
const formatRecorderLabel = (recorder) => {
  const name = recorder.name || recorder.id
  const startTime = recorder.start_time || '未知时间'
  const market = getMarketLabel(recorder.params?.market)
  return `${name} - ${market} - ${startTime}`
}

// 格式化指标
const formatMetric = (value) => {
  if (value === undefined || value === null) return '-'
  return value.toFixed(4)
}

// 格式化百分比
const formatPercent = (value) => {
  if (value === undefined || value === null) return '-'
  return (value * 100).toFixed(2) + '%'
}

// 获取指标样式类
const getMetricClass = (value) => {
  if (value === undefined || value === null) return ''
  return value >= 0 ? 'positive' : 'negative'
}

// 计算测试集分组收益表格数据
const testReturnTableData = computed(() => {
  if (!evaluationResult.value?.test_returns) return []

  const test_returns = evaluationResult.value.test_returns
  const data = []

  for (let i = 1; i <= 5; i++) {
    const groupKey = `Group${i}`
    const returns = test_returns[groupKey] || []
    const avgReturn = returns.length > 0 ? returns.reduce((a, b) => a + b, 0) / returns.length : 0
    const cumReturn = returns.length > 0 ? returns.reduce((a, b) => a + b, 0) : 0

    data.push({
      group: groupKey,
      avg_return: avgReturn,
      cum_return: cumReturn,
      description: i === 1 ? '预测分数最高组' : i === 5 ? '预测分数最低组' : '中间组'
    })
  }

  // 添加多空收益
  if (test_returns.long_short) {
    const lsReturns = test_returns.long_short
    const avgLsReturn = lsReturns.reduce((a, b) => a + b, 0) / lsReturns.length
    const cumLsReturn = lsReturns.reduce((a, b) => a + b, 0)
    data.push({
      group: 'Long-Short',
      avg_return: avgLsReturn,
      cum_return: cumLsReturn,
      description: 'Group1 - Group5 多空对冲'
    })
  }

  return data
})

// 获取训练记录列表
const fetchRecorders = async () => {
  loadingRecorders.value = true
  try {
    const response = await axios.get(`${API_BASE_URL}/qlib/recorders`)
    if (response.data.success) {
      recorders.value = response.data.recorders || []
    }
  } catch (error) {
    console.error('获取训练记录失败:', error)
  } finally {
    loadingRecorders.value = false
  }
}

// 当选择记录变化时
const onRecorderChange = () => {
  evaluationResult.value = null
  disposeCharts()
  // 自动加载评估
  if (selectedRecorderId.value) {
    loadEvaluation()
  }
}

// 销毁图表实例
const disposeCharts = () => {
  if (testChartInstance) {
    testChartInstance.dispose()
    testChartInstance = null
  }
  if (icChartInstance) {
    icChartInstance.dispose()
    icChartInstance = null
  }
  if (icDistChartInstance) {
    icDistChartInstance.dispose()
    icDistChartInstance = null
  }
  if (monthlyIcChartInstance) {
    monthlyIcChartInstance.dispose()
    monthlyIcChartInstance = null
  }
}

// 加载评估数据
const loadEvaluation = async () => {
  if (!selectedRecorderId.value) return

  loading.value = true
  disposeCharts()

  try {
    const response = await axios.get(`${API_BASE_URL}/qlib/train_result/${selectedRecorderId.value}`)
    console.log('API response:', response.data)
    if (response.data.success) {
      evaluationResult.value = response.data
      console.log('evaluationResult set, group_returns:', response.data.group_returns)
      console.log('evaluationResult set, ic_data:', response.data.ic_data)
      await nextTick()
      console.log('nextTick done, checking refs:', testReturnChart.value)
      renderCharts()
    } else {
      console.error('获取评估结果失败:', response.data.message)
    }
  } catch (error) {
    console.error('获取评估结果失败:', error)
  } finally {
    loading.value = false
  }
}

// 渲染图表
const renderCharts = () => {
  if (!evaluationResult.value) return

  console.log('renderCharts called, evaluationResult:', evaluationResult.value)

  // 安全获取数据
  const test_returns = evaluationResult.value.test_returns || {}
  const ic_data = evaluationResult.value.ic_data || {}

  console.log('test_returns:', test_returns)
  console.log('ic_data:', ic_data)

  // Debug: 检查DOM元素
  console.log('testReturnChart ref:', testReturnChart.value)
  console.log('icChart ref:', icChart.value)
  console.log('icDistChart ref:', icDistChart.value)
  console.log('monthlyIcChart ref:', monthlyIcChart.value)

  // 使用 setTimeout 确保 DOM 完全渲染并获取正确尺寸
  setTimeout(() => {
    renderTestReturnChart(test_returns)
    renderIcChart(ic_data)
    renderIcDistChart(ic_data)
    renderMonthlyIcChart(ic_data)
  }, 100)
}

// 1. 渲染测试集收益图
const renderTestReturnChart = (test_returns) => {
  if (!testReturnChart.value) return

  // 先销毁旧实例
  if (testChartInstance) {
    testChartInstance.dispose()
    testChartInstance = null
  }

  testChartInstance = echarts.init(testReturnChart.value)

  const series = []
  const colors = ['#ef4444', '#f97316', '#eab308', '#22c55e', '#3b82f6']
  const dates = test_returns.dates || []

  // 各组收益曲线
  for (let i = 1; i <= 5; i++) {
    const groupKey = `Group${i}`
    const groupData = test_returns[groupKey] || []
    if (groupData.length > 0 && dates.length === groupData.length) {
      const cumReturns = []
      let cum = 0
      for (const ret of groupData) {
        cum += ret || 0
        cumReturns.push((cum * 100).toFixed(2))
      }

      series.push({
        name: groupKey,
        type: 'line',
        data: cumReturns,
        smooth: true,
        lineStyle: { width: 2 },
        itemStyle: { color: colors[i - 1] }
      })
    }
  }

  // 多空收益曲线
  const longShortData = test_returns.long_short || []
  if (longShortData.length > 0 && dates.length === longShortData.length) {
    const cumLongShort = []
    let cum = 0
    for (const ret of longShortData) {
      cum += ret || 0
      cumLongShort.push((cum * 100).toFixed(2))
    }
    series.push({
      name: 'Long-Short',
      type: 'line',
      data: cumLongShort,
      smooth: true,
      lineStyle: { width: 3, type: 'dashed' },
      itemStyle: { color: '#8b5cf6' }
    })
  }

  console.log('Rendering test return chart, series count:', series.length)

  // 即使没有数据也显示空图表
  testChartInstance.setOption({
    tooltip: {
      trigger: 'axis',
      formatter: (params) => {
        let result = params[0]?.axisValue + '<br/>'
        params.forEach(p => {
          result += `${p.marker} ${p.seriesName}: ${p.value}%<br/>`
        })
        return result
      }
    },
    legend: { data: series.map(s => s.name), bottom: 0 },
    grid: { left: '3%', right: '4%', bottom: '15%', top: '10%', containLabel: true },
    xAxis: {
      type: 'category',
      data: dates,
      axisLabel: { rotate: 45, fontSize: 10 }
    },
    yAxis: {
      type: 'value',
      name: '累计收益(%)',
      axisLabel: { formatter: '{value}%' }
    },
    series: series.length > 0 ? series : [{
      name: '无数据',
      type: 'line',
      data: [],
      itemStyle: { color: '#ccc' }
    }],
    title: series.length === 0 ? {
      text: '暂无数据',
      left: 'center',
      top: 'center',
      textStyle: { color: '#999' }
    } : undefined
  })

  // 强制调整图表尺寸
  testChartInstance.resize()
}

// 2. 渲染IC时间序列图
const renderIcChart = (ic_data) => {
  if (!icChart.value) return

  // 先销毁旧实例
  if (icChartInstance) {
    icChartInstance.dispose()
    icChartInstance = null
  }

  icChartInstance = echarts.init(icChart.value)

  const dates = ic_data.dates || []
  const icValues = ic_data.ic || []
  const rankIcValues = ic_data.rank_ic || []

  // 过滤掉null值，只保留有效数据
  const validIC = icValues.filter(v => v !== null && v !== undefined && !Number.isNaN(v))
  const validRankIC = rankIcValues.filter(v => v !== null && v !== undefined && !Number.isNaN(v))

  // 安全处理toFixed，返回null值时为'-'
  const safeFixed = (val, digits = 4) => {
    if (val === null || val === undefined || Number.isNaN(val)) return null
    return val.toFixed(digits)
  }

  icChartInstance.setOption({
    tooltip: { trigger: 'axis' },
    legend: { data: ['IC', 'Rank IC'], bottom: 0 },
    grid: { left: '3%', right: '4%', bottom: '15%', top: '10%', containLabel: true },
    xAxis: {
      type: 'category',
      data: dates,
      axisLabel: { rotate: 45, fontSize: 10 }
    },
    yAxis: {
      type: 'value',
      name: 'IC值',
      axisLine: { lineStyle: { color: '#ccc' } },
      splitLine: { lineStyle: { color: '#eee' } }
    },
    series: [
      {
        name: 'IC',
        type: 'bar',
        data: icValues.map(v => safeFixed(v)),
        itemStyle: {
          color: (params) => {
            const val = params.value
            return val !== null && val !== undefined && val >= 0 ? '#22c55e' : '#ef4444'
          }
        }
      },
      {
        name: 'Rank IC',
        type: 'line',
        data: rankIcValues.map(v => safeFixed(v)),
        smooth: true,
        lineStyle: { width: 2 },
        itemStyle: { color: '#3b82f6' },
        symbol: 'none'
      }
    ],
    title: (validIC.length === 0 && validRankIC.length === 0) ? {
      text: '暂无数据',
      left: 'center',
      top: 'center',
      textStyle: { color: '#999' }
    } : undefined
  })

  // 强制调整图表尺寸
  icChartInstance.resize()
}

// 3. 渲染IC分布直方图
const renderIcDistChart = (ic_data) => {
  if (!icDistChart.value) return

  // 先销毁旧实例
  if (icDistChartInstance) {
    icDistChartInstance.dispose()
    icDistChartInstance = null
  }

  icDistChartInstance = echarts.init(icDistChart.value)

  // 过滤掉null值
  const icValues = (ic_data.ic || []).filter(v => v !== null && v !== undefined && !Number.isNaN(v))

  if (icValues.length > 0) {
    const min = Math.min(...icValues)
    const max = Math.max(...icValues)
    const binCount = Math.min(20, icValues.length)
    const binWidth = (max - min) / binCount || 1

    const bins = new Array(binCount).fill(0)
    icValues.forEach(v => {
      const binIndex = Math.min(Math.floor((v - min) / binWidth), binCount - 1)
      bins[binIndex]++
    })

    const binLabels = []
    for (let i = 0; i < binCount; i++) {
      const start = (min + i * binWidth).toFixed(3)
      const end = (min + (i + 1) * binWidth).toFixed(3)
      binLabels.push(`${start}~${end}`)
    }

    icDistChartInstance.setOption({
      tooltip: { trigger: 'axis' },
      grid: { left: '3%', right: '4%', bottom: '15%', top: '10%', containLabel: true },
      xAxis: {
        type: 'category',
        data: binLabels,
        axisLabel: { rotate: 45, fontSize: 9 }
      },
      yAxis: { type: 'value', name: '频次' },
      series: [{
        type: 'bar',
        data: bins,
        itemStyle: { color: '#3b82f6' }
      }]
    })
  } else {
    // 显示空图表
    icDistChartInstance.setOption({
      grid: { left: '3%', right: '4%', bottom: '15%', top: '10%', containLabel: true },
      xAxis: { type: 'category', data: [] },
      yAxis: { type: 'value', name: '频次' },
      series: [{
        type: 'bar',
        data: [],
        itemStyle: { color: '#3b82f6' }
      }],
      title: {
        text: '暂无数据',
        left: 'center',
        top: 'center',
        textStyle: { color: '#999' }
      }
    })
  }

  // 强制调整图表尺寸
  icDistChartInstance.resize()
}

// 4. 渲染月度IC热力图
const renderMonthlyIcChart = (ic_data) => {
  if (!monthlyIcChart.value) return

  // 先销毁旧实例
  if (monthlyIcChartInstance) {
    monthlyIcChartInstance.dispose()
    monthlyIcChartInstance = null
  }

  monthlyIcChartInstance = echarts.init(monthlyIcChart.value)

  const dates = ic_data.dates || []
  const icValues = ic_data.ic || []

  // 计算月度IC
  const monthlyData = {}
  dates.forEach((date, i) => {
    const icValue = icValues[i]
    if (icValue !== null && icValue !== undefined && !Number.isNaN(icValue)) {
      const month = date.slice(0, 7) // YYYY-MM
      if (!monthlyData[month]) {
        monthlyData[month] = []
      }
      monthlyData[month].push(icValue)
    }
  })

  const months = Object.keys(monthlyData).sort()
  const monthlyIc = months.map(month => {
    const values = monthlyData[month]
    return values.reduce((a, b) => a + b, 0) / values.length
  })

  monthlyIcChartInstance.setOption({
    tooltip: {
      trigger: 'axis',
      formatter: (params) => {
        const val = params[0]?.value
        return `${params[0]?.axisValue}<br/>月均IC: ${val !== null && val !== undefined ? val.toFixed(4) : '-'}`
      }
    },
    grid: { left: '3%', right: '4%', bottom: '15%', top: '10%', containLabel: true },
    xAxis: {
      type: 'category',
      data: months,
      axisLabel: { rotate: 45, fontSize: 10 }
    },
    yAxis: { type: 'value', name: '月均IC' },
    visualMap: {
      show: false,
      min: -0.1,
      max: 0.1,
      inRange: {
        color: ['#ef4444', '#ffffff', '#22c55e']
      }
    },
    series: [{
      type: 'bar',
      data: monthlyIc,
      itemStyle: {
        color: (params) => params.value >= 0 ? '#22c55e' : '#ef4444'
      }
    }],
    title: months.length === 0 ? {
      text: '暂无数据',
      left: 'center',
      top: 'center',
      textStyle: { color: '#999' }
    } : undefined
  })

  // 强制调整图表尺寸
  monthlyIcChartInstance.resize()
}

// 页面加载时获取训练记录
onMounted(() => {
  fetchRecorders()
  // 添加窗口大小变化监听
  window.addEventListener('resize', handleResize)
})

// 组件卸载时清理
onUnmounted(() => {
  window.removeEventListener('resize', handleResize)
  disposeCharts()
})

// 窗口大小变化处理
const handleResize = () => {
  if (testChartInstance) testChartInstance.resize()
  if (icChartInstance) icChartInstance.resize()
  if (icDistChartInstance) icDistChartInstance.resize()
  if (monthlyIcChartInstance) monthlyIcChartInstance.resize()
}

// 刷新训练记录列表
const refreshRecorders = async () => {
  await fetchRecorders()
  ElMessage.success('列表已刷新')
}

// 删除选中的记录
const deleteSelectedRecorder = async () => {
  if (!selectedRecorderId.value) return

  try {
    await ElMessageBox.confirm(
      '确定要删除这条训练记录吗？此操作不可恢复。',
      '确认删除',
      {
        confirmButtonText: '确定',
        cancelButtonText: '取消',
        type: 'warning'
      }
    )

    const response = await axios.delete(`${API_BASE_URL}/qlib/recorders/${selectedRecorderId.value}`)
    if (response.data.success) {
      ElMessage.success(response.data.message)
      selectedRecorderId.value = ''
      evaluationResult.value = null
      disposeCharts()
      await fetchRecorders()
    } else {
      ElMessage.error(response.data.message)
    }
  } catch (error) {
    if (error !== 'cancel') {
      ElMessage.error('删除记录失败')
      console.error('删除记录失败:', error)
    }
  }
}

// 删除所有记录
const deleteAllRecorders = async () => {
  try {
    await ElMessageBox.confirm(
      '确定要删除所有训练记录吗？此操作不可恢复！',
      '危险操作确认',
      {
        confirmButtonText: '确定删除',
        cancelButtonText: '取消',
        type: 'danger'
      }
    )

    const response = await axios.delete(`${API_BASE_URL}/qlib/recorders`)
    if (response.data.success) {
      ElMessage.success(response.data.message)
      selectedRecorderId.value = ''
      evaluationResult.value = null
      disposeCharts()
      await fetchRecorders()
    } else {
      ElMessage.error(response.data.message)
    }
  } catch (error) {
    if (error !== 'cancel') {
      ElMessage.error('删除所有记录失败')
      console.error('删除所有记录失败:', error)
    }
  }
}
</script>

<style scoped>
.eval-page {
  padding: 8px;
}

/* 页面标题 */
.page-header {
  margin-bottom: 24px;
}

.page-header h2 {
  font-size: 24px;
  font-weight: 600;
  color: #2c3e50;
  margin: 0 0 8px 0;
}

.subtitle {
  color: #6c757d;
  font-size: 14px;
  margin: 0;
}

/* 选择器区域 */
.selector-section {
  margin-bottom: 24px;
}

.selector-card {
  background: linear-gradient(135deg, #f8f9fa 0%, #e9ecef 100%);
  padding: 20px;
  border-radius: 12px;
  display: flex;
  align-items: center;
  gap: 16px;
}

.selector-label {
  font-size: 14px;
  font-weight: 500;
  color: #5a6d7e;
  white-space: nowrap;
}

.recorder-select {
  flex: 1;
  max-width: 500px;
}

/* 操作按钮 */
.action-btn {
  display: flex;
  align-items: center;
  gap: 6px;
  padding: 10px 16px;
  border-radius: 8px;
  border: none;
  font-size: 14px;
  font-weight: 500;
  cursor: pointer;
  transition: all 0.3s ease;
}

.action-btn:disabled {
  opacity: 0.5;
  cursor: not-allowed;
}

.refresh-btn {
  background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
  color: white;
}

.refresh-btn:hover:not(:disabled) {
  transform: translateY(-2px);
  box-shadow: 0 4px 12px rgba(102, 126, 234, 0.4);
}

.delete-btn {
  background: linear-gradient(135deg, #f093fb 0%, #f5576c 100%);
  color: white;
}

.delete-btn:hover:not(:disabled) {
  transform: translateY(-2px);
  box-shadow: 0 4px 12px rgba(245, 87, 108, 0.4);
}

.delete-all-btn {
  background: linear-gradient(135deg, #ff6b6b 0%, #ee5a24 100%);
  color: white;
}

.delete-all-btn:hover:not(:disabled) {
  transform: translateY(-2px);
  box-shadow: 0 4px 12px rgba(238, 90, 36, 0.4);
}

.btn-spinner-small {
  width: 14px;
  height: 14px;
  border: 2px solid rgba(255, 255, 255, 0.3);
  border-top-color: white;
  border-radius: 50%;
  animation: spin 0.8s linear infinite;
}

@keyframes spin {
  to { transform: rotate(360deg); }
}

.load-btn {
  background: linear-gradient(135deg, #11998e 0%, #38ef7d 100%);
  color: white;
}

.load-btn:hover:not(:disabled) {
  transform: translateY(-2px);
  box-shadow: 0 4px 12px rgba(17, 153, 142, 0.4);
}

.btn-divider {
  width: 1px;
  height: 24px;
  background: #dee2e6;
  margin: 0 4px;
}

/* 加载中 */
.loading-section {
  padding: 40px;
  background: white;
  border-radius: 12px;
}

/* 结果区域 */
.result-section {
  display: flex;
  flex-direction: column;
  gap: 24px;
}

/* 信息卡片 */
.info-cards {
  display: grid;
  grid-template-columns: repeat(4, 1fr);
  gap: 16px;
}

.info-card {
  background: white;
  padding: 16px 20px;
  border-radius: 12px;
  box-shadow: 0 2px 8px rgba(0, 0, 0, 0.06);
  display: flex;
  flex-direction: column;
  gap: 6px;
}

.info-label {
  font-size: 13px;
  color: #6c757d;
}

.info-value {
  font-size: 15px;
  font-weight: 600;
  color: #2c3e50;
}

/* 指标区域 */
.metrics-section,
.charts-section,
.detail-section {
  background: white;
  padding: 24px;
  border-radius: 12px;
  box-shadow: 0 2px 8px rgba(0, 0, 0, 0.06);
}

.metrics-section h3,
.charts-section h3,
.detail-section h3 {
  font-size: 18px;
  font-weight: 600;
  color: #2c3e50;
  margin: 0 0 20px 0;
  padding-bottom: 12px;
  border-bottom: 2px solid #e9ecef;
}

.metrics-grid {
  display: grid;
  grid-template-columns: repeat(6, 1fr);
  gap: 16px;
}

.metric-card {
  background: linear-gradient(135deg, #f8f9fa 0%, #e9ecef 100%);
  padding: 20px;
  border-radius: 12px;
  display: flex;
  align-items: center;
  gap: 12px;
  transition: transform 0.2s ease;
}

.metric-card:hover {
  transform: translateY(-2px);
}

.metric-icon {
  font-size: 28px;
}

.metric-content {
  display: flex;
  flex-direction: column;
  gap: 4px;
}

.metric-header {
  display: flex;
  align-items: center;
  gap: 6px;
}

.help-icon {
  font-size: 14px;
  color: #8a94a6;
  cursor: pointer;
  transition: color 0.2s ease;
}

.help-icon:hover {
  color: #3b82f6;
}

.metric-tooltip {
  max-width: 280px;
  line-height: 1.6;
  font-size: 13px;
}

.metric-tooltip strong {
  color: #2c3e50;
}

.metric-label {
  font-size: 13px;
  color: #6c757d;
}

.metric-value {
  font-size: 22px;
  font-weight: 700;
  color: #2c3e50;
}

.metric-value.positive {
  color: #22c55e;
}

.metric-value.negative {
  color: #ef4444;
}

.metric-desc {
  font-size: 12px;
  color: #8a94a6;
}

/* 图表区域 */
.charts-grid {
  display: grid;
  grid-template-columns: repeat(2, 1fr);
  gap: 24px;
}

.chart-card {
  background: #f8f9fa;
  padding: 20px;
  border-radius: 12px;
}

.chart-header {
  display: flex;
  align-items: center;
  justify-content: space-between;
  margin-bottom: 16px;
}

.chart-header h4 {
  font-size: 15px;
  font-weight: 600;
  color: #2c3e50;
  margin: 0;
}

.chart-container {
  height: 280px;
}

/* 详细数据表格 */
.detail-section :deep(.positive) {
  color: #22c55e;
  font-weight: 600;
}

.detail-section :deep(.negative) {
  color: #ef4444;
  font-weight: 600;
}

/* 空状态 */
.empty-state {
  padding: 80px 40px;
  text-align: center;
}

.empty-icon {
  font-size: 80px;
  margin-bottom: 16px;
}

/* 响应式 */
@media (max-width: 1200px) {
  .info-cards {
    grid-template-columns: repeat(2, 1fr);
  }

  .metrics-grid {
    grid-template-columns: repeat(3, 1fr);
  }

  .charts-grid {
    grid-template-columns: 1fr;
  }
}

@media (max-width: 768px) {
  .info-cards {
    grid-template-columns: 1fr;
  }

  .metrics-grid {
    grid-template-columns: repeat(2, 1fr);
  }

  .selector-card {
    flex-direction: column;
    align-items: stretch;
  }

  .recorder-select {
    max-width: none;
  }
}
</style>
