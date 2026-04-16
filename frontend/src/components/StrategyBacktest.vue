<template>
  <div class="backtest-page">
    <!-- 配置区域 -->
    <div class="config-section">
      <div class="config-block">
        <div class="block-header">
          <span class="block-icon">📊</span>
          <h3>回测基础配置</h3>
        </div>
        <div class="config-grid">
          <div class="config-item">
            <label class="config-label">回测市场</label>
            <el-select v-model="config.market" class="config-select">
              <el-option label="沪深300" value="csi300" />
              <el-option label="中证500" value="csi500" />
              <el-option label="中证800" value="csi800" />
              <el-option label="中证1000" value="csi1000" />
              <el-option label="全部市场" value="all" />
            </el-select>
          </div>
          <div class="config-item">
            <label class="config-label">基准指数</label>
            <el-select v-model="config.benchmark" class="config-select">
              <el-option label="沪深300" value="SH000300" />
              <el-option label="上证50" value="SH000016" />
              <el-option label="中证500" value="SH000852" />
              <el-option label="中证1000" value="SH000905" />
            </el-select>
          </div>
          <div class="config-item">
            <label class="config-label">开始日期</label>
            <el-date-picker v-model="config.start_date" type="date" value-format="YYYY-MM-DD" class="config-picker" />
          </div>
          <div class="config-item">
            <label class="config-label">结束日期</label>
            <el-date-picker v-model="config.end_date" type="date" value-format="YYYY-MM-DD" class="config-picker" />
          </div>
        </div>
      </div>

      <div class="config-block">
        <div class="block-header">
          <span class="block-icon">⚙️</span>
          <h3>回测参数配置</h3>
        </div>
        <div class="params-grid">
          <div class="config-item">
            <label class="config-label">初始资金 (万)</label>
            <el-input-number v-model="config.initial_account" :min="10" :step="10" />
          </div>
          <div class="config-item">
            <label class="config-label">策略类型</label>
            <el-select v-model="config.strategy_type" class="config-select">
              <el-option label="TopkDropout策略" value="TopkDropoutStrategy" />
              <el-option label="权重策略" value="WeightStrategyBase" />
            </el-select>
          </div>
          <div class="param-slider">
            <div class="slider-header">
              <label class="config-label">持仓数量 (Topk)</label>
              <span class="slider-value">{{ config.topk }}</span>
            </div>
            <el-slider v-model="config.topk" :min="1" :max="50" :step="1" />
          </div>
          <div class="param-slider">
            <div class="slider-header">
              <label class="config-label">调仓卖出数</label>
              <span class="slider-value">{{ config.n_drop }}</span>
            </div>
            <el-slider v-model="config.n_drop" :min="1" :max="10" :step="1" />
          </div>
        </div>
      </div>

      <!-- 训练记录选择 -->
      <div class="config-block">
        <div class="block-header">
          <span class="block-icon">🎯</span>
          <h3>训练记录选择</h3>
        </div>
        <div class="recorder-row">
          <el-select v-model="config.recorder_id" class="recorder-select" placeholder="选择训练记录">
            <el-option v-for="rec in trainRecorders" :key="rec.id" :label="rec.name || `${rec.start_time} - ${rec.id}`" :value="rec.id" />
          </el-select>
          <button class="refresh-btn" @click="loadRecorders" :disabled="loadingRecorders">
            <span v-if="!loadingRecorders">🔄</span>
            <span v-else class="btn-spinner-small"></span>
            <span>刷新</span>
          </button>
          <button class="delete-btn" :disabled="!config.recorder_id" @click="deleteRecorder">
            <span>🗑️</span>
            <span>删除</span>
          </button>
          <button class="delete-all-btn" :disabled="!trainRecorders.length" @click="deleteAllRecorders">
            <span>🗑️🗑️</span>
            <span>全部删除</span>
          </button>
        </div>
      </div>
    </div>

    <!-- 操作区 -->
    <div class="action-section">
      <button class="run-btn" :class="{ running: running }" :disabled="running || !config.recorder_id" @click="runBacktest">
        <span v-if="!running" class="btn-content">
          <span class="btn-icon">▶️</span>
          <span>执行回测分析</span>
        </span>
        <span v-else class="btn-content">
          <span class="btn-spinner"></span>
          <span>回测中...</span>
        </span>
      </button>

      <div v-if="message" :class="['message-box', messageType]">
        <span class="message-icon">{{ messageType === 'success' ? '✅' : '❌' }}</span>
        <span>{{ message }}</span>
      </div>
    </div>
  </div>
</template>

<script setup>
import { ref, reactive, onMounted } from 'vue'
import { ElMessage, ElMessageBox } from 'element-plus'
import axios from 'axios'

const API_BASE_URL = '/api'

const running = ref(false)
const message = ref('')
const messageType = ref('success')
const trainRecorders = ref([])
const loadingRecorders = ref(false)

const config = reactive({
  market: 'csi300',
  benchmark: 'SH000300',
  start_date: '',
  end_date: '',
  initial_account: 100,
  topk: 5,
  n_drop: 1,
  strategy_type: 'TopkDropoutStrategy',
  recorder_id: ''
})

onMounted(() => {
  initDates()
  loadRecorders()
})

const initDates = () => {
  const today = new Date()
  const threeMonthsAgo = new Date()
  threeMonthsAgo.setMonth(threeMonthsAgo.getMonth() - 3)
  config.start_date = formatDate(threeMonthsAgo)
  config.end_date = formatDate(today)
}

const formatDate = (d) => d.toISOString().split('T')[0]

const loadRecorders = async () => {
  loadingRecorders.value = true
  try {
    const res = await axios.get(`${API_BASE_URL}/qlib/recorders`)
    if (res.data.success) {
      trainRecorders.value = res.data.recorders
      if (trainRecorders.value.length) config.recorder_id = trainRecorders.value[0].id
    }
  } catch (e) {
    ElMessage.error('加载训练记录失败')
  } finally {
    loadingRecorders.value = false
  }
}

const runBacktest = async () => {
  if (!config.recorder_id) {
    ElMessage.warning('请先选择训练记录')
    return
  }
  running.value = true
  message.value = ''
  try {
    const payload = { ...config, initial_account: config.initial_account * 10000 }
    const res = await axios.post(`${API_BASE_URL}/qlib/backtest`, payload)
    if (res.data.success) {
      message.value = res.data.message
      messageType.value = 'success'
    } else {
      message.value = res.data.message
      messageType.value = 'error'
    }
  } catch (e) {
    message.value = `回测失败: ${e.message}`
    messageType.value = 'error'
  } finally {
    running.value = false
  }
}

const deleteRecorder = async () => {
  try {
    await ElMessageBox.confirm(`确定删除记录 ${config.recorder_id}？`, '确认删除', { type: 'warning' })
    const res = await axios.delete(`${API_BASE_URL}/qlib/recorders/${config.recorder_id}`)
    if (res.data.success) {
      ElMessage.success(res.data.message)
      loadRecorders()
    } else {
      ElMessage.error(res.data.message)
    }
  } catch (e) {
    if (e !== 'cancel') ElMessage.error(`删除失败: ${e.message}`)
  }
}

const deleteAllRecorders = async () => {
  try {
    await ElMessageBox.confirm(`确定删除所有 ${trainRecorders.value.length} 条训练记录？此操作不可恢复！`, '确认全部删除', { type: 'warning' })
    let successCount = 0
    let failCount = 0
    for (const rec of trainRecorders.value) {
      try {
        const res = await axios.delete(`${API_BASE_URL}/qlib/recorders/${rec.id}`)
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
    loadRecorders()
  } catch (e) {
    if (e !== 'cancel') ElMessage.error(`删除失败`)
  }
}
</script>

<style scoped>
.backtest-page {
  padding: 8px;
}

.config-section {
  display: flex;
  flex-direction: column;
  gap: 20px;
}

.config-block {
  background: #f8f9fa;
  border-radius: 12px;
  padding: 20px;
}

.block-header {
  display: flex;
  align-items: center;
  gap: 8px;
  margin-bottom: 16px;
}

.block-icon {
  font-size: 20px;
}

.block-header h3 {
  font-size: 16px;
  font-weight: 600;
  color: #2c3e50;
}

.config-grid {
  display: grid;
  grid-template-columns: repeat(4, 1fr);
  gap: 16px;
}

.params-grid {
  display: grid;
  grid-template-columns: repeat(2, 1fr);
  gap: 20px;
}

.config-item {
  display: flex;
  flex-direction: column;
  gap: 8px;
}

.config-label {
  font-size: 14px;
  color: #5a6d7e;
  font-weight: 500;
}

.config-select,
.config-picker {
  width: 100%;
}

.param-slider {
  display: flex;
  flex-direction: column;
  gap: 8px;
}

.slider-header {
  display: flex;
  justify-content: space-between;
  align-items: center;
}

.slider-value {
  font-size: 14px;
  font-weight: 600;
  color: #667eea;
  background: rgba(102, 126, 234, 0.1);
  padding: 4px 12px;
  border-radius: 6px;
}

.recorder-row {
  display: flex;
  gap: 12px;
  align-items: center;
}

.recorder-select {
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
  font-size: 14px;
  transition: all 0.2s;
}

.delete-btn:hover:not(:disabled) {
  background: #dc3545;
  color: white;
}

.delete-btn:disabled {
  opacity: 0.5;
  cursor: not-allowed;
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

.action-section {
  margin-top: 32px;
  display: flex;
  flex-direction: column;
  align-items: center;
  gap: 20px;
}

.run-btn {
  background: linear-gradient(135deg, #11998e 0%, #38ef7d 100%);
  color: white;
  border: none;
  padding: 16px 48px;
  border-radius: 12px;
  font-size: 16px;
  font-weight: 600;
  cursor: pointer;
  transition: all 0.3s;
  box-shadow: 0 4px 20px rgba(17, 153, 142, 0.4);
}

.run-btn:hover:not(:disabled) {
  transform: translateY(-2px);
  box-shadow: 0 6px 24px rgba(17, 153, 142, 0.5);
}

.run-btn:disabled {
  cursor: not-allowed;
  opacity: 0.7;
}

.btn-content {
  display: flex;
  align-items: center;
  gap: 8px;
}

.btn-spinner {
  width: 20px;
  height: 20px;
  border: 2px solid rgba(255,255,255,0.3);
  border-top-color: white;
  border-radius: 50%;
  animation: spin 0.8s linear infinite;
}

@keyframes spin { to { transform: rotate(360deg); } }

.message-box {
  display: flex;
  align-items: center;
  gap: 12px;
  padding: 16px 24px;
  border-radius: 12px;
  max-width: 600px;
}

.message-box.success {
  background: linear-gradient(135deg, #d4edda 0%, #c3e6cb 100%);
  color: #155724;
}

.message-box.error {
  background: linear-gradient(135deg, #f8d7da 0%, #f5c6cb 100%);
  color: #721c24;
}
</style>