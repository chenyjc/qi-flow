<template>
  <div class="train-page">
    <!-- 顶部统计卡片 -->
    <div class="stats-row">
      <div class="stat-card">
        <div class="stat-icon model">🤖</div>
        <div class="stat-content">
          <span class="stat-label">模型类型</span>
          <span class="stat-value">{{ config.model_type }}</span>
        </div>
      </div>
      <div class="stat-card">
        <div class="stat-icon market">📊</div>
        <div class="stat-content">
          <span class="stat-label">训练市场</span>
          <span class="stat-value">{{ getMarketLabel(config.market) }}</span>
        </div>
      </div>
      <div class="stat-card">
        <div class="stat-icon benchmark">🎯</div>
        <div class="stat-content">
          <span class="stat-label">基准指数</span>
          <span class="stat-value">{{ config.benchmark }}</span>
        </div>
      </div>
    </div>

    <!-- 配置区域 -->
    <div class="config-section">
      <!-- 基础配置 -->
      <div class="config-block">
        <div class="block-header">
          <span class="block-icon">⚙️</span>
          <h3>训练基础配置</h3>
        </div>
        <div class="config-grid">
          <div class="config-item">
            <label class="config-label">训练市场</label>
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
        </div>
      </div>

      <!-- 时间配置 -->
      <div class="config-block">
        <div class="block-header">
          <span class="block-icon">📅</span>
          <h3>时间范围配置</h3>
        </div>
        <div class="config-grid time-grid">
          <div class="config-item">
            <label class="config-label">训练开始</label>
            <el-date-picker v-model="config.train_start_date" type="date" value-format="YYYY-MM-DD" class="config-picker" />
          </div>
          <div class="config-item">
            <label class="config-label">训练结束</label>
            <el-date-picker v-model="config.train_end_date" type="date" value-format="YYYY-MM-DD" class="config-picker" />
          </div>
          <div class="config-item">
            <label class="config-label">验证开始</label>
            <el-date-picker v-model="config.valid_start_date" type="date" value-format="YYYY-MM-DD" class="config-picker" />
          </div>
          <div class="config-item">
            <label class="config-label">验证结束</label>
            <el-date-picker v-model="config.valid_end_date" type="date" value-format="YYYY-MM-DD" class="config-picker" />
          </div>
          <div class="config-item">
            <label class="config-label">测试开始</label>
            <el-date-picker v-model="config.test_start_date" type="date" value-format="YYYY-MM-DD" class="config-picker" />
          </div>
          <div class="config-item">
            <label class="config-label">测试结束</label>
            <el-date-picker v-model="config.test_end_date" type="date" value-format="YYYY-MM-DD" class="config-picker" />
          </div>
        </div>
      </div>

      <!-- 模型参数 -->
      <div class="config-block">
        <div class="block-header">
          <span class="block-icon">🔧</span>
          <h3>模型参数配置</h3>
        </div>
        <div class="params-grid">
          <div class="config-item">
            <label class="config-label">模型类型</label>
            <el-select v-model="config.model_type" class="config-select">
              <el-option label="LightGBM" value="LGBModel" />
              <el-option label="XGBoost" value="XGBModel" />
              <el-option label="线性模型" value="Linear" />
            </el-select>
          </div>
          <div class="param-slider">
            <div class="slider-header">
              <label class="config-label">学习率</label>
              <span class="slider-value">{{ config.lr.toFixed(4) }}</span>
            </div>
            <el-slider v-model="config.lr" :min="0.001" :max="0.1" :step="0.0001" show-stops />
          </div>
          <div class="param-slider">
            <div class="slider-header">
              <label class="config-label">最大深度</label>
              <span class="slider-value">{{ config.max_depth }}</span>
            </div>
            <el-slider v-model="config.max_depth" :min="3" :max="15" :step="1" />
          </div>
          <div class="param-slider">
            <div class="slider-header">
              <label class="config-label">叶子数量</label>
              <span class="slider-value">{{ config.num_leaves }}</span>
            </div>
            <el-slider v-model="config.num_leaves" :min="30" :max="300" :step="5" />
          </div>
          <div class="param-slider">
            <div class="slider-header">
              <label class="config-label">采样比例</label>
              <span class="slider-value">{{ config.subsample.toFixed(4) }}</span>
            </div>
            <el-slider v-model="config.subsample" :min="0.5" :max="1" :step="0.0001" />
          </div>
          <div class="param-slider">
            <div class="slider-header">
              <label class="config-label">列采样</label>
              <span class="slider-value">{{ config.colsample_bytree.toFixed(4) }}</span>
            </div>
            <el-slider v-model="config.colsample_bytree" :min="0.5" :max="1" :step="0.0001" />
          </div>
        </div>
      </div>
    </div>

    <!-- 操作区 -->
    <div class="action-section">
      <button class="train-btn" :class="{ training: training }" :disabled="training" @click="trainModel">
        <span v-if="!training" class="btn-content">
          <span class="btn-icon">🚀</span>
          <span>开始训练模型</span>
        </span>
        <span v-else class="btn-content">
          <span class="btn-spinner"></span>
          <span>训练中...</span>
        </span>
      </button>

      <!-- 进度区 -->
      <div v-if="showProgress" class="progress-section">
        <div class="progress-bar-container">
          <div class="progress-bar" :style="{ width: progress + '%' }">
            <span class="progress-fill"></span>
          </div>
          <span class="progress-percent">{{ progress }}%</span>
        </div>
        <p class="progress-message">{{ progressMessage }}</p>
      </div>

      <!-- 消息 -->
      <div v-if="message" :class="['message-box', messageType]">
        <span class="message-icon">{{ messageType === 'success' ? '✅' : '❌' }}</span>
        <span>{{ message }}</span>
      </div>
    </div>
  </div>
</template>

<script setup>
import { ref, reactive, onMounted } from 'vue'
import axios from 'axios'

const API_BASE_URL = '/api'

const training = ref(false)
const showProgress = ref(false)
const progress = ref(0)
const progressMessage = ref('')
const message = ref('')
const messageType = ref('success')

const config = reactive({
  market: 'csi300',
  benchmark: 'SH000300',
  train_start_date: '2016-01-01',
  train_end_date: '',
  valid_start_date: '',
  valid_end_date: '',
  test_start_date: '',
  test_end_date: '',
  model_type: 'LGBModel',
  lr: 0.0421,
  max_depth: 8,
  num_leaves: 210,
  subsample: 0.8789,
  colsample_bytree: 0.8879
})

const marketLabels = {
  csi300: '沪深300',
  csi500: '中证500',
  csi800: '中证800',
  csi1000: '中证1000',
  all: '全部市场'
}

const getMarketLabel = (market) => marketLabels[market] || market

onMounted(() => {
  const today = new Date()
  const oneYearAgo = new Date(today)
  oneYearAgo.setFullYear(oneYearAgo.getFullYear() - 1)
  const threeMonthsAgo = new Date(today)
  threeMonthsAgo.setMonth(threeMonthsAgo.getMonth() - 3)

  config.train_end_date = formatDate(oneYearAgo)
  config.valid_start_date = formatDate(oneYearAgo)
  config.valid_end_date = formatDate(threeMonthsAgo)
  config.test_start_date = formatDate(threeMonthsAgo)
  config.test_end_date = formatDate(today)
})

const formatDate = (date) => date.toISOString().split('T')[0]

const trainModel = async () => {
  training.value = true
  showProgress.value = true
  progress.value = 5
  progressMessage.value = '正在启动训练...'
  message.value = ''

  try {
    const response = await fetch(`${API_BASE_URL}/qlib/train_stream`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(config)
    })

    if (!response.ok) throw new Error(`HTTP error! status: ${response.status}`)

    const reader = response.body.getReader()
    const decoder = new TextDecoder()

    while (true) {
      const { done, value } = await reader.read()
      if (done) break

      const chunk = decoder.decode(value)
      const lines = chunk.split('\n')

      for (const line of lines) {
        if (line.startsWith('data: ')) {
          try {
            const data = JSON.parse(line.substring(6))
            if (data.progress >= 0) {
              progress.value = data.progress
              progressMessage.value = data.message + ` (${data.progress}%)`
            }
            if (data.progress === 100) {
              training.value = false
              showProgress.value = false
              message.value = `训练完成！记录ID: ${data.recorder_id}`
              messageType.value = 'success'
              return
            }
            if (data.progress === -1) {
              training.value = false
              showProgress.value = false
              message.value = data.message
              messageType.value = 'error'
              return
            }
          } catch (e) {}
        }
      }
    }
  } catch (error) {
    training.value = false
    showProgress.value = false
    message.value = `训练失败: ${error.message}`
    messageType.value = 'error'
  }
}
</script>

<style scoped>
.train-page {
  padding: 8px;
}

/* 统计卡片 */
.stats-row {
  display: grid;
  grid-template-columns: repeat(3, 1fr);
  gap: 16px;
  margin-bottom: 24px;
}

.stat-card {
  background: linear-gradient(135deg, #f8f9fa 0%, #e9ecef 100%);
  padding: 16px 20px;
  border-radius: 12px;
  display: flex;
  align-items: center;
  gap: 16px;
  transition: transform 0.2s ease;
}

.stat-card:hover {
  transform: translateY(-2px);
}

.stat-icon {
  width: 48px;
  height: 48px;
  border-radius: 12px;
  display: flex;
  align-items: center;
  justify-content: center;
  font-size: 24px;
}

.stat-icon.model { background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); }
.stat-icon.market { background: linear-gradient(135deg, #11998e 0%, #38ef7d 100%); }
.stat-icon.benchmark { background: linear-gradient(135deg, #f093fb 0%, #f5576c 100%); }

.stat-content {
  display: flex;
  flex-direction: column;
}

.stat-label {
  font-size: 13px;
  color: #6c757d;
}

.stat-value {
  font-size: 18px;
  font-weight: 600;
  color: #2c3e50;
}

/* 配置区域 */
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
  grid-template-columns: repeat(2, 1fr);
  gap: 16px;
}

.time-grid {
  grid-template-columns: repeat(3, 1fr);
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

/* 参数滑块 */
.params-grid {
  display: grid;
  grid-template-columns: repeat(2, 1fr);
  gap: 20px;
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

/* 操作区 */
.action-section {
  margin-top: 32px;
  display: flex;
  flex-direction: column;
  align-items: center;
  gap: 24px;
}

.train-btn {
  background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
  color: white;
  border: none;
  padding: 16px 48px;
  border-radius: 12px;
  font-size: 16px;
  font-weight: 600;
  cursor: pointer;
  transition: all 0.3s ease;
  box-shadow: 0 4px 20px rgba(102, 126, 234, 0.4);
}

.train-btn:hover:not(:disabled) {
  transform: translateY(-2px);
  box-shadow: 0 6px 24px rgba(102, 126, 234, 0.5);
}

.train-btn:disabled {
  cursor: not-allowed;
  opacity: 0.8;
}

.train-btn.training {
  background: linear-gradient(135deg, #5a6fd6 0%, #6a4190 100%);
}

.btn-content {
  display: flex;
  align-items: center;
  gap: 8px;
}

.btn-icon {
  font-size: 20px;
}

.btn-spinner {
  width: 20px;
  height: 20px;
  border: 2px solid rgba(255, 255, 255, 0.3);
  border-top-color: white;
  border-radius: 50%;
  animation: spin 0.8s linear infinite;
}

@keyframes spin {
  to { transform: rotate(360deg); }
}

/* 进度条 */
.progress-section {
  width: 100%;
  max-width: 600px;
}

.progress-bar-container {
  display: flex;
  align-items: center;
  gap: 16px;
}

.progress-bar {
  flex: 1;
  height: 8px;
  background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
  border-radius: 4px;
  position: relative;
  overflow: hidden;
}

.progress-fill {
  position: absolute;
  inset: 0;
  background: linear-gradient(90deg, transparent, rgba(255, 255, 255, 0.3), transparent);
  animation: shimmer 2s infinite;
}

@keyframes shimmer {
  0% { transform: translateX(-100%); }
  100% { transform: translateX(100%); }
}

.progress-percent {
  font-size: 16px;
  font-weight: 600;
  color: #667eea;
  min-width: 50px;
}

.progress-message {
  margin-top: 12px;
  font-size: 14px;
  color: #5a6d7e;
  text-align: center;
}

/* 消息框 */
.message-box {
  display: flex;
  align-items: center;
  gap: 12px;
  padding: 16px 24px;
  border-radius: 12px;
  width: 100%;
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

.message-icon {
  font-size: 20px;
}
</style>