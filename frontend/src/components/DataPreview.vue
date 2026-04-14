<template>
  <div class="preview-page">
    <!-- 选择器 -->
    <div class="selector-bar">
      <div class="selector-item">
        <label>市场选择</label>
        <el-select v-model="market">
          <el-option label="沪深300" value="csi300" />
          <el-option label="中证500" value="csi500" />
          <el-option label="中证800" value="csi800" />
          <el-option label="中证1000" value="csi1000" />
          <el-option label="全部市场" value="all" />
        </el-select>
      </div>
      <button class="action-btn" :disabled="loading" @click="previewData">
        <span v-if="!loading">🔍 预览数据</span>
        <span v-else class="loading-content">
          <span class="mini-spinner"></span>
          <span>加载中...</span>
        </span>
      </button>
      <button class="action-btn download" :disabled="downloading" @click="downloadData">
        <span v-if="!downloading">📥 下载数据</span>
        <span v-else class="loading-content">
          <span class="mini-spinner"></span>
          <span>下载中...</span>
        </span>
      </button>
      <button class="action-btn update" :disabled="updating" @click="updateStockDB">
        <span v-if="!updating">🔄 更新股票</span>
        <span v-else class="loading-content">
          <span class="mini-spinner"></span>
          <span>更新中...</span>
        </span>
      </button>
    </div>

    <!-- 进度区域 -->
    <div v-if="showProgress" class="progress-section">
      <div class="progress-header">
        <span class="progress-icon">⚡</span>
        <span>数据下载进度</span>
      </div>
      <div class="progress-bar-wrap">
        <div class="progress-track">
          <div class="progress-fill" :style="{ width: progress + '%' }">
            <div class="progress-glow"></div>
          </div>
        </div>
        <span class="progress-text">{{ progress }}%</span>
      </div>
      <p class="progress-message">{{ progressMessage }}</p>
    </div>

    <!-- 消息提示 -->
    <div v-if="message" :class="['message-bar', messageType]">
      <span class="message-icon">{{ messageType === 'success' ? '✅' : '❌' }}</span>
      <span>{{ message }}</span>
    </div>

    <!-- 信息提示 -->
    <div v-if="info" class="info-bar">
      <span class="info-icon">ℹ️</span>
      <span>{{ info }}</span>
    </div>

    <!-- 数据表格 -->
    <div v-if="rows.length" class="data-table-wrap">
      <div class="table-scroll">
        <table class="data-table">
          <thead>
            <tr>
              <th>日期</th>
              <th>股票</th>
              <th>开盘</th>
              <th>收盘</th>
              <th>最高</th>
              <th>最低</th>
              <th>成交量</th>
              <th>成交额</th>
            </tr>
          </thead>
          <tbody>
            <tr v-for="row in rows" :key="row.date">
              <td>{{ row.date }}</td>
              <td>{{ row.stock }}</td>
              <td>{{ row.open }}</td>
              <td>{{ row.close }}</td>
              <td>{{ row.high }}</td>
              <td>{{ row.low }}</td>
              <td>{{ row.volume }}</td>
              <td>{{ row.amount }}</td>
            </tr>
          </tbody>
        </table>
      </div>
    </div>

    <!-- 错误 -->
    <div v-if="error" class="error-bar">
      <span>❌</span>
      <span>{{ error }}</span>
    </div>
  </div>
</template>

<script setup>
import { ref } from 'vue'
import axios from 'axios'

const API = '/api'

const market = ref('csi300')
const loading = ref(false)
const info = ref('')
const rows = ref([])
const error = ref('')
const downloading = ref(false)
const updating = ref(false)
const showProgress = ref(false)
const progress = ref(0)
const progressMessage = ref('')
const message = ref('')
const messageType = ref('success')

const previewData = async () => {
  loading.value = true
  info.value = ''
  rows.value = []
  error.value = ''

  try {
    const res = await axios.post(`${API}/qlib/preview_data`, { market: market.value })
    if (res.data.success) {
      info.value = `市场: ${res.data.market} | 日期: ${res.data.start_date} ~ ${res.data.end_date} | 股票数: ${res.data.stock_count}`
      const d = res.data.data
      if (d?.data?.length) {
        const sd = d.data[0]
        const dates = d.dates || []
        for (let i = 0; i < Math.min(dates.length, sd.open?.length || 0); i++) {
          rows.value.push({
            date: dates[i],
            stock: sd.stock || '',
            open: sd.open?.[i]?.toFixed(2) || 'N/A',
            close: sd.close?.[i]?.toFixed(2) || 'N/A',
            high: sd.high?.[i]?.toFixed(2) || 'N/A',
            low: sd.low?.[i]?.toFixed(2) || 'N/A',
            volume: sd.volume?.[i]?.toFixed(0) || 'N/A',
            amount: sd.amount?.[i]?.toFixed(2) || 'N/A'
          })
        }
      }
    } else {
      error.value = res.data.message
    }
  } catch (e) {
    error.value = `获取失败: ${e.message}`
  } finally {
    loading.value = false
  }
}

const downloadData = async () => {
  downloading.value = true
  showProgress.value = true
  progress.value = 0
  progressMessage.value = '正在启动下载...'
  message.value = ''

  try {
    const es = new EventSource(`${API}/qlib/download_data_stream`)
    es.onmessage = (e) => {
      const data = JSON.parse(e.data)
      if (data.progress >= 0) {
        progress.value = data.progress
        progressMessage.value = data.message
      }
      if (data.progress === 100) {
        es.close()
        downloading.value = false
        showProgress.value = false
        message.value = data.message
        messageType.value = 'success'
      }
      if (data.progress === -1) {
        es.close()
        downloading.value = false
        showProgress.value = false
        message.value = data.message
        messageType.value = 'error'
      }
    }
    es.onerror = () => {
      es.close()
      downloading.value = false
      showProgress.value = false
      message.value = '连接失败，请重试'
      messageType.value = 'error'
    }
  } catch (e) {
    downloading.value = false
    showProgress.value = false
    message.value = `下载失败: ${e.message}`
    messageType.value = 'error'
  }
}

const updateStockDB = async () => {
  updating.value = true
  message.value = ''
  try {
    const res = await axios.post(`${API}/qlib/update_stock_db`)
    if (res.data.success) {
      message.value = res.data.message
      messageType.value = 'success'
    } else {
      message.value = res.data.message
      messageType.value = 'error'
    }
  } catch (e) {
    message.value = `更新失败: ${e.message}`
    messageType.value = 'error'
  } finally {
    updating.value = false
  }
}
</script>

<style scoped>
.preview-page {
  padding: 8px;
}

.selector-bar {
  display: flex;
  align-items: center;
  gap: 16px;
  background: #f8f9fa;
  padding: 16px;
  border-radius: 12px;
  margin-bottom: 16px;
}

.selector-item {
  display: flex;
  align-items: center;
  gap: 8px;
}

.selector-item label {
  font-size: 14px;
  color: #5a6d7e;
  min-width: 70px;
}

.selector-item .el-select {
  width: 180px;
}

.action-btn {
  background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
  color: white;
  border: none;
  padding: 10px 20px;
  border-radius: 8px;
  font-size: 14px;
  cursor: pointer;
  transition: all 0.2s;
}

.action-btn:hover:not(:disabled) {
  transform: translateY(-1px);
}

.action-btn:disabled {
  opacity: 0.7;
}

.action-btn.download {
  background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
}

.action-btn.update {
  background: linear-gradient(135deg, #11998e 0%, #38ef7d 100%);
}

.loading-content {
  display: flex;
  align-items: center;
  gap: 8px;
}

.mini-spinner {
  width: 16px;
  height: 16px;
  border: 2px solid rgba(255,255,255,0.3);
  border-top-color: white;
  border-radius: 50%;
  animation: spin 0.8s linear infinite;
}

@keyframes spin { to { transform: rotate(360deg); } }

.progress-section {
  background: #f8f9fa;
  border-radius: 12px;
  padding: 16px;
  margin-bottom: 16px;
}

.progress-header {
  display: flex;
  align-items: center;
  gap: 8px;
  margin-bottom: 12px;
  font-weight: 600;
  color: #2c3e50;
}

.progress-bar-wrap {
  display: flex;
  align-items: center;
  gap: 16px;
}

.progress-track {
  flex: 1;
  height: 10px;
  background: #e9ecef;
  border-radius: 5px;
  overflow: hidden;
}

.progress-fill {
  height: 100%;
  background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
  border-radius: 5px;
  position: relative;
}

.progress-glow {
  position: absolute;
  right: 0;
  top: 0;
  bottom: 0;
  width: 20px;
  background: linear-gradient(90deg, transparent, rgba(255,255,255,0.5));
  animation: glow 1s infinite;
}

@keyframes glow {
  0%, 100% { opacity: 0.5; }
  50% { opacity: 1; }
}

.progress-text {
  font-size: 14px;
  font-weight: 600;
  color: #667eea;
  min-width: 50px;
}

.progress-message {
  margin-top: 8px;
  font-size: 13px;
  color: #5a6d7e;
  text-align: center;
}

.message-bar {
  display: flex;
  align-items: center;
  gap: 12px;
  padding: 16px;
  border-radius: 12px;
  margin-bottom: 16px;
  animation: fadeIn 0.3s ease;
}

@keyframes fadeIn {
  from { opacity: 0; transform: translateY(-10px); }
  to { opacity: 1; transform: translateY(0); }
}

.message-bar.success {
  background: linear-gradient(135deg, #d4edda 0%, #c3e6cb 100%);
  color: #155724;
}

.message-bar.error {
  background: linear-gradient(135deg, #f8d7da 0%, #f5c6cb 100%);
  color: #721c24;
}

.message-icon {
  font-size: 18px;
}

.info-bar {
  display: flex;
  align-items: center;
  gap: 8px;
  background: #e7f3ff;
  padding: 12px 16px;
  border-radius: 8px;
  margin-bottom: 16px;
  font-size: 13px;
  color: #0066cc;
}

.data-table-wrap {
  background: white;
  border-radius: 12px;
  border: 1px solid #e9ecef;
}

.table-scroll {
  overflow-x: auto;
}

.data-table {
  width: 100%;
  border-collapse: collapse;
}

.data-table th, .data-table td {
  padding: 12px 16px;
  text-align: left;
  font-size: 13px;
  white-space: nowrap;
}

.data-table th {
  background: #f8f9fa;
  color: #5a6d7e;
  font-weight: 600;
  border-bottom: 2px solid #e9ecef;
}

.data-table td {
  border-bottom: 1px solid #e9ecef;
}

.data-table tr:last-child td {
  border-bottom: none;
}

.data-table tr:hover td {
  background: #f5f7fa;
}

.error-bar {
  display: flex;
  align-items: center;
  gap: 12px;
  padding: 16px;
  background: #f8d7da;
  color: #721c24;
  border-radius: 12px;
}
</style>