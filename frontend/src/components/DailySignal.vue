<template>
  <div class="signal-page">
    <!-- 顶部说明 -->
    <div class="page-intro">
      <div class="intro-icon">📡</div>
      <div class="intro-text">
        <h2>每日预测信号</h2>
        <p>使用训练好的模型，对当日市场数据生成预测评分，输出推荐买入的 Top-K 股票排名。无需跑完整回测，几秒内返回结果。</p>
      </div>
    </div>

    <!-- 配置区 -->
    <div class="config-block">
      <div class="block-header">
        <span class="block-icon">⚙️</span>
        <h3>信号参数</h3>
      </div>
      <div class="config-grid">
        <div class="config-item">
          <label class="config-label">训练记录</label>
          <div class="recorder-row">
            <el-select v-model="config.recorder_id" class="config-select" placeholder="选择训练记录">
              <el-option
                v-for="rec in trainRecorders"
                :key="rec.id"
                :label="rec.name || rec.id"
                :value="rec.id"
              />
            </el-select>
            <button class="refresh-btn" @click="loadRecorders" :disabled="loadingRecorders">
              <span v-if="!loadingRecorders">🔄</span>
              <span v-else class="btn-spinner-small"></span>
            </button>
          </div>
        </div>
        <div class="config-item">
          <label class="config-label">市场</label>
          <el-select v-model="config.market" class="config-select">
            <el-option label="沪深300" value="csi300" />
            <el-option label="中证500" value="csi500" />
            <el-option label="中证800" value="csi800" />
            <el-option label="中证1000" value="csi1000" />
          </el-select>
        </div>
        <div class="config-item">
          <label class="config-label">持仓数量 (Topk)</label>
          <el-input-number v-model="config.topk" :min="5" :max="50" :step="5" />
        </div>
        <div class="config-item">
          <label class="config-label">每日换仓数 (n_drop)</label>
          <el-input-number v-model="config.n_drop" :min="1" :max="10" :step="1" />
        </div>
      </div>
      <div class="action-row">
        <button class="predict-btn" :disabled="loading || !config.recorder_id" @click="runPredict">
          <span v-if="!loading" class="btn-content">
            <span class="btn-icon">🔮</span>
            <span>生成今日信号</span>
          </span>
          <span v-else class="btn-content">
            <span class="btn-spinner"></span>
            <span>预测中...</span>
          </span>
        </button>
      </div>
    </div>

    <!-- 错误消息 -->
    <div v-if="error" class="message-box error">
      <span class="message-icon">❌</span>
      <span>{{ error }}</span>
    </div>

    <!-- 结果区域 -->
    <div v-if="result" class="result-section">
      <!-- 信号概要 -->
      <div class="result-header">
        <div class="signal-meta">
          <span class="meta-item">📅 信号日期: <strong>{{ result.prediction_date }}</strong></span>
          <span class="meta-item">🤖 模型: <strong>{{ result.model_type }}</strong></span>
          <span class="meta-item">📊 因子: <strong>{{ result.handler_type }}</strong></span>
          <span class="meta-item">🏪 市场: <strong>{{ result.market }}</strong></span>
          <span class="meta-item">📈 评估股票数: <strong>{{ result.total_stocks }}</strong></span>
        </div>
      </div>

      <!-- 推荐买入列表 -->
      <div class="config-block">
        <div class="block-header">
          <span class="block-icon">🏆</span>
          <h3>推荐持仓 Top {{ result.topk }}</h3>
        </div>
        <div class="stock-table-wrap">
          <table class="stock-table">
            <thead>
              <tr>
                <th class="col-rank">排名</th>
                <th class="col-code">代码</th>
                <th class="col-name">名称</th>
                <th class="col-score">预测分数</th>
                <th class="col-bar">信号强度</th>
              </tr>
            </thead>
            <tbody>
              <tr v-for="(stock, idx) in result.buy_list" :key="stock.stock_code">
                <td class="col-rank">
                  <span :class="['rank-badge', idx < 3 ? 'top3' : '']">{{ idx + 1 }}</span>
                </td>
                <td class="col-code">{{ stock.stock_code }}</td>
                <td class="col-name">{{ stock.stock_name }}</td>
                <td class="col-score">{{ stock.score.toFixed(4) }}</td>
                <td class="col-bar">
                  <div class="score-bar">
                    <div class="score-fill" :style="{ width: getBarWidth(stock.score) + '%' }"></div>
                  </div>
                </td>
              </tr>
            </tbody>
          </table>
        </div>
      </div>

      <!-- 完整评分（可折叠） -->
      <div class="config-block">
        <div class="block-header clickable" @click="showFullScores = !showFullScores">
          <span class="block-icon">📋</span>
          <h3>完整评分 (前100)</h3>
          <span class="toggle-icon">{{ showFullScores ? '▼' : '▶' }}</span>
        </div>
        <div v-if="showFullScores" class="stock-table-wrap full-scores">
          <table class="stock-table compact">
            <thead>
              <tr>
                <th>排名</th>
                <th>代码</th>
                <th>名称</th>
                <th>预测分数</th>
              </tr>
            </thead>
            <tbody>
              <tr v-for="stock in result.full_scores" :key="stock.stock_code"
                  :class="{ 'in-topk': stock.rank <= result.topk }">
                <td>{{ stock.rank }}</td>
                <td>{{ stock.stock_code }}</td>
                <td>{{ stock.stock_name }}</td>
                <td>{{ stock.score.toFixed(4) }}</td>
              </tr>
            </tbody>
          </table>
        </div>
      </div>
    </div>
  </div>
</template>

<script setup>
import { ref, reactive, onMounted } from 'vue'
import axios from 'axios'
import { ElMessage } from 'element-plus'

const API_BASE_URL = '/api'

const loading = ref(false)
const error = ref('')
const result = ref(null)
const showFullScores = ref(false)
const trainRecorders = ref([])
const loadingRecorders = ref(false)

const config = reactive({
  recorder_id: '',
  market: 'csi300',
  topk: 10,
  n_drop: 1,
})

onMounted(() => {
  loadRecorders()
})

const loadRecorders = async () => {
  loadingRecorders.value = true
  try {
    const res = await axios.get(`${API_BASE_URL}/qlib/recorders`)
    if (res.data.success) {
      trainRecorders.value = res.data.recorders
      if (trainRecorders.value.length && !config.recorder_id) {
        config.recorder_id = trainRecorders.value[0].id
      }
    }
  } catch (e) {
    ElMessage.error('加载训练记录失败')
  } finally {
    loadingRecorders.value = false
  }
}

const runPredict = async () => {
  loading.value = true
  error.value = ''
  result.value = null
  try {
    const res = await axios.post(`${API_BASE_URL}/qlib/predict`, config)
    if (res.data.success) {
      result.value = res.data
      ElMessage.success(res.data.message)
    } else {
      error.value = res.data.message
    }
  } catch (e) {
    error.value = `请求失败: ${e.response?.data?.detail || e.message}`
  } finally {
    loading.value = false
  }
}

const getBarWidth = (score) => {
  if (!result.value || !result.value.buy_list.length) return 0
  const maxScore = result.value.buy_list[0].score
  const minScore = result.value.buy_list[result.value.buy_list.length - 1].score
  if (maxScore === minScore) return 100
  return Math.max(10, ((score - minScore) / (maxScore - minScore)) * 100)
}
</script>

<style scoped>
.signal-page {
  padding: 8px;
}

.page-intro {
  display: flex;
  align-items: flex-start;
  gap: 16px;
  padding: 20px 24px;
  background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
  border-radius: 12px;
  margin-bottom: 20px;
  color: white;
}

.intro-icon {
  font-size: 36px;
}

.intro-text h2 {
  margin: 0 0 6px 0;
  font-size: 18px;
}

.intro-text p {
  margin: 0;
  font-size: 13px;
  opacity: 0.9;
  line-height: 1.6;
}

.config-block {
  background: white;
  border-radius: 12px;
  padding: 20px;
  margin-bottom: 16px;
  box-shadow: 0 1px 3px rgba(0, 0, 0, 0.06);
}

.block-header {
  display: flex;
  align-items: center;
  gap: 8px;
  margin-bottom: 16px;
}

.block-header h3 {
  margin: 0;
  font-size: 15px;
  font-weight: 600;
}

.block-icon {
  font-size: 18px;
}

.config-grid {
  display: grid;
  grid-template-columns: repeat(auto-fill, minmax(220px, 1fr));
  gap: 16px;
}

.config-item {
  display: flex;
  flex-direction: column;
  gap: 6px;
}

.config-label {
  font-size: 12px;
  font-weight: 500;
  color: #64748b;
}

.config-select {
  width: 100%;
}

.recorder-row {
  display: flex;
  gap: 8px;
}

.recorder-row .config-select {
  flex: 1;
}

.refresh-btn {
  border: 1px solid #e2e8f0;
  background: white;
  border-radius: 6px;
  padding: 0 10px;
  cursor: pointer;
  font-size: 14px;
}

.refresh-btn:hover {
  background: #f8fafc;
}

.action-row {
  margin-top: 20px;
  display: flex;
  justify-content: center;
}

.predict-btn {
  padding: 12px 32px;
  border: none;
  border-radius: 8px;
  background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
  color: white;
  font-size: 15px;
  font-weight: 600;
  cursor: pointer;
  transition: all 0.2s;
}

.predict-btn:hover:not(:disabled) {
  transform: translateY(-1px);
  box-shadow: 0 4px 12px rgba(102, 126, 234, 0.4);
}

.predict-btn:disabled {
  opacity: 0.6;
  cursor: not-allowed;
}

.btn-content {
  display: flex;
  align-items: center;
  gap: 8px;
}

.btn-icon {
  font-size: 18px;
}

.btn-spinner {
  width: 16px;
  height: 16px;
  border: 2px solid rgba(255, 255, 255, 0.3);
  border-top-color: white;
  border-radius: 50%;
  animation: spin 0.8s linear infinite;
}

.btn-spinner-small {
  width: 14px;
  height: 14px;
  border: 2px solid #ccc;
  border-top-color: #409eff;
  border-radius: 50%;
  animation: spin 0.8s linear infinite;
  display: inline-block;
}

@keyframes spin {
  to { transform: rotate(360deg); }
}

.message-box {
  padding: 12px 16px;
  border-radius: 8px;
  margin-bottom: 16px;
  display: flex;
  align-items: center;
  gap: 8px;
  font-size: 13px;
}

.message-box.error {
  background: #fef2f2;
  color: #dc2626;
  border: 1px solid #fecaca;
}

.message-icon {
  font-size: 16px;
}

/* 结果区域 */
.result-section {
  animation: fadeIn 0.3s ease;
}

@keyframes fadeIn {
  from { opacity: 0; transform: translateY(8px); }
  to { opacity: 1; transform: translateY(0); }
}

.result-header {
  background: linear-gradient(135deg, #f0f9ff 0%, #e0f2fe 100%);
  border-radius: 12px;
  padding: 16px 20px;
  margin-bottom: 16px;
}

.signal-meta {
  display: flex;
  flex-wrap: wrap;
  gap: 16px;
  font-size: 13px;
  color: #334155;
}

.meta-item strong {
  color: #0f172a;
}

/* 股票表格 */
.stock-table-wrap {
  overflow-x: auto;
}

.stock-table {
  width: 100%;
  border-collapse: collapse;
  font-size: 13px;
}

.stock-table th {
  background: #f8fafc;
  padding: 10px 12px;
  text-align: left;
  font-weight: 600;
  color: #475569;
  border-bottom: 2px solid #e2e8f0;
  white-space: nowrap;
}

.stock-table td {
  padding: 10px 12px;
  border-bottom: 1px solid #f1f5f9;
  color: #334155;
}

.stock-table tr:hover td {
  background: #f8fafc;
}

.col-rank { width: 60px; text-align: center; }
.col-code { width: 100px; }
.col-name { width: 120px; }
.col-score { width: 100px; font-family: monospace; }
.col-bar { width: 200px; }

.rank-badge {
  display: inline-flex;
  align-items: center;
  justify-content: center;
  width: 26px;
  height: 26px;
  border-radius: 50%;
  font-size: 12px;
  font-weight: 600;
  background: #f1f5f9;
  color: #64748b;
}

.rank-badge.top3 {
  background: linear-gradient(135deg, #f59e0b, #d97706);
  color: white;
}

.score-bar {
  height: 8px;
  background: #f1f5f9;
  border-radius: 4px;
  overflow: hidden;
}

.score-fill {
  height: 100%;
  background: linear-gradient(90deg, #667eea, #764ba2);
  border-radius: 4px;
  transition: width 0.5s ease;
}

/* 完整评分 */
.clickable {
  cursor: pointer;
  user-select: none;
}

.clickable:hover {
  opacity: 0.8;
}

.toggle-icon {
  margin-left: auto;
  font-size: 12px;
  color: #94a3b8;
}

.full-scores {
  max-height: 500px;
  overflow-y: auto;
}

.stock-table.compact td,
.stock-table.compact th {
  padding: 6px 10px;
  font-size: 12px;
}

.in-topk td {
  background: #f0fdf4 !important;
  font-weight: 500;
}
</style>
