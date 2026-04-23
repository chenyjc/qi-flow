<template>
  <div class="train-page">
    <!-- 顶部统计卡片 -->
    <div class="stats-row">
      <div class="stat-card">
        <div class="stat-icon factor">📈</div>
        <div class="stat-content">
          <span class="stat-label">因子类型</span>
          <span class="stat-value">{{ config.handler_type }}</span>
        </div>
      </div>
      <div class="stat-card">
        <div class="stat-icon model">🤖</div>
        <div class="stat-content">
          <span class="stat-label">模型类型</span>
          <span class="stat-value">{{ getModelLabel(config.model_type) }}</span>
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
            <div class="label-with-help">
              <label class="config-label">因子类型</label>
              <el-tooltip placement="top" effect="light">
                <template #content>
                  <div class="param-tooltip">
                    <strong>因子类型选择</strong><br/>
                    选择用于训练的特征因子集。<br/><br/>
                    <strong>选项说明：</strong><br/>
                    • Alpha158：158个经典Alpha因子，适合所有模型<br/>
                    • Alpha360：360个扩展因子，特征更丰富<br/>
                    • Alpha158vwap：基于VWAP的Alpha158<br/>
                    • Alpha360vwap：基于VWAP的Alpha360<br/><br/>
                    <strong>推荐：</strong> Alpha158（经典因子集，效果稳定）
                  </div>
                </template>
                <el-icon class="help-icon"><QuestionFilled /></el-icon>
              </el-tooltip>
            </div>
            <el-select v-model="config.handler_type" class="config-select">
              <el-option-group label="经典因子">
                <el-option label="Alpha158 (158因子)" value="Alpha158" />
                <el-option label="Alpha360 (360因子)" value="Alpha360" />
              </el-option-group>
              <el-option-group label="VWAP因子">
                <el-option label="Alpha158vwap" value="Alpha158vwap" />
                <el-option label="Alpha360vwap" value="Alpha360vwap" />
              </el-option-group>
            </el-select>
          </div>
          <div class="config-item">
            <div class="label-with-help">
              <label class="config-label">模型类型</label>
              <el-tooltip placement="top" effect="light">
                <template #content>
                  <div class="param-tooltip">
                    <strong>模型类型选择</strong><br/>
                    选择用于训练的机器学习模型。<br/><br/>
                    <strong>传统机器学习模型：</strong><br/>
                    • 双重集成：Qlib benchmark表现最佳，IC=0.0521，年化收益11.58%<br/>
                    • LightGBM：微软开发的高效梯度提升树，速度快、内存占用低<br/>
                    • XGBoost：经典的梯度提升树，精度高但速度较慢<br/>
                    • CatBoost：Yandex开发，对类别特征处理优秀<br/>
                    • 线性模型：简单的线性回归，适合快速验证<br/><br/>
                    <strong>深度学习模型（需要GPU）：</strong><br/>
                    • LSTM：长短期记忆网络，适合时间序列<br/>
                    • GRU：门控循环单元，比LSTM更轻量<br/>
                    • ALSTM：带注意力机制的LSTM<br/>
                    • DNN：全连接神经网络<br/>
                    • GATs：图注意力网络<br/>
                    • TCN：时间卷积网络<br/>
                    • SFM：状态频率模型<br/>
                    • TabNet：表格数据神经网络<br/><br/>
                    <strong>推荐：</strong> 双重集成（Qlib官方benchmark表现最佳）
                  </div>
                </template>
                <el-icon class="help-icon"><QuestionFilled /></el-icon>
              </el-tooltip>
            </div>
            <el-select v-model="config.model_type" class="config-select">
              <el-option-group label="传统机器学习">
                <el-option label="DoubleEnsemble (推荐)" value="DEnsembleModel" />
                <el-option label="LightGBM" value="LGBModel" />
                <el-option label="XGBoost" value="XGBModel" />
                <el-option label="CatBoost" value="CatBoostModel" />
                <el-option label="Linear" value="Linear" />
              </el-option-group>
              <el-option-group label="深度学习 (需要GPU)">
                <el-option label="LSTM" value="LSTM" />
                <el-option label="GRU" value="GRU" />
                <el-option label="ALSTM" value="ALSTM" />
                <el-option label="DNN" value="DNN" />
                <el-option label="GATs" value="GATs" />
                <el-option label="TCN" value="TCN" />
                <el-option label="SFM" value="SFM" />
                <el-option label="TabNet" value="TabnetModel" />
              </el-option-group>
            </el-select>
          </div>
          <div class="config-item" v-if="isDLModel()">
            <div class="label-with-help">
              <label class="config-label">特征选择</label>
              <el-tooltip placement="top" effect="light">
                <template #content>
                  <div class="param-tooltip">
                    <strong>特征选择策略</strong><br/>
                    选择深度学习模型使用的特征数量。<br/><br/>
                    <strong>选项说明：</strong><br/>
                    • 精选20特征：基于LightGBM特征重要性筛选，推荐用于深度学习模型<br/>
                    • 全部特征：使用全部158/360个特征，可能增加噪声和过拟合风险<br/><br/>
                    <strong>Qlib最佳实践：</strong><br/>
                    深度学习模型推荐使用精选20特征，树模型推荐使用全部特征。<br/>
                    精选特征可减少噪声干扰、降低过拟合风险、加快训练速度。
                  </div>
                </template>
                <el-icon class="help-icon"><QuestionFilled /></el-icon>
              </el-tooltip>
            </div>
            <el-select v-model="config.use_all_features" class="config-select">
              <el-option label="精选20特征 (推荐)" :value="false" />
              <el-option label="全部特征" :value="true" />
            </el-select>
          </div>
          <div class="param-slider">
            <div class="slider-header">
              <div class="label-with-help">
                <label class="config-label">学习率</label>
                <el-tooltip placement="top" effect="light">
                  <template #content>
                    <div class="param-tooltip">
                      <strong>学习率 (Learning Rate)</strong><br/>
                      控制每一步迭代时模型参数更新的幅度。<br/><br/>
                      <strong>解读：</strong><br/>
                      • 0.01-0.05：常用范围，平衡速度和精度<br/>
                      • 较小值：收敛慢但更稳定，精度可能更高<br/>
                      • 较大值：收敛快但可能过拟合或震荡<br/><br/>
                      <strong>推荐：</strong> 0.03-0.05（配合更多迭代次数）
                    </div>
                  </template>
                  <el-icon class="help-icon"><QuestionFilled /></el-icon>
                </el-tooltip>
              </div>
              <span class="slider-value">{{ config.lr.toFixed(4) }}</span>
            </div>
            <el-slider v-model="config.lr" :min="0.001" :max="0.1" :step="0.0001" show-stops />
          </div>
          <div class="param-slider">
            <div class="slider-header">
              <div class="label-with-help">
                <label class="config-label">最大深度</label>
                <el-tooltip placement="top" effect="light">
                  <template #content>
                    <div class="param-tooltip">
                      <strong>最大深度 (Max Depth)</strong><br/>
                      树的最大深度，控制模型复杂度。<br/><br/>
                      <strong>解读：</strong><br/>
                      • 3-6：浅树，防止过拟合，泛化能力强<br/>
                      • 6-10：中等深度，平衡精度和泛化<br/>
                      • 10+：深树，可能过拟合，训练时间长<br/><br/>
                      <strong>推荐：</strong> 6-8（量化数据噪声较大，不宜过深）
                    </div>
                  </template>
                  <el-icon class="help-icon"><QuestionFilled /></el-icon>
                </el-tooltip>
              </div>
              <span class="slider-value">{{ config.max_depth }}</span>
            </div>
            <el-slider v-model="config.max_depth" :min="3" :max="15" :step="1" />
          </div>
          <div class="param-slider">
            <div class="slider-header">
              <div class="label-with-help">
                <label class="config-label">叶子数量</label>
                <el-tooltip placement="top" effect="light">
                  <template #content>
                    <div class="param-tooltip">
                      <strong>叶子数量 (Num Leaves)</strong><br/>
                      每棵树的叶子节点数量，控制模型复杂度。<br/><br/>
                      <strong>解读：</strong><br/>
                      • 31-63：简单模型，防止过拟合<br/>
                      • 127-255：中等复杂度，常用设置<br/>
                      • 255+：复杂模型，可能过拟合<br/><br/>
                      <strong>关系：</strong> 叶子数 ≤ 2^(max_depth)<br/>
                      <strong>推荐：</strong> 127-255
                    </div>
                  </template>
                  <el-icon class="help-icon"><QuestionFilled /></el-icon>
                </el-tooltip>
              </div>
              <span class="slider-value">{{ config.num_leaves }}</span>
            </div>
            <el-slider v-model="config.num_leaves" :min="30" :max="300" :step="5" />
          </div>
          <div class="param-slider">
            <div class="slider-header">
              <div class="label-with-help">
                <label class="config-label">采样比例</label>
                <el-tooltip placement="top" effect="light">
                  <template #content>
                    <div class="param-tooltip">
                      <strong>采样比例 (Subsample)</strong><br/>
                      每次迭代时随机采样训练数据的比例。<br/><br/>
                      <strong>解读：</strong><br/>
                      • 0.5-0.7：强正则化，防止过拟合<br/>
                      • 0.8-0.9：常用设置，平衡精度和稳定性<br/>
                      • 1.0：使用全部数据，可能过拟合<br/><br/>
                      <strong>作用：</strong> 增加模型多样性，降低过拟合风险<br/>
                      <strong>推荐：</strong> 0.8-0.9
                    </div>
                  </template>
                  <el-icon class="help-icon"><QuestionFilled /></el-icon>
                </el-tooltip>
              </div>
              <span class="slider-value">{{ config.subsample.toFixed(4) }}</span>
            </div>
            <el-slider v-model="config.subsample" :min="0.5" :max="1" :step="0.0001" />
          </div>
          <div class="param-slider">
            <div class="slider-header">
              <div class="label-with-help">
                <label class="config-label">列采样</label>
                <el-tooltip placement="top" effect="light">
                  <template #content>
                    <div class="param-tooltip">
                      <strong>列采样 (Colsample Bytree)</strong><br/>
                      每棵树构建时随机选择特征的比例。<br/><br/>
                      <strong>解读：</strong><br/>
                      • 0.5-0.7：强正则化，适合高维特征<br/>
                      • 0.8-0.9：常用设置<br/>
                      • 1.0：使用全部特征<br/><br/>
                      <strong>作用：</strong> 降低特征间相关性影响，防止过拟合<br/>
                      <strong>Alpha158特征：</strong> 158个特征，建议0.8-0.9
                    </div>
                  </template>
                  <el-icon class="help-icon"><QuestionFilled /></el-icon>
                </el-tooltip>
              </div>
              <span class="slider-value">{{ config.colsample_bytree.toFixed(4) }}</span>
            </div>
            <el-slider v-model="config.colsample_bytree" :min="0.5" :max="1" :step="0.0001" />
          </div>
          <div class="param-slider">
            <div class="slider-header">
              <div class="label-with-help">
                <label class="config-label">随机种子</label>
                <el-tooltip placement="top" effect="light">
                  <template #content>
                    <div class="param-tooltip">
                      <strong>随机种子 (Random Seed)</strong><br/>
                      控制随机过程的初始值，确保结果可复现。<br/><br/>
                      <strong>解读：</strong><br/>
                      • 固定值：每次训练结果相同，便于调试和对比<br/>
                      • 不同值：可测试模型稳定性<br/><br/>
                      <strong>作用：</strong> 保证实验可复现性<br/>
                      <strong>推荐：</strong> 42（机器学习常用默认值）
                    </div>
                  </template>
                  <el-icon class="help-icon"><QuestionFilled /></el-icon>
                </el-tooltip>
              </div>
              <span class="slider-value">{{ config.seed }}</span>
            </div>
            <el-slider v-model="config.seed" :min="0" :max="1000" :step="1" />
          </div>
          <div class="param-slider">
            <div class="slider-header">
              <div class="label-with-help">
                <label class="config-label">多线程数量</label>
                <el-tooltip placement="top" effect="light">
                  <template #content>
                    <div class="param-tooltip">
                      <strong>多线程数量 (Num Threads)</strong><br/>
                      并行训练使用的CPU线程数。<br/><br/>
                      <strong>解读：</strong><br/>
                      • 1：单线程，适合调试，结果稳定<br/>
                      • 4-8：常用设置，平衡速度和资源<br/>
                      • CPU核心数：最大化速度<br/><br/>
                      <strong>注意：</strong> 多线程可能导致轻微数值差异<br/>
                      <strong>推荐：</strong> 1（确保结果可复现）或 CPU核心数
                    </div>
                  </template>
                  <el-icon class="help-icon"><QuestionFilled /></el-icon>
                </el-tooltip>
              </div>
              <span class="slider-value">{{ config.num_threads }}</span>
            </div>
            <el-slider v-model="config.num_threads" :min="1" :max="16" :step="1" />
          </div>
        </div>
        
        <!-- 深度学习模型参数（仅在选择深度学习模型时显示） -->
        <div v-if="isDLModel()" class="dl-params-section">
          <div class="section-header">
            <span class="section-icon">🧠</span>
            <h4>深度学习模型参数</h4>
            <span class="params-info">当前模型: {{ config.model_type }}</span>
          </div>
          <div class="dl-params-grid">
            <div class="dl-param-item">
              <label>隐藏层大小</label>
              <span class="param-value">{{ currentDLParams.hidden_size }}</span>
            </div>
            <div class="dl-param-item">
              <label>网络层数</label>
              <span class="param-value">{{ currentDLParams.num_layers }}</span>
            </div>
            <div class="dl-param-item">
              <label>Dropout</label>
              <span class="param-value">{{ currentDLParams.dropout }}</span>
            </div>
            <div class="dl-param-item">
              <label>训练轮数</label>
              <span class="param-value">{{ currentDLParams.n_epochs }}</span>
            </div>
            <div class="dl-param-item">
              <label>批次大小</label>
              <span class="param-value">{{ currentDLParams.batch_size }}</span>
            </div>
            <div class="dl-param-item">
              <label>早停轮数</label>
              <span class="param-value">{{ currentDLParams.early_stop }}</span>
            </div>
            <div class="dl-param-item">
              <label>学习率</label>
              <span class="param-value">{{ currentDLParams.lr }}</span>
            </div>
            <div class="dl-param-item">
              <label>时间步长</label>
              <span class="param-value">{{ currentDLParams.step_len }}</span>
            </div>
          </div>
          <div class="params-note">
            <el-icon><InfoFilled /></el-icon>
            <span>参数基于Qlib官方benchmark最佳配置，如需调整请修改源码 dlModelParams</span>
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
import { ref, reactive, computed, onMounted } from 'vue'
import { QuestionFilled, InfoFilled } from '@element-plus/icons-vue'

// 获取当前日期前一天的日期字符串
const getYesterdayDate = () => {
  const today = new Date()
  const yesterday = new Date(today)
  yesterday.setDate(yesterday.getDate() - 1)
  return yesterday.toISOString().split('T')[0]
}

const yesterdayDate = getYesterdayDate()
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
  test_end_date: yesterdayDate,
  handler_type: 'Alpha158',
  model_type: 'DEnsembleModel',
  lr: 0.2,
  max_depth: 8,
  num_leaves: 210,
  subsample: 0.8789,
  colsample_bytree: 0.8879,
  seed: 42,
  num_threads: 1,
  use_all_features: false
})

const dlModels = ['LSTM', 'GRU', 'ALSTM', 'DNN', 'GATs', 'TCN', 'SFM', 'TabnetModel']
const isDLModel = () => dlModels.includes(config.model_type)

const marketLabels = {
  csi300: '沪深300',
  csi500: '中证500',
  csi800: '中证800',
  csi1000: '中证1000',
  all: '全部市场'
}

const modelLabels = {
  LGBModel: 'LightGBM',
  XGBModel: 'XGBoost',
  CatBoostModel: 'CatBoost',
  Linear: 'Linear',
  DEnsembleModel: 'DoubleEnsemble',
  LSTM: 'LSTM',
  GRU: 'GRU',
  ALSTM: 'ALSTM',
  DNN: 'DNN',
  GATs: 'GATs',
  TCN: 'TCN',
  SFM: 'SFM',
  TabnetModel: 'TabNet'
}

// 深度学习模型默认参数配置（基于Qlib官方benchmark最佳参数）
// GPU优化：batch_size增大可提高GPU利用率
const dlModelParams = {
  LSTM: {
    d_feat: 20,
    hidden_size: 64,
    num_layers: 2,
    dropout: 0.0,
    n_epochs: 200,
    batch_size: 2000,  // 增大batch_size以提高GPU利用率
    early_stop: 10,
    step_len: 20,
    lr: 0.001
  },
  GRU: {
    d_feat: 20,
    hidden_size: 64,
    num_layers: 2,
    dropout: 0.0,
    n_epochs: 200,
    batch_size: 2000,  // 增大batch_size以提高GPU利用率
    early_stop: 10,
    step_len: 20,
    lr: 0.0002
  },
  ALSTM: {
    d_feat: 20,
    hidden_size: 64,
    num_layers: 2,
    dropout: 0.0,
    n_epochs: 200,
    batch_size: 2000,
    early_stop: 10,
    step_len: 20,
    lr: 0.001
  },
  DNN: {
    d_feat: 20,
    hidden_size: 64,
    num_layers: 2,
    dropout: 0.0,
    n_epochs: 200,
    batch_size: 2000,
    early_stop: 10,
    step_len: 20,
    lr: 0.001
  },
  GATs: {
    d_feat: 20,
    hidden_size: 64,
    num_layers: 2,
    dropout: 0.7,
    n_epochs: 200,
    batch_size: 2000,
    early_stop: 10,
    step_len: 20,
    lr: 0.0001
  },
  TCN: {
    d_feat: 20,
    hidden_size: 32,
    num_layers: 5,
    dropout: 0.5,
    n_epochs: 200,
    batch_size: 2000,
    early_stop: 20,
    step_len: 20,
    lr: 0.0001
  },
  SFM: {
    d_feat: 20,
    hidden_size: 64,
    num_layers: 2,
    dropout: 0.0,
    n_epochs: 200,
    batch_size: 2000,
    early_stop: 10,
    step_len: 20,
    lr: 0.001
  },
  TabnetModel: {
    d_feat: 20,
    hidden_size: 64,
    num_layers: 2,
    dropout: 0.0,
    n_epochs: 200,
    batch_size: 2000,
    early_stop: 10,
    step_len: 20,
    lr: 0.001
  }
}

const currentDLParams = computed(() => {
  return dlModelParams[config.model_type] || dlModelParams.LSTM
})

const getMarketLabel = (market) => marketLabels[market] || market
const getModelLabel = (model) => modelLabels[model] || model

onMounted(() => {
  const today = new Date()
  const yesterday = new Date(today)
  yesterday.setDate(yesterday.getDate() - 1)
  const oneYearAgo = new Date(yesterday)
  oneYearAgo.setFullYear(oneYearAgo.getFullYear() - 1)
  const threeMonthsAgo = new Date(yesterday)
  threeMonthsAgo.setMonth(threeMonthsAgo.getMonth() - 3)

  config.train_end_date = formatDate(oneYearAgo)
  config.valid_start_date = formatDate(oneYearAgo)
  config.valid_end_date = formatDate(threeMonthsAgo)
  config.test_start_date = formatDate(threeMonthsAgo)
  config.test_end_date = formatDate(yesterday)
})

const formatDate = (date) => date.toISOString().split('T')[0]

const trainModel = async () => {
  training.value = true
  showProgress.value = true
  progress.value = 5
  progressMessage.value = '正在启动训练...'
  message.value = ''

  try {
    // 判断是否为深度学习模型
    const dlModels = ['LSTM', 'GRU', 'ALSTM', 'DNN', 'GATs', 'TCN', 'SFM', 'TabnetModel']
    const isDLModel = dlModels.includes(config.model_type)
    
    let apiUrl
    let requestBody
    
    if (isDLModel) {
      const modelParams = dlModelParams[config.model_type] || dlModelParams.LSTM
      
      const d_feat = config.use_all_features 
        ? (config.handler_type.includes('360') ? 360 : 158)
        : modelParams.d_feat
      
      apiUrl = `${API_BASE_URL}/dl/train_dl_stream`
      requestBody = {
        market: config.market,
        benchmark: config.benchmark,
        train_start_date: config.train_start_date,
        train_end_date: config.train_end_date,
        valid_start_date: config.valid_start_date,
        valid_end_date: config.valid_end_date,
        test_start_date: config.test_start_date,
        test_end_date: config.test_end_date,
        model_type: config.model_type,
        handler_type: config.handler_type,
        d_feat: d_feat,
        hidden_size: modelParams.hidden_size,
        num_layers: modelParams.num_layers,
        dropout: modelParams.dropout,
        lr: modelParams.lr,
        n_epochs: modelParams.n_epochs,
        batch_size: modelParams.batch_size,
        early_stop: modelParams.early_stop,
        step_len: modelParams.step_len,
        seed: config.seed,
        GPU: 0,
        use_all_features: config.use_all_features
      }
    } else {
      // 传统机器学习模型使用原有 API
      apiUrl = `${API_BASE_URL}/qlib/train_stream`
      requestBody = config
    }
    
    const response = await fetch(apiUrl, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(requestBody)
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
              message.value = `训练完成！记录ID: ${data.recorder_id}。请前往"模型评估"页面查看详细结果。`
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
  grid-template-columns: repeat(4, 1fr);
  gap: 16px;
  margin-bottom: 24px;
}

.stat-icon.factor { background: linear-gradient(135deg, #f5af19 0%, #f12711 100%); }

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

.label-with-help {
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
  color: #667eea;
}

.param-tooltip {
  max-width: 300px;
  line-height: 1.6;
  font-size: 13px;
}

.param-tooltip strong {
  color: #2c3e50;
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

/* 深度学习模型参数区域 */
.dl-params-section {
  margin-top: 20px;
  padding: 16px;
  background: linear-gradient(135deg, #f0f4ff 0%, #e8f0fe 100%);
  border-radius: 12px;
  border: 1px solid #d0e3ff;
}

.section-header {
  display: flex;
  align-items: center;
  gap: 12px;
  margin-bottom: 16px;
}

.section-icon {
  font-size: 24px;
}

.section-header h4 {
  font-size: 16px;
  font-weight: 600;
  color: #3b5998;
  margin: 0;
}

.params-info {
  font-size: 13px;
  color: #667eea;
  background: #e8f0fe;
  padding: 4px 12px;
  border-radius: 6px;
}

.dl-params-grid {
  display: grid;
  grid-template-columns: repeat(4, 1fr);
  gap: 12px;
}

.dl-param-item {
  display: flex;
  flex-direction: column;
  gap: 4px;
  padding: 12px;
  background: white;
  border-radius: 8px;
  border: 1px solid #e0e8f0;
}

.dl-param-item label {
  font-size: 12px;
  color: #6b7c93;
  font-weight: 500;
}

.dl-param-item .param-value {
  font-size: 16px;
  font-weight: 600;
  color: #3b5998;
}

.params-note {
  display: flex;
  align-items: center;
  gap: 8px;
  margin-top: 12px;
  padding: 8px 12px;
  background: #fff8e6;
  border-radius: 6px;
  font-size: 12px;
  color: #8b6914;
}
</style>