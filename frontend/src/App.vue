<template>
  <div class="app-container">
    <!-- 顶部导航 -->
    <header class="app-header">
      <div class="header-left">
        <div class="logo-container">
          <div class="logo-icon">
            <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2">
              <path d="M3 3v18h18" stroke-linecap="round"/>
              <path d="M7 16l4-8 4 4 6-6" stroke-linecap="round" stroke-linejoin="round"/>
            </svg>
          </div>
          <h1 class="logo-text">QiFlow量化交易策略系统</h1>
        </div>
      </div>
      <div class="header-right">
        <el-button class="header-btn" @click="openApiDocs">
          <el-icon><Document /></el-icon>
          <span>API文档</span>
        </el-button>
        <el-button class="header-btn primary" @click="showAbout">
          <el-icon><InfoFilled /></el-icon>
          <span>关于</span>
        </el-button>
      </div>
    </header>

    <!-- 主内容区 -->
    <main class="app-main">
      <nav class="tab-nav">
        <button
          v-for="tab in tabs"
          :key="tab.name"
          :class="['tab-item', { active: activeTab === tab.name }]"
          @click="activeTab = tab.name"
        >
          <span class="tab-icon">{{ tab.icon }}</span>
          <span class="tab-label">{{ tab.label }}</span>
        </button>
      </nav>

      <div class="content-area">
        <transition name="fade" mode="out-in">
          <keep-alive>
            <component :is="currentComponent" :key="activeTab" />
          </keep-alive>
        </transition>
      </div>
    </main>

    <!-- 关于对话框 -->
    <el-dialog
      v-model="aboutVisible"
      title=""
      width="480px"
      class="about-dialog"
      :show-close="false"
    >
      <div class="about-content">
        <div class="about-header">
          <div class="about-icon">
            <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2">
              <path d="M3 3v18h18" stroke-linecap="round"/>
              <path d="M7 16l4-8 4 4 6-6" stroke-linecap="round" stroke-linejoin="round"/>
            </svg>
          </div>
          <h2>关于 QiFlow</h2>
        </div>
        <p class="about-desc">
          基于 Qlib 和 FastAPI 的专业量化交易策略回测平台，提供数据管理、模型训练、策略回测和结果分析等功能。
        </p>
        <div class="name-explanation">
          <div class="name-title">QiFlow 名称含义</div>
          <div class="name-items">
            <div class="name-item"><span class="name-letter">Q</span>量化 / Qlib</div>
            <div class="name-item"><span class="name-letter">i</span>投资</div>
            <div class="name-item"><span class="name-letter">Flow</span>Pipeline 工作流</div>
          </div>
        </div>
        <div class="tech-stack">
          <div class="tech-item">
            <span class="tech-label">后端</span>
            <span class="tech-value">FastAPI + Qlib</span>
          </div>
          <div class="tech-item">
            <span class="tech-label">前端</span>
            <span class="tech-value">Vue 3 + Element Plus</span>
          </div>
          <div class="tech-item">
            <span class="tech-label">数据</span>
            <span class="tech-value">Qlib A股数据</span>
          </div>
        </div>
      </div>
      <template #footer>
        <el-button type="primary" class="close-btn" @click="aboutVisible = false">确定</el-button>
      </template>
    </el-dialog>

    <!-- Copyright Footer -->
    <footer class="app-footer">
      <span>© 2026 QiFlow量化交易策略系统 | 基于 Qlib & FastAPI</span>
    </footer>
  </div>
</template>

<script setup>
import { ref, computed, onMounted, watch } from 'vue'
import { Document, InfoFilled } from '@element-plus/icons-vue'
import DataPreview from './components/DataPreview.vue'
import ModelTrain from './components/ModelTrain.vue'
import StrategyBacktest from './components/StrategyBacktest.vue'
import BacktestResults from './components/BacktestResults.vue'
import ModelEvaluation from './components/ModelEvaluation.vue'

const tabs = [
  { name: 'data', label: '数据管理', icon: '📦', component: DataPreview },
  { name: 'train', label: '模型训练', icon: '🚀', component: ModelTrain },
  { name: 'evaluation', label: '模型评估', icon: '📊', component: ModelEvaluation },
  { name: 'backtest', label: '策略回测', icon: '🎯', component: StrategyBacktest },
  { name: 'results', label: '回测结果', icon: '📈', component: BacktestResults }
]

// 从 URL hash 初始化 activeTab
const getTabFromHash = () => {
  const hash = window.location.hash.slice(1)
  const validTabs = tabs.map(t => t.name)
  return validTabs.includes(hash) ? hash : 'train'
}

const activeTab = ref(getTabFromHash())
const aboutVisible = ref(false)

// 监听 activeTab 变化，更新 URL hash
watch(activeTab, (newTab) => {
  window.location.hash = newTab
})

// 监听浏览器前进/后退
onMounted(() => {
  window.addEventListener('hashchange', () => {
    activeTab.value = getTabFromHash()
  })
})

const currentComponent = computed(() => {
  return tabs.find(t => t.name === activeTab.value)?.component
})

const showAbout = () => {
  aboutVisible.value = true
}

const openApiDocs = () => {
  window.open('/docs', '_blank')
}
</script>

<style>
/* 全局样式 */
* {
  margin: 0;
  padding: 0;
  box-sizing: border-box;
}

body {
  font-family: 'Inter', 'PingFang SC', 'Microsoft YaHei', sans-serif;
  background: linear-gradient(135deg, #f5f7fa 0%, #e4e8ec 100%);
  min-height: 100vh;
  color: #2c3e50;
}

.app-container {
  min-height: 100vh;
  display: flex;
  flex-direction: column;
}

/* 头部样式 */
.app-header {
  background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
  padding: 0 24px;
  height: 64px;
  display: flex;
  align-items: center;
  justify-content: space-between;
  box-shadow: 0 4px 20px rgba(102, 126, 234, 0.3);
  position: sticky;
  top: 0;
  z-index: 100;
}

.header-left {
  display: flex;
  align-items: center;
}

.logo-container {
  display: flex;
  align-items: center;
  gap: 12px;
}

.logo-icon {
  width: 36px;
  height: 36px;
  background: rgba(255, 255, 255, 0.2);
  border-radius: 10px;
  display: flex;
  align-items: center;
  justify-content: center;
}

.logo-icon svg {
  width: 24px;
  height: 24px;
  color: white;
}

.logo-text {
  font-size: 20px;
  font-weight: 600;
  color: white;
  letter-spacing: 0.5px;
}

.header-right {
  display: flex;
  gap: 12px;
}

.header-btn {
  background: rgba(255, 255, 255, 0.15) !important;
  border: 1px solid rgba(255, 255, 255, 0.25) !important;
  color: white !important;
  padding: 8px 16px;
  border-radius: 8px;
  font-size: 14px;
  transition: all 0.3s ease;
}

.header-btn:hover {
  background: rgba(255, 255, 255, 0.25) !important;
  transform: translateY(-1px);
}

.header-btn.primary {
  background: rgba(255, 255, 255, 0.95) !important;
  color: #667eea !important;
  border: none !important;
}

.header-btn.primary:hover {
  background: white !important;
}

/* 主内容区 */
.app-main {
  flex: 1;
  padding: 24px;
  max-width: 1400px;
  margin: 0 auto;
  width: 100%;
}

/* Tab导航 */
.tab-nav {
  display: flex;
  gap: 8px;
  background: white;
  padding: 8px;
  border-radius: 12px;
  box-shadow: 0 2px 12px rgba(0, 0, 0, 0.08);
  margin-bottom: 24px;
}

.tab-item {
  display: flex;
  align-items: center;
  gap: 8px;
  padding: 12px 20px;
  border-radius: 8px;
  border: none;
  background: transparent;
  color: #5a6d7e;
  font-size: 15px;
  font-weight: 500;
  cursor: pointer;
  transition: all 0.3s ease;
}

.tab-item:hover {
  background: #f0f4f8;
}

.tab-item.active {
  background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
  color: white;
  box-shadow: 0 4px 12px rgba(102, 126, 234, 0.4);
}

.tab-icon {
  font-size: 18px;
}

/* 内容区域 */
.content-area {
  background: white;
  border-radius: 16px;
  padding: 24px;
  box-shadow: 0 4px 24px rgba(0, 0, 0, 0.06);
  min-height: calc(100vh - 200px);
}

/* 过渡动画 */
.fade-enter-active,
.fade-leave-active {
  transition: opacity 0.2s ease, transform 0.2s ease;
}

.fade-enter-from {
  opacity: 0;
  transform: translateY(10px);
}

.fade-leave-to {
  opacity: 0;
  transform: translateY(-10px);
}

/* 关于对话框 */
.about-dialog .el-dialog__header {
  display: none;
}

.about-dialog .el-dialog__body {
  padding: 32px;
}

.about-content {
  text-align: center;
}

.about-header {
  display: flex;
  align-items: center;
  justify-content: center;
  gap: 12px;
  margin-bottom: 16px;
}

.about-icon {
  width: 48px;
  height: 48px;
  background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
  border-radius: 12px;
  display: flex;
  align-items: center;
  justify-content: center;
}

.about-icon svg {
  width: 28px;
  height: 28px;
  color: white;
}

.about-header h2 {
  font-size: 24px;
  color: #2c3e50;
}

.about-desc {
  color: #5a6d7e;
  font-size: 15px;
  line-height: 1.6;
  margin-bottom: 20px;
}

.name-explanation {
  background: #f5f7fa;
  border-radius: 12px;
  padding: 16px;
  margin-bottom: 20px;
}

.name-title {
  font-size: 14px;
  font-weight: 600;
  color: #2c3e50;
  margin-bottom: 12px;
  text-align: center;
}

.name-items {
  display: flex;
  justify-content: center;
  gap: 24px;
}

.name-item {
  display: flex;
  align-items: center;
  gap: 8px;
  font-size: 14px;
  color: #5a6d7e;
}

.name-letter {
  display: inline-flex;
  align-items: center;
  justify-content: center;
  width: 28px;
  height: 28px;
  background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
  color: white;
  border-radius: 6px;
  font-weight: 700;
  font-size: 14px;
}

.tech-stack {
  display: flex;
  gap: 16px;
  justify-content: center;
}

.tech-item {
  background: #f5f7fa;
  padding: 12px 20px;
  border-radius: 8px;
  display: flex;
  flex-direction: column;
  gap: 4px;
}

.tech-label {
  font-size: 12px;
  color: #8a94a6;
}

.tech-value {
  font-size: 14px;
  color: #2c3e50;
  font-weight: 500;
}

.close-btn {
  width: 100%;
  height: 44px;
  font-size: 16px;
  border-radius: 10px;
}

/* Footer样式 */
.app-footer {
  text-align: center;
  padding: 16px;
  color: #8a94a6;
  font-size: 13px;
  background: rgba(255, 255, 255, 0.8);
  border-top: 1px solid #eef2f6;
}

/* Element Plus 样式覆盖 */
.el-card {
  border: none;
  border-radius: 12px;
  box-shadow: 0 2px 12px rgba(0, 0, 0, 0.08);
}

.el-card__header {
  border-bottom: 1px solid #eef2f6;
  padding: 16px 20px;
}

.el-button--primary {
  background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
  border: none;
}

.el-button--primary:hover {
  background: linear-gradient(135deg, #5a6fd6 0%, #6a4190 100%);
}

.el-input__wrapper,
.el-select__wrapper {
  border-radius: 8px;
}

.el-progress-bar__outer {
  border-radius: 8px;
}

.el-progress-bar__inner {
  border-radius: 8px;
}
</style>