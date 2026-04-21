<template>
  <span
    class="stock-link"
    @click="openStockChart"
    :title="`查看 ${code} 行情`"
  >
    {{ code }}
  </span>
</template>

<script setup>
const props = defineProps({
  code: {
    type: String,
    required: true
  }
})

const emit = defineEmits(['click'])

const openStockChart = () => {
  // 切换到行情页面
  window.location.hash = 'chart'

  // 使用自定义事件传递股票代码
  window.dispatchEvent(new CustomEvent('select-stock', {
    detail: { code: props.code }
  }))

  emit('click', props.code)
}
</script>

<style scoped>
.stock-link {
  color: #667eea;
  cursor: pointer;
  text-decoration: none;
  font-weight: 500;
  transition: all 0.2s;
  padding: 2px 6px;
  border-radius: 4px;
}

.stock-link:hover {
  color: #764ba2;
  background: rgba(102, 126, 234, 0.1);
  text-decoration: underline;
}
</style>
