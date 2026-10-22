<script setup>
import { computed } from 'vue'
import { House, Map as MapIcon, ChartColumn, User, LogIn } from '@lucide/vue'
import { authState, isAdmin } from '../auth'

// แถบนำทางล่างสำหรับมือถือ (≤720px) แสดงผลเฉพาะจอแคบด้วย CSS เดสก์ท็อปใช้ top nav ใน AppHeader
// "เปรียบเทียบ" เป็น admin-only (route guard อยู่ที่ router/index.js)
const tabs = computed(() => {
  const list = [
    { to: '/', label: 'พยากรณ์', icon: House },
    { to: '/map', label: 'แผนที่', icon: MapIcon },
  ]
  if (isAdmin()) list.push({ to: '/compare', label: 'เปรียบเทียบ', icon: ChartColumn })
  list.push(
    authState.user
      ? { to: '/profile', label: 'โปรไฟล์', icon: User }
      : { to: '/login', label: 'เข้าสู่ระบบ', icon: LogIn },
  )
  return list
})
</script>

<template>
  <nav class="tab-bar" aria-label="เมนูหลัก">
    <router-link v-for="t in tabs" :key="t.to" :to="t.to" class="tab" exact-active-class="tab-active">
      <component :is="t.icon" :size="24" :stroke-width="2" aria-hidden="true" />
      <span class="tab-label">{{ t.label }}</span>
    </router-link>
  </nav>
</template>

<style scoped>
.tab-bar {
  display: none;
}

@media (max-width: 720px) {
  .tab-bar {
    display: flex;
    position: fixed;
    left: 0;
    right: 0;
    bottom: 0;
    z-index: 100;
    padding: 8px 8px calc(8px + env(safe-area-inset-bottom, 0px));
    background: var(--auth-card-bg);
    backdrop-filter: blur(20px);
    -webkit-backdrop-filter: blur(20px);
    border-top: 1px solid var(--auth-border);
    box-shadow: 0 -4px 30px rgba(0, 0, 0, 0.06);
  }
}

.tab {
  flex: 1;
  min-height: 52px;
  display: flex;
  flex-direction: column;
  align-items: center;
  justify-content: center;
  gap: 4px;
  color: var(--auth-muted);
  text-decoration: none;
  border-radius: 12px;
  font-size: 0.75rem;
  font-weight: 500;
}
.tab-active {
  color: var(--auth-accent);
  font-weight: 700;
}
.tab:focus-visible {
  outline: 2px solid var(--auth-accent);
  outline-offset: -2px;
}
</style>
