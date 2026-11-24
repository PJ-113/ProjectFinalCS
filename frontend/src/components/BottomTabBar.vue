<script setup>
import { computed } from 'vue'
import { useRoute } from 'vue-router'
import { House, ChartColumn, Settings } from '@lucide/vue'
import { isAdmin } from '../auth'
import { t } from '../i18n'

// แถบนำทางล่างสำหรับมือถือ (≤720px) แสดงผลเฉพาะจอแคบด้วย CSS เดสก์ท็อปใช้ top nav ใน AppHeader
// "เปรียบเทียบ" เป็น admin-only (route guard อยู่ที่ router/index.js)
// แท็บ "แผนที่" ถูกเอาออก — รวมเข้าเป็นส่วนหนึ่งของหน้า "/" แล้ว (DashboardMap.vue)
// แท็บ "ตั้งค่า" แทน "โปรไฟล์/เข้าสู่ระบบ" เดิม — แสดงตลอดไม่ว่าจะ login หรือไม่ (เข้าสู่ระบบอยู่ในหน้าตั้งค่า)
const route = useRoute()

const tabs = computed(() => {
  const list = [
    { to: '/', label: t('พยากรณ์'), icon: House },
  ]
  if (isAdmin()) list.push({ to: '/compare', label: t('เปรียบเทียบ'), icon: ChartColumn })
  list.push({ to: '/settings', label: t('ตั้งค่า'), icon: Settings })
  return list
})

// ไฮไลต์แท็บตาม prefix ของ path (เช่น /settings/profile ยังนับเป็นแท็บตั้งค่า) — exact-active-class เดิม
// ไม่ครอบคลุมหน้าย่อย เพราะ route ของหน้าย่อยไม่ได้เป็น nested route ของ /settings; หน้า login/register
// นับเป็นแท็บตั้งค่าด้วย เพราะเข้ามาจากการ์ดในหน้าตั้งค่า
function isActive(to) {
  if (to === '/') return route.path === '/'
  if (to === '/settings') {
    return route.path.startsWith('/settings') || route.name === 'login' || route.name === 'register'
  }
  return route.path.startsWith(to)
}
</script>

<template>
  <nav class="tab-bar" :aria-label="t('เมนูหลัก')">
    <router-link
      v-for="t in tabs"
      :key="t.to"
      :to="t.to"
      class="tab"
      :class="{ 'tab-active': isActive(t.to) }"
      :aria-current="isActive(t.to) ? 'page' : undefined"
    >
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
