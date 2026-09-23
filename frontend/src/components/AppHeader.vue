<script setup>
import { computed, onMounted, onUnmounted, ref, watch } from 'vue'
import { Bell, BellDot } from '@lucide/vue'
import { authState, fetchCurrentUser, isAdmin } from '../auth'
import {
  notificationState,
  fetchForecast,
  fetchHomeProvinceStatus,
  buildWarningMessage,
  notificationHistory,
  fetchNotificationHistory,
  summarizeHistoryItem,
  formatCardTime,
} from '../notifications'
import { provinceState, fetchProvinces } from '../provinces'
import logoIcon from '../assets/logo-icon.svg'

// เปิดใช้งานแล้ว 2026-09-23 (ตั้งค่าเครื่องใหม่เสร็จ: Postgres/SECRET_KEY/GEMINI_KEY พร้อมหมด) —
// login/register/profile + กระดิ่งแจ้งเตือนกลับมาโชว์
const SHOW_EXTRA_FEATURES = true
// หน้าเปรียบเทียบ (/compare) จำกัดเฉพาะ admin เท่านั้น (route guard ที่ router/index.js เป็นคนบังคับจริง
// อันนี้แค่ซ่อนลิงก์ไม่ให้ user ทั่วไปงงว่ากดแล้วโดนเด้งกลับ)
const SHOW_COMPARE = true
// หน้าแผนที่ภาพรวม (/map) เช่นเดียวกัน — อ่านจาก notification_log ล้วนๆ ไม่ต้องพึ่ง auth/Gemini
const SHOW_MAP = true

const notifOpen = ref(false)
const notifWrapperEl = ref(null)

function toggleNotif() {
  notifOpen.value = !notifOpen.value
  // ดึงประวัติตอนเปิด dropdown เท่านั้น (ไม่ prewarm ตอน mount เหมือน fetchForecast() — ประวัติไม่
  // จำเป็นสำหรับ badge จุดแดง กันยิง request เพิ่มทุกหน้าโดยไม่จำเป็น) ใช้จังหวัดเดียวกับที่ badge
  // สถานะกำลังเฝ้าอยู่ (home_province ถ้าตั้งไว้ ไม่งั้น dropdown ที่เลือกอยู่)
  if (notifOpen.value) fetchNotificationHistory(5, authState.user?.home_province || provinceState.selected)
}

// เลือกว่ากระดิ่งจะเฝ้าจังหวัดไหน: home_province ของ user (ถ้าตั้งไว้) แยกจาก dropdown ที่ใช้ดู
// พยากรณ์อยู่ตอนนี้ เพื่อไม่ให้สลับ dropdown ไปดูจังหวัดอื่นแล้วกระดิ่งเปลี่ยนไปเฝ้าจังหวัดนั้นด้วย —
// ใช้ /api/notifications/status (ถูก, อ่านจาก DB ล่าสุด) แทน /api/forecast (แพง) เมื่อมี home_province
function refreshNotificationWatch() {
  if (authState.user?.home_province) {
    fetchHomeProvinceStatus(authState.user.home_province)
  } else {
    fetchForecast()
  }
}

// ปิด dropdown เมื่อคลิกนอกกล่อง — ไม่มี pattern นี้อยู่ก่อนในโปรเจกต์ ใช้ document click listener
// ตรงๆ (ไม่มี modal/dropdown component ให้ reuse)
function handleOutsideClick(e) {
  if (notifOpen.value && notifWrapperEl.value && !notifWrapperEl.value.contains(e.target)) {
    notifOpen.value = false
  }
}

onMounted(() => {
  fetchProvinces() // ต้องมีเสมอ ไม่ผูกกับ SHOW_EXTRA_FEATURES — โหลด list ไว้ให้หน้าพยากรณ์/เปรียบเทียบ
  // ใช้ (dropdown เลือกจังหวัดย้ายไปอยู่ในแต่ละหน้าแล้ว ไม่ได้อยู่ใน header)
  if (SHOW_EXTRA_FEATURES) {
    // ต้องรอ fetchCurrentUser() เสร็จก่อน ถึงจะรู้ authState.user?.home_province แน่ชัด (ไม่งั้นจะ
    // เห็นเป็น undefined ชั่วคราวแล้วเลือกผิดทาง)
    fetchCurrentUser().then(refreshNotificationWatch)
  }
  document.addEventListener('click', handleOutsideClick)
})
onUnmounted(() => {
  document.removeEventListener('click', handleOutsideClick)
})

// เผื่อ login/logout หรือแก้ home_province ในหน้าโปรไฟล์ระหว่างเปิดหน้าเว็บค้างไว้ — ให้กระดิ่งอัปเดต
// ตามจังหวัดใหม่โดยไม่ต้องรีเฟรชหน้าเอง
watch(() => authState.user?.home_province, () => {
  if (SHOW_EXTRA_FEATURES) refreshNotificationWatch()
})

// คำนวณ summary/timeLabel ครั้งเดียวต่อ item แทนที่จะเรียกฟังก์ชันซ้ำหลายครั้งในเทมเพลต
const historyItems = computed(() =>
  notificationHistory.items.map((item) => ({
    ...item,
    timeLabel: formatCardTime(item.recorded_at_th),
    summary: summarizeHistoryItem(item),
  })),
)
</script>

<template>
  <header class="app-header">
    <nav class="app-header-nav">
      <router-link to="/" class="brand">
        PM2.5 Forecast
      </router-link>
      <router-link to="/" class="nav-link nav-tab">พยากรณ์</router-link>
      <router-link v-if="SHOW_COMPARE && isAdmin()" to="/compare" class="nav-link nav-tab">เปรียบเทียบ</router-link>
      <router-link v-if="SHOW_MAP" to="/map" class="nav-link nav-tab">แผนที่</router-link>
    </nav>
    <nav v-if="SHOW_EXTRA_FEATURES" class="app-header-auth">
      <div class="notif-wrapper" ref="notifWrapperEl">
        <button
          class="notif-bell"
          :class="{ 'notif-bell-alert': notificationState.highPollutionRisk }"
          @click="toggleNotif"
          aria-label="การแจ้งเตือน"
        >
          <component :is="notificationState.highPollutionRisk ? BellDot : Bell" :size="20" :stroke-width="2" />
        </button>
        <div v-if="notifOpen" class="notif-dropdown glass-card">
          <div class="notif-dropdown-title">การแจ้งเตือน</div>
          <div v-if="notificationState.status === 'loading'" class="notif-empty">กำลังโหลด...</div>
          <template v-else>
            <div v-if="notificationState.highPollutionRisk" class="notif-item notif-item-alert">
              ⚠️ {{ buildWarningMessage(notificationState.riskGroup) }}
            </div>
            <div v-if="notificationState.fallbackHorizons.length > 0" class="notif-item notif-item-info">
              ℹ️ ข้อมูล input ไม่สมบูรณ์ในบางช่วงเวลา ใช้ค่าพยากรณ์แบบคงที่ชั่วคราว (ความเชื่อมั่นต่ำ)
            </div>
            <div
              v-if="!notificationState.highPollutionRisk && notificationState.fallbackHorizons.length === 0"
              class="notif-empty"
            >
              ✅ ไม่มีการแจ้งเตือนในขณะนี้ คุณภาพอากาศอยู่ในเกณฑ์ปกติ
            </div>
          </template>

          <div class="notif-dropdown-title notif-history-title">ประวัติล่าสุด</div>
          <div v-if="notificationHistory.status === 'loading'" class="notif-empty">กำลังโหลด...</div>
          <div v-else-if="historyItems.length === 0" class="notif-empty">ยังไม่มีประวัติการแจ้งเตือน</div>
          <ul v-else class="notif-history-list">
            <li v-for="item in historyItems" :key="item.recorded_at_th" class="notif-history-item">
              <span class="notif-history-time">{{ item.timeLabel }}</span>
              <span class="notif-history-text">{{ item.summary.headline }}</span>
              <span v-if="item.summary.ranges.length" class="notif-history-ranges">
                <span v-for="(r, i) in item.summary.ranges" :key="i" class="notif-history-chip">{{ r }}</span>
              </span>
              <span v-if="item.summary.note" class="notif-history-note">{{ item.summary.note }}</span>
            </li>
          </ul>

          <router-link to="/notifications" class="notif-dropdown-link" @click="notifOpen = false">ดูทั้งหมด →</router-link>
        </div>
      </div>
      <template v-if="authState.status === 'loading'">
        <span class="nav-muted">...</span>
      </template>
      <template v-else-if="authState.user">
        <router-link to="/profile" class="nav-link">Profile</router-link>
      </template>
      <template v-else>
        <router-link to="/login" class="nav-link">เข้าสู่ระบบ</router-link>
        <router-link to="/register" class="nav-link nav-link-accent">สมัครสมาชิก</router-link>
      </template>
    </nav>
  </header>
</template>

<style scoped>
.app-header {
  display: flex;
  align-items: center;
  justify-content: space-between;
  flex-wrap: wrap;
  gap: 8px;
  padding: 12px 24px;
  background: var(--auth-card-bg);
  backdrop-filter: blur(20px);
  -webkit-backdrop-filter: blur(20px);
  border: 1px solid var(--auth-border);
  box-shadow: 0 4px 30px rgba(0, 0, 0, 0.05);
  border-radius: 100px;
  font-size: 0.95rem;
  position: fixed;
  top: 16px;
  left: 16px;
  right: 16px;
  z-index: 100;
  transition: all 0.3s ease;
}

.app-header:hover {
  box-shadow: 0 6px 40px rgba(0, 0, 0, 0.08);
}

.app-header-nav,
.app-header-auth {
  display: flex;
  align-items: center;
  gap: 16px;
}

.brand {
  display: flex;
  align-items: center;
  gap: 8px;
  font-weight: 800;
  color: var(--auth-accent);
  text-decoration: none;
  margin-right: 8px;
  font-size: 1.05rem;
  letter-spacing: -0.02em;
}

.brand-logo {
  width: 28px;
  height: 28px;
  flex-shrink: 0;
}

.nav-link {
  color: var(--auth-muted);
  text-decoration: none;
  font-weight: 500;
  transition: color 0.2s ease, transform 0.2s ease;
}

.nav-link:hover {
  color: var(--auth-accent);
  transform: translateY(-1px);
}

.nav-tab {
  padding: 4px 8px;
  border-radius: 8px;
}

.nav-tab.router-link-exact-active {
  color: var(--auth-accent);
  font-weight: 700;
  background: rgba(59, 130, 246, 0.1);
}

@media (prefers-color-scheme: dark) {
  .nav-tab.router-link-exact-active {
    background: rgba(96, 165, 250, 0.15);
  }
}

.nav-link-accent {
  background: var(--auth-accent);
  color: #fff !important;
  font-weight: 600;
  padding: 8px 20px;
  border-radius: 999px;
  box-shadow: 0 4px 14px rgba(59, 130, 246, 0.4);
  transition: all 0.2s ease;
}

.nav-link-accent:hover {
  background: var(--auth-accent-hover);
  box-shadow: 0 6px 20px rgba(59, 130, 246, 0.6);
  transform: translateY(-2px);
}

.nav-muted {
  color: var(--auth-muted);
}

/* จอแคบ (มือถือ): brand+nav (+auth links ถ้า login แล้ว) อาจกว้างเกินแถวเดียว ให้ .app-header-nav
   ยอม wrap เป็น fallback กันชนเนื้อหาหน้าด้านล่าง (padding-top ของแต่ละหน้าเผื่อไว้แล้ว — ดู
   .forecast-page/.compare-page/.map-page) */
@media (max-width: 600px) {
  .app-header { padding: 10px 16px; gap: 6px; }
  .app-header-nav { gap: 8px; flex-wrap: wrap; row-gap: 4px; }
  .brand { font-size: 0.95rem; margin-right: 2px; }
}

.notif-wrapper {
  position: relative;
  display: flex;
  align-items: center;
}

.notif-bell {
  position: relative;
  display: flex;
  align-items: center;
  justify-content: center;
  width: 34px;
  height: 34px;
  border: none;
  background: transparent;
  color: var(--auth-muted);
  border-radius: 50%;
  cursor: pointer;
  transition: color 0.2s ease, background 0.2s ease;
}

.notif-bell:hover {
  color: var(--auth-accent);
  background: rgba(59, 130, 246, 0.1);
}

/* pulse dot เดียวกับ .live-dot ใน ForecastView.vue แค่เปลี่ยนสีเป็น error แทน good */
.notif-bell-alert::after {
  content: '';
  position: absolute;
  top: 4px;
  right: 4px;
  width: 8px;
  height: 8px;
  border-radius: 50%;
  background: var(--auth-error);
  animation: notif-pulse-dot 2s ease-in-out infinite;
}

@keyframes notif-pulse-dot {
  0%, 100% { opacity: 1; box-shadow: 0 0 0 0 rgba(239, 68, 68, 0.4); }
  50% { opacity: 0.7; box-shadow: 0 0 0 5px rgba(239, 68, 68, 0); }
}

.notif-dropdown {
  position: absolute;
  top: calc(100% + 12px);
  right: 0;
  width: 320px;
  max-width: min(320px, 90vw);
  padding: 16px;
  border-radius: 16px;
  display: flex;
  flex-direction: column;
  gap: 10px;
  color: var(--auth-text);
  text-align: left;
  font-size: 0.85rem;
  box-shadow: 0 10px 40px rgba(0, 0, 0, 0.15);
}

.notif-dropdown-title {
  font-weight: 700;
  font-size: 0.95rem;
  color: var(--auth-text);
}

.notif-empty {
  color: var(--auth-muted);
  line-height: 1.5;
}

.notif-item {
  padding: 10px 12px;
  border-radius: 10px;
  line-height: 1.5;
}

.notif-item-alert {
  background: rgba(239, 68, 68, 0.1);
  border: 1px solid rgba(239, 68, 68, 0.25);
  color: var(--auth-text);
}

.notif-item-info {
  background: rgba(59, 130, 246, 0.1);
  border: 1px solid rgba(59, 130, 246, 0.25);
  color: var(--auth-text);
}

.notif-history-title {
  margin-top: 4px;
  padding-top: 10px;
  border-top: 1px solid var(--auth-border);
}

.notif-history-list {
  list-style: none;
  margin: 0;
  padding: 0;
  display: flex;
  flex-direction: column;
  gap: 8px;
  max-height: 180px;
  overflow-y: auto;
}

.notif-history-item {
  display: flex;
  flex-direction: column;
  gap: 2px;
}

.notif-history-time {
  font-size: 0.72rem;
  color: var(--auth-muted);
  font-weight: 600;
}

.notif-history-text {
  line-height: 1.4;
  color: var(--auth-text);
}

.notif-history-ranges {
  display: flex;
  flex-wrap: wrap;
  gap: 4px;
  margin-top: 2px;
}

.notif-history-chip {
  background: rgba(59, 130, 246, 0.12);
  color: var(--auth-accent);
  font-size: 0.68rem;
  font-weight: 600;
  padding: 2px 7px;
  border-radius: 999px;
  white-space: nowrap;
}

.notif-history-note {
  font-size: 0.72rem;
  color: var(--auth-muted);
}

.notif-dropdown-link {
  color: var(--auth-accent);
  text-decoration: none;
  font-weight: 600;
  font-size: 0.8rem;
  margin-top: 2px;
}

.notif-dropdown-link:hover {
  text-decoration: underline;
}
</style>
