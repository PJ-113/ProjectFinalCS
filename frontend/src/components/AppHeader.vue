<script setup>
import { computed, onMounted, onUnmounted, ref, watch } from 'vue'
import { Bell, BellDot, TriangleAlert, Info, Sun, Moon } from '@lucide/vue'
import { themeState, toggleTheme } from '../theme'
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

// เปิด/ปิดการแสดง login/register/profile + กระดิ่งแจ้งเตือน
const SHOW_EXTRA_FEATURES = true
// เปิด/ปิดลิงก์หน้าเปรียบเทียบ (/compare) — สิทธิ์เข้าหน้าจริงบังคับโดย route guard ใน router/index.js
const SHOW_COMPARE = true
// เปิด/ปิดลิงก์หน้าแผนที่ภาพรวม (/map)
const SHOW_MAP = true

const notifOpen = ref(false)
const notifWrapperEl = ref(null)

function toggleNotif() {
  notifOpen.value = !notifOpen.value
  // ดึงประวัติเมื่อเปิด dropdown เท่านั้น ใช้จังหวัดเดียวกับที่ badge สถานะกำลังเฝ้าอยู่
  // (home_province ถ้าตั้งไว้ ไม่งั้น dropdown ที่เลือกอยู่)
  if (notifOpen.value) fetchNotificationHistory(5, authState.user?.home_province || provinceState.selected)
}

// เลือกจังหวัดที่กระดิ่งจะเฝ้า: ถ้า user ตั้ง home_province ไว้ ใช้ /api/notifications/status
// (แยกจาก dropdown เลือกจังหวัดที่ใช้ดูพยากรณ์อยู่) ไม่งั้นใช้ /api/forecast ตามจังหวัดที่เลือกดูอยู่
function refreshNotificationWatch() {
  if (authState.user?.home_province) {
    fetchHomeProvinceStatus(authState.user.home_province)
  } else {
    fetchForecast()
  }
}

// ปิด dropdown เมื่อคลิกนอกกล่อง
function handleOutsideClick(e) {
  if (notifOpen.value && notifWrapperEl.value && !notifWrapperEl.value.contains(e.target)) {
    notifOpen.value = false
  }
}

onMounted(() => {
  fetchProvinces() // โหลดรายชื่อจังหวัดไว้ให้หน้าพยากรณ์/เปรียบเทียบใช้ (ไม่ขึ้นกับ SHOW_EXTRA_FEATURES)
  if (SHOW_EXTRA_FEATURES) {
    // รอ fetchCurrentUser() เสร็จก่อน แล้วค่อยเช็ค authState.user?.home_province
    fetchCurrentUser().then(refreshNotificationWatch)
  }
  document.addEventListener('click', handleOutsideClick)
})
onUnmounted(() => {
  document.removeEventListener('click', handleOutsideClick)
})

// เมื่อ home_province เปลี่ยน อัปเดตจังหวัดที่กระดิ่งเฝ้าโดยไม่ต้องรีเฟรชหน้า
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
  <header class="app-header" :class="{ 'app-header-open': notifOpen }">
    <nav class="app-header-nav">
      <router-link to="/" class="brand">
        PM2.5 Forecast
      </router-link>
      <router-link to="/" class="nav-link nav-tab desktop-only">Dashboard</router-link>
      <router-link v-if="SHOW_COMPARE && isAdmin()" to="/compare" class="nav-link nav-tab desktop-only">Compare</router-link>
      <router-link v-if="SHOW_MAP" to="/map" class="nav-link nav-tab desktop-only">Maps</router-link>
    </nav>
    <nav v-if="SHOW_EXTRA_FEATURES" class="app-header-auth">
      <button
        type="button"
        class="theme-toggle"
        :aria-label="themeState.mode === 'dark' ? 'เปลี่ยนเป็นธีมสว่าง' : 'เปลี่ยนเป็นธีมมืด'"
        :title="themeState.mode === 'dark' ? 'ธีมสว่าง' : 'ธีมมืด'"
        @click="toggleTheme"
      >
        <component :is="themeState.mode === 'dark' ? Sun : Moon" :size="20" :stroke-width="2" />
      </button>
      <div class="notif-wrapper" ref="notifWrapperEl">
        <button
          class="notif-bell"
          :class="{ 'notif-bell-alert': notificationState.highPollutionRisk }"
          @click="toggleNotif"
          aria-label="การแจ้งเตือน"
        >
          <component :is="notificationState.highPollutionRisk ? BellDot : Bell" :size="20" :stroke-width="2" />
        </button>
        <div v-if="notifOpen" class="notif-backdrop" @click="notifOpen = false"></div>
        <div v-if="notifOpen" class="notif-dropdown" role="dialog" aria-label="การแจ้งเตือน">
          <div class="notif-grip" aria-hidden="true"></div>
          <div class="notif-dropdown-title">การแจ้งเตือน</div>
          <div v-if="notificationState.status === 'loading'" class="notif-empty">กำลังโหลด...</div>
          <template v-else>
            <div v-if="notificationState.highPollutionRisk" class="notif-item notif-item-alert">
              <TriangleAlert :size="22" :stroke-width="2" aria-hidden="true" />
              <span>{{ buildWarningMessage(notificationState.riskGroup) }}</span>
            </div>
            <div v-if="notificationState.fallbackHorizons.length > 0" class="notif-item notif-item-info">
              <Info :size="22" :stroke-width="2" aria-hidden="true" />
              <span>ข้อมูลบางช่วงไม่ครบ ใช้ค่าพยากรณ์ชั่วคราว (ความเชื่อมั่นต่ำ)</span>
            </div>
            <div
              v-if="!notificationState.highPollutionRisk && notificationState.fallbackHorizons.length === 0"
              class="notif-empty"
            >
              ไม่มีการแจ้งเตือนในขณะนี้ คุณภาพอากาศอยู่ในเกณฑ์ปกติ
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
        <span class="nav-muted desktop-only">...</span>
      </template>
      <template v-else-if="authState.user">
        <router-link to="/profile" class="nav-link desktop-only">Profile</router-link>
      </template>
      <template v-else>
        <router-link to="/login" class="nav-link desktop-only">เข้าสู่ระบบ</router-link>
        <router-link to="/register" class="nav-link nav-link-accent desktop-only">สมัครสมาชิก</router-link>
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

/* ธีมมืด (data-theme="dark" บน <html>) */
  :root[data-theme="dark"] .nav-tab.router-link-exact-active {
    background: rgba(96, 165, 250, 0.15);
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

/* ปุ่มสลับธีม — หน้าตาเดียวกับกระดิ่ง */
.theme-toggle {
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
.theme-toggle:hover {
  color: var(--auth-accent);
  background: rgba(59, 130, 246, 0.1);
}
.theme-toggle:focus-visible { outline: 2px solid var(--auth-accent); outline-offset: 2px; }

.notif-bell:hover {
  color: var(--auth-accent);
  background: rgba(59, 130, 246, 0.1);
}

/* จุดกระพริบแจ้งเตือนบนไอคอนกระดิ่ง */
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

.notif-backdrop { display: none; }
.notif-grip { display: none; }

.notif-dropdown {
  position: absolute;
  top: calc(100% + 12px);
  right: 0;
  width: 340px;
  max-width: min(340px, 90vw);
  padding: 16px;
  border-radius: 20px;
  display: flex;
  flex-direction: column;
  gap: 10px;
  background: var(--card);
  border: 1px solid var(--line);
  color: var(--ink);
  text-align: left;
  font-size: 0.9rem;
  box-shadow: 0 10px 40px rgba(0, 0, 0, 0.18);
}

.notif-dropdown-title {
  font-weight: 700;
  font-size: 1rem;
  color: var(--ink);
}

.notif-empty {
  color: var(--muted);
  line-height: 1.5;
}

/* กล่องไอคอน + ข้อความ ใช้สี warn/info */
.notif-item {
  display: flex;
  gap: 12px;
  padding: 14px 16px;
  border-radius: 16px;
  border: 1px solid;
  line-height: 1.55;
  font-size: 0.9rem;
}
.notif-item > svg { flex-shrink: 0; margin-top: 1px; }

.notif-item-alert {
  background: var(--warn-bg);
  border-color: var(--warn-line);
  color: var(--warn-ink);
}

.notif-item-info {
  background: var(--info-bg);
  border-color: var(--info-line);
  color: var(--info-ink);
}

.notif-history-title {
  margin-top: 6px;
  padding-top: 14px;
  border-top: 1px solid var(--line);
}

.notif-history-list {
  list-style: none;
  margin: 0;
  padding: 0;
  display: flex;
  flex-direction: column;
  gap: 12px;
  max-height: 220px;
  overflow-y: auto;
}

.notif-history-item {
  display: flex;
  flex-direction: column;
  gap: 2px;
}

.notif-history-time {
  font-size: 0.75rem;
  color: var(--muted);
  font-weight: 600;
}

.notif-history-text {
  line-height: 1.45;
  color: var(--ink);
}

.notif-history-ranges {
  display: flex;
  flex-wrap: wrap;
  gap: 4px;
  margin-top: 4px;
}

.notif-history-chip {
  background: var(--info-bg);
  color: var(--info-ink);
  font-size: 0.72rem;
  font-weight: 600;
  padding: 3px 9px;
  border-radius: 999px;
  white-space: nowrap;
}

.notif-history-note {
  font-size: 0.75rem;
  color: var(--muted);
}

.notif-dropdown-link {
  color: var(--accent);
  text-decoration: none;
  font-weight: 600;
  font-size: 0.85rem;
  margin-top: 2px;
}

.notif-dropdown-link:hover {
  text-decoration: underline;
}

/* ต้องอยู่หลังกฎฐานของ .notif-dropdown/.notif-backdrop/.notif-grip เพื่อให้ media query นี้ทับได้ */
/* มือถือ (≤720px): เหลือแค่ brand + กระดิ่งแถวเดียว — ลิงก์ tab/login/register อยู่ใน BottomTabBar.vue */
@media (max-width: 720px) {
  .desktop-only { display: none !important; }
  .app-header {
    top: 8px;
    left: 8px;
    right: 8px;
    flex-wrap: nowrap;
    padding: 6px 8px 6px 16px;
    /* backdrop-filter ทำให้ลูก position:fixed อ้างอิงกล่อง header แทนหน้าจอ — ปิดบนมือถือและใช้พื้นทึบแทน */
    backdrop-filter: none;
    -webkit-backdrop-filter: none;
    background: var(--card);
    border-color: var(--line);
  }
  .brand { font-size: 1rem; margin-right: 0; }
  .notif-bell, .theme-toggle { width: 44px; height: 44px; }
  .app-header-auth { gap: 0; }
  .notif-bell-alert::after { top: 9px; right: 10px; }

  /* dropdown แจ้งเตือน → bottom sheet เต็มความกว้าง + backdrop มืด; ยก header เหนือ tab bar ตอนเปิดด้วย z-index */
  .app-header-open { z-index: 250; }
  .notif-backdrop {
    display: block;
    position: fixed;
    inset: 0;
    background: rgba(15, 18, 22, 0.5);
    z-index: 1;
  }
  .notif-dropdown {
    position: fixed;
    top: auto;
    left: 0;
    right: 0;
    bottom: 0;
    width: auto;
    max-width: none;
    max-height: 78vh;
    overflow-y: auto;
    border-radius: 24px 24px 0 0;
    padding: 8px 16px calc(28px + env(safe-area-inset-bottom, 0px));
    box-shadow: 0 -10px 40px rgba(0, 0, 0, 0.25);
    font-size: 0.95rem;
    z-index: 2;
  }
  .notif-grip {
    display: block;
    width: 40px;
    height: 4px;
    border-radius: 2px;
    background: var(--line);
    margin: 4px auto 8px;
  }
  .notif-dropdown-title { font-size: 1.15rem; }
  .notif-history-list { max-height: 40vh; }
  .notif-history-chip { font-size: 0.75rem; padding: 3px 9px; }
  .notif-dropdown-link { display: inline-block; padding: 12px 0 0; font-size: 0.95rem; }
}
</style>
