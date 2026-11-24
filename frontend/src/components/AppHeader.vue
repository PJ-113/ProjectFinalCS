<script setup>
import { computed, onMounted, onUnmounted, ref, watch } from 'vue'
import { Bell, BellDot, TriangleAlert, Info, Sparkles, Settings } from '@lucide/vue'
import { authState, fetchCurrentUser, isAdmin } from '../auth'
import {
  notificationState,
  fetchForecast,
  fetchHomeProvinceStatus,
  buildRiskAlert,
  RISK_ALERT_NOTE,
  buildImprovingTrendMessage,
  buildStationSilenceMessage,
  notificationHistory,
  fetchNotificationHistory,
  summarizeHistoryItem,
  formatCardTime,
} from '../notifications'
import { provinceState, fetchProvinces } from '../provinces'
import { t } from '../i18n'
// โลโก้ DustCast24 ไฟล์เดียวกับไอคอน PWA/favicon (public/) — ไม่ใช้ src/assets/logo-icon.svg เดิม (โลโก้เก่า)
const logoIcon = '/pwa-192x192.png'

// เปิด/ปิดการแสดง login/register/ตั้งค่า + กระดิ่งแจ้งเตือน
const SHOW_EXTRA_FEATURES = true
// เปิด/ปิดลิงก์หน้าเปรียบเทียบ (/compare) — สิทธิ์เข้าหน้าจริงบังคับโดย route guard ใน router/index.js
const SHOW_COMPARE = true
// เปิด/ปิดลิงก์หน้าแผนที่ภาพรวม (/map) — ปิดไว้: รวมเข้า Dashboard ที่ "/" แล้ว (DashboardMap.vue)
const SHOW_MAP = false

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
const riskAlertText = computed(() =>
  notificationState.riskAlert ? buildRiskAlert(notificationState.riskAlert, notificationState.riskGroup) : null,
)

const avatarLetter = computed(() => (authState.user?.email || '?').charAt(0).toUpperCase())

const historyItems = computed(() =>
  notificationHistory.items.map((item) => ({
    ...item,
    timeLabel: formatCardTime(item.recorded_at_th),
    summary: summarizeHistoryItem(item),
  })),
)
</script>

<template>
  <!-- Header แบบ A "แถบเต็มความกว้าง" (เลือกจาก mockup บน Design canvas): แถบทึบติดขอบบน เส้นคั่นล่าง
       ไม่ลอยทับเนื้อหา — ซ้าย โลโก้+DustCast24 และเมนู (ขีดเส้นใต้หน้าที่อยู่), ขวา กระดิ่ง | ปุ่มตั้งค่า
       มือถือเหลือโลโก้+กระดิ่ง (เมนูอยู่ BottomTabBar.vue) -->
  <header class="app-header" :class="{ 'app-header-open': notifOpen }">
    <div class="app-header-left">
      <router-link to="/" class="brand">
        <img :src="logoIcon" alt="" class="brand-logo" aria-hidden="true" />
        <span class="brand-name">DustCast24</span>
      </router-link>
      <nav class="app-header-nav desktop-only" :aria-label="t('เมนูหลัก')">
        <router-link to="/" class="nav-tab" exact-active-class="nav-tab-active">{{ t('พยากรณ์') }}</router-link>
        <router-link v-if="SHOW_COMPARE && isAdmin()" to="/compare" class="nav-tab" active-class="nav-tab-active">{{ t('เปรียบเทียบ') }}</router-link>
        <router-link v-if="SHOW_MAP" to="/map" class="nav-tab" active-class="nav-tab-active">Maps</router-link>
      </nav>
    </div>
    <div v-if="SHOW_EXTRA_FEATURES" class="app-header-auth">
      <div class="notif-wrapper" ref="notifWrapperEl">
        <button
          class="notif-bell"
          :class="{ 'notif-bell-alert': notificationState.highPollutionRisk }"
          @click="toggleNotif"
          :aria-label="t('การแจ้งเตือน')"
        >
          <component :is="notificationState.highPollutionRisk ? BellDot : Bell" :size="20" :stroke-width="2" aria-hidden="true" />
        </button>
        <div v-if="notifOpen" class="notif-backdrop" @click="notifOpen = false"></div>
        <div v-if="notifOpen" class="notif-dropdown" role="dialog" :aria-label="t('การแจ้งเตือน')">
          <div class="notif-grip" aria-hidden="true"></div>
          <div class="notif-dropdown-title">{{ t('การแจ้งเตือน') }}</div>
          <div v-if="notificationState.status === 'loading'" class="notif-empty">{{ t('กำลังโหลด...') }}</div>
          <template v-else>
            <!-- คำเตือนฝุ่นสูงข้อความเดียว (CHANGELOG 272) แทน 2 กล่องเดิม (ใกล้ถึงช่วงเสี่ยง + ช่วงนี้ฝุ่นสูง) —
                 builder เดียวกับแบนเนอร์หน้าแรก; เดิมกล่อง "ใกล้ถึงช่วงเสี่ยง" ส่ง notificationState (camelCase)
                 เข้า builder ที่อ่าน snake_case จึงขึ้น undefined/{time} — แก้ไปในตัวด้วย -->
            <div v-if="riskAlertText" class="notif-item notif-item-alert">
              <TriangleAlert :size="22" :stroke-width="2" aria-hidden="true" />
              <span class="notif-alert-body">
                <strong>{{ riskAlertText.title }}</strong>
                <span>{{ riskAlertText.sub }}</span>
                <span class="notif-alert-note">{{ t(RISK_ALERT_NOTE) }}</span>
              </span>
            </div>
            <div v-if="notificationState.fallbackHorizons.length > 0" class="notif-item notif-item-info">
              <Info :size="22" :stroke-width="2" aria-hidden="true" />
              <span>{{ t('ข้อมูลบางช่วงไม่ครบ ใช้ค่าพยากรณ์ชั่วคราว (ความเชื่อมั่นต่ำ)') }}</span>
            </div>
            <!-- ข้อ 4 "สถานีเงียบนานผิดปกติ" — ตั้งเฉพาะตอนเฝ้า home_province ผ่าน fetchHomeProvinceStatus() -->
            <div v-if="notificationState.dataStale" class="notif-item notif-item-info">
              <Info :size="22" :stroke-width="2" aria-hidden="true" />
              <span>{{ buildStationSilenceMessage(notificationState.staleHours) }}</span>
            </div>
            <!-- ข้อ 3 "แนวโน้มฝุ่นจะดีขึ้น" (ข่าวดี) วางหลังสุด เพราะการแจ้งเตือนด้านความเสี่ยง/คุณภาพข้อมูลสำคัญกว่า -->
            <div v-if="notificationState.trendDirection === 'down'" class="notif-item notif-item-good">
              <Sparkles :size="22" :stroke-width="2" aria-hidden="true" />
              <span>{{ buildImprovingTrendMessage(notificationState.trendPeriod) }}</span>
            </div>
            <div
              v-if="!riskAlertText
                && !notificationState.dataStale && notificationState.trendDirection !== 'down'
                && notificationState.fallbackHorizons.length === 0"
              class="notif-empty"
            >
              {{ t('ไม่มีการแจ้งเตือนในขณะนี้ คุณภาพอากาศอยู่ในเกณฑ์ปกติ') }}
            </div>
          </template>

          <div class="notif-dropdown-title notif-history-title">{{ t('ประวัติล่าสุด') }}</div>
          <div v-if="notificationHistory.status === 'loading'" class="notif-empty">{{ t('กำลังโหลด...') }}</div>
          <div v-else-if="historyItems.length === 0" class="notif-empty">{{ t('ยังไม่มีประวัติการแจ้งเตือน') }}</div>
          <ul v-else class="notif-history-list">
            <li v-for="item in historyItems" :key="item.recorded_at_th" class="notif-history-item">
              <span class="notif-history-time">{{ item.timeLabel }}</span>
              <span class="notif-history-text">{{ item.summary.headline }}</span>
              <span v-if="item.summary.ranges.length" class="notif-history-ranges">
                <span v-for="(r, i) in item.summary.ranges" :key="i" class="notif-history-chip">{{ r }}</span>
              </span>
              <span v-if="item.summary.note" class="notif-history-note">{{ item.summary.note }}</span>
              <span v-if="item.summary.transitionNote" class="notif-history-note">{{ item.summary.transitionNote }}</span>
              <span v-if="item.summary.goodNews" class="notif-history-note notif-history-note-good">{{ item.summary.goodNews }}</span>
            </li>
          </ul>

          <router-link to="/notifications" class="notif-dropdown-link" @click="notifOpen = false">{{ t('ดูทั้งหมด →') }}</router-link>
        </div>
      </div>
      <span class="header-divider desktop-only" aria-hidden="true"></span>
      <router-link v-if="!authState.user && authState.status !== 'loading'" to="/login" class="header-link desktop-only">
        {{ t('เข้าสู่ระบบ') }}
      </router-link>
      <!-- ปุ่มตั้งค่า: login แล้วมีรูปย่อ (ตัวอักษรแรกของอีเมล), ยังไม่ login เป็นไอคอนเฟือง -->
      <router-link to="/settings" class="settings-pill desktop-only" active-class="settings-pill-active">
        <span v-if="authState.user" class="settings-avatar" aria-hidden="true">{{ avatarLetter }}</span>
        <Settings v-else :size="18" :stroke-width="2" class="settings-gear" aria-hidden="true" />
        <span>{{ t('ตั้งค่า') }}</span>
      </router-link>
    </div>
  </header>
</template>

<style scoped>
.app-header {
  position: fixed;
  top: 0;
  left: 0;
  right: 0;
  z-index: 100;
  /* iOS แบบติดตั้ง (standalone) เนื้อหาเริ่มใต้ status bar — ดันลงด้วย safe-area */
  padding: env(safe-area-inset-top, 0px) max(32px, env(safe-area-inset-right, 0px)) 0 max(32px, env(safe-area-inset-left, 0px));
  height: calc(var(--app-header-h) + env(safe-area-inset-top, 0px));
  box-sizing: border-box;
  display: flex;
  align-items: center;
  justify-content: space-between;
  gap: 16px;
  background: var(--card);
  border-bottom: 1px solid var(--line);
  color: var(--ink);
  font-size: 0.95rem;
}

.app-header-left {
  display: flex;
  align-items: center;
  gap: 32px;
  min-width: 0;
  height: 100%;
}

.brand {
  display: flex;
  align-items: center;
  gap: 10px;
  text-decoration: none;
  color: var(--ink);
  flex-shrink: 0;
}
.brand-logo {
  width: 32px;
  height: 32px;
  border-radius: 8px;
  flex-shrink: 0;
}
.brand-name {
  font-family: var(--font-display);
  font-weight: 600;
  font-size: 1.125rem;
  letter-spacing: -0.01em;
}

.app-header-nav {
  display: flex;
  align-items: stretch;
  gap: 8px;
  height: 100%;
}
.nav-tab {
  display: flex;
  align-items: center;
  padding: 0 12px;
  color: var(--muted);
  text-decoration: none;
  font-weight: 500;
  border-bottom: 2px solid transparent;
  /* ให้เส้นใต้ทับเส้นคั่นล่างของ header พอดี */
  margin-bottom: -1px;
  transition: color 0.15s ease, border-color 0.15s ease;
}
.nav-tab:hover { color: var(--ink); }
.nav-tab-active {
  color: var(--ink);
  font-weight: 600;
  border-bottom-color: var(--accent);
}

.app-header-auth {
  display: flex;
  align-items: center;
  gap: 8px;
}

.header-divider {
  width: 1px;
  height: 24px;
  background: var(--line);
  margin: 0 4px;
}

.header-link {
  color: var(--muted);
  text-decoration: none;
  font-weight: 500;
  padding: 8px 6px;
}
.header-link:hover { color: var(--ink); }

.settings-pill {
  display: flex;
  align-items: center;
  gap: 8px;
  min-height: 40px;
  padding: 4px 14px 4px 4px;
  border: 1px solid var(--line);
  border-radius: 999px;
  color: var(--ink);
  text-decoration: none;
  font-weight: 500;
  font-size: 0.9rem;
  transition: background 0.15s ease, border-color 0.15s ease;
}
.settings-pill:hover { background: var(--track); }
.settings-pill-active { border-color: var(--accent); }
.settings-avatar {
  width: 30px;
  height: 30px;
  border-radius: 50%;
  background: var(--ink);
  color: var(--card);
  display: flex;
  align-items: center;
  justify-content: center;
  font-family: var(--font-display);
  font-weight: 600;
  font-size: 0.8rem;
}
.settings-gear {
  width: 30px;
  margin-left: 2px;
  color: var(--muted);
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
  width: 40px;
  height: 40px;
  border: none;
  background: transparent;
  color: var(--ink);
  border-radius: 50%;
  cursor: pointer;
  transition: background 0.15s ease;
}
.notif-bell:hover { background: var(--track); }

.brand:focus-visible,
.nav-tab:focus-visible,
.header-link:focus-visible,
.settings-pill:focus-visible,
.notif-bell:focus-visible {
  outline: 2px solid var(--accent);
  outline-offset: 2px;
  border-radius: 8px;
}
.settings-pill:focus-visible,
.notif-bell:focus-visible { border-radius: 999px; }

/* จุดแจ้งเตือนบนไอคอนกระดิ่ง (ขอบสีเดียวกับพื้น header ให้แยกจากไอคอนชัด) */
.notif-bell-alert::after {
  content: '';
  position: absolute;
  top: 8px;
  right: 9px;
  width: 8px;
  height: 8px;
  border-radius: 50%;
  background: var(--danger);
  border: 2px solid var(--card);
  animation: notif-pulse-dot 2s ease-in-out infinite;
}

@keyframes notif-pulse-dot {
  0%, 100% { opacity: 1; }
  50% { opacity: 0.6; }
}
@media (prefers-reduced-motion: reduce) {
  .notif-bell-alert::after { animation: none; }
}

.notif-backdrop { display: none; }
.notif-grip { display: none; }

.notif-dropdown {
  position: absolute;
  top: calc(100% + 10px);
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
.notif-alert-body { display: flex; flex-direction: column; gap: 2px; }
.notif-alert-note { font-size: 0.78rem; opacity: 0.8; margin-top: 2px; }

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

.notif-item-good {
  background: var(--success-bg);
  border-color: color-mix(in srgb, var(--success-ink) 35%, transparent);
  color: var(--success-ink);
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

.notif-history-note-good {
  color: var(--success-ink);
}

.notif-dropdown-link {
  align-self: flex-end;
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
    padding-left: max(16px, env(safe-area-inset-left, 0px));
    padding-right: max(8px, env(safe-area-inset-right, 0px));
  }
  .brand-logo { width: 28px; height: 28px; border-radius: 7px; }
  .brand-name { font-size: 1.0625rem; }
  .notif-bell { width: 44px; height: 44px; }
  .app-header-auth { gap: 0; }
  .notif-bell-alert::after { top: 10px; right: 11px; }

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
