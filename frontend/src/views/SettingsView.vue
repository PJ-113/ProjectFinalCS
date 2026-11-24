<script setup>
import { computed, ref } from 'vue'
import { useRouter } from 'vue-router'
import { ChevronRight, Lock, LogOut, Bell, BellOff, User, MapPin } from '@lucide/vue'
import { authState, logout } from '../auth'
import { provinceState } from '../provinces'
import { themeState, setTheme } from '../theme'
import { usePushStatus } from '../pushStatus'
import ProvincePickerSheet from '../components/ProvincePickerSheet.vue'
import { t, localeState, setLocale, provinceName } from '../i18n'

// หน้าตั้งค่า (แทนหน้าโปรไฟล์เดิม) — เปิดได้โดยไม่ต้อง login เพราะคนที่ไม่ได้ login ก็ต้องเปลี่ยนธีมได้
// ส่วนบัญชี (ข้อมูลส่วนตัว/เปลี่ยนรหัสผ่าน/ออกจากระบบ) แสดงเฉพาะตอน login แล้ว หน้าย่อยอยู่ที่
// /settings/profile (SettingsProfileView.vue) และ /settings/password (SettingsPasswordView.vue)
const router = useRouter()

const avatarLetter = computed(() => (authState.user?.email || '?').charAt(0).toUpperCase())
const homeProvinceName = computed(
  () => provinceName(provinceState.list.find((p) => p.province_en === authState.user?.home_province)),
)
const profileSub = computed(() => {
  const parts = [authState.user?.email]
  parts.push(homeProvinceName.value || t('ยังไม่ได้ตั้งจังหวัดหลัก'))
  return parts.join(' · ')
})

// ธีม: ภาพตัวอย่างหน้าจอย่อของแต่ละแบบ ใช้สีจริงของธีมนั้น (ค่าเดียวกับ --ground/--card/--ink/--line ใน main.css)
// "ตามระบบ" เป็นภาพครึ่งสว่างครึ่งมืด
const LIGHT = { ground: '#f6f4ef', card: '#ffffff', ink: '#16181d', line: '#e6e1d6' }
const DARK = { ground: '#0e1013', card: '#171a1f', ink: '#f3f4f6', line: '#272b33' }
// ภาษา: ชื่อแต่ละภาษาเขียนด้วยภาษานั้นเอง (ไม่แปล) ให้คนที่อ่านอีกภาษาไม่ออกก็ยังหาเจอ
const LANGUAGE_OPTIONS = [
  { id: 'th', label: 'ไทย' },
  { id: 'en', label: 'English' },
]

const THEME_OPTIONS = [
  { id: 'light', label: 'สว่าง', halves: [LIGHT, LIGHT] },
  { id: 'dark', label: 'มืด', halves: [DARK, DARK] },
  { id: 'system', label: 'ตามระบบ', halves: [LIGHT, DARK] },
]

// Web Push: watchedProvince = จังหวัดเริ่มต้นตอนกดเปิดครั้งแรกเท่านั้น (home_province ถ้าตั้งไว้ ไม่งั้นจังหวัดที่
// เลือกดูอยู่) — หลังเปิดแล้ว จังหวัดที่ได้รับแจ้งเตือนจริงคือ push.pushedProvince (ค่าในฐานข้อมูล) ซึ่งเปลี่ยนเองได้
// จากแถว "จังหวัดที่แจ้งเตือน" ไม่เปลี่ยนตามจังหวัดที่ดูอยู่ (เดิมโชว์ watchedProvince ซึ่งอาจไม่ตรงกับความจริง)
const watchedProvince = computed(() => authState.user?.home_province || provinceState.selected)
const push = usePushStatus(watchedProvince)
const pushedProvinceName = computed(
  () => provinceName(provinceState.list.find((p) => p.province_en === push.pushedProvince.value)),
)
const pushPickerOpen = ref(false)
const pushOn = computed(() => push.status.value === 'subscribed')
const pushToggleable = computed(() => ['subscribed', 'subscribable', 'checking'].includes(push.status.value))
const pushSub = computed(() => {
  switch (push.status.value) {
    case 'subscribed':
      return t('เปิดอยู่')
    case 'subscribable':
      return t('รับแจ้งเตือนแม้ไม่ได้เปิดแอปอยู่')
    case 'ios-not-installed':
      return t('ติดตั้งแอปลงหน้าจอโฮมก่อน (iOS รองรับเฉพาะแอปที่ติดตั้งแล้ว)')
    case 'denied':
      return t('ถูกปิดไว้ — ไปเปิดในการตั้งค่าการแจ้งเตือนของอุปกรณ์')
    case 'unsupported':
      return t('อุปกรณ์/เบราว์เซอร์นี้ยังไม่รองรับ')
    default:
      return t('กำลังตรวจสอบ...')
  }
})

function togglePush() {
  if (push.busy.value || push.status.value === 'checking') return
  if (pushOn.value) push.disable()
  else push.enable()
}

async function handleLogout() {
  await logout()
  router.push('/')
}
</script>

<template>
  <div class="settings-page">
    <div class="wrap">
      <h1>{{ t('ตั้งค่า') }}</h1>

      <!-- บัญชี -->
      <div class="section">
        <div class="section-head">
          <h2 class="group-label">{{ t('บัญชี') }}</h2>
          <p class="section-desc">{{ t('ข้อมูลส่วนตัว จังหวัดหลัก และรหัสผ่าน') }}</p>
        </div>
        <section v-if="authState.user" class="group">
          <router-link to="/settings/profile" class="row row-profile">
            <span class="avatar" aria-hidden="true">{{ avatarLetter }}</span>
            <span class="row-text">
              <span class="row-title">{{ t('ข้อมูลส่วนตัว') }}</span>
              <span class="row-sub">{{ profileSub }}</span>
            </span>
            <ChevronRight :size="20" :stroke-width="2" class="row-chevron" aria-hidden="true" />
          </router-link>
          <router-link to="/settings/password" class="row">
            <Lock :size="20" :stroke-width="2" class="row-icon" aria-hidden="true" />
            <span class="row-text"><span class="row-title">{{ t('เปลี่ยนรหัสผ่าน') }}</span></span>
            <ChevronRight :size="20" :stroke-width="2" class="row-chevron" aria-hidden="true" />
          </router-link>
        </section>
        <section v-else class="group guest-card">
        <div class="guest-head">
          <span class="avatar avatar-guest" aria-hidden="true"><User :size="24" :stroke-width="2" /></span>
          <span class="row-text">
            <span class="row-title">{{ t('ยังไม่ได้เข้าสู่ระบบ') }}</span>
            <span class="row-sub">{{ t('เข้าสู่ระบบเพื่อตั้งจังหวัดหลักและกลุ่มเสี่ยง ให้การแจ้งเตือนตรงกับคุณ') }}</span>
          </span>
        </div>
        <div class="guest-actions">
          <router-link to="/login" class="pill pill-primary">{{ t('เข้าสู่ระบบ') }}</router-link>
          <router-link to="/register" class="pill pill-outline">{{ t('สมัครสมาชิก') }}</router-link>
        </div>
        </section>
      </div>

      <!-- ธีม -->
      <div class="section">
        <div class="section-head">
          <h2 class="group-label" id="theme-label">{{ t('ธีม') }}</h2>
          <p class="section-desc">{{ t('"ตามระบบ" จะสลับสว่าง/มืดตามที่ตั้งไว้ในเครื่อง') }}</p>
        </div>
      <section class="group theme-group" role="radiogroup" aria-labelledby="theme-label">
        <button
          v-for="opt in THEME_OPTIONS"
          :key="opt.id"
          type="button"
          role="radio"
          class="theme-option"
          :class="{ active: themeState.preference === opt.id }"
          :aria-checked="themeState.preference === opt.id"
          @click="setTheme(opt.id)"
        >
          <span class="theme-preview" aria-hidden="true">
            <span
              v-for="(h, i) in opt.halves"
              :key="i"
              class="theme-half"
              :style="{ background: h.ground }"
            >
              <span class="theme-bar" :style="{ background: h.ink }"></span>
              <span class="theme-card" :style="{ background: h.card, borderColor: h.line }"></span>
              <span class="theme-card" :style="{ background: h.card, borderColor: h.line }"></span>
            </span>
          </span>
          <span class="theme-label">{{ t(opt.label) }}</span>
        </button>
      </section>
      </div>

      <!-- ภาษา -->
      <div class="section">
        <div class="section-head">
          <h2 class="group-label" id="lang-label">{{ t('ภาษา') }}</h2>
          <p class="section-desc">{{ t('สรุปรายวันและข้อความแจ้งเตือนจากระบบยังเป็นภาษาไทย') }}</p>
        </div>
        <section class="group lang-group" role="radiogroup" aria-labelledby="lang-label">
          <button
            v-for="opt in LANGUAGE_OPTIONS"
            :key="opt.id"
            type="button"
            role="radio"
            class="lang-option"
            :class="{ active: localeState.locale === opt.id }"
            :aria-checked="localeState.locale === opt.id"
            :lang="opt.id"
            @click="setLocale(opt.id)"
          >{{ opt.label }}</button>
        </section>
      </div>

      <!-- การแจ้งเตือน -->
      <div class="section">
        <div class="section-head">
          <h2 class="group-label">{{ t('การแจ้งเตือน') }}</h2>
          <p class="section-desc">{{ t('แจ้งเตือนคุณภาพอากาศแม้ไม่ได้เปิดแอปอยู่') }}</p>
        </div>
        <div class="section-body">
      <section class="group">
        <div class="row row-static">
          <component :is="pushToggleable ? Bell : BellOff" :size="20" :stroke-width="2" class="row-icon" aria-hidden="true" />
          <span class="row-text">
            <span class="row-title" id="push-label">{{ t('แจ้งเตือนบนเครื่อง') }}</span>
            <span class="row-sub">{{ pushSub }}</span>
          </span>
          <button
            v-if="pushToggleable"
            type="button"
            role="switch"
            class="switch"
            :class="{ on: pushOn }"
            :aria-checked="pushOn"
            aria-labelledby="push-label"
            :disabled="push.busy.value || push.status.value === 'checking'"
            @click="togglePush"
          >
            <span class="switch-knob"></span>
          </button>
        </div>
        <button
          v-if="pushOn"
          type="button"
          class="row"
          aria-haspopup="dialog"
          :disabled="push.busy.value"
          @click="pushPickerOpen = true"
        >
          <MapPin :size="20" :stroke-width="2" class="row-icon" aria-hidden="true" />
          <span class="row-text"><span class="row-title">{{ t('จังหวัดที่แจ้งเตือน') }}</span></span>
          <span class="row-value">{{ pushedProvinceName || '—' }}</span>
          <ChevronRight :size="20" :stroke-width="2" class="row-chevron" aria-hidden="true" />
        </button>
      </section>
      <p v-if="push.errorMsg.value" class="push-error" role="alert">{{ push.errorMsg.value }}</p>
        </div>
      </div>

      <!-- ออกจากระบบ: แยกการ์ดไว้ล่างสุด ไม่ให้กดโดนง่าย -->
      <div v-if="authState.user" class="section section-logout">
        <div class="section-head" aria-hidden="true"></div>
      <section class="group">
        <button type="button" class="row row-danger" @click="handleLogout">
          <LogOut :size="20" :stroke-width="2" aria-hidden="true" />
          <span class="row-title">{{ t('ออกจากระบบ') }}</span>
        </button>
      </section>
      </div>

      <p class="footer">{{ t('DustCast24 · พยากรณ์ PM2.5 ภาคอีสาน') }}</p>
    </div>

    <ProvincePickerSheet
      :open="pushPickerOpen"
      :selected="push.pushedProvince.value || ''"
      @select="push.changeProvince"
      @close="pushPickerOpen = false"
    />
  </div>
</template>

<style scoped>
/* โทเคนสีมาจาก :root ใน main.css (ชุดเดียวกับหน้าแจ้งเตือน/หน้าย่อยตั้งค่า) */
.settings-page {
  min-height: 100vh;
  padding: var(--app-top-offset) 16px calc(var(--app-bottom-offset) + 24px);
  background: var(--ground);
  color: var(--ink);
}
.wrap {
  max-width: 480px;
  margin: 0 auto;
  display: flex;
  flex-direction: column;
  gap: 10px;
}
h1 { font-size: 1.6rem; font-weight: 700; line-height: 1.3; margin: 0 0 4px; }

.group-label {
  margin: 10px 0 0 8px;
  font-size: 0.8rem;
  font-weight: 600;
  color: var(--muted);
}
.group {
  background: var(--card);
  border: 1px solid var(--line);
  border-radius: 24px;
  overflow: hidden;
}

/* แถวรายการ */
.row {
  width: 100%;
  box-sizing: border-box;
  min-height: 56px;
  padding: 10px 16px;
  display: flex;
  align-items: center;
  gap: 12px;
  border: 0;
  background: transparent;
  color: var(--ink);
  font-family: inherit;
  text-align: left;
  text-decoration: none;
  cursor: pointer;
}
.row + .row { border-top: 1px solid var(--line); }
.row:not(.row-static):hover { background: var(--track); }
.row-static { cursor: default; min-height: 64px; }
.row-profile { min-height: 68px; }
.row-icon { flex-shrink: 0; }
.row-chevron { flex-shrink: 0; color: var(--muted); }
.row-text { flex: 1; min-width: 0; display: flex; flex-direction: column; gap: 2px; }
.row-title { font-size: 0.95rem; font-weight: 500; }
.row-sub { font-size: 0.8rem; color: var(--muted); line-height: 1.45; overflow-wrap: anywhere; }
.row-value { flex-shrink: 0; font-size: 0.9rem; color: var(--muted); }
.row:disabled { opacity: 0.6; cursor: default; }
.row-danger { color: var(--danger); }
.row-danger .row-title { font-weight: 600; }

.avatar {
  width: 36px;
  height: 36px;
  flex-shrink: 0;
  border-radius: 50%;
  background: var(--ink);
  color: var(--card);
  display: flex;
  align-items: center;
  justify-content: center;
  font-family: var(--font-display);
  font-weight: 600;
  font-size: 0.95rem;
}

/* ยังไม่ login */
.guest-card { padding: 20px; display: flex; flex-direction: column; gap: 14px; }
.guest-head { display: flex; align-items: center; gap: 14px; }
.avatar-guest { width: 52px; height: 52px; background: var(--track); color: var(--muted); }
.guest-actions { display: flex; gap: 8px; }
.pill {
  flex: 1;
  min-height: 48px;
  box-sizing: border-box;
  display: flex;
  align-items: center;
  justify-content: center;
  border-radius: 24px;
  border: 1.5px solid transparent;
  text-decoration: none;
  font-size: 0.95rem;
  font-weight: 600;
}
.pill-primary { background: var(--ink); color: var(--card); border-color: var(--ink); }
.pill-outline { color: var(--ink); border-color: var(--line); }

/* ธีม */
.theme-group {
  padding: 12px;
  display: grid;
  grid-template-columns: repeat(3, minmax(0, 1fr));
  gap: 10px;
}
.theme-option {
  padding: 0;
  border: 0;
  background: transparent;
  color: var(--muted);
  font-family: inherit;
  cursor: pointer;
  display: flex;
  flex-direction: column;
  align-items: center;
  gap: 8px;
}
.theme-preview {
  width: 100%;
  height: 76px;
  box-sizing: border-box;
  display: flex;
  overflow: hidden;
  border-radius: 14px;
  border: 1.5px solid var(--line);
}
.theme-option.active .theme-preview { border: 2.5px solid var(--accent); }
.theme-half { flex: 1; display: flex; flex-direction: column; gap: 4px; padding: 6px; }
.theme-bar { height: 6px; width: 60%; border-radius: 3px; }
.theme-card { height: 14px; border-radius: 4px; border: 1px solid; }
.theme-label { font-size: 0.875rem; font-weight: 500; }
.theme-option.active .theme-label { color: var(--ink); font-weight: 700; }

/* ภาษา: segmented 2 ปุ่ม (หน้าตาเดียวกับ .segmented ในหน้าข้อมูลส่วนตัว) */
.lang-group {
  display: flex;
  gap: 4px;
  padding: 4px;
  background: var(--track);
  border-color: transparent;
}
.lang-option {
  flex: 1;
  min-height: 44px;
  border: 0;
  border-radius: 20px;
  background: transparent;
  color: var(--muted);
  font-family: inherit;
  font-size: 0.95rem;
  font-weight: 600;
  cursor: pointer;
}
.lang-option.active { background: var(--ink); color: var(--card); }

/* สวิตช์ */
.switch {
  width: 52px;
  height: 32px;
  flex-shrink: 0;
  padding: 3px;
  box-sizing: border-box;
  border: 0;
  border-radius: 16px;
  background: var(--track);
  display: flex;
  justify-content: flex-start;
  cursor: pointer;
  transition: background 0.2s ease;
}
.switch.on { background: var(--accent); justify-content: flex-end; }
.switch:disabled { opacity: 0.6; cursor: default; }
.switch-knob {
  width: 26px;
  height: 26px;
  border-radius: 50%;
  background: #ffffff;
  box-shadow: 0 1px 3px rgba(0, 0, 0, 0.25);
}

.push-error { margin: 0 0 0 8px; font-size: 0.8rem; color: var(--danger); }

/* แต่ละกลุ่ม: มือถือ = หัวข้อเหนือการ์ด, จอกว้าง = หัวข้อ+คำอธิบายซ้าย การ์ดขวา (ดู media query ด้านล่าง) */
.section { display: flex; flex-direction: column; gap: 10px; }
.section-body { display: flex; flex-direction: column; gap: 10px; min-width: 0; }
.section-desc { display: none; margin: 4px 0 0; font-size: 0.85rem; color: var(--muted); line-height: 1.5; }
.section-logout { margin-top: 14px; }
.section-logout .section-head { display: none; }

.footer { margin: 8px 0 0; text-align: center; font-size: 0.75rem; color: var(--muted); }

/* เดสก์ท็อป: กว้างขึ้นเป็น 2 คอลัมน์ ไม่อัดกันกลางจอ */
@media (min-width: 900px) {
  .wrap { max-width: 880px; gap: 0; }
  h1 { font-size: 1.9rem; margin-bottom: 16px; }
  .section {
    display: grid;
    grid-template-columns: 240px minmax(0, 1fr);
    gap: 40px;
    padding: 28px 0;
    /* เส้นคั่นกลุ่มเข้มกว่า --line (ซึ่งจางเกินบนพื้น --ground) — ผสมจาก --ink จึงเข้ากับทั้งธีมสว่างและมืด */
    border-top: 1px solid color-mix(in srgb, var(--ink) 20%, var(--ground));
  }
  .section > .group,
  .section-body { min-width: 0; }
  .section-head { padding-top: 4px; }
  .group-label { margin: 0; font-size: 1rem; color: var(--ink); }
  .section-desc { display: block; }
  .section-logout { margin-top: 0; }
  .section-logout .section-head { display: block; }
  .theme-group { padding: 16px; gap: 16px; }
  .theme-preview { height: 104px; }
  .footer { margin-top: 24px; }
}

.row:focus-visible,
.pill:focus-visible,
.theme-option:focus-visible,
.lang-option:focus-visible,
.switch:focus-visible {
  outline: 2px solid var(--accent);
  outline-offset: 2px;
}
.row:focus-visible { outline-offset: -2px; }
</style>
