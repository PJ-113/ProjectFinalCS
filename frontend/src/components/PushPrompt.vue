<script setup>
import { onMounted, ref } from "vue";
import { Bell, BellOff, BellRing } from "@lucide/vue";
import { subscribeToPush, unsubscribeFromPush, getExistingPushSubscription } from "../notifications";

// ตั้งใจ "ไม่ popup อัตโนมัติ" แบบ InstallPrompt.vue — การ์ดนี้ฝังอยู่กับเนื้อหาปกติของ NotificationsView.vue
// (ไม่ใช่ fixed-position banner ลอยทับหน้าจอ) และต้องรอผู้ใช้กดปุ่มเองเท่านั้นถึงจะเรียก requestPermission()
// เหตุผล: permission ขอซ้ำไม่ได้ง่ายๆถ้าโดนปฏิเสธ (ต่างจาก install prompt ที่ขอใหม่ได้เรื่อยๆ) และ iOS
// บังคับต้องมาจาก user gesture อยู่แล้ว
const props = defineProps({
  province: { type: String, required: true },
});

const status = ref("checking"); // 'checking' | 'unsupported' | 'denied' | 'ios-not-installed' | 'subscribable' | 'subscribed'
const busy = ref(false);
const errorMsg = ref("");

function isStandalone() {
  // เหมือนกับ InstallPrompt.vue เป๊ะ — iOS Safari ต้องติดตั้งเป็น PWA ก่อนถึงจะรับ push ได้จริง
  return window.matchMedia("(display-mode: standalone)").matches || window.navigator.standalone === true;
}

function isIOS() {
  return /iphone|ipad|ipod/i.test(navigator.userAgent);
}

// เขียนแยกเป็นฟังก์ชันแทนที่จะอ้าง window/navigator ตรงๆ ในเทมเพลต — เทมเพลตของ Vue คอมไพล์ expression
// ในสโคปที่ไม่รับประกันว่าจะ resolve global `window`/`navigator` ได้ถูกต้องเสมอ (เจอจริง: `'Notification' in
// window` ในเทมเพลตพัง เพราะ window กลายเป็น undefined ตอน runtime ทำให้ทั้ง component render พังไปด้วย)
function hasNotificationApi() {
  return "Notification" in window;
}
function hasPushManagerApi() {
  return "PushManager" in window;
}
function hasServiceWorkerApi() {
  return "serviceWorker" in navigator;
}

async function refreshStatus() {
  // เช็ค iOS-ไม่ได้ติดตั้งก่อนเช็คว่า API มีจริงไหม — บน iOS Safari ที่ไม่ได้เปิดจากไอคอนหน้าจอโฮม
  // (standalone) ตัว Notification/PushManager จะ "ไม่มีอยู่จริง" ใน window เลย (ไม่ใช่แค่ permission ปิด)
  // ถ้าเช็ค unsupported ก่อน จะเจอเคสนี้แล้วไม่โชว์อะไรเลย ทั้งที่ควรโชว์ hint "ติดตั้งแอปก่อน" แทน
  if (isIOS() && !isStandalone()) {
    status.value = "ios-not-installed";
    return;
  }
  if (!hasNotificationApi() || !hasServiceWorkerApi() || !hasPushManagerApi()) {
    status.value = "unsupported";
    return;
  }
  if (Notification.permission === "denied") {
    status.value = "denied";
    return;
  }
  const existing = await getExistingPushSubscription().catch(() => null);
  if (existing) {
    status.value = "subscribed";
    return;
  }
  status.value = "subscribable";
}

async function enable() {
  errorMsg.value = "";
  busy.value = true;
  try {
    const permission = await Notification.requestPermission();
    if (permission !== "granted") {
      status.value = permission === "denied" ? "denied" : "subscribable";
      return;
    }
    await subscribeToPush(props.province);
    status.value = "subscribed";
  } catch (err) {
    errorMsg.value = "เปิดการแจ้งเตือนไม่สำเร็จ ลองใหม่อีกครั้ง";
  } finally {
    busy.value = false;
  }
}

async function disable() {
  busy.value = true;
  try {
    await unsubscribeFromPush();
    status.value = "subscribable";
  } finally {
    busy.value = false;
  }
}

onMounted(refreshStatus);
</script>

<template>
  <div v-if="status === 'subscribable'" class="push-card">
    <Bell :size="20" :stroke-width="2" class="push-icon" aria-hidden="true" />
    <div class="push-text">
      <p class="push-title">เปิดการแจ้งเตือนคุณภาพอากาศ</p>
      <p class="push-sub">รับแจ้งเตือนทันทีแม้ไม่ได้เปิดแอปอยู่</p>
    </div>
    <button type="button" class="push-btn" :disabled="busy" @click="enable">เปิด</button>
  </div>

  <div v-else-if="status === 'subscribed'" class="push-card push-card-active">
    <BellRing :size="20" :stroke-width="2" class="push-icon" aria-hidden="true" />
    <div class="push-text">
      <p class="push-title">เปิดการแจ้งเตือนอยู่</p>
      <p class="push-sub">จังหวัด{{ province }}</p>
    </div>
    <button type="button" class="push-btn-ghost" :disabled="busy" @click="disable">ปิดการแจ้งเตือน</button>
  </div>

  <div v-else-if="status === 'ios-not-installed'" class="push-card push-card-hint">
    <BellOff :size="20" :stroke-width="2" class="push-icon" aria-hidden="true" />
    <div class="push-text">
      <p class="push-title">ติดตั้งแอปก่อนเพื่อเปิดการแจ้งเตือน</p>
      <p class="push-sub">iOS รองรับการแจ้งเตือนเฉพาะแอปที่ติดตั้งลงหน้าจอโฮมแล้วเท่านั้น</p>
    </div>
  </div>

  <div v-else-if="status === 'denied'" class="push-card push-card-hint">
    <BellOff :size="20" :stroke-width="2" class="push-icon" aria-hidden="true" />
    <div class="push-text">
      <p class="push-title">การแจ้งเตือนถูกปิดไว้</p>
      <p class="push-sub">ไปเปิดใน Settings ของอุปกรณ์ > การแจ้งเตือน ของแอปนี้</p>
    </div>
  </div>

  <div v-else-if="status === 'unsupported'" class="push-card push-card-hint">
    <BellOff :size="20" :stroke-width="2" class="push-icon" aria-hidden="true" />
    <div class="push-text">
      <p class="push-title">อุปกรณ์/เบราว์เซอร์นี้ยังไม่รองรับการแจ้งเตือน</p>
      <p class="push-sub">iOS ต้องเป็นเวอร์ชัน 16.4 ขึ้นไป และต้องเปิดจากไอคอนที่ติดตั้งไว้หน้าจอโฮม</p>
    </div>
  </div>

  <p v-if="errorMsg" class="push-error">{{ errorMsg }}</p>
</template>

<style scoped>
.push-card {
  display: flex;
  align-items: center;
  gap: 12px;
  background: var(--card);
  border: 1px solid var(--line);
  border-radius: 20px;
  padding: 14px 16px;
  margin-bottom: 16px;
}
.push-card-active { border-color: color-mix(in srgb, var(--accent) 35%, var(--line)); }
.push-card-hint { background: var(--track); }

.push-icon { flex-shrink: 0; color: var(--accent); }

.push-text { flex: 1; min-width: 0; }
.push-title { margin: 0; font-weight: 700; font-size: 0.88rem; color: var(--ink); }
.push-sub { margin: 2px 0 0; font-size: 0.78rem; color: var(--muted); }

.push-btn {
  flex-shrink: 0;
  background: var(--accent);
  color: #fff;
  border: none;
  border-radius: 999px;
  padding: 8px 16px;
  font-weight: 600;
  font-size: 0.82rem;
  cursor: pointer;
  white-space: nowrap;
}
.push-btn:hover { filter: brightness(1.08); }
.push-btn:disabled { opacity: 0.6; cursor: default; }

.push-btn-ghost {
  flex-shrink: 0;
  background: transparent;
  color: var(--muted);
  border: 1px solid var(--line);
  border-radius: 999px;
  padding: 7px 14px;
  font-size: 0.78rem;
  cursor: pointer;
  white-space: nowrap;
}
.push-btn-ghost:hover { background: var(--track); color: var(--ink); }
.push-btn-ghost:disabled { opacity: 0.6; cursor: default; }

.push-error { margin: -8px 0 16px; font-size: 0.78rem; color: var(--danger); }
</style>
