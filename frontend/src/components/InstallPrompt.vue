<script setup>
import { computed, onMounted, onUnmounted, ref } from "vue";
import { useRoute } from "vue-router";
import { X, Download, Share } from "@lucide/vue";
import { t } from "../i18n";

// ใช้ path ตรงไปที่ public/pwa-192x192.png (ไฟล์เดียวกับไอคอน PWA/favicon/push notification)
// แทนที่จะ import จาก src/assets/logo-icon.svg (โลโก้เก่า คนละไฟล์กัน ไม่ sync กันอัตโนมัติ)
const logoIcon = "/pwa-192x192.png";

// ไม่โชว์ในหน้า login/register — ผู้ใช้กำลังโฟกัสกรอกฟอร์ม ไม่ควรมีแบนเนอร์มาบังหรือแย่งความสนใจ
const HIDDEN_ROUTES = ["login", "register"];
const route = useRoute();
const hiddenByRoute = computed(() => HIDDEN_ROUTES.includes(route.name));

// แบนเนอร์ชวนติดตั้งแอป (PWA) — เก็บ event `beforeinstallprompt` ไว้เอง เพราะเบราว์เซอร์ยิง event นี้
// แค่ครั้งเดียวตอนโหลดหน้า ถ้าไม่ดักไว้ (preventDefault) เบราว์เซอร์จะโชว์ popup ของตัวเองแทนและเราจะเรียก
// ปุ่ม "ติดตั้ง" ของเราเองไม่ได้อีกเลยจนกว่าจะโหลดหน้าใหม่
// iOS (Safari/Chrome-on-iOS ก็ยังใช้ WebKit) ไม่ส่ง event นี้เลย ต้องเดาจาก user agent แทน และบอกวิธีกดเอง
// เพราะ Apple ไม่เปิดให้เว็บเรียก prompt ติดตั้งเองได้

const SHOW_DELAY_MS = 2000; // หน่วงสักพักก่อนโผล่ ไม่ให้รู้สึกโดนยัดเยียดทันทีที่หน้าโหลดเสร็จ

// ไม่มีการจำการปิด (dismiss) ข้ามรอบเลย — กด "ไม่ใช่ตอนนี้" แค่ซ่อนแบนเนอร์ของรอบนี้เท่านั้น ตั้งใจให้ขึ้นใหม่
// ทุกครั้งที่เปิด/โหลดหน้าเว็บ ตราบใดที่ยังไม่ได้ติดตั้งแอปจริง (isStandalone() เท่านั้นที่หยุดมันถาวร)
const show = ref(false);
const mode = ref("android"); // "android" | "ios"
let deferredPrompt = null;

function isStandalone() {
  return window.matchMedia("(display-mode: standalone)").matches || window.navigator.standalone === true;
}

function isIOS() {
  return /iphone|ipad|ipod/i.test(navigator.userAgent);
}

function maybeShow() {
  if (isStandalone()) return;
  if (deferredPrompt) {
    mode.value = "android";
    setTimeout(() => { show.value = true; }, SHOW_DELAY_MS);
  } else if (isIOS()) {
    mode.value = "ios";
    setTimeout(() => { show.value = true; }, SHOW_DELAY_MS);
  }
}

function onBeforeInstallPrompt(e) {
  e.preventDefault();
  deferredPrompt = e;
  maybeShow();
}

function onAppInstalled() {
  show.value = false;
  deferredPrompt = null;
}

async function install() {
  if (!deferredPrompt) return;
  deferredPrompt.prompt();
  await deferredPrompt.userChoice; // ไม่ต้องสนใจผลว่า accepted/dismissed — ทั้งสองแบบปิดแบนเนอร์ของเราเหมือนกัน
  deferredPrompt = null;
  show.value = false;
}

function dismiss() {
  show.value = false; // แค่ซ่อนรอบนี้ — ไม่บันทึกอะไรไว้เลย เปิดหน้าใหม่/รีเฟรชแล้วจะขึ้นอีกถ้ายังไม่ได้ติดตั้ง
}

onMounted(() => {
  window.addEventListener("beforeinstallprompt", onBeforeInstallPrompt);
  window.addEventListener("appinstalled", onAppInstalled);
  // iOS ไม่มี beforeinstallprompt ให้รอ เช็คได้ทันที
  if (isIOS()) maybeShow();
});
onUnmounted(() => {
  window.removeEventListener("beforeinstallprompt", onBeforeInstallPrompt);
  window.removeEventListener("appinstalled", onAppInstalled);
});
</script>

<template>
  <Transition name="install-slide">
    <div v-if="show && !hiddenByRoute" class="install-banner" role="dialog" :aria-label="t('ติดตั้งแอป')">
      <img :src="logoIcon" alt="" class="install-logo" aria-hidden="true" />
      <div class="install-text">
        <p class="install-title">{{ t('ติดตั้งแอปพยากรณ์ PM2.5') }}</p>
        <p v-if="mode === 'android'" class="install-sub">{{ t('เข้าใช้งานได้เร็วขึ้นจากหน้าจอหลัก เหมือนแอปทั่วไป') }}</p>
        <p v-else class="install-sub">
          {{ t('แตะ') }} <Share :size="14" :stroke-width="2" class="inline-icon" aria-hidden="true" /> {{ t('แล้วเลือก "เพิ่มไปยังหน้าจอโฮม"') }}
        </p>
      </div>
      <button v-if="mode === 'android'" type="button" class="install-btn" @click="install">
        <Download :size="16" :stroke-width="2" aria-hidden="true" />
        {{ t('ติดตั้ง') }}
      </button>
      <button type="button" class="install-close" :aria-label="t('ปิด')" @click="dismiss">
        <X :size="18" :stroke-width="2" />
      </button>
    </div>
  </Transition>
</template>

<style scoped>
.install-banner {
  position: fixed;
  left: 12px;
  right: 12px;
  bottom: calc(12px + var(--app-bottom-offset, 0px) + env(safe-area-inset-bottom, 0px));
  z-index: 120;
  display: flex;
  align-items: center;
  gap: 12px;
  background: var(--card);
  border: 1px solid var(--line);
  border-radius: 16px;
  padding: 14px 14px;
  box-shadow: 0 10px 34px rgba(0, 0, 0, 0.16);
  max-width: 420px;
  margin: 0 auto;
}

.install-logo {
  width: 36px;
  height: 36px;
  flex-shrink: 0;
  border-radius: 9px;
}

.install-text {
  flex: 1;
  min-width: 0;
}
.install-title {
  margin: 0;
  font-weight: 700;
  font-size: 0.9rem;
  color: var(--ink);
}
.install-sub {
  margin: 2px 0 0;
  font-size: 0.78rem;
  color: var(--muted);
  display: flex;
  align-items: center;
  gap: 4px;
  flex-wrap: wrap;
}
.inline-icon {
  display: inline-block;
  vertical-align: -2px;
  color: var(--accent);
}

.install-btn {
  flex-shrink: 0;
  display: inline-flex;
  align-items: center;
  gap: 6px;
  background: var(--accent);
  color: #fff;
  border: none;
  border-radius: 999px;
  padding: 9px 16px;
  font-weight: 600;
  font-size: 0.85rem;
  cursor: pointer;
  white-space: nowrap;
}
.install-btn:hover { filter: brightness(1.08); }
.install-btn:focus-visible { outline: 2px solid var(--accent); outline-offset: 2px; }

.install-close {
  flex-shrink: 0;
  display: flex;
  align-items: center;
  justify-content: center;
  width: 30px;
  height: 30px;
  border: none;
  background: transparent;
  color: var(--muted);
  border-radius: 50%;
  cursor: pointer;
}
.install-close:hover { background: var(--track); color: var(--ink); }
.install-close:focus-visible { outline: 2px solid var(--accent); outline-offset: 2px; }

.install-slide-enter-active,
.install-slide-leave-active {
  transition: transform 0.28s ease, opacity 0.28s ease;
}
.install-slide-enter-from,
.install-slide-leave-to {
  transform: translateY(16px);
  opacity: 0;
}

@media (prefers-reduced-motion: reduce) {
  .install-slide-enter-active,
  .install-slide-leave-active { transition: none; }
}
</style>
