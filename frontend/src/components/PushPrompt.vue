<script setup>
import { Bell, BellOff, BellRing } from "@lucide/vue";
import { computed, ref } from "vue";
import { usePushStatus } from "../pushStatus";
import { t, provinceName } from "../i18n";
import { provinceState } from "../provinces";

// ตั้งใจ "ไม่ popup อัตโนมัติ" แบบ InstallPrompt.vue — การ์ดนี้ฝังอยู่กับเนื้อหาปกติของ NotificationsView.vue
// (ไม่ใช่ fixed-position banner ลอยทับหน้าจอ) และต้องรอผู้ใช้กดปุ่มเองเท่านั้นถึงจะเรียก requestPermission()
// เหตุผล: permission ขอซ้ำไม่ได้ง่ายๆถ้าโดนปฏิเสธ (ต่างจาก install prompt ที่ขอใหม่ได้เรื่อยๆ) และ iOS
// บังคับต้องมาจาก user gesture อยู่แล้ว — logic สถานะอยู่ใน pushStatus.js (ใช้ร่วมกับสวิตช์ในหน้าตั้งค่า)
const props = defineProps({
  province: { type: String, required: true },
});

// ที่เปิด/ปิดหลักอยู่ในหน้าตั้งค่า (SettingsView.vue) ที่เดียว — การ์ดนี้เป็นแค่ "ตัวชวนเปิด" ในหน้าแจ้งเตือน:
// โชว์เฉพาะตอนยังไม่ได้เปิด (หรือเปิดไม่ได้ พร้อมเหตุผล) ถ้าเปิดอยู่แล้วไม่โชว์อะไรเลย และไม่มีปุ่มปิดที่นี่
// ยกเว้นเพิ่งกดเปิดจากการ์ดนี้ในรอบนี้ จะโชว์ยืนยันสั้นๆ + บอกว่าปิดได้ที่หน้าตั้งค่า (ไม่ให้การ์ดหายไปเฉยๆ
// จนผู้ใช้ไม่แน่ใจว่าเปิดสำเร็จไหม)
const { status, busy, errorMsg, enable } = usePushStatus(() => props.province);
const justEnabled = ref(false);

async function onEnable() {
  await enable();
  if (status.value === "subscribed") justEnabled.value = true;
}

// เดิมโชว์ slug ดิบ ("จังหวัดkhonkaen") — แปลงเป็นชื่อจังหวัดตามภาษาที่เลือก
const provinceLabel = computed(() =>
  provinceName(provinceState.list.find((p) => p.province_en === props.province)) || props.province,
);
</script>

<template>
  <div v-if="status === 'subscribable'" class="push-card">
    <Bell :size="20" :stroke-width="2" class="push-icon" aria-hidden="true" />
    <div class="push-text">
      <p class="push-title">{{ t("เปิดการแจ้งเตือนคุณภาพอากาศ") }}</p>
      <p class="push-sub">{{ t("รับแจ้งเตือนทันทีแม้ไม่ได้เปิดแอปอยู่") }}</p>
    </div>
    <button type="button" class="push-btn" :disabled="busy" @click="onEnable">{{ t("เปิด") }}</button>
  </div>

  <div v-else-if="status === 'subscribed' && justEnabled" class="push-card push-card-active" role="status">
    <BellRing :size="20" :stroke-width="2" class="push-icon" aria-hidden="true" />
    <div class="push-text">
      <p class="push-title">{{ t("เปิดอยู่ · เฝ้าจังหวัด{name}", { name: provinceLabel }) }}</p>
      <p class="push-sub">
        {{ t("ปิดหรือจัดการได้ที่") }} <router-link to="/settings" class="push-link">{{ t("ตั้งค่า") }}</router-link>
      </p>
    </div>
  </div>

  <div v-else-if="status === 'ios-not-installed'" class="push-card push-card-hint">
    <BellOff :size="20" :stroke-width="2" class="push-icon" aria-hidden="true" />
    <div class="push-text">
      <p class="push-title">{{ t("ติดตั้งแอปก่อนเพื่อเปิดการแจ้งเตือน") }}</p>
      <p class="push-sub">{{ t("iOS รองรับการแจ้งเตือนเฉพาะแอปที่ติดตั้งลงหน้าจอโฮมแล้วเท่านั้น") }}</p>
    </div>
  </div>

  <div v-else-if="status === 'denied'" class="push-card push-card-hint">
    <BellOff :size="20" :stroke-width="2" class="push-icon" aria-hidden="true" />
    <div class="push-text">
      <p class="push-title">{{ t("การแจ้งเตือนถูกปิดไว้") }}</p>
      <p class="push-sub">{{ t("ไปเปิดใน Settings ของอุปกรณ์ > การแจ้งเตือน ของแอปนี้") }}</p>
    </div>
  </div>

  <div v-else-if="status === 'unsupported'" class="push-card push-card-hint">
    <BellOff :size="20" :stroke-width="2" class="push-icon" aria-hidden="true" />
    <div class="push-text">
      <p class="push-title">{{ t("อุปกรณ์/เบราว์เซอร์นี้ยังไม่รองรับการแจ้งเตือน") }}</p>
      <p class="push-sub">{{ t("iOS ต้องเป็นเวอร์ชัน 16.4 ขึ้นไป และต้องเปิดจากไอคอนที่ติดตั้งไว้หน้าจอโฮม") }}</p>
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

.push-link { color: var(--accent); font-weight: 600; }

.push-error { margin: -8px 0 16px; font-size: 0.78rem; color: var(--danger); }
</style>
