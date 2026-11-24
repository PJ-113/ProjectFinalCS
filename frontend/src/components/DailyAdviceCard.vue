<script setup>
import { computed, ref, watch } from "vue";
import { Check, Eye, Fan, Pill } from "@lucide/vue";
import { t } from "../i18n";

// การ์ด "คำแนะนำวันนี้" (CHANGELOG 270) — ข้อป้องกันตัวเป็นไอคอน แตะแล้วเห็นประโยคเต็ม (ดีไซน์ Option D)
// ids มาจาก advice_config ของ /api/forecast (PROTECTIVE_ADVICE_RULES ใน app.py = แหล่งกฎที่เดียว) เลือกแถวตาม
// ระดับฝุ่น "ตอนนี้" โดย DashboardMap.vue — ไฟล์นี้รู้แค่ id -> ไอคอน/ข้อความ (แปลภาษาผ่าน t())
// ตั้งใจไม่ใช่ checklist: ไม่มีติ๊ก ไม่เก็บสถานะว่าทำแล้วหรือยัง
// สีไอคอน/กล่องประโยคเป็นสีกลาง (--track/--ink) ไม่เปลี่ยนตามระดับฝุ่น — เคยใช้ --hero-bg ของหน้าแล้วระดับ "ดี"
// ทำให้ไอคอนข้อควรระวังเป็นสีเขียว อ่านเหมือน "ปลอดภัย" และปนกับสถานะปกติ (CHANGELOG 271); สีเขียวสงวนไว้ให้
// สถานะ "ปกติ — ไม่มีข้อควรระวังพิเศษ" อย่างเดียว ระดับฝุ่นดูจากสีการ์ด PM2.5 ด้านบน
const props = defineProps({
  ids: { type: Array, required: true },
  riskGroup: { type: String, default: "ทั่วไป" },
});

// ไอคอนหน้ากาก/N95/หน้าต่างไม่มีใน lucide — วาดเป็น SVG เองใน template (kind = 'mask' | 'n95' | 'window')
const ITEMS = {
  mask: { kind: "mask", label: "สวมหน้ากาก", sentence: "สวมหน้ากากป้องกันฝุ่นเมื่อออกนอกอาคาร" },
  n95: { kind: "n95", label: "หน้ากาก N95", sentence: "สวมหน้ากาก N95 เมื่อจำเป็นต้องออกนอกอาคาร" },
  window: { kind: "window", label: "ปิดหน้าต่าง", sentence: "ปิดประตูหน้าต่าง ลดฝุ่นที่เข้ามาในบ้าน" },
  purifier: { icon: Fan, label: "เครื่องฟอกอากาศ", sentence: "เปิดเครื่องฟอกอากาศในห้องที่อยู่ประจำ (ถ้ามี)" },
  symptom: {
    icon: Eye,
    label: "สังเกตอาการ",
    sentenceByGroup: {
      "ทั่วไป": "สังเกตอาการระคายเคืองตา จมูก หรือไอ",
      "เด็ก-ผู้สูงอายุ": "คอยสังเกตอาการไอ หายใจลำบาก หรือระคายเคืองตาของเด็กและผู้สูงอายุ",
      "มีโรคประจำตัว": "สังเกตอาการแน่นหน้าอก หายใจลำบาก ถ้าผิดปกติให้พบแพทย์",
    },
  },
  medicine: { icon: Pill, label: "พกยาประจำตัว", sentence: "พกยาประจำตัว เช่น ยาพ่นขยายหลอดลม ติดตัวไว้เสมอ" },
};

const items = computed(() =>
  props.ids
    .filter((id) => ITEMS[id]) // id ที่ frontend ยังไม่รู้จัก (backend ใหม่กว่า) ข้ามไป ไม่ทำให้การ์ดพัง
    .map((id) => {
      const def = ITEMS[id];
      const sentence = def.sentenceByGroup
        ? def.sentenceByGroup[props.riskGroup] || def.sentenceByGroup["ทั่วไป"]
        : def.sentence;
      return { id, kind: def.kind, icon: def.icon, label: t(def.label), sentence: t(sentence) };
    }),
);

const picked = ref(0);
// ระดับฝุ่น/กลุ่มเสี่ยงเปลี่ยน (เช่น poll ค่าตอนนี้ทุก 2 นาที) -> ชุดไอคอนเปลี่ยน -> กลับไปเลือกอันแรก
watch(() => props.ids.join(","), () => { picked.value = 0; });

const detail = computed(() => items.value[Math.min(picked.value, items.value.length - 1)]?.sentence ?? "");
</script>

<template>
  <section class="advice-card" :aria-label="t('คำแนะนำวันนี้')">
    <div class="advice-head">
      <h2>{{ t("คำแนะนำวันนี้") }}</h2>
      <span class="advice-group">{{ t("สำหรับกลุ่ม{group}", { group: t(riskGroup) }) }}</span>
    </div>

    <div v-if="items.length === 0" class="advice-ok">
      <span class="advice-ok-circle" aria-hidden="true"><Check :size="26" :stroke-width="2.5" /></span>
      <span class="advice-ok-text">{{ t("ปกติ — ไม่มีข้อควรระวังพิเศษ") }}</span>
    </div>

    <template v-else>
      <div class="advice-strip" :class="{ 'advice-strip-full': items.length >= 4 }" role="group" :aria-label="t('คำแนะนำ แตะเพื่อดูรายละเอียด')">
        <button
          v-for="(it, i) in items"
          :key="it.id"
          type="button"
          class="advice-item"
          :class="{ active: picked === i }"
          :aria-pressed="picked === i"
          @click="picked = i"
        >
          <span class="advice-circle" aria-hidden="true">
            <component v-if="it.icon" :is="it.icon" :size="26" :stroke-width="2" />
            <svg v-else-if="it.kind === 'mask'" width="26" height="26" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round"><rect x="4" y="9" width="16" height="9" rx="4" /><path d="M4 12c-1.5 0-2 1-2 2M20 12c1.5 0 2 1 2 2M8 13h8" /></svg>
            <svg v-else-if="it.kind === 'n95'" width="26" height="26" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><path d="M4 11c0-1 1-2 2-2 2 0 4-2 6-2s4 2 6 2c1 0 2 1 2 2v2c0 3-4 6-8 6s-8-3-8-6z" /><path d="M9 13h6M10 16h4" /></svg>
            <svg v-else-if="it.kind === 'window'" width="26" height="26" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><rect x="4" y="4" width="16" height="16" rx="2" /><path d="M12 4v16M4 12h16" /></svg>
          </span>
          <span class="advice-label">{{ it.label }}</span>
        </button>
      </div>
      <p class="advice-detail" aria-live="polite">{{ detail }}</p>
    </template>
  </section>
</template>

<style scoped>
.advice-card {
  background: var(--card);
  border: 1px solid var(--line);
  border-radius: 24px;
  padding: 18px;
  display: flex;
  flex-direction: column;
  gap: 14px;
  color: var(--ink);
}

.advice-head {
  display: flex;
  align-items: baseline;
  justify-content: space-between;
  gap: 8px;
  flex-wrap: wrap;
}
.advice-head h2 {
  margin: 0;
  font-family: var(--font-display);
  font-size: 1.05rem;
  font-weight: 600;
}
.advice-group { font-size: 0.78rem; color: var(--muted); }

.advice-strip {
  display: flex;
  gap: 12px;
  justify-content: flex-start;
}
.advice-strip-full { justify-content: space-between; gap: 0; }

.advice-item {
  width: 76px;
  min-height: 44px;
  padding: 0;
  border: 0;
  background: transparent;
  color: var(--muted);
  font-family: inherit;
  cursor: pointer;
  display: flex;
  flex-direction: column;
  align-items: center;
  gap: 6px;
}
.advice-circle {
  width: 56px;
  height: 56px;
  box-sizing: border-box;
  border-radius: 50%;
  display: flex;
  align-items: center;
  justify-content: center;
  background: var(--track);
  color: var(--ink);
  border: 2.5px solid transparent;
  transition: border-color 0.15s ease;
}
.advice-item.active .advice-circle { border-color: var(--ink); }
.advice-label {
  font-size: 0.75rem;
  line-height: 1.3;
  text-align: center;
  font-weight: 500;
}
.advice-item.active .advice-label { color: var(--ink); font-weight: 700; }
.advice-item:focus-visible { outline: 2px solid var(--accent); outline-offset: 2px; border-radius: 12px; }

.advice-detail {
  margin: 0;
  padding: 12px 14px;
  border-radius: 14px;
  background: var(--track);
  color: var(--ink);
  font-size: 0.9rem;
  line-height: 1.5;
}

.advice-ok { display: flex; align-items: center; gap: 12px; }
.advice-ok-circle {
  width: 52px;
  height: 52px;
  flex-shrink: 0;
  border-radius: 50%;
  display: flex;
  align-items: center;
  justify-content: center;
  background: var(--success-bg);
  color: var(--success-ink);
}
.advice-ok-text { font-size: 0.95rem; font-weight: 600; color: var(--success-ink); }

@media (prefers-reduced-motion: reduce) {
  .advice-circle { transition: none; }
}
</style>
