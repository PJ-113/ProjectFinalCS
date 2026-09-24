<script setup>
import { computed, onUnmounted, ref, watch } from 'vue'
import { Search, X, Check } from '@lucide/vue'
import { provinceState } from '../provinces'

// bottom sheet บนมือถือ / dialog กลางจอบนเดสก์ท็อป — แทน <select> เดิมที่กดยากบนมือถือและไม่มีช่องค้นหา
// (20 จังหวัด) ไม่ตัดสินใจเองว่าเลือกแล้วทำอะไร: emit 'select' (ค่า province_en หรือ '' = ไม่ระบุ) ให้ผู้เรียก
// จัดการ — หน้าพยากรณ์ส่งเข้า selectProvince() (จังหวัดที่กำลังดู) ส่วนหน้าโปรไฟล์ใช้ตั้ง "จังหวัดหลัก"
// Teleport ไป body จึงใช้โทเคนสี global (:root ใน main.css) ไม่ inherit จากหน้าที่เรียกใช้
const props = defineProps({
  open: { type: Boolean, default: false },
  selected: { type: String, default: '' }, // province_en ที่ติ๊กถูกอยู่ ('' = ไม่ระบุ)
  allowNone: { type: Boolean, default: false }, // แสดงแถว "ไม่ระบุ" ด้านบน (ใช้ในโปรไฟล์)
})
const emit = defineEmits(['close', 'select'])

const query = ref('')

const filtered = computed(() => {
  const q = query.value.trim().toLowerCase()
  if (!q) return provinceState.list
  return provinceState.list.filter(
    (p) => p.province_th.includes(q) || p.province_en.toLowerCase().includes(q),
  )
})

function pick(provinceEn) {
  emit('select', provinceEn)
  emit('close')
}

function onKeydown(e) {
  if (e.key === 'Escape') emit('close')
}

// ผูก Esc เฉพาะตอนเปิด และคืนค่า query ทุกครั้งที่เปิดใหม่ (ไม่ค้างคำค้นเดิม)
watch(
  () => props.open,
  (isOpen) => {
    if (isOpen) {
      query.value = ''
      document.addEventListener('keydown', onKeydown)
    } else {
      document.removeEventListener('keydown', onKeydown)
    }
  },
)
onUnmounted(() => document.removeEventListener('keydown', onKeydown))
</script>

<template>
  <Teleport to="body">
    <div v-if="open" class="sheet-backdrop" @click.self="emit('close')">
      <div class="sheet" role="dialog" aria-modal="true" aria-label="เลือกจังหวัด">
        <div class="sheet-grip" aria-hidden="true"></div>
        <div class="sheet-head">
          <h2>เลือกจังหวัด</h2>
          <button type="button" class="icon-btn" aria-label="ปิด" @click="emit('close')">
            <X :size="22" :stroke-width="2" />
          </button>
        </div>
        <label class="search">
          <Search :size="18" :stroke-width="2" aria-hidden="true" />
          <input v-model="query" type="search" placeholder="ค้นหาจังหวัด" aria-label="ค้นหาจังหวัด" />
        </label>
        <ul class="province-list">
          <li v-if="allowNone && !query.trim()">
            <button type="button" class="province-row" :class="{ active: selected === '' }" @click="pick('')">
              <span class="province-name">ไม่ระบุ</span>
              <Check v-if="selected === ''" :size="20" :stroke-width="2.4" class="check" />
            </button>
          </li>
          <li v-for="p in filtered" :key="p.province_en">
            <button
              type="button"
              class="province-row"
              :class="{ active: p.province_en === selected }"
              @click="pick(p.province_en)"
            >
              <span class="province-name">{{ p.province_th }}</span>
              <span v-if="p.is_zero_shot" class="zs-badge">zero-shot</span>
              <Check v-if="p.province_en === selected" :size="20" :stroke-width="2.4" class="check" />
            </button>
          </li>
          <li v-if="filtered.length === 0" class="empty">ไม่พบจังหวัดที่ค้นหา</li>
        </ul>
      </div>
    </div>
  </Teleport>
</template>

<style scoped>
.sheet-backdrop {
  position: fixed;
  inset: 0;
  z-index: 300;
  background: rgba(15, 18, 22, 0.5);
  display: flex;
  align-items: flex-end;
  justify-content: center;
}

.sheet {
  width: 100%;
  max-width: 480px;
  max-height: 82vh;
  display: flex;
  flex-direction: column;
  background: var(--card);
  color: var(--ink);
  border-radius: 24px 24px 0 0;
  padding: 8px 16px calc(16px + env(safe-area-inset-bottom, 0px));
  box-shadow: 0 -10px 40px rgba(0, 0, 0, 0.25);
}
/* เดสก์ท็อป: dialog กลางจอ แทน sheet ชิดล่าง */
@media (min-width: 721px) {
  .sheet-backdrop { align-items: center; }
  .sheet { border-radius: 24px; max-height: 70vh; padding-top: 16px; }
  .sheet-grip { display: none; }
}

.sheet-grip {
  width: 40px;
  height: 4px;
  border-radius: 2px;
  background: var(--line);
  margin: 4px auto 8px;
}
.sheet-head {
  display: flex;
  align-items: center;
  justify-content: space-between;
  padding: 4px 0 8px;
}
h2 { margin: 0; font-size: 1.15rem; font-weight: 700; }

.icon-btn {
  width: 44px;
  height: 44px;
  border: 0;
  border-radius: 22px;
  background: transparent;
  color: var(--muted);
  display: flex;
  align-items: center;
  justify-content: center;
  cursor: pointer;
}

.search {
  display: flex;
  align-items: center;
  gap: 8px;
  min-height: 48px;
  padding: 0 14px;
  border-radius: 14px;
  background: var(--track);
  color: var(--muted);
}
.search input {
  flex: 1;
  min-width: 0;
  border: 0;
  outline: 0;
  background: transparent;
  color: var(--ink);
  font-family: inherit;
  font-size: 1rem; /* >=16px กัน iOS zoom เข้าช่องพิมพ์ */
}

.province-list {
  list-style: none;
  margin: 12px 0 0;
  padding: 0;
  overflow-y: auto;
  overscroll-behavior: contain;
}
.province-row {
  width: 100%;
  min-height: 52px;
  display: flex;
  align-items: center;
  gap: 10px;
  padding: 0 8px;
  border: 0;
  border-bottom: 1px solid var(--line);
  background: transparent;
  color: var(--ink);
  font-family: inherit;
  font-size: 1rem;
  text-align: left;
  cursor: pointer;
}
.province-row.active { font-weight: 700; color: var(--accent); }
.province-name { flex: 1; }
.check { flex-shrink: 0; }
.zs-badge {
  font-size: 0.72rem;
  font-weight: 600;
  padding: 2px 8px;
  border-radius: 999px;
  background: var(--track);
  color: var(--muted);
}
.empty { padding: 20px 8px; color: var(--muted); }
</style>
