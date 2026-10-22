<script setup>
import { computed, onMounted } from 'vue'
import { BellOff } from '@lucide/vue'
import { authState } from '../auth'
import { provinceState } from '../provinces'
import { notificationHistory, fetchNotificationHistory, summarizeHistoryItem, formatCardTime } from '../notifications'

// province ที่กระดิ่งเฝ้าอยู่จริง: home_province ของ user ถ้ามี ไม่งั้นใช้ provinceState.selected
const watchedProvince = computed(() => authState.user?.home_province || provinceState.selected)

onMounted(() => fetchNotificationHistory(50, watchedProvince.value))

const provinceName = computed(
  () => provinceState.list.find((p) => p.province_en === watchedProvince.value)?.province_th || '',
)

// คำนวณ summary/timeLabel ล่วงหน้าต่อ item
const items = computed(() =>
  notificationHistory.items.map((item) => ({
    ...item,
    timeLabel: formatCardTime(item.recorded_at_th),
    summary: summarizeHistoryItem(item),
  })),
)
</script>

<template>
  <div class="notifications-page">
    <div class="wrap">
      <h1>ประวัติการแจ้งเตือน</h1>
      <p class="subtitle">
        ช่วงเวลาที่ PM2.5 อยู่ในระดับที่ควรระวัง<template v-if="provinceName"> · จังหวัด{{ provinceName }}</template>
      </p>

      <div v-if="notificationHistory.status === 'loading'" class="stack">
        <div v-for="n in 3" :key="n" class="history-card skeleton-card">
          <div class="skeleton skeleton-line" style="width: 40%"></div>
          <div class="skeleton skeleton-line" style="width: 90%"></div>
          <div class="skeleton skeleton-line" style="width: 65%"></div>
        </div>
      </div>
      <div v-else-if="notificationHistory.status === 'error'" class="state-box" role="alert">
        ไม่สามารถโหลดข้อมูลได้ ลองรีเฟรชหน้าอีกครั้ง
      </div>
      <div v-else-if="items.length === 0" class="state-box">
        <BellOff :size="32" :stroke-width="1.8" aria-hidden="true" />
        <p>ยังไม่มีประวัติการแจ้งเตือน</p>
      </div>
      <ul v-else class="history-list">
        <li v-for="item in items" :key="item.recorded_at_th" class="history-card">
          <div class="history-top">
            <span class="history-time">{{ item.timeLabel }}</span>
            <span class="history-pm25">PM2.5 ตอนนั้น {{ item.latest_pm25 }} µg/m³</span>
          </div>
          <div class="history-headline">{{ item.summary.headline }}</div>
          <div v-if="item.summary.ranges.length" class="history-ranges">
            <span v-for="(r, i) in item.summary.ranges" :key="i" class="history-range-chip">{{ r }}</span>
          </div>
          <div v-if="item.summary.note" class="history-note">{{ item.summary.note }}</div>
        </li>
      </ul>
    </div>
  </div>
</template>

<style scoped>
/* โทเคนสีธีมมาจาก :root ใน main.css (ใช้ชุดเดียวกับหน้าพยากรณ์/แผนที่) */
.notifications-page {
  min-height: 100vh;
  padding: var(--app-top-offset) 16px calc(var(--app-bottom-offset) + 24px);
  background: var(--ground);
  color: var(--ink);
}

.wrap { max-width: 640px; margin: 0 auto; }

h1 { font-size: 1.6rem; font-weight: 700; line-height: 1.3; margin: 0 0 4px; }
.subtitle { color: var(--muted); font-size: 0.9rem; line-height: 1.55; margin: 0 0 16px; }

.stack, .history-list { list-style: none; margin: 0; padding: 0; display: flex; flex-direction: column; gap: 12px; }

.history-card {
  background: var(--card);
  border: 1px solid var(--line);
  border-radius: 24px;
  padding: 16px 18px;
}

.history-top { display: flex; justify-content: space-between; align-items: center; gap: 8px; }
.history-time { font-size: 0.85rem; font-weight: 700; color: var(--accent); }
.history-pm25 { font-size: 0.8rem; color: var(--muted); text-align: right; }

.history-headline { font-size: 1rem; font-weight: 600; line-height: 1.5; margin-top: 6px; }

.history-ranges { display: flex; flex-wrap: wrap; gap: 6px; margin-top: 10px; }
.history-range-chip {
  background: var(--info-bg);
  color: var(--info-ink);
  font-size: 0.8rem;
  font-weight: 600;
  padding: 5px 12px;
  border-radius: 999px;
  white-space: nowrap;
}

.history-note { font-size: 0.8rem; color: var(--muted); line-height: 1.5; margin-top: 8px; }

.state-box {
  background: var(--card);
  border: 1px solid var(--line);
  border-radius: 24px;
  padding: 32px 20px;
  text-align: center;
  color: var(--muted);
  display: flex;
  flex-direction: column;
  align-items: center;
  gap: 8px;
}
.state-box p { margin: 0; }

.skeleton-card { display: flex; flex-direction: column; gap: 10px; }
.skeleton {
  background: linear-gradient(90deg, var(--line) 25%, transparent 37%, var(--line) 63%);
  background-size: 400% 100%;
  animation: shimmer 1.4s ease infinite;
  border-radius: 8px;
}
.skeleton-line { height: 14px; }
@keyframes shimmer {
  0% { background-position: 100% 50%; }
  100% { background-position: 0 50%; }
}
@media (prefers-reduced-motion: reduce) {
  .skeleton { animation: none; }
}
</style>
