<script setup>
import { computed, onMounted } from 'vue'
import { authState } from '../auth'
import { provinceState } from '../provinces'
import { notificationHistory, fetchNotificationHistory, summarizeHistoryItem, formatCardTime } from '../notifications'

// province: ให้ตรงกับจังหวัดหลักที่กระดิ่งเฝ้าอยู่จริง (เดิมหน้านี้ไม่ส่ง province เลยจึงใช้
// provinceState.selected เสมอ ไม่ตรงกับ dropdown ใน AppHeader.vue ที่ใช้ home_province เป็นหลัก)
onMounted(() => fetchNotificationHistory(50, authState.user?.home_province || provinceState.selected))

// คำนวณ summary/timeLabel ครั้งเดียวต่อ item แทนที่จะเรียกฟังก์ชันซ้ำหลายครั้งในเทมเพลต
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
      <p class="subtitle">รายการช่วงเวลาที่ PM2.5 อยู่ในระดับที่ควรระวัง ย้อนหลังล่าสุด</p>

      <div v-if="notificationHistory.status === 'loading'" class="state-msg">กำลังโหลด...</div>
      <div v-else-if="notificationHistory.status === 'error'" class="state-msg">
        ไม่สามารถโหลดข้อมูลได้ ลองรีเฟรชหน้าอีกครั้ง
      </div>
      <div v-else-if="items.length === 0" class="state-msg">
        ยังไม่มีประวัติการแจ้งเตือน
      </div>
      <ul v-else class="history-list">
        <li v-for="item in items" :key="item.recorded_at_th" class="history-card glass-card">
          <div class="history-time">{{ item.timeLabel }}</div>
          <div class="history-headline">{{ item.summary.headline }}</div>
          <div v-if="item.summary.ranges.length" class="history-ranges">
            <span v-for="(r, i) in item.summary.ranges" :key="i" class="history-range-chip">{{ r }}</span>
          </div>
          <div v-if="item.summary.note" class="history-note">{{ item.summary.note }}</div>
          <div class="history-pm25">PM2.5 ล่าสุดตอนนั้น: {{ item.latest_pm25 }} µg/m³</div>
        </li>
      </ul>
    </div>
  </div>
</template>

<style scoped>
/* พื้นหลังเดียวกับหน้าตระกูล auth (Login/Register/Profile) — var(--auth-bg) + glowing orb */
.notifications-page {
  min-height: 100vh;
  padding: 100px 24px 40px;
  color: var(--auth-text);
  background: var(--auth-bg);
  position: relative;
  overflow: hidden;
}

.notifications-page::before {
  content: "";
  position: absolute;
  top: -10%;
  left: -10%;
  width: 50vw;
  height: 50vw;
  background: radial-gradient(circle, rgba(59, 130, 246, 0.15) 0%, rgba(255, 255, 255, 0) 70%);
  border-radius: 50%;
  z-index: 0;
}

.wrap {
  position: relative;
  z-index: 1;
  max-width: 700px;
  margin: 0 auto;
}

h1 {
  font-size: 1.6rem;
  font-weight: 800;
  margin: 0 0 4px;
}

.subtitle {
  color: var(--auth-muted);
  margin: 0 0 24px;
}

.state-msg {
  color: var(--auth-muted);
  padding: 24px 0;
}

.history-list {
  list-style: none;
  margin: 0;
  padding: 0;
  display: flex;
  flex-direction: column;
  gap: 12px;
}

.history-card {
  padding: 16px 18px;
  border-radius: 14px;
}

.history-time {
  font-size: 0.8rem;
  font-weight: 700;
  color: var(--auth-accent);
  margin-bottom: 6px;
}

.history-headline {
  line-height: 1.5;
  margin-bottom: 8px;
}

.history-ranges {
  display: flex;
  flex-wrap: wrap;
  gap: 6px;
  margin-bottom: 8px;
}

.history-range-chip {
  background: rgba(59, 130, 246, 0.12);
  color: var(--auth-accent);
  font-size: 0.8rem;
  font-weight: 600;
  padding: 4px 10px;
  border-radius: 999px;
  white-space: nowrap;
}

.history-note {
  font-size: 0.85rem;
  color: var(--auth-muted);
  margin-bottom: 6px;
}

.history-pm25 {
  font-size: 0.85rem;
  color: var(--auth-muted);
}
</style>
