<script setup>
import { onMounted } from 'vue'
import { notificationHistory, fetchNotificationHistory, summarizeHistoryItem } from '../notifications'

onMounted(() => fetchNotificationHistory(50))
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
      <div v-else-if="notificationHistory.items.length === 0" class="state-msg">
        ยังไม่มีประวัติการแจ้งเตือน
      </div>
      <ul v-else class="history-list">
        <li v-for="item in notificationHistory.items" :key="item.recorded_at_th" class="history-card glass-card">
          <div class="history-time">{{ item.recorded_at_th }}</div>
          <div class="history-text">{{ summarizeHistoryItem(item) }}</div>
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

.history-text {
  line-height: 1.5;
  margin-bottom: 6px;
}

.history-pm25 {
  font-size: 0.85rem;
  color: var(--auth-muted);
}
</style>
