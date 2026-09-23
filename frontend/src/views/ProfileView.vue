<script setup>
import { ref, watchEffect } from 'vue'
import { useRouter } from 'vue-router'
import { authState, changePassword, logout, updateProfile } from '../auth'
import { provinceState } from '../provinces'

const RISK_GROUPS = ['ทั่วไป', 'เด็ก-ผู้สูงอายุ', 'มีโรคประจำตัว']
const FAR_THRESHOLD_KM = 200 // ไกลกว่านี้ = อาจอยู่นอกภาคอีสาน เตือนแต่ไม่บล็อกการบันทึก

const router = useRouter()

const email = ref('')
const riskGroup = ref('ทั่วไป')
const age = ref('')
const homeProvince = ref('')
const profileMessage = ref('')
const profileError = ref('')
const savingProfile = ref(false)
const geoLocating = ref(false)
const geoMessage = ref('')

watchEffect(() => {
  if (authState.user) {
    email.value = authState.user.email
    riskGroup.value = authState.user.risk_group
    age.value = authState.user.age ?? ''
    homeProvince.value = authState.user.home_province || ''
  }
})

async function handleSaveProfile() {
  profileMessage.value = ''
  profileError.value = ''
  savingProfile.value = true
  try {
    await updateProfile({
      email: email.value,
      risk_group: riskGroup.value,
      age: age.value ? Number(age.value) : null,
      home_province: homeProvince.value || null,
    })
    profileMessage.value = 'บันทึกโปรไฟล์เรียบร้อยแล้ว'
  } catch (e) {
    profileError.value = e.message
  } finally {
    savingProfile.value = false
  }
}

// ระยะทางวงกลมใหญ่ (กม.) ระหว่างสองพิกัด — ใช้หาจังหวัดที่ใกล้ตำแหน่งปัจจุบันที่สุด ไม่มี dependency
// เพิ่ม เป็นแค่สูตรคณิตศาสตร์ทั่วไป
function haversineKm(lat1, lon1, lat2, lon2) {
  const R = 6371
  const dLat = (lat2 - lat1) * Math.PI / 180
  const dLon = (lon2 - lon1) * Math.PI / 180
  const a = Math.sin(dLat / 2) ** 2
    + Math.cos(lat1 * Math.PI / 180) * Math.cos(lat2 * Math.PI / 180) * Math.sin(dLon / 2) ** 2
  return R * 2 * Math.atan2(Math.sqrt(a), Math.sqrt(1 - a))
}

// ปุ่ม GPS แค่ "ช่วยเดา" เติมค่าใน dropdown ให้เฉยๆ ไม่ auto-save — ผู้ใช้ต้องกดบันทึกโปรไฟล์เองเสมอ
// (ตัดสินใจไว้ตั้งแต่ตอนวางแผน กัน GPS หยาบ/ผิดพลาดไปบันทึกทับโดยไม่ได้ตรวจสอบก่อน)
function handleUseLocation() {
  geoMessage.value = ''
  if (!navigator.geolocation) {
    geoMessage.value = 'เบราว์เซอร์นี้ไม่รองรับการระบุตำแหน่ง — เลือกจังหวัดเองจากรายการได้เลย'
    return
  }
  geoLocating.value = true
  navigator.geolocation.getCurrentPosition(
    (pos) => {
      geoLocating.value = false
      const { latitude, longitude } = pos.coords
      let nearest = null
      for (const p of provinceState.list) {
        const d = haversineKm(latitude, longitude, p.lat, p.lon)
        if (!nearest || d < nearest.d) nearest = { p, d }
      }
      if (!nearest) {
        geoMessage.value = 'ยังโหลดรายชื่อจังหวัดไม่เสร็จ ลองใหม่อีกครั้ง'
        return
      }
      homeProvince.value = nearest.p.province_en
      geoMessage.value = nearest.d > FAR_THRESHOLD_KM
        ? `ตรวจพบตำแหน่งห่างจาก${nearest.p.province_th}ประมาณ ${Math.round(nearest.d)} กม. (อาจอยู่นอกภาคอีสาน) ตรวจสอบให้แน่ใจก่อนกดบันทึก`
        : `เลือก${nearest.p.province_th}ให้แล้ว (ใกล้ตำแหน่งปัจจุบันที่สุด) กด "บันทึกโปรไฟล์" เพื่อยืนยัน`
    },
    (err) => {
      geoLocating.value = false
      geoMessage.value = err.code === err.PERMISSION_DENIED
        ? 'ไม่ได้รับอนุญาตให้เข้าถึงตำแหน่ง — เลือกจังหวัดเองจากรายการได้เลย'
        : 'ไม่สามารถระบุตำแหน่งได้ตอนนี้ ลองใหม่อีกครั้งหรือเลือกจังหวัดเอง'
    },
    { timeout: 10000 },
  )
}

const currentPassword = ref('')
const newPassword = ref('')
const confirmNewPassword = ref('')
const passwordMessage = ref('')
const passwordError = ref('')
const savingPassword = ref(false)

async function handleChangePassword() {
  passwordMessage.value = ''
  passwordError.value = ''
  if (newPassword.value !== confirmNewPassword.value) {
    passwordError.value = 'รหัสผ่านใหม่ไม่ตรงกัน'
    return
  }
  savingPassword.value = true
  try {
    await changePassword({
      current_password: currentPassword.value,
      new_password: newPassword.value,
    })
    passwordMessage.value = 'เปลี่ยนรหัสผ่านเรียบร้อยแล้ว'
    currentPassword.value = ''
    newPassword.value = ''
    confirmNewPassword.value = ''
  } catch (e) {
    passwordError.value = e.message
  } finally {
    savingPassword.value = false
  }
}

async function handleLogout() {
  await logout()
  router.push('/')
}
</script>

<template>
  <div class="auth-page">
    <div class="auth-card">
      <h1>โปรไฟล์ของฉัน</h1>

      <form class="section" @submit.prevent="handleSaveProfile">
        <label>
          อีเมล
          <input v-model="email" type="email" required />
        </label>
        <label>
          กลุ่มความเสี่ยง
          <select v-model="riskGroup">
            <option v-for="g in RISK_GROUPS" :key="g" :value="g">{{ g }}</option>
          </select>
        </label>
        <label>
          อายุ
          <input v-model="age" type="number" min="1" max="120" />
        </label>
        <label>
          จังหวัดหลัก (ไม่บังคับ — ใช้ให้กระดิ่งแจ้งเตือนเฝ้าจังหวัดนี้แยกจากจังหวัดที่กำลังดูพยากรณ์อยู่)
          <div class="province-row">
            <select v-model="homeProvince">
              <option value="">-- ไม่ระบุ --</option>
              <option v-for="p in provinceState.list" :key="p.province_en" :value="p.province_en">
                {{ p.province_th }}{{ p.is_zero_shot ? ' (zero-shot)' : '' }}
              </option>
            </select>
            <button type="button" class="geo-btn" @click="handleUseLocation" :disabled="geoLocating">
              {{ geoLocating ? 'กำลังค้นหา...' : '📍 ใช้ตำแหน่งปัจจุบัน' }}
            </button>
          </div>
          <span v-if="geoMessage" class="geo-message">{{ geoMessage }}</span>
        </label>
        <p v-if="profileMessage" class="success-text">{{ profileMessage }}</p>
        <p v-if="profileError" class="error-text">{{ profileError }}</p>
        <button type="submit" :disabled="savingProfile">{{ savingProfile ? 'กำลังบันทึก...' : 'บันทึกโปรไฟล์' }}</button>
      </form>

      <hr />

      <form class="section" @submit.prevent="handleChangePassword">
        <h2>เปลี่ยนรหัสผ่าน</h2>
        <label>
          รหัสผ่านปัจจุบัน
          <input v-model="currentPassword" type="password" required autocomplete="current-password" />
        </label>
        <label>
          รหัสผ่านใหม่ (อย่างน้อย 8 ตัวอักษร)
          <input v-model="newPassword" type="password" required minlength="8" autocomplete="new-password" />
        </label>
        <label>
          ยืนยันรหัสผ่านใหม่
          <input v-model="confirmNewPassword" type="password" required autocomplete="new-password" />
        </label>
        <p v-if="passwordMessage" class="success-text">{{ passwordMessage }}</p>
        <p v-if="passwordError" class="error-text">{{ passwordError }}</p>
        <button type="submit" :disabled="savingPassword">{{ savingPassword ? 'กำลังเปลี่ยน...' : 'เปลี่ยนรหัสผ่าน' }}</button>
      </form>

      <hr />

      <button type="button" class="logout-btn" @click="handleLogout">ออกจากระบบ</button>
    </div>
  </div>
</template>

<style scoped>
.auth-page {
  min-height: 100vh;
  display: flex;
  align-items: flex-start;
  justify-content: center;
  background: var(--auth-bg);
  padding: 100px 16px 80px;
  position: relative;
  overflow: hidden;
}

.auth-page::before {
  content: "";
  position: absolute;
  top: -10%;
  left: -10%;
  width: 50vw;
  height: 50vw;
  background: radial-gradient(circle, rgba(59,130,246,0.15) 0%, rgba(255,255,255,0) 70%);
  border-radius: 50%;
  z-index: 0;
}

.auth-card {
  position: relative;
  z-index: 1;
  width: 100%;
  max-width: 440px;
  background: var(--auth-card-bg);
  backdrop-filter: blur(24px);
  -webkit-backdrop-filter: blur(24px);
  border: 1px solid var(--auth-border);
  box-shadow: 0 20px 40px rgba(0, 0, 0, 0.08);
  border-radius: 20px;
  padding: 32px;
  display: flex;
  flex-direction: column;
  gap: 24px;
  color: var(--auth-text);
  margin-top: 40px;
}

h1 {
  font-size: 1.6rem;
  font-weight: 700;
  margin: 0;
  text-align: center;
}

h2 {
  font-size: 1.2rem;
  font-weight: 600;
  margin: 0 0 4px;
}

.section {
  display: flex;
  flex-direction: column;
  gap: 16px;
}

hr {
  border: none;
  border-top: 1px solid var(--auth-border);
  margin: 0;
}

label {
  display: flex;
  flex-direction: column;
  gap: 8px;
  font-size: 0.95rem;
  font-weight: 500;
  color: var(--auth-muted);
}

input,
select {
  padding: 12px 16px;
  border: 1px solid var(--auth-border);
  border-radius: 12px;
  background: rgba(255, 255, 255, 0.5);
  color: var(--auth-text);
  font-size: 1rem;
  transition: all 0.2s ease;
}

input:focus, select:focus {
  outline: none;
  border-color: var(--auth-accent);
  box-shadow: 0 0 0 3px rgba(59, 130, 246, 0.2);
  background: var(--auth-card-bg);
}

@media (prefers-color-scheme: dark) {
  input, select { background: rgba(0, 0, 0, 0.2); }
  input:focus, select:focus { background: rgba(0, 0, 0, 0.4); }
}

.province-row {
  display: flex;
  gap: 8px;
  flex-wrap: wrap;
}
.province-row select { flex: 1 1 160px; }
.geo-btn {
  flex: 0 0 auto;
  padding: 0 14px;
  border: 1px solid var(--auth-border);
  border-radius: 12px;
  background: rgba(255, 255, 255, 0.5);
  color: var(--auth-text);
  font-size: 0.85rem;
  font-weight: 600;
  cursor: pointer;
  white-space: nowrap;
}
.geo-btn:hover:not(:disabled) { border-color: var(--auth-accent); color: var(--auth-accent); }
.geo-btn:disabled { opacity: 0.6; cursor: default; }
@media (prefers-color-scheme: dark) {
  .geo-btn { background: rgba(0, 0, 0, 0.2); }
}
.geo-message {
  font-size: 0.78rem;
  color: var(--auth-muted);
  font-weight: 400;
}

button[type='submit'] {
  margin-top: 4px;
  padding: 12px;
  border: none;
  border-radius: 12px;
  background: linear-gradient(135deg, var(--auth-accent), var(--auth-accent-hover));
  color: #fff;
  font-size: 1rem;
  font-weight: 600;
  cursor: pointer;
  transition: all 0.2s ease;
  box-shadow: 0 4px 12px rgba(59, 130, 246, 0.3);
}

button[type='submit']:hover {
  transform: translateY(-2px);
  box-shadow: 0 6px 16px rgba(59, 130, 246, 0.5);
}

button[type='submit']:disabled {
  opacity: 0.7;
  cursor: default;
  transform: none;
  box-shadow: none;
}

.logout-btn {
  padding: 12px;
  border: 1.5px solid var(--auth-error);
  border-radius: 12px;
  background: transparent;
  color: var(--auth-error);
  font-size: 1rem;
  font-weight: 600;
  cursor: pointer;
  transition: all 0.2s ease;
}

.logout-btn:hover {
  background: rgba(239, 68, 68, 0.05);
}

.error-text {
  color: var(--auth-error);
  font-size: 0.9rem;
  background: rgba(239, 68, 68, 0.1);
  padding: 10px;
  border-radius: 8px;
  margin: 0;
}

.success-text {
  color: #16a34a;
  font-size: 0.9rem;
  background: rgba(22, 163, 74, 0.1);
  padding: 10px;
  border-radius: 8px;
  margin: 0;
}
</style>
