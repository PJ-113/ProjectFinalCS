<script setup>
import { computed, ref, watchEffect } from 'vue'
import { useRouter } from 'vue-router'
import { LocateFixed, LogOut, Check, ChevronDown, TriangleAlert } from '@lucide/vue'
import { authState, changePassword, logout, updateProfile } from '../auth'
import { provinceState } from '../provinces'
import ProvincePickerSheet from '../components/ProvincePickerSheet.vue'

const RISK_GROUPS = ['ทั่วไป', 'เด็ก-ผู้สูงอายุ', 'มีโรคประจำตัว']
// ป้ายบนปุ่ม segmented แบบย่อให้พอดีจอ 390px — ค่าที่เก็บ/ส่ง backend ยังเป็น RISK_GROUPS เดิมทุกตัว
const RISK_GROUP_LABEL = { 'มีโรคประจำตัว': 'โรคประจำตัว' }
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
const pickerOpen = ref(false)

watchEffect(() => {
  if (authState.user) {
    email.value = authState.user.email
    riskGroup.value = authState.user.risk_group
    age.value = authState.user.age ?? ''
    homeProvince.value = authState.user.home_province || ''
  }
})

const homeProvinceName = computed(
  () => provinceState.list.find((p) => p.province_en === homeProvince.value)?.province_th || '',
)
// การ์ดหัวสรุปค่าที่บันทึกไว้จริง (จาก authState) ไม่ใช่ค่าที่กำลังแก้ในฟอร์ม
const savedProvinceName = computed(
  () => provinceState.list.find((p) => p.province_en === authState.user?.home_province)?.province_th || '',
)
const avatarLetter = computed(() => (authState.user?.email || '?').charAt(0).toUpperCase())
const headerSubtitle = computed(() => {
  const parts = []
  parts.push(savedProvinceName.value ? `จังหวัดหลัก: ${savedProvinceName.value}` : 'ยังไม่ได้ตั้งจังหวัดหลัก')
  if (authState.user?.risk_group) parts.push(`กลุ่ม${authState.user.risk_group}`)
  return parts.join(' · ')
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

// ระยะทางวงกลมใหญ่ (กม.) ระหว่างสองพิกัด — ใช้หาจังหวัดที่ใกล้ตำแหน่งปัจจุบันที่สุด
function haversineKm(lat1, lon1, lat2, lon2) {
  const R = 6371
  const dLat = (lat2 - lat1) * Math.PI / 180
  const dLon = (lon2 - lon1) * Math.PI / 180
  const a = Math.sin(dLat / 2) ** 2
    + Math.cos(lat1 * Math.PI / 180) * Math.cos(lat2 * Math.PI / 180) * Math.sin(dLon / 2) ** 2
  return R * 2 * Math.atan2(Math.sqrt(a), Math.sqrt(1 - a))
}

// ปุ่ม GPS แค่ "ช่วยเดา" เติมค่าในช่องจังหวัดให้เฉยๆ ไม่ auto-save — ผู้ใช้ต้องกดบันทึกโปรไฟล์เองเสมอ
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
  <div class="profile-page">
    <div class="wrap">
      <h1>โปรไฟล์ของฉัน</h1>

      <section class="card user-card">
        <div class="avatar" aria-hidden="true">{{ avatarLetter }}</div>
        <div class="user-text">
          <div class="user-email">{{ authState.user?.email }}</div>
          <div class="user-sub">{{ headerSubtitle }}</div>
        </div>
      </section>

      <form class="card form" @submit.prevent="handleSaveProfile">
        <h2>ข้อมูลของคุณ</h2>

        <label class="field">
          <span class="field-label">อีเมล</span>
          <input v-model="email" type="email" required autocomplete="email" />
        </label>

        <div class="field" role="radiogroup" aria-label="กลุ่มความเสี่ยง">
          <span class="field-label">กลุ่มความเสี่ยง</span>
          <div class="segmented">
            <button
              v-for="g in RISK_GROUPS"
              :key="g"
              type="button"
              role="radio"
              class="segmented-btn"
              :class="{ active: riskGroup === g }"
              :aria-checked="riskGroup === g"
              @click="riskGroup = g"
            >{{ RISK_GROUP_LABEL[g] || g }}</button>
          </div>
        </div>

        <label class="field">
          <span class="field-label">อายุ</span>
          <input v-model="age" type="number" min="1" max="120" inputmode="numeric" />
        </label>

        <div class="field">
          <span class="field-label" id="home-province-label">จังหวัดหลัก</span>
          <button
            type="button"
            class="field-btn"
            aria-haspopup="dialog"
            aria-labelledby="home-province-label"
            @click="pickerOpen = true"
          >
            <span :class="{ placeholder: !homeProvinceName }">{{ homeProvinceName || 'ไม่ระบุ' }}</span>
            <ChevronDown :size="20" :stroke-width="2" aria-hidden="true" />
          </button>
          <span class="hint">กระดิ่งแจ้งเตือนจะเฝ้าจังหวัดนี้ แยกจากจังหวัดที่กำลังดูพยากรณ์</span>
          <button type="button" class="pill-btn" :disabled="geoLocating" @click="handleUseLocation">
            <LocateFixed :size="18" :stroke-width="2" aria-hidden="true" />
            {{ geoLocating ? 'กำลังค้นหา...' : 'ใช้ตำแหน่งปัจจุบัน' }}
          </button>
          <span v-if="geoMessage" class="hint">{{ geoMessage }}</span>
        </div>

        <p v-if="profileMessage" class="msg msg-success" role="status"><Check :size="18" :stroke-width="2.2" aria-hidden="true" />{{ profileMessage }}</p>
        <p v-if="profileError" class="msg msg-error" role="alert"><TriangleAlert :size="18" :stroke-width="2" aria-hidden="true" />{{ profileError }}</p>

        <button type="submit" class="btn btn-primary" :disabled="savingProfile">
          {{ savingProfile ? 'กำลังบันทึก...' : 'บันทึกโปรไฟล์' }}
        </button>
      </form>

      <form class="card form" @submit.prevent="handleChangePassword">
        <h2>เปลี่ยนรหัสผ่าน</h2>

        <label class="field">
          <span class="field-label">รหัสผ่านปัจจุบัน</span>
          <input v-model="currentPassword" type="password" required autocomplete="current-password" />
        </label>
        <label class="field">
          <span class="field-label">รหัสผ่านใหม่</span>
          <input v-model="newPassword" type="password" required minlength="8" autocomplete="new-password" />
          <span class="hint">อย่างน้อย 8 ตัวอักษร</span>
        </label>
        <label class="field">
          <span class="field-label">ยืนยันรหัสผ่านใหม่</span>
          <input v-model="confirmNewPassword" type="password" required autocomplete="new-password" />
        </label>

        <p v-if="passwordMessage" class="msg msg-success" role="status"><Check :size="18" :stroke-width="2.2" aria-hidden="true" />{{ passwordMessage }}</p>
        <p v-if="passwordError" class="msg msg-error" role="alert"><TriangleAlert :size="18" :stroke-width="2" aria-hidden="true" />{{ passwordError }}</p>

        <button type="submit" class="btn btn-outline" :disabled="savingPassword">
          {{ savingPassword ? 'กำลังเปลี่ยน...' : 'เปลี่ยนรหัสผ่าน' }}
        </button>
      </form>

      <button type="button" class="btn btn-danger" @click="handleLogout">
        <LogOut :size="20" :stroke-width="2" aria-hidden="true" />
        ออกจากระบบ
      </button>
    </div>

    <ProvincePickerSheet
      :open="pickerOpen"
      :selected="homeProvince"
      allow-none
      @select="homeProvince = $event"
      @close="pickerOpen = false"
    />
  </div>
</template>

<style scoped>
/* โทเคนสีมาจาก :root ใน main.css (ใช้ชุดเดียวกับหน้าพยากรณ์/แผนที่) */
.profile-page {
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
  gap: 16px;
}

h1 { font-size: 1.6rem; font-weight: 700; line-height: 1.3; margin: 0; }
h2 { font-size: 1.05rem; font-weight: 600; margin: 0; }

.card {
  background: var(--card);
  border: 1px solid var(--line);
  border-radius: 24px;
  padding: 20px;
}

.user-card { display: flex; align-items: center; gap: 14px; }
.avatar {
  width: 52px;
  height: 52px;
  flex-shrink: 0;
  border-radius: 50%;
  background: var(--ink);
  color: var(--card);
  display: flex;
  align-items: center;
  justify-content: center;
  font-family: var(--font-display);
  font-weight: 600;
  font-size: 1.35rem;
}
.user-text { min-width: 0; }
.user-email { font-size: 1.05rem; font-weight: 600; overflow-wrap: anywhere; }
.user-sub { font-size: 0.8rem; color: var(--muted); margin-top: 2px; line-height: 1.5; }

.form { display: flex; flex-direction: column; gap: 16px; }

.field { display: flex; flex-direction: column; gap: 8px; }
.field-label { font-size: 0.9rem; font-weight: 600; color: var(--ink); }
.hint { font-size: 0.8rem; color: var(--muted); line-height: 1.5; }

input {
  min-height: 52px;
  padding: 0 16px;
  border: 1.5px solid var(--line);
  border-radius: 14px;
  background: var(--card);
  color: var(--ink);
  font-family: inherit;
  font-size: 1rem; /* >=16px กัน iOS zoom เข้าช่องพิมพ์ */
}
input:focus-visible,
.field-btn:focus-visible,
.segmented-btn:focus-visible,
.pill-btn:focus-visible,
.btn:focus-visible {
  outline: 2px solid var(--accent);
  outline-offset: 2px;
}

.field-btn {
  min-height: 52px;
  padding: 0 16px;
  display: flex;
  align-items: center;
  justify-content: space-between;
  gap: 8px;
  border: 1.5px solid var(--line);
  border-radius: 14px;
  background: var(--card);
  color: var(--ink);
  font-family: inherit;
  font-size: 1rem;
  text-align: left;
  cursor: pointer;
}
.field-btn .placeholder { color: var(--muted); }

.segmented {
  display: flex;
  gap: 4px;
  padding: 4px;
  border-radius: 16px;
  background: var(--track);
}
.segmented-btn {
  flex: 1;
  min-height: 44px;
  padding: 0 4px;
  border: 0;
  border-radius: 12px;
  background: transparent;
  color: var(--muted);
  font-family: inherit;
  font-size: 0.9rem;
  font-weight: 600;
  cursor: pointer;
}
.segmented-btn.active { background: var(--ink); color: var(--card); }

.pill-btn {
  align-self: flex-start;
  min-height: 44px;
  display: inline-flex;
  align-items: center;
  gap: 8px;
  padding: 0 16px;
  border: 1.5px solid var(--line);
  border-radius: 22px;
  background: transparent;
  color: var(--ink);
  font-family: inherit;
  font-size: 0.9rem;
  font-weight: 600;
  cursor: pointer;
}
.pill-btn:disabled { opacity: 0.6; cursor: default; }

.msg {
  display: flex;
  align-items: flex-start;
  gap: 8px;
  margin: 0;
  padding: 12px 14px;
  border-radius: 14px;
  font-size: 0.9rem;
  line-height: 1.5;
}
.msg > svg { flex-shrink: 0; margin-top: 2px; }
.msg-success { background: var(--success-bg); color: var(--success-ink); }
.msg-error { background: var(--warn-bg); color: var(--danger); border: 1px solid var(--danger-line); }

.btn {
  min-height: 52px;
  width: 100%;
  display: flex;
  align-items: center;
  justify-content: center;
  gap: 8px;
  border-radius: 26px;
  border: 1.5px solid transparent;
  font-family: inherit;
  font-size: 1rem;
  font-weight: 600;
  cursor: pointer;
}
.btn:disabled { opacity: 0.6; cursor: default; }
.btn-primary { background: var(--ink); color: var(--card); border-color: var(--ink); }
.btn-outline { background: transparent; color: var(--ink); border-color: var(--line); }
.btn-danger { background: transparent; color: var(--danger); border-color: var(--danger-line); }
</style>
