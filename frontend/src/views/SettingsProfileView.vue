<script setup>
import { computed, ref, watchEffect } from 'vue'
import { LocateFixed, Check, ChevronDown, ChevronLeft, TriangleAlert } from '@lucide/vue'
import { authState, updateProfile } from '../auth'
import { provinceState } from '../provinces'
import ProvincePickerSheet from '../components/ProvincePickerSheet.vue'
import { t, provinceName } from '../i18n'

const RISK_GROUPS = ['ทั่วไป', 'เด็ก-ผู้สูงอายุ', 'มีโรคประจำตัว']

const RISK_GROUP_LABEL = { 'มีโรคประจำตัว': 'โรคประจำตัว' }
const FAR_THRESHOLD_KM = 200 

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
  () => provinceName(provinceState.list.find((p) => p.province_en === homeProvince.value)),
)

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
    profileMessage.value = t('บันทึกข้อมูลเรียบร้อยแล้ว')
  } catch (e) {
    profileError.value = e.message
  } finally {
    savingProfile.value = false
  }
}

function haversineKm(lat1, lon1, lat2, lon2) {
  const R = 6371
  const dLat = (lat2 - lat1) * Math.PI / 180
  const dLon = (lon2 - lon1) * Math.PI / 180
  const a = Math.sin(dLat / 2) ** 2
    + Math.cos(lat1 * Math.PI / 180) * Math.cos(lat2 * Math.PI / 180) * Math.sin(dLon / 2) ** 2
  return R * 2 * Math.atan2(Math.sqrt(a), Math.sqrt(1 - a))
}

function handleUseLocation() {
  geoMessage.value = ''
  if (!navigator.geolocation) {
    geoMessage.value = t('เบราว์เซอร์นี้ไม่รองรับการระบุตำแหน่ง — เลือกจังหวัดเองจากรายการได้เลย')
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
        geoMessage.value = t('ยังโหลดรายชื่อจังหวัดไม่เสร็จ ลองใหม่อีกครั้ง')
        return
      }
      homeProvince.value = nearest.p.province_en
      geoMessage.value = nearest.d > FAR_THRESHOLD_KM
        ? t('ตรวจพบตำแหน่งห่างจาก{name}ประมาณ {km} กม. (อาจอยู่นอกภาคอีสาน) ตรวจสอบให้แน่ใจก่อนกดบันทึก', { name: provinceName(nearest.p), km: Math.round(nearest.d) })
        : t('เลือก{name}ให้แล้ว (ใกล้ตำแหน่งปัจจุบันที่สุด) กด "บันทึก" เพื่อยืนยัน', { name: provinceName(nearest.p) })
    },
    (err) => {
      geoLocating.value = false
      geoMessage.value = err.code === err.PERMISSION_DENIED
        ? t('ไม่ได้รับอนุญาตให้เข้าถึงตำแหน่ง — เลือกจังหวัดเองจากรายการได้เลย')
        : t('ไม่สามารถระบุตำแหน่งได้ตอนนี้ ลองใหม่อีกครั้งหรือเลือกจังหวัดเอง')
    },
    { timeout: 10000 },
  )
}

</script>

<template>
  <div class="settings-sub-page">
    <div class="wrap">
      <router-link to="/settings" class="back-link">
        <ChevronLeft :size="22" :stroke-width="2" aria-hidden="true" />
        {{ t('ตั้งค่า') }}
      </router-link>
      <h1>{{ t('ข้อมูลส่วนตัว') }}</h1>

      <form class="card form" @submit.prevent="handleSaveProfile">
        <label class="field">
          <span class="field-label">{{ t('อีเมล') }}</span>
          <input v-model="email" type="email" required autocomplete="email" />
        </label>

        <div class="field" role="radiogroup" :aria-label="t('กลุ่มความเสี่ยง')">
          <span class="field-label">{{ t('กลุ่มความเสี่ยง') }}</span>
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
            >{{ t(RISK_GROUP_LABEL[g] || g) }}</button>
          </div>
        </div>

        <label class="field">
          <span class="field-label">{{ t('อายุ') }}</span>
          <input v-model="age" type="number" min="1" max="120" inputmode="numeric" />
        </label>

        <div class="field">
          <span class="field-label" id="home-province-label">{{ t('จังหวัดหลัก') }}</span>
          <button
            type="button"
            class="field-btn"
            aria-haspopup="dialog"
            aria-labelledby="home-province-label"
            @click="pickerOpen = true"
          >
            <span :class="{ placeholder: !homeProvinceName }">{{ homeProvinceName || t('ไม่ระบุ') }}</span>
            <ChevronDown :size="20" :stroke-width="2" aria-hidden="true" />
          </button>
          <span class="hint">{{ t('กระดิ่งแจ้งเตือนจะเฝ้าจังหวัดนี้ แยกจากจังหวัดที่กำลังดูพยากรณ์') }}</span>
          <button type="button" class="pill-btn" :disabled="geoLocating" @click="handleUseLocation">
            <LocateFixed :size="18" :stroke-width="2" aria-hidden="true" />
            {{ geoLocating ? t('กำลังค้นหา...') : t('ใช้ตำแหน่งปัจจุบัน') }}
          </button>
          <span v-if="geoMessage" class="hint">{{ geoMessage }}</span>
        </div>

        <p v-if="profileMessage" class="msg msg-success" role="status"><Check :size="18" :stroke-width="2.2" aria-hidden="true" />{{ profileMessage }}</p>
        <p v-if="profileError" class="msg msg-error" role="alert"><TriangleAlert :size="18" :stroke-width="2" aria-hidden="true" />{{ profileError }}</p>

        <button type="submit" class="btn btn-primary" :disabled="savingProfile">
          {{ savingProfile ? t('กำลังบันทึก...') : t('บันทึก') }}
        </button>
      </form>

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
.settings-sub-page {
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

.card {
  background: var(--card);
  border: 1px solid var(--line);
  border-radius: 24px;
  padding: 20px;
}

.back-link {
  align-self: flex-start;
  min-height: 44px;
  display: inline-flex;
  align-items: center;
  gap: 2px;
  margin: -8px 0 -8px -6px;
  padding-right: 8px;
  color: var(--muted);
  text-decoration: none;
  font-size: 0.95rem;
  font-weight: 600;
}
.back-link:hover { color: var(--ink); }

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
.btn:focus-visible,
.back-link:focus-visible {
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
</style>
