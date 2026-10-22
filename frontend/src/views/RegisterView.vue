<script setup>
import { computed, ref } from 'vue'
import { useRouter } from 'vue-router'
import { ChevronDown, TriangleAlert } from '@lucide/vue'
import { register } from '../auth'
import { provinceState } from '../provinces'
import ProvincePickerSheet from '../components/ProvincePickerSheet.vue'

const RISK_GROUPS = ['ทั่วไป', 'เด็ก-ผู้สูงอายุ', 'มีโรคประจำตัว']
// ป้ายบนปุ่ม segmented แบบย่อให้พอดีจอ 390px — ค่าที่ส่ง backend ยังเป็น RISK_GROUPS เดิมทุกตัว
const RISK_GROUP_LABEL = { 'มีโรคประจำตัว': 'โรคประจำตัว' }

const router = useRouter()

const email = ref('')
const password = ref('')
const confirmPassword = ref('')
const riskGroup = ref('ทั่วไป')
const age = ref('')
const homeProvince = ref('')
const errorMessage = ref('')
const submitting = ref(false)
const pickerOpen = ref(false)

const passwordMismatch = computed(
  () => confirmPassword.value.length > 0 && password.value !== confirmPassword.value,
)
const homeProvinceName = computed(
  () => provinceState.list.find((p) => p.province_en === homeProvince.value)?.province_th || '',
)

async function handleSubmit() {
  errorMessage.value = ''
  if (passwordMismatch.value) {
    errorMessage.value = 'รหัสผ่านไม่ตรงกัน'
    return
  }
  submitting.value = true
  try {
    await register({
      email: email.value,
      password: password.value,
      risk_group: riskGroup.value,
      age: age.value ? Number(age.value) : null,
      home_province: homeProvince.value || null,
    })
    router.push('/')
  } catch (e) {
    errorMessage.value = e.message
  } finally {
    submitting.value = false
  }
}
</script>

<template>
  <div class="auth-page">
    <div class="wrap">
      <div class="intro">
        <h1>สมัครสมาชิก</h1>
        <p class="lead">ใช้เวลาไม่ถึงนาที ข้อมูลที่ไม่บังคับตั้งทีหลังได้ในหน้าโปรไฟล์</p>
      </div>

      <form class="card form" @submit.prevent="handleSubmit">
        <label class="field">
          <span class="field-label">อีเมล</span>
          <input v-model="email" type="email" required autocomplete="email" placeholder="name@example.com" />
        </label>

        <label class="field">
          <span class="field-label">รหัสผ่าน</span>
          <input v-model="password" type="password" required minlength="8" autocomplete="new-password" />
          <span class="hint">อย่างน้อย 8 ตัวอักษร</span>
        </label>

        <label class="field">
          <span class="field-label">ยืนยันรหัสผ่าน</span>
          <input
            v-model="confirmPassword"
            type="password"
            required
            autocomplete="new-password"
            :class="{ invalid: passwordMismatch }"
            :aria-invalid="passwordMismatch"
          />
          <span v-if="passwordMismatch" class="hint hint-error" role="alert">รหัสผ่านไม่ตรงกัน</span>
        </label>

        <div class="field" role="radiogroup" aria-label="กลุ่มความเสี่ยง">
          <span class="field-label">กลุ่มความเสี่ยง <span class="optional">(ไม่บังคับ)</span></span>
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
          <span class="field-label">อายุ <span class="optional">(ไม่บังคับ)</span></span>
          <input v-model="age" type="number" min="1" max="120" inputmode="numeric" placeholder="เช่น 35" />
        </label>

        <div class="field">
          <span class="field-label" id="reg-home-province-label">
            จังหวัดหลัก <span class="optional">(ไม่บังคับ)</span>
          </span>
          <button
            type="button"
            class="field-btn"
            aria-haspopup="dialog"
            aria-labelledby="reg-home-province-label"
            @click="pickerOpen = true"
          >
            <span :class="{ placeholder: !homeProvinceName }">{{ homeProvinceName || 'เลือกจังหวัด' }}</span>
            <ChevronDown :size="20" :stroke-width="2" aria-hidden="true" />
          </button>
          <!--<span class="hint">ใช้ให้กระดิ่งแจ้งเตือนเฝ้าจังหวัดนี้ ตั้งทีหลังได้</span>-->
        </div>

        <p v-if="errorMessage" class="msg msg-error" role="alert">
          <TriangleAlert :size="18" :stroke-width="2" aria-hidden="true" />{{ errorMessage }}
        </p>

        <button type="submit" class="btn btn-primary" :disabled="submitting">
          {{ submitting ? 'กำลังสมัคร...' : 'สมัครสมาชิก' }}
        </button>
      </form>

      <p class="switch-link">
        มีบัญชีอยู่แล้ว? <router-link to="/login">เข้าสู่ระบบ</router-link>
      </p>
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
/* โทเคนสีธีมมาจาก :root ใน main.css (ใช้ชุดเดียวกับหน้าพยากรณ์/แผนที่/โปรไฟล์) */
.auth-page {
  min-height: 100vh;
  padding: var(--app-top-offset) 16px calc(var(--app-bottom-offset) + 24px);
  background: var(--ground);
  color: var(--ink);
}

.wrap {
  max-width: 440px;
  margin: 0 auto;
  display: flex;
  flex-direction: column;
  gap: 16px;
}

.intro { padding: 8px 4px 0; }
h1 { font-size: 1.9rem; font-weight: 700; line-height: 1.25; margin: 0; }
.lead { margin: 4px 0 0; font-size: 0.95rem; line-height: 1.55; color: var(--muted); }

.card {
  background: var(--card);
  border: 1px solid var(--line);
  border-radius: 24px;
  padding: 20px;
}
.form { display: flex; flex-direction: column; gap: 16px; }

.field { display: flex; flex-direction: column; gap: 8px; }
.field-label { font-size: 0.9rem; font-weight: 600; color: var(--ink); }
.optional { font-weight: 400; color: var(--muted); }
.hint { font-size: 0.8rem; color: var(--muted); line-height: 1.5; }
.hint-error { color: var(--danger); }

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
input.invalid { border-color: var(--danger); }
input:focus-visible,
.field-btn:focus-visible,
.segmented-btn:focus-visible,
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
.msg-error { background: var(--warn-bg); color: var(--danger); border: 1px solid var(--danger-line); }

.btn {
  min-height: 52px;
  width: 100%;
  display: flex;
  align-items: center;
  justify-content: center;
  border-radius: 26px;
  border: 1.5px solid var(--ink);
  background: var(--ink);
  color: var(--card);
  font-family: inherit;
  font-size: 1rem;
  font-weight: 600;
  cursor: pointer;
}
.btn:disabled { opacity: 0.6; cursor: default; }

.switch-link { text-align: center; margin: 0; padding: 4px 0; font-size: 0.95rem; color: var(--muted); }
.switch-link a { color: var(--accent); font-weight: 600; text-decoration: none; }
.switch-link a:hover { text-decoration: underline; }

/* เดสก์ท็อป: จัดกล่องฟอร์มให้อยู่กึ่งกลางหน้าจอทั้งแนวตั้งและแนวนอน (margin: auto ใน flex column —
   ถ้าเนื้อหาสูงกว่าจอจะตกลงเป็นชิดบนเองโดยไม่ถูกตัด) มือถือชิดบน */
@media (min-width: 721px) {
  .auth-page { display: flex; flex-direction: column; }
  .wrap { margin: auto; width: 100%; }
}
</style>
