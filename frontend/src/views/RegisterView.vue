<script setup>
import { computed, ref } from 'vue'
import { useRouter } from 'vue-router'
import { register } from '../auth'
import { provinceState } from '../provinces'

const RISK_GROUPS = ['ทั่วไป', 'เด็ก-ผู้สูงอายุ', 'มีโรคประจำตัว']

const router = useRouter()

const email = ref('')
const password = ref('')
const confirmPassword = ref('')
const riskGroup = ref('ทั่วไป')
const age = ref('')
const homeProvince = ref('')
const errorMessage = ref('')
const submitting = ref(false)

const passwordMismatch = computed(
  () => confirmPassword.value.length > 0 && password.value !== confirmPassword.value,
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
    <form class="auth-card" @submit.prevent="handleSubmit">
      <h1>สมัครสมาชิก</h1>

      <label>
        อีเมล
        <input v-model="email" type="email" required autocomplete="email" />
      </label>

      <label>
        รหัสผ่าน (อย่างน้อย 8 ตัวอักษร)
        <input v-model="password" type="password" required minlength="8" autocomplete="new-password" />
      </label>

      <label>
        ยืนยันรหัสผ่าน
        <input v-model="confirmPassword" type="password" required autocomplete="new-password" />
      </label>

      <label>
        กลุ่มความเสี่ยง (ไม่บังคับ)
        <select v-model="riskGroup">
          <option v-for="g in RISK_GROUPS" :key="g" :value="g">{{ g }}</option>
        </select>
      </label>

      <label>
        อายุ (ไม่บังคับ)
        <input v-model="age" type="number" min="1" max="120" />
      </label>

      <label>
        จังหวัดหลัก (ไม่บังคับ — ตั้งทีหลังในหน้าโปรไฟล์ได้ ใช้ให้กระดิ่งแจ้งเตือนเฝ้าจังหวัดนี้)
        <select v-model="homeProvince">
          <option value="">-- ไม่ระบุ --</option>
          <option v-for="p in provinceState.list" :key="p.province_en" :value="p.province_en">
            {{ p.province_th }}{{ p.is_zero_shot ? ' (zero-shot)' : '' }}
          </option>
        </select>
      </label>

      <p v-if="errorMessage" class="error-text">{{ errorMessage }}</p>

      <button type="submit" :disabled="submitting">{{ submitting ? 'กำลังสมัคร...' : 'สมัครสมาชิก' }}</button>

      <p class="switch-link">
        มีบัญชีอยู่แล้ว? <router-link to="/login">เข้าสู่ระบบ</router-link>
      </p>
    </form>
  </div>
</template>

<style scoped>
.auth-page {
  min-height: 100vh;
  display: flex;
  align-items: center;
  justify-content: center;
  background: var(--auth-bg);
  padding: 100px 16px 24px;
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
  max-width: 400px;
  background: var(--auth-card-bg);
  backdrop-filter: blur(24px);
  -webkit-backdrop-filter: blur(24px);
  border: 1px solid var(--auth-border);
  box-shadow: 0 20px 40px rgba(0, 0, 0, 0.08);
  border-radius: 20px;
  padding: 32px;
  display: flex;
  flex-direction: column;
  gap: 16px;
  color: var(--auth-text);
}

h1 {
  font-size: 1.6rem;
  font-weight: 700;
  margin: 0 0 8px;
  text-align: center;
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

button[type='submit'] {
  margin-top: 8px;
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

.error-text {
  color: var(--auth-error);
  font-size: 0.9rem;
  background: rgba(239, 68, 68, 0.1);
  padding: 10px;
  border-radius: 8px;
  margin: 0;
  text-align: center;
}

.switch-link {
  text-align: center;
  font-size: 0.9rem;
  color: var(--auth-muted);
  margin: 8px 0 0;
}

.switch-link a {
  color: var(--auth-accent);
  font-weight: 600;
  text-decoration: none;
}

.switch-link a:hover {
  text-decoration: underline;
}
</style>
