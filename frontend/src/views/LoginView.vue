<script setup>
import { ref } from 'vue'
import { useRoute, useRouter } from 'vue-router'
import { TriangleAlert } from '@lucide/vue'
import { login } from '../auth'

const route = useRoute()
const router = useRouter()

const email = ref('')
const password = ref('')
const errorMessage = ref('')
const submitting = ref(false)

async function handleSubmit() {
  errorMessage.value = ''
  submitting.value = true
  try {
    await login(email.value, password.value)
    router.push(route.query.redirect || '/')
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
        <h1>เข้าสู่ระบบ</h1>
        <p class="lead">รับการแจ้งเตือนฝุ่นของจังหวัดหลัก และปรับคำแนะนำให้ตรงกับกลุ่มเสี่ยงของคุณ</p>
      </div>

      <form class="card form" @submit.prevent="handleSubmit">
        <label class="field">
          <span class="field-label">อีเมล</span>
          <input v-model="email" type="email" required autocomplete="email" placeholder="name@example.com" />
        </label>

        <label class="field">
          <span class="field-label">รหัสผ่าน</span>
          <input v-model="password" type="password" required autocomplete="current-password" />
        </label>

        <p v-if="errorMessage" class="msg msg-error" role="alert">
          <TriangleAlert :size="18" :stroke-width="2" aria-hidden="true" />{{ errorMessage }}
        </p>

        <button type="submit" class="btn btn-primary" :disabled="submitting">
          {{ submitting ? 'กำลังเข้าสู่ระบบ...' : 'เข้าสู่ระบบ' }}
        </button>
      </form>

      <p class="switch-link">
        ยังไม่มีบัญชี? <router-link to="/register">สมัครสมาชิก</router-link>
      </p>
    </div>
  </div>
</template>

<style scoped>
/* โทเคนสีธีมใหม่มาจาก :root ใน main.css (ใช้ชุดเดียวกับหน้าพยากรณ์/แผนที่/โปรไฟล์) */
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

.intro { display: flex; flex-direction: column; gap: 4px; padding: 8px 4px 0; }
h1 { font-size: 1.9rem; font-weight: 700; line-height: 1.25; margin: 0; }
.lead { margin: 0; font-size: 0.95rem; line-height: 1.55; color: var(--muted); }

.card {
  background: var(--card);
  border: 1px solid var(--line);
  border-radius: 24px;
  padding: 20px;
}
.form { display: flex; flex-direction: column; gap: 16px; }

.field { display: flex; flex-direction: column; gap: 8px; }
.field-label { font-size: 0.9rem; font-weight: 600; color: var(--ink); }

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
.btn:focus-visible {
  outline: 2px solid var(--accent);
  outline-offset: 2px;
}

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
   ถ้าเนื้อหาสูงกว่าจอจะตกลงเป็นชิดบนเองโดยไม่ถูกตัด) มือถือคงชิดบนตามเดิม */
@media (min-width: 721px) {
  .auth-page { display: flex; flex-direction: column; }
  .wrap { margin: auto; width: 100%; }
}
</style>
