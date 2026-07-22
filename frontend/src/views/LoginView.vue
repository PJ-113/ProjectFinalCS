<script setup>
import { ref } from 'vue'
import { useRoute, useRouter } from 'vue-router'
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
    <form class="auth-card" @submit.prevent="handleSubmit">
      <h1>เข้าสู่ระบบ</h1>

      <label>
        อีเมล
        <input v-model="email" type="email" required autocomplete="email" />
      </label>

      <label>
        รหัสผ่าน
        <input v-model="password" type="password" required autocomplete="current-password" />
      </label>

      <p v-if="errorMessage" class="error-text">{{ errorMessage }}</p>

      <button type="submit" :disabled="submitting">{{ submitting ? 'กำลังเข้าสู่ระบบ...' : 'เข้าสู่ระบบ' }}</button>

      <p class="switch-link">
        ยังไม่มีบัญชี? <router-link to="/register">สมัครสมาชิก</router-link>
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

/* Subtle background glowing orb effect */
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

input {
  padding: 12px 16px;
  border: 1px solid var(--auth-border);
  border-radius: 12px;
  background: rgba(255, 255, 255, 0.5);
  color: var(--auth-text);
  font-size: 1rem;
  transition: all 0.2s ease;
}

input:focus {
  outline: none;
  border-color: var(--auth-accent);
  box-shadow: 0 0 0 3px rgba(59, 130, 246, 0.2);
  background: var(--auth-card-bg);
}

@media (prefers-color-scheme: dark) {
  input { background: rgba(0, 0, 0, 0.2); }
  input:focus { background: rgba(0, 0, 0, 0.4); }
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
