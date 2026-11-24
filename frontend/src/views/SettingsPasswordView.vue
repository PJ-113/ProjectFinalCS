<script setup>
import { ref } from 'vue'
import { Check, ChevronLeft, TriangleAlert } from '@lucide/vue'
import { changePassword } from '../auth'
import { t } from '../i18n'

// แยกออกมาจากหน้าโปรไฟล์เดิม (ProfileView.vue → SettingsProfileView.vue) เป็นหน้าย่อยของ /settings
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
    passwordError.value = t('รหัสผ่านใหม่ไม่ตรงกัน')
    return
  }
  savingPassword.value = true
  try {
    await changePassword({
      current_password: currentPassword.value,
      new_password: newPassword.value,
    })
    passwordMessage.value = t('เปลี่ยนรหัสผ่านเรียบร้อยแล้ว')
    currentPassword.value = ''
    newPassword.value = ''
    confirmNewPassword.value = ''
  } catch (e) {
    passwordError.value = e.message
  } finally {
    savingPassword.value = false
  }
}
</script>

<template>
  <div class="settings-sub-page">
    <div class="wrap">
      <router-link to="/settings" class="back-link">
        <ChevronLeft :size="22" :stroke-width="2" aria-hidden="true" />
        {{ t('ตั้งค่า') }}
      </router-link>
      <h1>{{ t('เปลี่ยนรหัสผ่าน') }}</h1>

      <form class="card form" @submit.prevent="handleChangePassword">
        <label class="field">
          <span class="field-label">{{ t('รหัสผ่านปัจจุบัน') }}</span>
          <input v-model="currentPassword" type="password" required autocomplete="current-password" />
        </label>
        <label class="field">
          <span class="field-label">{{ t('รหัสผ่านใหม่') }}</span>
          <input v-model="newPassword" type="password" required minlength="8" autocomplete="new-password" />
          <span class="hint">{{ t('อย่างน้อย 8 ตัวอักษร') }}</span>
        </label>
        <label class="field">
          <span class="field-label">{{ t('ยืนยันรหัสผ่านใหม่') }}</span>
          <input v-model="confirmNewPassword" type="password" required autocomplete="new-password" />
        </label>

        <p v-if="passwordMessage" class="msg msg-success" role="status"><Check :size="18" :stroke-width="2.2" aria-hidden="true" />{{ passwordMessage }}</p>
        <p v-if="passwordError" class="msg msg-error" role="alert"><TriangleAlert :size="18" :stroke-width="2" aria-hidden="true" />{{ passwordError }}</p>

        <button type="submit" class="btn btn-primary" :disabled="savingPassword">
          {{ savingPassword ? t('กำลังเปลี่ยน...') : t('เปลี่ยนรหัสผ่าน') }}
        </button>
      </form>
    </div>
  </div>
</template>

<style scoped>
/* โทเคนสีมาจาก :root ใน main.css — หน้าตาชุดเดียวกับ SettingsProfileView.vue */
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

.card {
  background: var(--card);
  border: 1px solid var(--line);
  border-radius: 24px;
  padding: 20px;
}
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
.btn:focus-visible,
.back-link:focus-visible {
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
