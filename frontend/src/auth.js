import { reactive } from 'vue'

export const authState = reactive({ user: null, status: 'idle' }) // 'idle' | 'loading' | 'ready'

async function apiFetch(url, options) {
  const res = await fetch(url, {
    headers: { 'Content-Type': 'application/json' },
    ...options,
  })
  const data = await res.json().catch(() => ({}))
  if (!res.ok) throw new Error(data.error || 'เกิดข้อผิดพลาด')
  return data
}

let _fetchPromise = null

export function fetchCurrentUser() {
  if (_fetchPromise) return _fetchPromise
  authState.status = 'loading'
  _fetchPromise = apiFetch('/api/auth/me')
    .then((u) => { authState.user = u })
    .catch(() => { authState.user = null })
    .finally(() => { authState.status = 'ready' })
  return _fetchPromise
}

export async function login(email, password) {
  const user = await apiFetch('/api/auth/login', {
    method: 'POST',
    body: JSON.stringify({ email, password }),
  })
  authState.user = user
  authState.status = 'ready'
  return user
}

export async function register(payload) {
  const user = await apiFetch('/api/auth/register', {
    method: 'POST',
    body: JSON.stringify(payload),
  })
  authState.user = user
  authState.status = 'ready'
  return user
}

export async function logout() {
  await apiFetch('/api/auth/logout', { method: 'POST' })
  authState.user = null
}

export async function updateProfile(payload) {
  const user = await apiFetch('/api/auth/profile', {
    method: 'PUT',
    body: JSON.stringify(payload),
  })
  authState.user = user
  return user
}

export async function changePassword(payload) {
  return apiFetch('/api/auth/change-password', {
    method: 'POST',
    body: JSON.stringify(payload),
  })
}

export function isLoggedIn() {
  return authState.user !== null
}
