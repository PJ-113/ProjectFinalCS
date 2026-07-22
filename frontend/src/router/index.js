import { createRouter, createWebHistory } from 'vue-router'
import ForecastView from '../views/ForecastView.vue'
import { authState, fetchCurrentUser, isLoggedIn } from '../auth'

const router = createRouter({
  history: createWebHistory(import.meta.env.BASE_URL),
  routes: [
    {
      path: '/',
      name: 'forecast',
      component: ForecastView,
    },
    {
      path: '/compare',
      name: 'compare',
      component: () => import('../views/CompareView.vue'),
    },
    {
      path: '/notifications',
      name: 'notifications',
      component: () => import('../views/NotificationsView.vue'),
    },
    {
      path: '/login',
      name: 'login',
      component: () => import('../views/LoginView.vue'),
      meta: { guestOnly: true },
    },
    {
      path: '/register',
      name: 'register',
      component: () => import('../views/RegisterView.vue'),
      meta: { guestOnly: true },
    },
    {
      path: '/profile',
      name: 'profile',
      component: () => import('../views/ProfileView.vue'),
      meta: { requiresAuth: true },
    },
  ],
})

router.beforeEach(async (to) => {
  // เรียก fetchCurrentUser() เสมอ (ไม่เช็ค status === 'idle' ก่อน) เพราะ AppHeader's onMounted()
  // อาจเริ่มเรียกไปแล้วก่อน guard นี้ทำงาน (mount ของ AppHeader ไม่ถูก router บล็อก เพราะมันอยู่
  // นอก <RouterView/>) — ถ้าเช็ค status ก่อน จะข้ามการ await ไปเจอ authState.user ที่ยังเป็น null
  // (fetch ยังไม่ resolve) แล้วรีไดเรกต์ผิดพลาดไป /login ทั้งที่ล็อกอินอยู่จริง (race condition ที่
  // เจอจริงตอนเทส) fetchCurrentUser() เองมี in-flight promise cache อยู่แล้ว เรียกซ้ำจึงราคาถูก
  if (to.meta.requiresAuth || to.meta.guestOnly) {
    await fetchCurrentUser()
  }
  if (to.meta.requiresAuth && !isLoggedIn()) {
    return { path: '/login', query: { redirect: to.fullPath } }
  }
  if (to.meta.guestOnly && isLoggedIn()) {
    return { path: '/profile' }
  }
  return true
})

export default router
