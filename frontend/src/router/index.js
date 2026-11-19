import { createRouter, createWebHistory } from 'vue-router'
import ForecastView from '../views/ForecastView.vue'
import { authState, fetchCurrentUser, isLoggedIn, isAdmin } from '../auth'

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
      meta: { requiresAdmin: true },
    },
    {
      path: '/map',
      name: 'map',
      component: () => import('../views/MapView.vue'),
    },
    {
      // พรีวิวชั่วคราว: Dashboard ที่ฝังการ์ด "ภาพรวมทั้งภาค" (ของเดิมจาก /map) เข้ามาเป็นส่วนหนึ่งของหน้า
      // ไม่ใช่แท็บแยก — ยังไม่ผูกกับ nav ไหนทั้งนั้น เข้าถึงได้แค่พิมพ์ URL ตรงๆ ให้ดูของจริงก่อนตัดสินใจ
      // ย้ายเข้า "/" ถาวร (ลบ route นี้ทิ้งถ้าตัดสินใจไม่เอา หรือแทนที่ ForecastView ถ้าตัดสินใจเอา)
      path: '/dashboard-map-preview',
      name: 'dashboard-map-preview',
      component: () => import('../views/DashboardMapPreview.vue'),
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
  // เรียก fetchCurrentUser() เสมอสำหรับ route ที่มี requiresAuth/guestOnly/requiresAdmin และ await
  // ผลให้เสร็จก่อนตัดสิน (มี in-flight promise cache อยู่แล้ว เรียกซ้ำจึงราคาถูก)
  if (to.meta.requiresAuth || to.meta.guestOnly || to.meta.requiresAdmin) {
    await fetchCurrentUser()
  }
  if (to.meta.requiresAuth && !isLoggedIn()) {
    return { path: '/login', query: { redirect: to.fullPath } }
  }
  if (to.meta.guestOnly && isLoggedIn()) {
    return { path: '/profile' }
  }
  // requiresAdmin ที่ไม่ใช่ admin เด้งกลับหน้าแรกเงียบๆ ไม่ redirect ไป /login
  if (to.meta.requiresAdmin && !isAdmin()) {
    return { path: '/' }
  }
  return true
})

export default router
