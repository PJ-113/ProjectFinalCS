import './assets/main.css'
import './assets/auth-theme.css'

import { createApp } from 'vue'
import App from './App.vue'
import router from './router'
import { initTheme } from './theme'

initTheme()

const app = createApp(App)

app.use(router)

app.mount('#app')
