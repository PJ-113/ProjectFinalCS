<script setup>
import { RouterView } from 'vue-router'
import AppHeader from './components/AppHeader.vue'
import BottomTabBar from './components/BottomTabBar.vue'
import InstallPrompt from './components/InstallPrompt.vue'
</script>

<template>
  <AppHeader />
  <RouterView />
  <BottomTabBar />
  <InstallPrompt />
</template>
