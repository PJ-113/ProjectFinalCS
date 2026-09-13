import { reactive } from 'vue'

// รายชื่อ 20 จังหวัดภาคอีสาน (17 เทรน + 3 zero-shot) มาจาก /api/provinces (อ่านจาก
// data/province_station_registry.json ฝั่ง backend) — จำจังหวัดที่เลือกไว้ผ่าน localStorage
// อย่างเดียว (per-viewer preference) ไม่มี backend เกี่ยวข้อง เหมือน riskScore.js
const STORAGE_KEY = 'pm25_selected_province'
const DEFAULT_PROVINCE = 'khonkaen'

export const provinceState = reactive({
  status: 'idle', // 'idle' | 'loading' | 'ready' | 'error'
  list: [],
  selected: localStorage.getItem(STORAGE_KEY) || DEFAULT_PROVINCE,
})

export function selectProvince(provinceEn) {
  provinceState.selected = provinceEn
  localStorage.setItem(STORAGE_KEY, provinceEn)
}

let _fetchPromise = null

export async function fetchProvinces() {
  if (_fetchPromise) return _fetchPromise
  provinceState.status = 'loading'
  _fetchPromise = (async () => {
    const res = await fetch('/api/provinces')
    if (!res.ok) throw new Error(`Server error (${res.status})`)
    const data = await res.json()
    provinceState.list = data
    // ถ้าจังหวัดที่เคยเลือกไว้ (localStorage) ไม่อยู่ใน registry แล้วด้วยเหตุผลใดก็ตาม กลับไป default
    if (!data.some((p) => p.province_en === provinceState.selected)) {
      selectProvince(DEFAULT_PROVINCE)
    }
    provinceState.status = 'ready'
    return data
  })()
  try {
    return await _fetchPromise
  } catch (err) {
    provinceState.status = 'error'
    throw err
  } finally {
    _fetchPromise = null
  }
}
