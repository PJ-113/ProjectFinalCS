"""
ดึงความสูงจากระดับน้ำทะเล (elevation) ของแต่ละจังหวัดเข้า data/province_station_registry.json
— ตอบ feedback อาจารย์เรื่องภูมิประเทศ (ภาคอีสานที่สูง อาจมีผลต่อ pattern ของ PM2.5)

ใช้ Open-Meteo Elevation API (https://open-meteo.com/en/docs/elevation-api) — ฟรี ไม่ต้อง API key
สำหรับการใช้งานที่ไม่ใช่เชิงพาณิชย์ รับได้สูงสุด 100 พิกัดต่อ 1 request (comma-separated) จึงดึงครบ
ทั้ง 20 จังหวัดได้ในการเรียกครั้งเดียว ข้อมูลจาก Copernicus DEM 2021 (ความละเอียด 90m)

เขียน elevation_m กลับเข้า registry JSON เดิม in-place (ไฟล์นี้เป็น reference/config ไม่ใช่
model/data output — province_registry.py เองก็ overwrite ไฟล์นี้เวลารันใหม่อยู่แล้ว)
"""
import json
import os

import requests

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
REGISTRY_PATH = os.path.join(_ROOT, "data", "province_station_registry.json")


def main():
    with open(REGISTRY_PATH, encoding="utf-8") as f:
        registry = json.load(f)

    provinces = registry["provinces"]
    lats = ",".join(str(p["lat"]) for p in provinces)
    lons = ",".join(str(p["lon"]) for p in provinces)

    print(f"เรียก Open-Meteo Elevation API สำหรับ {len(provinces)} จังหวัด...")
    r = requests.get("https://api.open-meteo.com/v1/elevation",
                      params={"latitude": lats, "longitude": lons}, timeout=30)
    r.raise_for_status()
    elevations = r.json()["elevation"]
    assert len(elevations) == len(provinces), \
        f"จำนวน elevation ที่ได้ ({len(elevations)}) ไม่ตรงกับจำนวนจังหวัด ({len(provinces)})"

    for p, elev in zip(provinces, elevations):
        p["elevation_m"] = elev
        print(f"  {p['province_th']} ({p['province_en']}): {elev:.1f} m")

    with open(REGISTRY_PATH, "w", encoding="utf-8") as f:
        json.dump(registry, f, ensure_ascii=False, indent=2)

    print(f"\nบันทึกกลับเข้า: {REGISTRY_PATH}")
    print(f"elevation ต่ำสุด: {min(elevations):.1f}m, สูงสุด: {max(elevations):.1f}m, "
          f"เฉลี่ย: {sum(elevations)/len(elevations):.1f}m")


if __name__ == "__main__":
    main()
