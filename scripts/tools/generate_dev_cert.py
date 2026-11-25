"""
สร้าง self-signed TLS cert แบบถาวร (ไม่ใช่ adhoc ที่ Werkzeug สุ่มใหม่ทุกครั้งที่ server restart) สำหรับทดสอบ
Web Push บนมือถือจริงผ่าน HTTPS — ต้องมี IP ของเครื่อง (SAN) ตรงกับ URL ที่เปิดจริง ไม่งั้นเบราว์เซอร์จะฟ้อง
"ใบรับรองไม่ตรงกับโดเมน" แยกต่างหากจากคำเตือน self-signed อยู่ดี

cert นี้ "ไม่น่าเชื่อถือ" เสมอ (ไม่มี CA ไหนเซ็นให้) เบราว์เซอร์จะเตือนครั้งแรกที่เปิดแน่นอน แต่พอกด "ดำเนินการต่อ"
("Visit this website" / "ขั้นสูง > ไปที่ ... (ไม่ปลอดภัย)") ครั้งเดียว จะไม่เตือนซ้ำอีกตราบใดที่ cert ไฟล์เดิม
ไม่เปลี่ยน (ต่างจาก adhoc ที่สุ่มใหม่ทุกรอบ server restart ทำให้ต้องกดผ่านคำเตือนซ้ำทุกครั้ง)

Usage: python scripts/tools/generate_dev_cert.py [IP อื่นๆที่จะเข้าถึงนอกจาก localhost เช่น 10.62.124.154]
"""
import ipaddress
import os
import sys
from datetime import datetime, timedelta, timezone

from cryptography import x509
from cryptography.hazmat.primitives import hashes, serialization
from cryptography.hazmat.primitives.asymmetric import rsa
from cryptography.x509.oid import NameOID

_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
CERT_PATH = os.path.join(_ROOT, "dev_cert.pem")
KEY_PATH = os.path.join(_ROOT, "dev_key.pem")


def main():
    extra_ip = sys.argv[1] if len(sys.argv) > 1 else None

    key = rsa.generate_private_key(public_exponent=65537, key_size=2048)
    subject = issuer = x509.Name([x509.NameAttribute(NameOID.COMMON_NAME, "localhost")])

    san_entries = [x509.DNSName("localhost"), x509.IPAddress(ipaddress.ip_address("127.0.0.1"))]
    if extra_ip:
        san_entries.append(x509.IPAddress(ipaddress.ip_address(extra_ip)))

    cert = (
        x509.CertificateBuilder()
        .subject_name(subject)
        .issuer_name(issuer)
        .public_key(key.public_key())
        .serial_number(x509.random_serial_number())
        .not_valid_before(datetime.now(timezone.utc))
        .not_valid_after(datetime.now(timezone.utc) + timedelta(days=365))
        .add_extension(x509.SubjectAlternativeName(san_entries), critical=False)
        .sign(key, hashes.SHA256())
    )

    with open(CERT_PATH, "wb") as f:
        f.write(cert.public_bytes(serialization.Encoding.PEM))
    with open(KEY_PATH, "wb") as f:
        f.write(key.private_bytes(
            encoding=serialization.Encoding.PEM,
            format=serialization.PrivateFormat.TraditionalOpenSSL,
            encryption_algorithm=serialization.NoEncryption(),
        ))

    print(f"สร้างเสร็จแล้ว: {CERT_PATH} / {KEY_PATH}")
    print(f"ครอบคลุม: localhost, 127.0.0.1" + (f", {extra_ip}" if extra_ip else ""))
    print("ไฟล์นี้ถาวร ไม่ต้องสร้างใหม่ทุกครั้งที่ restart server — รันสคริปต์นี้ซ้ำเฉพาะตอน IP เครื่องเปลี่ยน")


if __name__ == "__main__":
    main()
