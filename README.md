# 271101_workshop_openCV
import libraries ที่เราต้องใช้ cv2(openCV) numpy mediapipe math
เปิดกล้อง เพื่อรับภาพมา process มาเป็น Video
ตั้งตัวแปล function ของ mediapipe เพื่อง่ายต่อการเรียกใช้งาน
และเขียน function หาขนาดของ Vector, หา Vector ,และ  Hand process
สร้าง numpy array เพื่อนำมารับค่าตำแหน่งของแต่ละนิ้ว
while loop เพื่อ ตรวจจับนิ้ว แล้วนำมาประมวลผล
นำค่าของแต่ละนิ้วมาทำ list เพื่อความง่านในการใช้งาน
Hand process หลักการคือ เราจะนำเวคเตอร์ มา cross product เพื่อแยกมือซ้าย,มือขวา
และ นำ Vector จาก จุดจฝ่ามือ ถึงปลายนิ้วของแต่ละนิ้ว มาหาขนาด เพื่อเปรียบเทียบเงื่อนไข และใช้ในการนับ และบอกว่า มีนิ้วไหนชูขึ้น
โดยเราจะคูณ ค่าคงที่และ หารด้วยขนาด ของเวคเตอร์ที่คงที่ (จากจุดที่ไม่ได้ขยับถึงกัน) เพื่อให้เป็น อัตราส่วนคงที่ เพื่อแก้ไข Error เวลาระยะของมือไกลขึ้น

# ⚠️ Condition  ของ โปรแกรม
ไม่สามารถจำแนกหน้ามือหลังมือได้
และไม่สามารถทำท่าพิสดารเกินไปได้
