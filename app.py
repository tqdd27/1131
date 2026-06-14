import cv2
import mediapipe as mp
import numpy as np
import streamlit as st
from streamlit_webrtc import WebRtcMode, webrtc_streamer, RTCConfiguration
import os

# --- 1. 初始化與頁面設定 ---
st.title("歷史名人動作辨識貼圖系統")
st.subheader("請選擇您想變身的角色，做出指定動作後點擊拍照！")

role = st.selectbox(
    "選擇變身角色",
    ["愛因斯坦 (張開嘴巴)", "孔子 (手心朝己攤平)", "秦始皇 (比讚)", "釋迦牟尼佛 (閉眼)", "路易十六 (直接拍照)"]
)

# 配置 WebRTC STUN 伺服器
RTC_CONFIGURATION = RTCConfiguration(
    {"iceServers": [{"urls": ["stun:stun.l.google.com:19302", "stun:stun1.l.google.com:19302"]}]}
)

# 初始化 MediaPipe
mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(max_num_faces=1, refine_landmarks=True, min_detection_confidence=0.4)

mp_hands = mp.solutions.hands
hands = mp_hands.Hands(static_image_mode=True, max_num_hands=2, min_detection_confidence=0.4)

# --- 2. 輔助函式：安全貼圖疊加 (支援透明度 PNG) ---
def overlay_transparent(background, overlay, x, y, overlay_size=None):
    if overlay is None:
        return background
    bg_h, bg_w, _ = background.shape
    
    # 縮放貼圖
    if overlay_size is not None:
        overlay = cv2.resize(overlay, overlay_size, interpolation=cv2.INTER_AREA)
    
    o_h, o_w, o_c = overlay.shape
    if o_c < 4:
        # 如果不是透明 PNG，強制轉為帶有 Alpha 通道
        overlay = cv2.cvtColor(overlay, cv2.COLOR_BGR2BGRA)

    # 計算覆蓋邊界，防止邊緣溢出崩潰
    x1, y1 = max(int(x), 0), max(int(y), 0)
    x2, y2 = min(int(x + o_w), bg_w), min(int(y + o_h), bg_h)
    
    o_x1, o_y1 = x1 - int(x), y1 - int(y)
    o_x2, o_y2 = o_x1 + (x2 - x1), o_y1 + (y2 - y1)
    
    if x2 - x1 <= 0 or y2 - y1 <= 0:
        return background

    # 進行 Alpha 混合疊加
    img_crop = background[y1:y2, x1:x2]
    overlay_crop = overlay[o_y1:o_y2, o_x1:o_x2]
    
    alpha_overlay = overlay_crop[:, :, 3] / 255.0
    alpha_bg = 1.0 - alpha_overlay
    
    for c in range(3):
        img_crop[:, :, c] = (alpha_overlay * overlay_crop[:, :, c] + alpha_bg * img_crop[:, :, c])
        
    background[y1:y2, x1:x2] = img_crop
    return background

# --- 3. 讀取 assets 資料夾內貼圖的輔助函式 ---
def load_overlay(filename):
    # 自動加上 assets/ 路徑
    full_path = os.path.join("assets", filename)
    if os.path.exists(full_path):
        return cv2.imread(full_path, cv2.IMREAD_UNCHANGED)
    return None

# --- 4. 啟動視訊鏡頭 ---
ctx = webrtc_streamer(
    key="historical-snapshot",
    mode=WebRtcMode.SENDRECV,
    rtc_configuration=RTC_CONFIGURATION,
    media_stream_constraints={"video": True, "audio": False},
    async_processing=True
)

# --- 5. 拍照與核心辨識邏輯 ---
if ctx.video_receiver:
    if st.button("📸 按下拍照並進行特徵辨識"):
        img_frame = ctx.video_receiver.get_frame()
        
        if img_frame is not None:
            bgr_img = img_frame.to_ndarray(format="bgr24")
            h, w, _ = bgr_img.shape
            rgb_img = cv2.cvtColor(bgr_img, cv2.COLOR_BGR2RGB)
            
            # 進行偵測
            face_res = face_mesh.process(rgb_img)
            hand_res = hands.process(rgb_img)
            
            output_img = bgr_img.copy()
            action_detected = False
            
            # ────────────────────────────────────────────────────────
            # 歷史人物 A：愛因斯坦 (張開嘴巴)
            # ────────────────────────────────────────────────────────
            if "愛因斯坦" in role:
                if face_res.multi_face_landmarks:
                    face_landmarks = face_res.multi_face_landmarks[0].landmark
                    mouth_top = face_landmarks[13].y * h
                    mouth_bottom = face_landmarks[14].y * h
                    face_height = (face_landmarks[152].y - face_landmarks[10].y) * h
                    
                    if (mouth_bottom - mouth_top) > (face_height * 0.03):
                        action_detected = True
                        
                        # 畫面轉黑白濾鏡
                        gray = cv2.cvtColor(output_img, cv2.COLOR_BGR2GRAY)
                        output_img = cv2.merge([gray, gray, gray])
                        
                        f_w = int((face_landmarks[454].x - face_landmarks[234].x) * w)
                        hair_w = int(f_w * 1.8)
                        hair_h = int(face_height * 1.3)
                        
                        tx, ty = int(face_landmarks[10].x * w), int(face_landmarks[10].y * h)
                        mx, my = int(face_landmarks[13].x * w), int(face_landmarks[13].y * h)
                        
                        hair = load_overlay("Einstein_hair.png")
                        tongue = load_overlay("Einstein_tongue.png")
                        
                        output_img = overlay_transparent(output_img, hair, tx - hair_w//2, ty - int(hair_h * 0.7), (hair_w, hair_h))
                        output_img = overlay_transparent(output_img, tongue, mx - int(f_w * 0.3), my, (int(f_w * 0.6), int(f_w * 0.6)))
                
                if not action_detected:
                    st.warning("偵測失敗：請在大聲張開嘴巴的同時按下拍照！")

            # ────────────────────────────────────────────────────────
            # 歷史人物 B：孔子 (手心朝向自己，手攤平)
            # ────────────────────────────────────────────────────────
            elif "孔子" in role:
                if hand_res.multi_hand_landmarks:
                    action_detected = True
                    hand_landmarks = hand_res.multi_hand_landmarks[0].landmark
                    hx, hy = int(hand_landmarks[9].x * w), int(hand_landmarks[9].y * h)
                    
                    if face_res.multi_face_landmarks:
                        fl = face_res.multi_face_landmarks[0].landmark
                        f_w = int((fl[454].x - fl[234].x) * w)
                        
                        tx, ty = int(fl[10].x * w), int(fl[10].y * h)
                        bx, by = int(fl[152].x * w), int(fl[152].y * h)
                        
                        # 已對應正確的 assets 內檔名
                        cap = load_overlay("kongzi_cap.png")
                        beard = load_overlay("kongzi_beard.png")
                        
                        output_img = overlay_transparent(output_img, cap, tx - f_w, ty - int(f_w * 1.4), (f_w * 2, f_w * 1.5))
                        output_img = overlay_transparent(output_img, beard, bx - int(f_w * 0.5), by - int(f_w * 0.2), (f_w, int(f_w * 1.2)))
                    
                    sleeves = load_overlay("kongzi_sleeves.png")
                    output_img = overlay_transparent(output_img, sleeves, hx - 120, hy - 120, (240, 240))
                
                if not action_detected:
                    st.warning("偵測失敗：請將手平舉、手心朝向自己再拍照！")

            # ────────────────────────────────────────────────────────
            # 歷史人物 C：秦始皇 (比讚)
            # ────────────────────────────────────────────────────────
            elif "秦始皇" in role:
                if hand_res.multi_hand_landmarks:
                    hl = hand_res.multi_hand_landmarks[0].landmark
                    if hl[4].y < hl[2].y and hl[8].y > hl[6].y:
                        action_detected = True
                        
                        if face_res.multi_face_landmarks:
                            fl = face_res.multi_face_landmarks[0].landmark
                            f_w = int((fl[454].x - fl[234].x) * w)
                            
                            tx, ty = int(fl[10].x * w), int(fl[10].y * h)
                            bx, by = int(fl[152].x * w), int(fl[152].y * h)
                            
                            cap = load_overlay("qinshihuang_cap.png")
                            bear = load_overlay("polar_bear.png")
                            
                            output_img = overlay_transparent(output_img, cap, tx - f_w, ty - int(f_w * 1.4), (f_w * 2, f_w * 1.5))
                            output_img = overlay_transparent(output_img, bear, bx - int(f_w * 0.7), by, (int(f_w * 1.4), int(f_w * 1.4)))
                
                if not action_detected:
                    st.warning("偵測失敗：請在鏡頭前明確比個「讚 👍」！")

            # ────────────────────────────────────────────────────────
            # 歷史人物 D：釋迦牟尼佛 (閉眼)
            # ────────────────────────────────────────────────────────
            elif "釋迦牟尼佛" in role:
                if face_res.multi_face_landmarks:
                    fl = face_res.multi_face_landmarks[0].landmark
                    eye_dist = abs(fl[159].y - fl[145].y)
                    
                    if eye_dist < 0.022:
                        action_detected = True
                        tx, ty = int(fl[10].x * w), int(fl[10].y * h)
                        f_w = int((fl[454].x - fl[234].x) * w)
                        light_size = int(f_w * 3.0)
                        
                        light = load_overlay("holy_light.png")
                        output_img = overlay_transparent(output_img, light, tx - light_size//2, ty - int(light_size * 0.75), (light_size, light_size))
                
                if not action_detected:
                    st.warning("偵測失敗：請閉上雙眼再按下拍照！")

            # ────────────────────────────────────────────────────────
            # 歷史人物 E：路易十六 (直接拍照)
            # ────────────────────────────────────────────────────────
            elif "路易十六" in role:
                action_detected = True
                if face_res.multi_face_landmarks:
                    fl = face_res.multi_face_landmarks[0].landmark
                    cx, cy = int(fl[1].x * w), int(fl[1].y * h)
                    f_w = int((fl[454].x - fl[234].x) * w)
                    tomato_size = int(f_w * 1.8)
                    
                    # 已修正對應正確的 assets/tomato.png 檔名
                    tomato = load_overlay("tomato.png")
                    output_img = overlay_transparent(output_img, tomato, cx - tomato_size//2, cy - tomato_size//2, (tomato_size, tomato_size))
                else:
                    tomato = load_overlay("tomato.png")
                    output_img = overlay_transparent(output_img, tomato, w//3, h//3, (w//3, w//3))

            # --- 最終成品呈現 ---
            st.subheader("🎉 辨識與貼圖結果")
            st.image(cv2.cvtColor(output_img, cv2.COLOR_BGR2RGB), use_container_width=True)
            
        else:
            st.error("未能從網頁相機擷取到畫面，請重試。")
else:
    st.info("請先點擊上方的『Start』按鈕開啟網頁相機。")
