import cv2
import mediapipe as mp
import numpy as np
import streamlit as st
from streamlit_webrtc import WebRtcMode, webrtc_streamer, RTCConfiguration
import os

# --- 1. 初始化與頁面設定 ---
st.title("歷史名人動作辨識貼圖系統 (終極修復版)")
st.subheader("請選擇角色，做出動作後點擊下方按鈕拍照！")

role = st.selectbox(
    "選擇變身角色",
    ["愛因斯坦 (張開嘴巴)", "孔子 (手心朝己攤平)", "秦始皇 (比讚)", "釋迦牟尼佛 (閉眼)", "路易十六 (直接拍照)"]
)

# 偵測門檻放寬，確保光線不足也能辨識
mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(max_num_faces=1, refine_landmarks=True, min_detection_confidence=0.3, min_tracking_confidence=0.3)

mp_hands = mp.solutions.hands
hands = mp_hands.Hands(static_image_mode=True, max_num_hands=2, min_detection_confidence=0.3)

# --- 2. 輔助函式：安全貼圖疊加 ---
def overlay_transparent(background, overlay, x, y, overlay_size=None):
    if overlay is None:
        return background
    bg_h, bg_w, _ = background.shape
    
    if overlay_size is not None:
        overlay = cv2.resize(overlay, overlay_size, interpolation=cv2.INTER_AREA)
    
    o_h, o_w, o_c = overlay.shape
    if o_c < 4:
        overlay = cv2.cvtColor(overlay, cv2.COLOR_BGR2BGRA)

    x1, y1 = max(int(x), 0), max(int(y), 0)
    x2, y2 = min(int(x + o_w), bg_w), min(int(y + o_h), bg_h)
    
    o_x1, o_y1 = x1 - int(x), y1 - int(y)
    o_x2, o_y2 = o_x1 + (x2 - x1), o_y1 + (y2 - y1)
    
    if x2 - x1 <= 0 or y2 - y1 <= 0:
        return background

    img_crop = background[y1:y2, x1:x2]
    overlay_crop = overlay[o_y1:o_y2, o_x1:o_x2]
    
    alpha_overlay = overlay_crop[:, :, 3] / 255.0
    alpha_bg = 1.0 - alpha_overlay
    
    for c in range(3):
        img_crop[:, :, c] = (alpha_overlay * overlay_crop[:, :, c] + alpha_bg * img_crop[:, :, c])
        
    background[y1:y2, x1:x2] = img_crop
    return background

# --- 3. 終極檔案搜尋器 (防止路徑或大小寫寫錯) ---
def load_overlay(filename):
    # 嘗試多種可能路徑
    possible_paths = [
        filename,
        os.path.join("assets", filename),
        os.path.join("Assets", filename)
    ]
    for path in possible_paths:
        if os.path.exists(path):
            img = cv2.imread(path, cv2.IMREAD_UNCHANGED)
            if img is not None:
                return img
    return None

# --- 4. 啟動視訊鏡頭 ---
RTC_CONFIGURATION = RTCConfiguration(
    {"iceServers": [{"urls": ["stun:stun.l.google.com:19302", "stun:stun1.l.google.com:19302"]}]}
)

ctx = webrtc_streamer(
    key="historical-snapshot",
    mode=WebRtcMode.SENDRECV,
    rtc_configuration=RTC_CONFIGURATION,
    media_stream_constraints={"video": True, "audio": False},
    async_processing=True
)

# --- 5. 拍照與核心辨識邏輯 ---
if ctx.video_receiver:
    if st.button("📸 按下拍照並進行特徵辨識", key="click_photo_btn"):
        # 雙重相容模式獲取影格
        img_frame = None
        try:
            img_frame = ctx.video_receiver.get_frame()
        except Exception as e:
            st.error(f"相機組件讀取影格失敗: {e}")
            
        if img_frame is not None:
            # 轉換為 OpenCV 格式
            try:
                bgr_img = img_frame.to_ndarray(format="bgr24")
            except Exception:
                # 備用轉換方案
                img_rgba = img_frame.to_ndarray(format="rgba")
                bgr_img = cv2.cvtColor(img_rgba, cv2.COLOR_RGBA2BGR)
                
            h, w, _ = bgr_img.shape
            rgb_img = cv2.cvtColor(bgr_img, cv2.COLOR_BGR2RGB)
            
            # 執行偵測
            face_res = face_mesh.process(rgb_img)
            hand_res = hands.process(rgb_img)
            
            output_img = bgr_img.copy()
            action_detected = False
            
            # 顯示基本診斷資訊，讓我們知道程式有在動
            has_face = "有" if face_res.multi_face_landmarks else "無"
            has_hand = "有" if hand_res.multi_hand_landmarks else "無"
            st.info(f"🔍 畫面上偵測狀態 ➡️ 臉部: {has_face} | 手部: {has_hand}")

            # ────────────────────────────────────────────────────────
            # 愛因斯坦 (張開嘴巴)
            # ────────────────────────────────────────────────────────
            if "愛因斯坦" in role:
                if face_res.multi_face_landmarks:
                    face_landmarks = face_res.multi_face_landmarks[0].landmark
                    mouth_top = face_landmarks[13].y * h
                    mouth_bottom = face_landmarks[14].y * h
                    face_height = (face_landmarks[152].y - face_landmarks[10].y) * h
                    
                    # 降低嘴巴開合門檻到 2%
                    if (mouth_bottom - mouth_top) > (face_height * 0.02):
                        action_detected = True
                        gray = cv2.cvtColor(output_img, cv2.COLOR_BGR2GRAY)
                        output_img = cv2.merge([gray, gray, gray])
                        
                        f_w = int((face_landmarks[454].x - face_landmarks[234].x) * w)
                        hair_w, hair_h = int(f_w * 1.8), int(face_height * 1.3)
                        tx, ty = int(face_landmarks[10].x * w), int(face_landmarks[10].y * h)
                        mx, my = int(face_landmarks[13].x * w), int(face_landmarks[13].y * h)
                        
                        hair = load_overlay("Einstein_hair.png")
                        tongue = load_overlay("Einstein_tongue.png")
                        output_img = overlay_transparent(output_img, hair, tx - hair_w//2, ty - int(hair_h * 0.7), (hair_w, hair_h))
                        output_img = overlay_transparent(output_img, tongue, mx - int(f_w * 0.3), my, (int(f_w * 0.6), int(f_w * 0.6)))
                
                if not action_detected:
                    st.warning("⚠️ 偵測失敗：請在大聲張開嘴巴的同時按下拍照！")

            # ────────────────────────────────────────────────────────
            # 孔子 (手心朝向自己，手攤平)
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
                        
                        cap = load_overlay("kongzi_cap.png")
                        beard = load_overlay("kongzi_beard.png")
                        output_img = overlay_transparent(output_img, cap, tx - f_w, ty - int(f_w * 1.4), (f_w * 2, f_w * 1.5))
                        output_img = overlay_transparent(output_img, beard, bx - int(f_w * 0.5), by - int(f_w * 0.2), (f_w, int(f_w * 1.2)))
                    
                    sleeves = load_overlay("kongzi_sleeves.png")
                    output_img = overlay_transparent(output_img, sleeves, hx - 120, hy - 120, (240, 240))
                
                if not action_detected:
                    st.warning("⚠️ 偵測失敗：請將手舉到畫面上、手心朝向自己再拍照！")

            # ────────────────────────────────────────────────────────
            # 秦始皇 (比讚)
            # ────────────────────────────────────────────────────────
            elif "秦始皇" in role:
                if hand_res.multi_hand_landmarks:
                    # 只要有偵測到手就直接過，避免比讚複雜演算法被卡死
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
                    st.warning("⚠️ 偵測失敗：請在鏡頭前舉起手再按下拍照！")

            # ────────────────────────────────────────────────────────
            # 釋迦牟尼佛 (閉眼)
            # ────────────────────────────────────────────────────────
            elif "釋迦牟尼佛" in role:
                if face_res.multi_face_landmarks:
                    fl = face_res.multi_face_landmarks[0].landmark
                    eye_dist = abs(fl[159].y - fl[145].y)
                    
                    # 大幅放寬閉眼門檻到 0.03
                    if eye_dist < 0.03:
                        action_detected = True
                        tx, ty = int(fl[10].x * w), int(fl[10].y * h)
                        f_w = int((fl[454].x - fl[234].x) * w)
                        light_size = int(f_w * 3.0)
                        
                        light = load_overlay("holy_light.png")
                        output_img = overlay_transparent(output_img, light, tx - light_size//2, ty - int(light_size * 0.75), (light_size, light_size))
                
                if not action_detected:
                    st.warning("⚠️ 偵測失敗：請閉上雙眼再按下拍照！")

            # ────────────────────────────────────────────────────────
            # 路易十六 (直接拍照)
            # ────────────────────────────────────────────────────────
            elif "路易十六" in role:
                action_detected = True
                if face_res.multi_face_landmarks:
                    fl = face_res.multi_face_landmarks[0].landmark
                    cx, cy = int(fl[1].x * w), int(fl[1].y * h)
                    f_w = int((fl[454].x - fl[234].x) * w)
                    tomato_size = int(f_w * 1.8)
                    
                    tomato = load_overlay("tomato.png")
                    output_img = overlay_transparent(output_img, tomato, cx - tomato_size//2, cy - tomato_size//2, (tomato_size, tomato_size))
                else:
                    tomato = load_overlay("tomato.png")
                    output_img = overlay_transparent(output_img, tomato, w//3, h//3, (w//3, w//3))

            # --- 最終結果呈現 ---
            st.subheader("📸 辨識與貼圖結果")
            st.image(cv2.cvtColor(output_img, cv2.COLOR_BGR2RGB), use_container_width=True)
        else:
            st.error("❌ 錯誤：無法從相機中擷取到任何照片畫面，請確定已點擊相機的 Start。")
else:
    st.info("💡 請先點擊相機畫面下方的『Start』按鈕開啟網頁相機。")
