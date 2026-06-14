import cv2
import mediapipe as mp
import numpy as np
import streamlit as st
from streamlit_webrtc import WebRtcMode, webrtc_streamer

# --- 1. 初始化 Mediapipe 解決方案 ---
mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(
    max_num_faces=1, 
    refine_landmarks=True, 
    min_detection_confidence=0.4,
    min_tracking_confidence=0.4
)

mp_hands = mp.solutions.hands
hands = mp_hands.Hands(
    max_num_hands=1, 
    min_detection_confidence=0.4, 
    min_tracking_confidence=0.4
)

# --- 2. 輔助函式：貼圖覆蓋 (支援透明度 PNG) ---
def overlay_image(background, overlay, x, y, w, h):
    if overlay is None or w <= 0 or h <= 0:
        return background
    try:
        overlay_resized = cv2.resize(overlay, (w, h), interpolation=cv2.INTER_AREA)
        h_bg, w_bg, _ = background.shape
        
        # 計算邊界與裁切，防止溢出崩潰
        x1, y1 = max(0, x), max(0, y)
        x2, y2 = min(w_bg, x + w), min(h_bg, y + h)
        
        x1_o, y1_o = x1 - x, y1 - y
        x2_o, y2_o = x1_o + (x2 - x1), y1_o + (y2 - y1)
        
        if x2 - x1 <= 0 or y2 - y1 <= 0:
            return background
            
        crop_background = background[y1:y2, x1:x2]
        crop_overlay = overlay_resized[y1_o:y2_o, x1_o:x2_o]
        
        # 若無 Alpha 通道，強制補滿
        if crop_overlay.shape[2] < 4:
            crop_overlay = cv2.cvtColor(crop_overlay, cv2.COLOR_BGR2BGRA)
            
        alpha = crop_overlay[:, :, 3] / 255.0
        alpha = np.expand_dims(alpha, axis=2)
        
        composite = crop_overlay[:, :, :3] * alpha + crop_background * (1 - alpha)
        background[y1:y2, x1:x2] = composite.astype(np.uint8)
    except Exception:
        pass
    return background

# --- 3. 讀取貼圖資產 ---
@st.cache_resource
def load_assets():
    return {
        "hair": cv2.imread("assets/Einstein_hair.png", cv2.IMREAD_UNCHANGED),
        "tongue": cv2.imread("assets/Einstein_tongue.png", cv2.IMREAD_UNCHANGED),
        "confucius_hat": cv2.imread("assets/kongzi_cap.png", cv2.IMREAD_UNCHANGED),
        "beard": cv2.imread("assets/kongzi_beard.png", cv2.IMREAD_UNCHANGED),
        "sleeve": cv2.imread("assets/kongzi_sleeves.png", cv2.IMREAD_UNCHANGED),
        "emperor_hat": cv2.imread("assets/qinshihuang_cap.png", cv2.IMREAD_UNCHANGED),
        "bear": cv2.imread("assets/polar_bear.png", cv2.IMREAD_UNCHANGED),
        "holy_light": cv2.imread("assets/holy_light.png", cv2.IMREAD_UNCHANGED),
        "tomato": cv2.imread("assets/tomato.png", cv2.IMREAD_UNCHANGED),
    }

assets = load_assets()

# --- 4. Streamlit 介面設定 ---
st.title("歷史名人動作偵測相機 📸")
st.text("選擇你想挑戰的角色，做出指定動作即可觸發特效與拍照！")

character = st.selectbox(
    "選擇角色：", ["愛因斯坦", "孔子", "秦始皇", "釋迦牟尼佛", "路易十六"]
)

hints = {
    "愛因斯坦": "😮 張開嘴巴：畫面轉黑白，長出瘋狂頭髮與大舌頭！",
    "孔子": "✋ 手心朝向自己且攤平：戴上儒帽、長出鬍鬚、衣袖遮手！",
    "秦始皇": "👍 比個讚 (大拇指朝上)：戴上皇冠，召喚北極熊！",
    "釋迦牟尼佛": "😑 閉上雙眼：佛光普照，頭頂出現聖光！",
    "路易十六": "😐 不做任何動作：直接將頭部變成大番茄！",
}
st.info(hints[character])

# --- 5. 影像處理核心邏輯 (WebRTC Callback) ---
def video_frame_callback(frame):
    img = frame.to_ndarray(format="bgr24")
    h, w, _ = img.shape
    
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    face_results = face_mesh.process(img_rgb)
    hand_results = hands.process(img_rgb)
    
    # 1. 愛因斯坦：張嘴
    if character == "愛因斯坦" and face_results.multi_face_landmarks:
        landmarks = face_results.multi_face_landmarks[0].landmark
        top_lip = np.array([landmarks[13].x * w, landmarks[13].y * h])
        bottom_lip = np.array([landmarks[14].x * w, landmarks[14].y * h])
        left_mouth = np.array([landmarks[78].x * w, landmarks[78].y * h])
        right_mouth = np.array([landmarks[308].x * w, landmarks[308].y * h])
        
        mouth_height = np.linalg.norm(top_lip - bottom_lip)
        mouth_width = np.linalg.norm(left_mouth - right_mouth)
        
        # 放寬張嘴比例判定 (0.35 降低至 0.22 更易觸發)
        if mouth_height / max(1, mouth_width) > 0.22:
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            img = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
            
            hx, hy = int(landmarks[10].x * w), int(landmarks[10].y * h)
            face_width = int(np.linalg.norm(
                np.array([landmarks[234].x * w, landmarks[234].y * h]) - 
                np.array([landmarks[454].x * w, landmarks[454].y * h])
            ))
            
            img = overlay_image(img, assets["hair"], hx - face_width, hy - face_width, face_width * 2, face_width * 2)
            
            tx, ty = int(landmarks[14].x * w), int(landmarks[14].y * h)
            img = overlay_image(img, assets["tongue"], tx - int(mouth_width / 2), ty, int(mouth_width), int(mouth_width))

    # 2. 孔子：手心朝內且攤平
    elif character == "孔子" and hand_results.multi_hand_landmarks:
        hand_landmarks = hand_results.multi_hand_landmarks[0]
        if face_results.multi_face_landmarks:
            landmarks = face_results.multi_face_landmarks[0].landmark
            fx, fy = int(landmarks[10].x * w), int(landmarks[10].y * h)
            face_width = int(abs(landmarks[234].x - landmarks[454].x) * w)
            
            img = overlay_image(img, assets["confucius_hat"], fx - int(face_width * 0.7), fy - face_width, int(face_width * 1.4), face_width)
            
            bx, by = int(landmarks[152].x * w), int(landmarks[152].y * h)
            img = overlay_image(img, assets["beard"], bx - int(face_width / 2), by - 10, face_width, face_width)
            
            hx, hy = int(hand_landmarks.landmark[0].x * w), int(hand_landmarks.landmark[0].y * h)
            img = overlay_image(img, assets["sleeve"], hx - 150, hy - 150, 300, 300)

    # 3. 秦始皇：比讚
    elif character == "秦始皇" and hand_results.multi_hand_landmarks:
        hl = hand_results.multi_hand_landmarks[0].landmark
        # 修正原代碼變數未定義錯誤，並放寬比讚判定
        is_thumb_up = hl[4].y < hl[3].y and hl[8].y > hl[6].y
        
        if is_thumb_up and face_results.multi_face_landmarks:
            landmarks = face_results.multi_face_landmarks[0].landmark
            fx, fy = int(landmarks[10].x * w), int(landmarks[10].y * h)
            face_width = int(abs(landmarks[234].x - landmarks[454].x) * w)
            
            img = overlay_image(img, assets["emperor_hat"], fx - face_width, fy - int(face_width * 1.2), face_width * 2, int(face_width * 1.5))
            
            # 依需求移到下巴下方
            bx, by = int(landmarks[152].x * w), int(landmarks[152].y * h)
            img = overlay_image(img, assets["bear"], bx - 100, by + 10, 200, 150)

    # 4. 釋迦牟尼佛：閉眼
    elif character == "釋迦牟尼佛" and face_results.multi_face_landmarks:
        landmarks = face_results.multi_face_landmarks[0].landmark
        left_eye_top = np.array([landmarks[159].x * w, landmarks[159].y * h])
        left_eye_bot = np.array([landmarks[145].x * w, landmarks[145].y * h])
        eye_dist = np.linalg.norm(left_eye_top - left_eye_bot)
        
        # 放寬眼瞼距離門檻，避免不同螢幕解析度造成無法觸發
        if eye_dist < 8.0:
            fx, fy = int(landmarks[10].x * w), int(landmarks[10].y * h)
            face_width = int(abs(landmarks[234].x - landmarks[454].x) * w)
            img = overlay_image(img, assets["holy_light"], fx - face_width * 2, fy - face_width * 3, face_width * 4, face_width * 4)

    # 5. 路易十六：換番茄
    elif character == "路易十六" and face_results.multi_face_landmarks:
        landmarks = face_results.multi_face_landmarks[0].landmark
        cx, cy = int(landmarks[4].x * w), int(landmarks[4].y * h)
        face_width = int(abs(landmarks[234].x - landmarks[454].x) * w)
        
        img = overlay_image(img, assets["tomato"], cx - int(face_width * 0.9), cy - int(face_width * 1.0), int(face_width * 1.8), int(face_width * 1.8))

    return frame.from_ndarray(img, format="bgr24")

# --- 6. 啟動網頁相機視訊流 ---
ctx = webrtc_streamer(
    key="historical-camera",
    mode=WebRtcMode.SENDRECV,
    video_frame_callback=video_frame_callback,
    rtc_configuration={
        "iceServers": [{"urls": ["stun:stun.l.google.com:19302", "stun:stun1.l.google.com:19302"]}]
    },
    media_stream_constraints={"video": True, "audio": False},
)

st.write("💡 **如何拍照？**：使用電腦或手機自帶的螢幕截圖，或是點擊網頁視訊畫面右下角的內建功能即可儲存照片喔！")
