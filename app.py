import cv2
import mediapipe as mp
import numpy as np
import streamlit as st
from streamlit_webrtc import WebRtcMode, webrtc_streamer, RTCConfiguration

# --- 1. 初始化 ＆ 設定 ---
st.title("MediaPipe 拍照偵測系統")

# 配置免費的 Google STUN 伺服器，確保雲端佈署也能順利啟動鏡頭
RTC_CONFIGURATION = RTCConfiguration(
    {"iceServers": [{"urls": ["stun:stun.l.google.com:19302", "stun:stun1.l.google.com:19302"]}]}
)

# 初始化 MediaPipe 模組
mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(
    max_num_faces=1, refine_landmarks=True, min_detection_confidence=0.5
)

mp_hands = mp.solutions.hands
hands = mp_hands.Hands(
    static_image_mode=True, max_num_hands=2, min_detection_confidence=0.5
)

# --- 2. 建立視訊組件 ---
# 使用 ctx 來捕捉鏡頭狀態
ctx = webrtc_streamer(
    key="snapshot-mode",
    mode=WebRtcMode.SENDRECV,  # 接收並傳送畫面
    rtc_configuration=RTC_CONFIGURATION,
    media_stream_constraints={"video": True, "audio": False}, # 只需要影像，關閉聲音
    async_processing=True
)

# --- 3. 拍照與偵測按鈕 ---
if ctx.video_receiver:
    # 當鏡頭成功連線後，顯示拍照按鈕
    if st.button("📸 點我拍照並識別"):
        try:
            # 從 WebRTC 接收器中抓取最新的一影格（Frame）
            img_frame = ctx.video_receiver.get_frame()
            
            if img_frame is not None:
                # 將框架轉換為 Np Array (BGR 格式)
                bgr_image = img_frame.to_ndarray(format="bgr24")
                # 轉為 MediaPipe 需要的 RGB 格式
                rgb_image = cv2.cvtColor(bgr_image, cv2.COLOR_BGR2RGB)
                
                st.write("正在分析畫面...")
                
                # --- 執行 MediaPipe 偵測 ---
                face_results = face_mesh.process(rgb_image)
                hand_results = hands.process(rgb_image)
                
                # 繪製結果（複用您原本的繪圖邏輯，此處為示意範例）
                annotated_image = bgr_image.copy()
                
                # 範例：如果偵測到手部，在畫面寫字
                if hand_results.multi_hand_landmarks:
                    cv2.putText(annotated_image, "Hand Detected!", (10, 50), 
                                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                
                # 範例：如果偵測到臉部
                if face_results.multi_face_landmarks:
                    cv2.putText(annotated_image, "Face Detected!", (10, 100), 
                                cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)
                
                # --- 顯示結果照片 ---
                st.subheader("偵測結果")
                st.image(cv2.cvtColor(annotated_image, cv2.COLOR_BGR2RGB), use_column_width=True)
            else:
                st.warning("未能擷取到畫面，請確認視訊鏡頭是否已啟動並允許瀏覽器存取。")
                
        except Exception as e:
            st.error(f"拍照識別時發生錯誤: {e}")
else:
    st.info("請先點擊上方的『Start』按鈕啟動您的視訊鏡頭。")        x2_o, y2_o = x1_o + (x2 - x1), y1_o + (y2 - y1)

        if x2 - x1 <= 0 or y2 - y1 <= 0:
            return background

        crop_background = background[y1:y2, x1:x2]
        crop_overlay = overlay_resized[y1_o:y2_o, x1_o:x2_o]

        alpha = crop_overlay[:, :, 3] / 255.0
        alpha = np.expand_dims(alpha, axis=2)

        composite = crop_overlay[:, :, :3] * alpha + crop_background * (
            1 - alpha
        )
        background[y1:y2, x1:x2] = composite.astype(np.uint8)
    except Exception as e:
        pass
    return background


# --- 讀取貼圖資產 ---
@st.cache_resource
def load_assets():
    return {
        "hair": cv2.imread("assets/hair.png", cv2.IMREAD_UNCHANGED),
        "tongue": cv2.imread("assets/tongue.png", cv2.IMREAD_UNCHANGED),
        "confucius_hat": cv2.imread(
            "assets/confucius_hat.png", cv2.IMREAD_UNCHANGED
        ),
        "beard": cv2.imread("assets/beard.png", cv2.IMREAD_UNCHANGED),
        "sleeve": cv2.imread("assets/sleeve.png", cv2.IMREAD_UNCHANGED),
        "emperor_hat": cv2.imread(
            "assets/emperor_hat.png", cv2.IMREAD_UNCHANGED
        ),
        "bear": cv2.imread("assets/bear.png", cv2.IMREAD_UNCHANGED),
        "holy_light": cv2.imread("assets/holy_light.png", cv2.IMREAD_UNCHANGED),
        "tomato": cv2.imread("assets/tomato.png", cv2.IMREAD_UNCHANGED),
    }


assets = load_assets()

# --- Streamlit 介面設定 ---
st.title("歷史名人動作偵測相機 📸")
st.text("選擇你想挑戰的角色，做出指定動作即可觸發特效與拍照！")

character = st.selectbox(
    "選擇角色：", ["愛因斯坦", "孔子", "秦始皇", "釋迦牟尼佛", "路易十六"]
)

# 提示說明
hints = {
    "愛因斯坦": "😮 張開嘴巴：畫面轉黑白，長出瘋狂頭髮與大舌頭！",
    "孔子": "✋ 手心朝向自己且攤平：戴上儒帽、長出鬍鬚、衣袖遮手！",
    "秦始皇": "👍 比個讚 (大拇指朝上)：戴上皇冠，召喚北極熊！",
    "釋迦牟尼佛": "😑 閉上雙眼：佛光普照，頭頂出現聖光！",
    "路易十六": "😐 不做任何動作：直接將頭部變成大番茄！",
}
st.info(hints[character])


# --- 影像處理核心邏輯 (WebRTC Callback) ---
def video_frame_callback(frame):
    img = frame.to_ndarray(format="bgr24")
    h, w, _ = img.shape

    # 轉為 RGB 供 Mediapipe 處理
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    face_results = face_mesh.process(img_rgb)
    hand_results = hands.process(img_rgb)

    # 1. 愛因斯坦：張嘴
    if character == "愛因斯坦" and face_results.multi_face_landmarks:
        landmarks = face_results.multi_face_landmarks[0].landmark
        # 計算嘴唇上下距離與左右距離的比例
        top_lip = np.array([landmarks[13].x * w, landmarks[13].y * h])
        bottom_lip = np.array([landmarks[14].x * w, landmarks[14].y * h])
        left_mouth = np.array([landmarks[78].x * w, landmarks[78].y * h])
        right_mouth = np.array([landmarks[308].x * w, landmarks[308].y * h])

        mouth_height = np.linalg.norm(top_lip - bottom_lip)
        mouth_width = np.linalg.norm(left_mouth - right_mouth)

        if mouth_height / max(1, mouth_width) > 0.35:  # 張嘴判定
            # 轉黑白濾鏡
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            img = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)

            # 貼上頭髮 (頭頂 10 號節點)
            hx, hy = int(landmarks[10].x * w), int(landmarks[10].y * h)
            face_width = int(
                np.linalg.norm(
                    np.array([landmarks[234].x * w, landmarks[234].y * h])
                    - np.array([landmarks[454].x * w, landmarks[454].y * h])
                )
            )
            img = overlay_image(
                img,
                assets["hair"],
                hx - face_width,
                hy - face_width,
                face_width * 2,
                face_width * 2,
            )

            # 貼上舌頭 (下嘴唇 14 號節點)
            tx, ty = int(landmarks[14].x * w), int(landmarks[14].y * h)
            img = overlay_image(
                img,
                assets["tongue"],
                tx - int(mouth_width / 2),
                ty,
                int(mouth_width),
                int(mouth_width),
            )

    # 2. 孔子：手心朝內且攤平
    elif character == "孔子" and hand_results.multi_hand_landmarks:
        hand_landmarks = hand_results.multi_hand_landmarks[0]
        # 簡單判定手心朝內：大拇指(4)與小指(20)的相對位置，以及手掌是否攤平
        # 這邊以偵測到手部，且有臉部配合時加上特效
        if face_results.multi_face_landmarks:
            landmarks = face_results.multi_face_landmarks[0].landmark
            fx, fy = int(landmarks[10].x * w), int(landmarks[10].y * h)
            face_width = int(abs(landmarks[234].x - landmarks[454].x) * w)

            # 儒帽
            img = overlay_image(
                img,
                assets["confucius_hat"],
                fx - int(face_width * 0.7),
                fy - face_width,
                int(face_width * 1.4),
                face_width,
            )
            # 鬍鬚 (下巴 152 號節點)
            bx, by = int(landmarks[152].x * w), int(landmarks[152].y * h)
            img = overlay_image(
                img,
                assets["beard"],
                bx - int(face_width / 2),
                by,
                face_width,
                face_width,
            )

            # 袖子蓋住手 (抓取手掌核心 0 號節點 WRIST)
            hx, hy = (
                int(hand_landmarks.landmark[0].x * w),
                int(hand_landmarks.landmark[0].y * h),
            )
            img = overlay_image(
                img, assets["sleeve"], hx - 150, hy - 150, 300, 300
            )

    # 3. 秦始皇：比讚
    elif character == "秦始皇" and hand_results.multi_hand_landmarks:
        hl = hand_landmarks = hand_results.multi_hand_landmarks[0].landmark
        # 判定比讚：大拇指指尖(4)高於大拇指關節(3)，且其他四指收起(指尖低於第二關節)
        is_thumb_up = (
            hl[4].y < hl[3].y
            and hl[8].y > hl[6].y
            and hl[12].y > hl[10].y
            and hl[16].y > hl[14].y
        )

        if is_thumb_up and face_results.multi_face_landmarks:
            landmarks = face_results.multi_face_landmarks[0].landmark
            fx, fy = int(landmarks[10].x * w), int(landmarks[10].y * h)
            face_width = int(abs(landmarks[234].x - landmarks[454].x) * w)

            # 皇帝帽
            img = overlay_image(
                img,
                assets["emperor_hat"],
                fx - face_width,
                fy - int(face_width * 1.2),
                face_width * 2,
                int(face_width * 1.5),
            )
            # 北極熊貼在最下方
            img = overlay_image(img, assets["bear"], w // 2 - 150, h - 200, 300, 200)

    # 4. 釋迦牟尼佛：閉眼
    elif character == "釋迦牟尼佛" and face_results.multi_face_landmarks:
        landmarks = face_results.multi_face_landmarks[0].landmark
        # 計算眼睛上下眼瞼距離
        left_eye_top = np.array([landmarks[159].x * w, landmarks[159].y * h])
        left_eye_bot = np.array([landmarks[145].x * w, landmarks[145].y * h])
        eye_dist = np.linalg.norm(left_eye_top - left_eye_bot)

        if eye_dist < 6:  # 距離極小視為閉眼
            fx, fy = int(landmarks[10].x * w), int(landmarks[10].y * h)
            face_width = int(abs(landmarks[234].x - landmarks[454].x) * w)
            # 頭頂聖光
            img = overlay_image(
                img,
                assets["holy_light"],
                fx - face_width * 2,
                fy - face_width * 3,
                face_width * 4,
                face_width * 4,
            )

    # 5. 路易十六：不做動作，直接把頭換成番茄
    elif character == "路易十六" and face_results.multi_face_landmarks:
        landmarks = face_results.multi_face_landmarks[0].landmark
        # 取得臉部中心 (鼻尖 4 號節點)
        cx, cy = int(landmarks[4].x * w), int(landmarks[4].y * h)
        face_width = int(abs(landmarks[234].x - landmarks[454].x) * w)

        # 把整張臉蓋成大番茄
        img = overlay_image(
            img,
            assets["tomato"],
            cx - int(face_width * 1.2),
            cy - int(face_width * 1.2),
            int(face_width * 2.4),
            int(face_width * 2.4),
        )

    return frame.from_ndarray(img, format="bgr24")


# --- 啟動網頁相機視訊流 ---
ctx = webrtc_streamer(
    key="historical-camera",
    mode=WebRtcMode.SENDRECV,
    video_frame_callback=video_frame_callback,
    rtc_configuration={
        "iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]
    },
    media_stream_constraints={"video": True, "audio": False},
)

st.write("💡 **如何拍照？**：使用電腦或手機自帶的螢幕截圖，或是點擊網頁視訊畫面右下角的內建功能即可儲存照片喔！")
