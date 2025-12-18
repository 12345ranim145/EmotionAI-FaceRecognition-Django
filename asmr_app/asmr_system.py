import cv2
import mediapipe as mp
from gtts import gTTS
from playsound import playsound
import threading
import os
import time
import numpy as np
from collections import deque
import random
from datetime import datetime

# === Configuration Avancée ===
class Config:
    VOICE_COOLDOWN = 6
    EMOTION_HISTORY_SIZE = 15
    MIN_DETECTION_CONFIDENCE = 0.7
    MIN_TRACKING_CONFIDENCE = 0.7
    FULLSCREEN = True
    SHOW_FACE_MESH = True
    ENABLE_SCANNER_EFFECT = True
    ENABLE_PARTICLES = True

# === Initialisation ===
mp_face = mp.solutions.face_mesh
mp_hands = mp.solutions.hands
mp_pose = mp.solutions.pose
mp_draw = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles

face_mesh = mp_face.FaceMesh(
    min_detection_confidence=Config.MIN_DETECTION_CONFIDENCE,
    min_tracking_confidence=Config.MIN_TRACKING_CONFIDENCE,
    max_num_faces=1,
    refine_landmarks=True
)

hands = mp_hands.Hands(
    min_detection_confidence=0.7,
    min_tracking_confidence=0.7,
    max_num_hands=2
)

pose = mp_pose.Pose(
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
)

# === Détecteur d'Émotions Ultra Précis ===
class AdvancedEmotionDetector:
    def __init__(self):
        self.emotion_history = deque(maxlen=Config.EMOTION_HISTORY_SIZE)
        self.current_emotion = "neutre"
        self.confidence = 0.0
        self.emotion_start_time = {}
        self.micro_expressions = []
        
    def calculate_distance(self, p1, p2):
        return np.sqrt((p1.x - p2.x)**2 + (p1.y - p2.y)**2 + (p1.z - p2.z)**2)
    
    def calculate_ear(self, landmarks, eye_indices):
        """Eye Aspect Ratio - Précision améliorée"""
        p = landmarks
        
        # Vertical distances
        v1 = self.calculate_distance(p[eye_indices[1]], p[eye_indices[5]])
        v2 = self.calculate_distance(p[eye_indices[2]], p[eye_indices[4]])
        
        # Horizontal distance
        h = self.calculate_distance(p[eye_indices[0]], p[eye_indices[3]])
        
        ear = (v1 + v2) / (2.0 * h) if h > 0 else 0
        return ear
    
    def calculate_mar(self, landmarks):
        """Mouth Aspect Ratio - Version améliorée"""
        p = landmarks
        
        # Hauteur bouche (3 mesures)
        v1 = self.calculate_distance(p[13], p[14])  # Centre haut-bas
        v2 = self.calculate_distance(p[12], p[15])  # Gauche haut-bas
        v3 = self.calculate_distance(p[11], p[16])  # Droite haut-bas
        
        # Largeur bouche
        h = self.calculate_distance(p[61], p[291])
        
        mar = (v1 + v2 + v3) / (3.0 * h) if h > 0 else 0
        return mar
    
    def calculate_smile_asymmetry(self, landmarks):
        """Détecte l'asymétrie du sourire"""
        p = landmarks
        left_corner_y = p[61].y
        right_corner_y = p[291].y
        asymmetry = abs(left_corner_y - right_corner_y)
        return asymmetry
    
    def calculate_jaw_tension(self, landmarks):
        """Détecte la tension de la mâchoire"""
        p = landmarks
        jaw_width = self.calculate_distance(p[234], p[454])
        face_height = self.calculate_distance(p[10], p[152])
        tension = jaw_width / face_height if face_height > 0 else 0
        return tension
    
    def calculate_eyebrow_raise(self, landmarks):
        """Mesure l'élévation des sourcils"""
        p = landmarks
        
        # Sourcils
        left_brow = (p[70].y + p[63].y + p[105].y) / 3
        right_brow = (p[300].y + p[293].y + p[334].y) / 3
        
        # Yeux (référence)
        left_eye = (p[33].y + p[133].y) / 2
        right_eye = (p[362].y + p[263].y) / 2
        
        left_raise = left_eye - left_brow
        right_raise = right_eye - right_brow
        
        avg_raise = (left_raise + right_raise) / 2
        return avg_raise
    
    def calculate_lip_corner_pull(self, landmarks):
        """Détecte le sourire (coins des lèvres relevés)"""
        p = landmarks
        
        left_corner = p[61].y
        right_corner = p[291].y
        mouth_center = p[13].y
        
        pull = mouth_center - ((left_corner + right_corner) / 2)
        return pull
    
    def calculate_nose_wrinkle(self, landmarks):
        """Détecte le froncement du nez (dégoût)"""
        p = landmarks
        
        nose_bridge = p[6].y
        nose_tip = p[4].y
        cheek_left = p[205].y
        cheek_right = p[425].y
        
        wrinkle = (nose_tip - nose_bridge) * ((cheek_left + cheek_right) / 2)
        return wrinkle
    
    def calculate_head_tilt(self, landmarks):
        """Calcule l'inclinaison de la tête"""
        p = landmarks
        
        left_eye = p[33]
        right_eye = p[263]
        
        dx = right_eye.x - left_eye.x
        dy = right_eye.y - left_eye.y
        
        angle = np.degrees(np.arctan2(dy, dx))
        return angle
    
    def detect_micro_expression(self, metrics):
        """Détecte les micro-expressions (changements rapides)"""
        if len(self.micro_expressions) > 3:
            recent = self.micro_expressions[-3:]
            changes = sum(1 for i in range(len(recent)-1) 
                         if abs(recent[i] - recent[i+1]) > 0.02)
            if changes >= 2:
                return True
        return False
    
    def detect_emotion(self, face_landmarks):
        """Détection d'émotion avec IA avancée"""
        landmarks = face_landmarks.landmark
        
        # Indices des yeux
        left_eye = [33, 160, 158, 133, 153, 144]
        right_eye = [362, 385, 387, 263, 373, 380]
        
        # Calcul de toutes les métriques
        left_ear = self.calculate_ear(landmarks, left_eye)
        right_ear = self.calculate_ear(landmarks, right_eye)
        avg_ear = (left_ear + right_ear) / 2
        
        mar = self.calculate_mar(landmarks)
        smile_asymmetry = self.calculate_smile_asymmetry(landmarks)
        jaw_tension = self.calculate_jaw_tension(landmarks)
        eyebrow_raise = self.calculate_eyebrow_raise(landmarks)
        lip_pull = self.calculate_lip_corner_pull(landmarks)
        nose_wrinkle = self.calculate_nose_wrinkle(landmarks)
        head_tilt = self.calculate_head_tilt(landmarks)
        
        # Enregistrer pour micro-expressions
        self.micro_expressions.append(lip_pull)
        if len(self.micro_expressions) > 10:
            self.micro_expressions.pop(0)
        
        has_micro = self.detect_micro_expression(self.micro_expressions)
        
        # Système de scoring pour chaque émotion
        scores = {
            "heureux": 0,
            "très heureux": 0,
            "triste": 0,
            "confortable": 0,
            "stressé": 0,
            "fatigué": 0,
            "neutre": 50,
            "surpris": 0,
            "en colère": 0
        }
        
        # Heureux / Sourire
        if lip_pull > 0.008:
            scores["heureux"] += 40
        if lip_pull > 0.015 and mar > 0.25:
            scores["très heureux"] += 60
        if lip_pull > 0.012:
            scores["heureux"] += 30
        if smile_asymmetry < 0.01:
            scores["heureux"] += 20
            scores["très heureux"] += 20
        
        # Triste
        if lip_pull < -0.005:
            scores["triste"] += 40
        if eyebrow_raise < 0.045:
            scores["triste"] += 30
        if mar < 0.15:
            scores["triste"] += 20
        if abs(head_tilt) > 10:
            scores["triste"] += 15
        
        # Confortable / Relaxé
        if 0.20 < avg_ear < 0.25:
            scores["confortable"] += 40
        if mar < 0.20 and lip_pull > -0.002 and lip_pull < 0.005:
            scores["confortable"] += 30
        if jaw_tension < 0.85:
            scores["confortable"] += 20
        
        # Fatigué
        if avg_ear < 0.20:
            scores["fatigué"] += 50
        if eyebrow_raise < 0.040:
            scores["fatigué"] += 25
        if mar < 0.18:
            scores["fatigué"] += 20
        
        # Stressé
        if jaw_tension > 0.95:
            scores["stressé"] += 40
        if eyebrow_raise > 0.055:
            scores["stressé"] += 35
        if has_micro:
            scores["stressé"] += 25
        
        # Surpris
        if avg_ear > 0.30:
            scores["surpris"] += 40
        if eyebrow_raise > 0.060:
            scores["surpris"] += 40
        if mar > 0.35:
            scores["surpris"] += 30
        
        # En colère
        if eyebrow_raise < 0.042 and jaw_tension > 0.92:
            scores["en colère"] += 50
        if nose_wrinkle > 0.001:
            scores["en colère"] += 30
        if lip_pull < 0 and mar < 0.2:
            scores["en colère"] += 25
        
        # Trouver l'émotion dominante
        emotion = max(scores, key=scores.get)
        confidence = scores[emotion] / 100.0
        
        # Historique pour stabilité
        self.emotion_history.append((emotion, confidence))
        
        # Émotion finale avec vote pondéré
        if len(self.emotion_history) >= 5:
            recent_emotions = list(self.emotion_history)[-8:]
            weighted_votes = {}
            
            for i, (emo, conf) in enumerate(recent_emotions):
                weight = (i + 1) * conf
                weighted_votes[emo] = weighted_votes.get(emo, 0) + weight
            
            emotion = max(weighted_votes, key=weighted_votes.get)
            confidence = min(weighted_votes[emotion] / (len(recent_emotions) * 2), 1.0)
        
        self.current_emotion = emotion
        self.confidence = confidence
        
        metrics = {
            'ear': avg_ear,
            'mar': mar,
            'lip_pull': lip_pull,
            'eyebrow_raise': eyebrow_raise,
            'jaw_tension': jaw_tension,
            'head_tilt': head_tilt,
            'confidence': confidence
        }
        
        return emotion, metrics

# === Messages ASMR Étendus ===
class ASMRMessages:
    MESSAGES = {
        "heureux": [
            "Je vois ce sourire magnifique... continue à rayonner ainsi...",
            "Ton énergie positive illumine tout... profite de chaque instant...",
            "C'est merveilleux... garde cette joie en toi...",
            "Ta bonne humeur est contagieuse... savoure ce moment...",
        ],
        "très heureux": [
            "Quelle joie extraordinaire ! Tu rayonnes de bonheur...",
            "Je ressens ton enthousiasme... c'est magnifique...",
            "Continue à sourire... cette énergie est précieuse...",
            "Ton bonheur est inspirant... profite pleinement...",
        ],
        "triste": [
            "Je suis là avec toi... respire doucement... ça va aller...",
            "Chaque émotion a sa place... laisse couler tes sentiments...",
            "Tu n'es pas seul... prends ton temps... je suis là...",
            "C'est normal de se sentir ainsi... sois bienveillant envers toi-même...",
            "Ferme les yeux un instant... respire profondément... je t'accompagne...",
        ],
        "confortable": [
            "Tu es parfaitement détendu... continue ainsi... tout est calme...",
            "Laisse ton corps se relâcher complètement... tu es en sécurité...",
            "C'est un moment de paix... profite de chaque respiration...",
            "Sens la tranquillité t'envahir... tout va bien...",
        ],
        "stressé": [
            "Respire avec moi... inspire lentement... expire doucement...",
            "Relâche tes épaules... détends ta mâchoire... tout va s'arranger...",
            "Le stress va partir... concentre-toi sur l'instant présent...",
            "Ferme les yeux... compte jusqu'à cinq... respire calmement...",
            "Tu as le contrôle... laisse le calme revenir progressivement...",
        ],
        "fatigué": [
            "Ton corps a besoin de repos... écoute-le attentivement...",
            "Ferme les yeux doucement... laisse la fatigue s'évaporer...",
            "C'est l'heure de te reposer... détends chaque muscle...",
            "Respire lentement... laisse-toi aller... tu peux te reposer...",
        ],
        "surpris": [
            "Je vois ta surprise... prends un moment pour assimiler...",
            "Respire... tout va bien... laisse l'émotion passer...",
            "C'est intéressant ce qui se passe... reste calme...",
        ],
        "en colère": [
            "Je sens ta frustration... respire profondément... ça va passer...",
            "Prends un moment... inspire... expire... retrouve ton calme...",
            "La colère est normale... laisse-la s'exprimer puis s'apaiser...",
            "Respire... compte jusqu'à dix... tout va bien...",
        ],
        "neutre": [
            "Tout est calme... profite de cette sérénité...",
            "Respire naturellement... sois simplement présent...",
            "C'est un moment de tranquillité... savoure-le...",
        ]
    }
    
    @staticmethod
    def get_message(emotion):
        return random.choice(ASMRMessages.MESSAGES.get(emotion, ASMRMessages.MESSAGES["neutre"]))

# === Système Audio Avancé ===
class VoiceSystem:
    def __init__(self):
        self.last_voice_time = 0
        self.is_speaking = False
        self.lock = threading.Lock()
        self.voice_queue = []
    
    def can_speak(self):
        return time.time() - self.last_voice_time > Config.VOICE_COOLDOWN
    
    def play_voice(self, text, emotion):
        if self.is_speaking or not self.can_speak():
            return
        
        with self.lock:
            self.is_speaking = True
            self.last_voice_time = time.time()
        
        try:
            slow_speed = emotion in ["triste", "confortable", "fatigué", "stressé"]
            tts = gTTS(text=text, lang="fr", slow=slow_speed)
            filename = f"voice_{int(time.time()*1000)}.mp3"
            tts.save(filename)
            playsound(filename)
            os.remove(filename)
        except Exception as e:
            print(f"[AUDIO ERROR] {e}")
        finally:
            self.is_speaking = False
    
    def speak_async(self, text, emotion):
        thread = threading.Thread(target=self.play_voice, args=(text, emotion))
        thread.daemon = True
        thread.start()

# === Effets Visuels FBI/CIA Style ===
class VisualEffects:
    def __init__(self):
        self.scanner_y = 0
        self.particles = []
        self.grid_alpha = 0.3
        self.pulse = 0
        
    def draw_scanner_line(self, frame):
        """Ligne de scan style FBI"""
        h, w = frame.shape[:2]
        self.scanner_y = (self.scanner_y + 3) % h
        
        # Ligne principale
        cv2.line(frame, (0, self.scanner_y), (w, self.scanner_y), (0, 255, 0), 2)
        
        # Effet de glow
        for offset in range(1, 20):
            alpha = 1.0 - (offset / 20.0)
            color = (0, int(255 * alpha), 0)
            y_pos = (self.scanner_y - offset) % h
            cv2.line(frame, (0, y_pos), (w, y_pos), color, 1)
    
    def draw_grid(self, frame):
        """Grille de fond style analyse technique"""
        h, w = frame.shape[:2]
        overlay = frame.copy()
        
        # Lignes verticales
        for x in range(0, w, 50):
            cv2.line(overlay, (x, 0), (x, h), (0, 100, 0), 1)
        
        # Lignes horizontales
        for y in range(0, h, 50):
            cv2.line(overlay, (0, y), (w, y), (0, 100, 0), 1)
        
        cv2.addWeighted(overlay, self.grid_alpha, frame, 1 - self.grid_alpha, 0, frame)
    
    def draw_corner_brackets(self, frame, bbox):
        """Brackets de tracking style CIA"""
        x, y, w, h = bbox
        length = 30
        thickness = 3
        color = (0, 255, 255)
        
        # Coins supérieurs
        cv2.line(frame, (x, y), (x + length, y), color, thickness)
        cv2.line(frame, (x, y), (x, y + length), color, thickness)
        cv2.line(frame, (x + w, y), (x + w - length, y), color, thickness)
        cv2.line(frame, (x + w, y), (x + w, y + length), color, thickness)
        
        # Coins inférieurs
        cv2.line(frame, (x, y + h), (x + length, y + h), color, thickness)
        cv2.line(frame, (x, y + h), (x, y + h - length), color, thickness)
        cv2.line(frame, (x + w, y + h), (x + w - length, y + h), color, thickness)
        cv2.line(frame, (x + w, y + h), (x + w, y + h - length), color, thickness)
    
    def draw_hud(self, frame, emotion, metrics, fps):
        """HUD complet style interface militaire"""
        h, w = frame.shape[:2]
        
        # Panel principal (gauche)
        panel_w = 350
        overlay = frame.copy()
        cv2.rectangle(overlay, (0, 0), (panel_w, h), (0, 0, 0), -1)
        cv2.addWeighted(overlay, 0.7, frame, 0.3, 0, frame)
        
        # Bordure du panel
        cv2.rectangle(frame, (0, 0), (panel_w, h), (0, 255, 0), 2)
        
        y_pos = 40
        line_height = 35
        
        # Header
        cv2.putText(frame, "ASMR TACTICAL SYSTEM", (15, y_pos),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        cv2.line(frame, (10, y_pos + 10), (panel_w - 10, y_pos + 10), (0, 255, 0), 1)
        
        y_pos += 50
        
        # Timestamp
        timestamp = datetime.now().strftime("%H:%M:%S.%f")[:-3]
        cv2.putText(frame, f"TIME: {timestamp}", (15, y_pos),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 200, 0), 1)
        y_pos += line_height
        
        # FPS
        cv2.putText(frame, f"FPS: {fps:.1f}", (15, y_pos),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 200, 0), 1)
        y_pos += line_height + 10
        
        # Séparateur
        cv2.line(frame, (10, y_pos), (panel_w - 10, y_pos), (0, 255, 0), 1)
        y_pos += 30
        
        # Émotion détectée
        emotion_color = self.get_emotion_color(emotion)
        cv2.putText(frame, "EMOTION DETECTEE:", (15, y_pos),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1)
        y_pos += 30
        
        cv2.putText(frame, emotion.upper(), (15, y_pos),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.9, emotion_color, 2)
        y_pos += 40
        
        # Barre de confiance
        confidence = metrics.get('confidence', 0)
        self.draw_progress_bar(frame, 15, y_pos, panel_w - 30, 20, confidence, 
                               "CONFIANCE", emotion_color)
        y_pos += 50
        
        # Séparateur
        cv2.line(frame, (10, y_pos), (panel_w - 10, y_pos), (0, 255, 0), 1)
        y_pos += 30
        
        # Métriques biométriques
        cv2.putText(frame, "BIOMETRIE:", (15, y_pos),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1)
        y_pos += 30
        
        metrics_display = [
            ("EAR (Yeux)", metrics.get('ear', 0), 0.3),
            ("MAR (Bouche)", metrics.get('mar', 0), 0.5),
            ("Sourire", metrics.get('lip_pull', 0) * 100, 2.0),
            ("Sourcils", metrics.get('eyebrow_raise', 0) * 100, 6.0),
            ("Tension", metrics.get('jaw_tension', 0), 1.0),
        ]
        
        for label, value, max_val in metrics_display:
            norm_value = min(abs(value) / max_val, 1.0) if max_val > 0 else 0
            bar_color = (0, int(255 * (1 - norm_value)), int(255 * norm_value))
            
            cv2.putText(frame, label, (20, y_pos),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.45, (200, 200, 200), 1)
            y_pos += 20
            
            self.draw_progress_bar(frame, 20, y_pos, panel_w - 40, 15, 
                                  norm_value, f"{value:.3f}", bar_color)
            y_pos += 35
        
        # Panel droit (mini)
        mini_panel_w = 200
        mini_panel_x = w - mini_panel_w
        
        overlay = frame.copy()
        cv2.rectangle(overlay, (mini_panel_x, 0), (w, 150), (0, 0, 0), -1)
        cv2.addWeighted(overlay, 0.7, frame, 0.3, 0, frame)
        cv2.rectangle(frame, (mini_panel_x, 0), (w, 150), (0, 255, 0), 2)
        
        cv2.putText(frame, "STATUS", (mini_panel_x + 10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
        cv2.putText(frame, "ACTIVE", (mini_panel_x + 10, 65),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
        cv2.putText(frame, "TRACKING", (mini_panel_x + 10, 100),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 200, 0), 1)
        
        # Indicateur de pulse
        self.pulse = (self.pulse + 0.1) % (2 * np.pi)
        pulse_alpha = (np.sin(self.pulse) + 1) / 2
        pulse_color = (0, int(255 * pulse_alpha), 0)
        cv2.circle(frame, (mini_panel_x + 170, 35), 10, pulse_color, -1)
    
    def draw_progress_bar(self, frame, x, y, width, height, value, label, color):
        """Barre de progression style tech"""
        # Fond
        cv2.rectangle(frame, (x, y), (x + width, y + height), (50, 50, 50), -1)
        cv2.rectangle(frame, (x, y), (x + width, y + height), (0, 255, 0), 1)
        
        # Remplissage
        fill_width = int(width * value)
        if fill_width > 0:
            cv2.rectangle(frame, (x + 2, y + 2), 
                         (x + fill_width - 2, y + height - 2), color, -1)
        
        # Label
        text_x = x + width + 10
        cv2.putText(frame, label, (text_x, y + height - 5),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, (200, 200, 200), 1)
    
    def get_emotion_color(self, emotion):
        """Couleur selon l'émotion"""
        colors = {
            "heureux": (0, 255, 0),
            "très heureux": (0, 255, 255),
            "triste": (255, 0, 100),
            "confortable": (200, 150, 255),
            "stressé": (0, 100, 255),
            "fatigué": (150, 150, 150),
            "surpris": (255, 255, 0),
            "en colère": (0, 0, 255),
            "neutre": (255, 255, 255)
        }
        return colors.get(emotion, (255, 255, 255))

# === Boucle Principale ===
def main():
    cap = cv2.VideoCapture(0)
    
    if not cap.isOpened():
        print("[ERROR] Impossible d'accéder à la caméra")
        return
    
    # Configuration de la caméra
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
    cap.set(cv2.CAP_PROP_FPS, 30)
    
    detector = AdvancedEmotionDetector()
    voice_system = VoiceSystem()
    effects = VisualEffects()
    
    # Fenêtre en plein écran
    window_name = "ASMR TACTICAL SYSTEM - [ESC] to quit"
    cv2.namedWindow(window_name, cv2.WND_PROP_FULLSCREEN)
    if Config.FULLSCREEN:
        cv2.setWindowProperty(window_name, cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)
    
    print("=" * 70)
    print("🎯 ASMR TACTICAL SYSTEM - DETECTION AVANCÉE ACTIVÉE")
    print("=" * 70)
    print("📹 Caméra: ACTIVE")
    print("🎭 Détection: 9 émotions (Heureux, Triste, Confortable, Stressé, etc.)")
    print("🎤 Audio: ASMR adaptatif")
    print("👁️  Interface: FBI/CIA Style")
    print("⌨️  Commandes:")
    print("   - ESC: Quitter")
    print("   - G: Toggle Grille")
    print("   - S: Toggle Scanner")
    print("   - M: Toggle Maillage facial")
    print("   - F: Toggle Plein écran")
    print("=" * 70)
    
    last_emotion = None
    frame_count = 0
    fps = 0
    fps_time = time.time()
    fps_counter = 0
    
    show_grid = Config.ENABLE_PARTICLES
    show_scanner = Config.ENABLE_SCANNER_EFFECT
    show_mesh = Config.SHOW_FACE_MESH
    
    while True:
        loop_start = time.time()
        success, frame = cap.read()
        if not success:
            print("[ERROR] Échec de lecture de la caméra")
            break
        
        frame = cv2.flip(frame, 1)
        h, w = frame.shape[:2]
        
        # Convertir en RGB pour MediaPipe
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        # Traitement MediaPipe
        face_result = face_mesh.process(rgb)
        
        # Effets de fond
        if show_grid:
            effects.draw_grid(frame)
        
        if show_scanner:
            effects.draw_scanner_line(frame)
        
        emotion = "neutre"
        metrics = {}
        face_detected = False
        
        if face_result.multi_face_landmarks:
            for face_landmarks in face_result.multi_face_landmarks:
                face_detected = True
                
                # Dessiner le maillage facial si activé
                if show_mesh:
                    mp_draw.draw_landmarks(
                        frame,
                        face_landmarks,
                        mp_face.FACEMESH_TESSELATION,
                        landmark_drawing_spec=None,
                        connection_drawing_spec=mp_drawing_styles.get_default_face_mesh_tesselation_style()
                    )
                    
                    # Contours spéciaux
                    mp_draw.draw_landmarks(
                        frame,
                        face_landmarks,
                        mp_face.FACEMESH_CONTOURS,
                        landmark_drawing_spec=None,
                        connection_drawing_spec=mp_drawing_styles
                        .get_default_face_mesh_contours_style()
                    )
                
                # Détection d'émotion
                emotion, metrics = detector.detect_emotion(face_landmarks)
                
                # Calculer bounding box pour les brackets
                landmarks = face_landmarks.landmark
                x_coords = [lm.x * w for lm in landmarks]
                y_coords = [lm.y * h for lm in landmarks]
                
                bbox_x = int(min(x_coords))
                bbox_y = int(min(y_coords))
                bbox_w = int(max(x_coords) - bbox_x)
                bbox_h = int(max(y_coords) - bbox_y)
                
                # Dessiner les brackets de tracking
                effects.draw_corner_brackets(frame, (bbox_x, bbox_y, bbox_w, bbox_h))
                
                # Points clés avec indicateurs
                key_points = [
                    (33, "L_EYE"),   # Œil gauche
                    (263, "R_EYE"),  # Œil droit
                    (1, "NOSE"),     # Nez
                    (61, "L_MOUTH"), # Bouche gauche
                    (291, "R_MOUTH") # Bouche droite
                ]
                
                for idx, label in key_points:
                    lm = landmarks[idx]
                    x, y = int(lm.x * w), int(lm.y * h)
                    cv2.circle(frame, (x, y), 3, (0, 255, 255), -1)
                    cv2.circle(frame, (x, y), 8, (0, 255, 255), 1)
                
                # Message ASMR si changement d'émotion
                if emotion != last_emotion and frame_count % 30 == 0:
                    message = ASMRMessages.get_message(emotion)
                    voice_system.speak_async(message, emotion)
                    timestamp = datetime.now().strftime("%H:%M:%S")
                    print(f"[{timestamp}] 🎭 {emotion.upper()} ({metrics['confidence']:.0%}) → 🎤 {message}")
                    last_emotion = emotion
        
        # Calculer FPS
        fps_counter += 1
        if time.time() - fps_time >= 1.0:
            fps = fps_counter / (time.time() - fps_time)
            fps_counter = 0
            fps_time = time.time()
        
        # Dessiner le HUD
        if face_detected:
            effects.draw_hud(frame, emotion, metrics, fps)
        else:
            # Message si aucun visage
            overlay = frame.copy()
            cv2.rectangle(overlay, (w//2 - 250, h//2 - 80), 
                         (w//2 + 250, h//2 + 80), (0, 0, 0), -1)
            cv2.addWeighted(overlay, 0.7, frame, 0.3, 0, frame)
            
            cv2.rectangle(frame, (w//2 - 250, h//2 - 80), 
                         (w//2 + 250, h//2 + 80), (0, 100, 255), 2)
            
            cv2.putText(frame, "AUCUN VISAGE DETECTE", (w//2 - 220, h//2 - 20),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 100, 255), 2)
            cv2.putText(frame, "Positionnez-vous face a la camera", (w//2 - 220, h//2 + 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 1)
            
            # Afficher quand même le FPS
            cv2.putText(frame, f"FPS: {fps:.1f}", (20, 40),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        
        # Afficher instructions en bas
        instructions = [
            "[ESC] Quitter",
            "[G] Grille",
            "[S] Scanner",
            "[M] Maillage",
            "[F] Plein ecran"
        ]
        
        inst_y = h - 30
        inst_x = 20
        for inst in instructions:
            cv2.putText(frame, inst, (inst_x, inst_y),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 200, 0), 1)
            inst_x += 150
        
        # Afficher la frame
        cv2.imshow(window_name, frame)
        frame_count += 1
        
        # Gestion des touches
        key = cv2.waitKey(1) & 0xFF
        
        if key == 27:  # ESC
            print("\n[SYSTEM] Arrêt du système...")
            break
        elif key == ord('g') or key == ord('G'):
            show_grid = not show_grid
            print(f"[TOGGLE] Grille: {'ON' if show_grid else 'OFF'}")
        elif key == ord('s') or key == ord('S'):
            show_scanner = not show_scanner
            print(f"[TOGGLE] Scanner: {'ON' if show_scanner else 'OFF'}")
        elif key == ord('m') or key == ord('M'):
            show_mesh = not show_mesh
            print(f"[TOGGLE] Maillage facial: {'ON' if show_mesh else 'OFF'}")
        elif key == ord('f') or key == ord('F'):
            Config.FULLSCREEN = not Config.FULLSCREEN
            if Config.FULLSCREEN:
                cv2.setWindowProperty(window_name, cv2.WND_PROP_FULLSCREEN, 
                                     cv2.WINDOW_FULLSCREEN)
                print("[TOGGLE] Mode plein écran: ON")
            else:
                cv2.setWindowProperty(window_name, cv2.WND_PROP_FULLSCREEN, 
                                     cv2.WINDOW_NORMAL)
                print("[TOGGLE] Mode fenêtré: ON")
        
        # Limiter le framerate pour stabilité
        loop_time = time.time() - loop_start
        if loop_time < 0.033:  # ~30 FPS max
            time.sleep(0.033 - loop_time)
    
    # Nettoyage
    cap.release()
    cv2.destroyAllWindows()
    
    print("\n" + "=" * 70)
    print("✨ Merci d'avoir utilisé ASMR TACTICAL SYSTEM")
    print("💚 Session terminée avec succès")
    print("=" * 70)

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n[INTERRUPT] Arrêt d'urgence détecté")
        cv2.destroyAllWindows()
    except Exception as e:
        print(f"\n[CRITICAL ERROR] {e}")
        import traceback
        traceback.print_exc()
        cv2.destroyAllWindows()