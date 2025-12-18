from django.shortcuts import render
from django.http import JsonResponse
from django.views.decorators.csrf import csrf_exempt
from django.conf import settings
import base64
import cv2
import numpy as np
import face_recognition
import json
from .asmr_system import AdvancedEmotionDetector, ASMRMessages, VisualEffects, Config, mp_face, face_mesh, mp_draw, mp_drawing_styles
import os
import time
from gtts import gTTS
import random
from datetime import datetime

# Chemin de l'image de référence
KNOWN_IMAGE_PATH = os.path.join(settings.STATICFILES_DIRS[0], 'images/ranim.jpg')
known_image = face_recognition.load_image_file(KNOWN_IMAGE_PATH)
known_encoding = face_recognition.face_encodings(known_image)[0] if face_recognition.face_encodings(known_image) else None

# Variables globales pour persistance
last_emotion = None
frame_count = 0
fps = 0
fps_time = time.time()
fps_counter = 0
show_grid = Config.ENABLE_PARTICLES
show_scanner = Config.ENABLE_SCANNER_EFFECT
show_mesh = Config.SHOW_FACE_MESH

def home(request):
    return render(request, 'home.html')

@csrf_exempt
def recognize_face(request):
    if request.method == 'POST':
        data = request.body.decode('utf-8')
        image_data = json.loads(data)['image']
        image_bytes = base64.b64decode(image_data)
        np_arr = np.frombuffer(image_bytes, np.uint8)
        frame = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)

        # Reconnaissance avec face_recognition
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        face_locations = face_recognition.face_locations(rgb_frame)
        face_encodings = face_recognition.face_encodings(rgb_frame, face_locations)

        success = False
        name = None
        audio_url = None
        processed_frame = frame

        for (top, right, bottom, left), encoding in zip(face_locations, face_encodings):
            if known_encoding is not None:
                matches = face_recognition.compare_faces([known_encoding], encoding)
                if matches[0]:
                    success = True
                    name = "Ranim"
                    # Dessiner un cadre avec le nom
                    cv2.rectangle(processed_frame, (left, top), (right, bottom), (0, 255, 0), 2)
                    cv2.putText(processed_frame, name, (left, top - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
                    # Générer le message vocal
                    message = f"Welcome {name}, you can access"
                    tts = gTTS(text=message, lang="en", slow=False)
                    filename = f"welcome_{name.lower()}_{int(time.time())}.mp3"
                    file_path = os.path.join(settings.STATICFILES_DIRS[0], 'audio', filename)
                    os.makedirs(os.path.dirname(file_path), exist_ok=True)
                    print(f"Saving welcome audio to: {file_path}")  # Debug
                    tts.save(file_path)
                    if os.path.exists(file_path):
                        print(f"Welcome audio created: {file_path}")  # Debug
                    else:
                        print(f"Failed to create welcome audio: {file_path}")  # Debug
                    audio_url = settings.STATIC_URL + 'audio/' + filename
                    break

        # Encoder la frame traitée
        _, buffer = cv2.imencode('.jpg', processed_frame)
        processed_base64 = base64.b64encode(buffer).decode('utf-8')

        return JsonResponse({
            'success': success,
            'name': name,
            'processed_image': processed_base64,
            'audio_url': audio_url
        })
    return JsonResponse({'error': 'Méthode non autorisée'}, status=400)

def asmr_page(request):
    return render(request, 'asmr.html')

@csrf_exempt
def process_frame(request):
    global last_emotion, frame_count, fps, fps_time, fps_counter, show_grid, show_scanner, show_mesh

    if request.method == 'POST':
        print("Received POST to /process_frame/")  # Debug
        data = request.body.decode('utf-8')
        image_data = json.loads(data)['image']
        image_bytes = base64.b64decode(image_data)
        np_arr = np.frombuffer(image_bytes, np.uint8)
        frame = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)

        # Traitement avec le système ASMR
        frame = cv2.flip(frame, 1)
        h, w = frame.shape[:2]
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        face_result = face_mesh.process(rgb)

        detector = AdvancedEmotionDetector()
        effects = VisualEffects()
        emotion = "neutre"
        metrics = {}
        face_detected = False

        if face_result.multi_face_landmarks:
            for face_landmarks in face_result.multi_face_landmarks:
                face_detected = True
                if show_mesh:
                    mp_draw.draw_landmarks(
                        frame,
                        face_landmarks,
                        mp_face.FACEMESH_TESSELATION,
                        landmark_drawing_spec=None,
                        connection_drawing_spec=mp_drawing_styles.get_default_face_mesh_tesselation_style()
                    )
                    mp_draw.draw_landmarks(
                        frame,
                        face_landmarks,
                        mp_face.FACEMESH_CONTOURS,
                        landmark_drawing_spec=None,
                        connection_drawing_spec=mp_drawing_styles.get_default_face_mesh_contours_style()
                    )
                emotion, metrics = detector.detect_emotion(face_landmarks)
                landmarks = face_landmarks.landmark
                x_coords = [lm.x * w for lm in landmarks]
                y_coords = [lm.y * h for lm in landmarks]
                bbox_x = int(min(x_coords))
                bbox_y = int(min(y_coords))
                bbox_w = int(max(x_coords) - bbox_x)
                bbox_h = int(max(y_coords) - bbox_y)
                effects.draw_corner_brackets(frame, (bbox_x, bbox_y, bbox_w, bbox_h))
                key_points = [
                    (33, "L_EYE"),
                    (263, "R_EYE"),
                    (1, "NOSE"),
                    (61, "L_MOUTH"),
                    (291, "R_MOUTH")
                ]
                for idx, label in key_points:
                    lm = landmarks[idx]
                    x, y = int(lm.x * w), int(lm.y * h)
                    cv2.circle(frame, (x, y), 3, (0, 255, 255), -1)
                    cv2.circle(frame, (x, y), 8, (0, 255, 255), 1)
        if show_grid:
            effects.draw_grid(frame)
        if show_scanner:
            effects.draw_scanner_line(frame)
        fps_counter += 1
        if time.time() - fps_time >= 1.0:
            fps = fps_counter / (time.time() - fps_time)
            fps_counter = 0
            fps_time = time.time()
        if face_detected:
            effects.draw_hud(frame, emotion, metrics, fps)
        else:
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
            cv2.putText(frame, f"FPS: {fps:.1f}", (20, 40),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        _, buffer = cv2.imencode('.jpg', frame)
        processed_base64 = base64.b64encode(buffer).decode('utf-8')
        if emotion != last_emotion and frame_count % 30 == 0:
            last_emotion = emotion
            message = ASMRMessages.get_message(emotion)
            audio_url = f'/generate_audio/{emotion}/'
            timestamp = datetime.now().strftime("%H:%M:%S")
            print(f"[{timestamp}] 🎭 {emotion.upper()} ({metrics['confidence']:.0%}) → 🎤 {message}")
            return JsonResponse({
                'processed_image': processed_base64,
                'emotion': emotion,
                'audio_url': audio_url,
                'message': message
            })
        frame_count += 1
        return JsonResponse({'processed_image': processed_base64})
    return JsonResponse({'error': 'Méthode non autorisée'}, status=400)

@csrf_exempt
def generate_audio(request, emotion):
    message = ASMRMessages.get_message(emotion)
    slow_speed = emotion in ["triste", "confortable", "fatigué", "stressé"]
    tts = gTTS(text=message, lang="fr", slow=slow_speed)
    filename = f"asmr_{emotion}_{int(time.time())}.mp3"
    file_path = os.path.join(settings.STATICFILES_DIRS[0], 'audio', filename)
    os.makedirs(os.path.dirname(file_path), exist_ok=True)
    print(f"Saving audio to: {file_path}")  # Debug
    tts.save(file_path)
    if os.path.exists(file_path):
        print(f"Audio file created: {file_path}")  # Debug
    else:
        print(f"Failed to create audio file: {file_path}")  # Debug
    audio_url = settings.STATIC_URL + 'audio/' + filename
    print(f"Returning audio URL: {audio_url}")  # Debug
    return JsonResponse({'audio_url': audio_url})