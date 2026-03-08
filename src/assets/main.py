"""
YOLOv11 Motor Control - Hauptprogramm
Automatische Motorsteuerung basierend auf Objekterkennung
"""

import cv2
import numpy as np
import time
import sys

from video_stream import VideoStream
from motor_controller import MotorController
from object_tracker import ObjectTracker
from yolo_detector import YOLOv11TRT
from utils import get_position_description, calculate_motor_speeds, find_lowest_object


def main():
    """Hauptprogramm"""
    
    # Konfiguration
    ENGINE_PATH = "best.engine"
    CAMERA_ID = 0
    
    class_names = ["Muell"]
    
    # YOLOv11 TRT Modell laden
    print("="*80)
    print("SYSTEM-INITIALISIERUNG")
    print("="*80)
    print("\n1. Lade TensorRT Engine...")
    try:
        yolo = YOLOv11TRT(ENGINE_PATH, conf_thres=0.25, iou_thres=0.45)
        print("✓ Engine erfolgreich geladen!\n")
    except Exception as e:
        print(f"✗ Fehler beim Laden der Engine: {e}")
        return
    
    # Object Tracker initialisieren
    print("2. Initialisiere Object Tracker...")
    tracker = ObjectTracker(
        max_age=20,
        min_hits=3,
        iou_threshold=0.15,
        max_distance=300
    )
    print("✓ Object Tracker bereit\n")
    
    # Motor-Controller initialisieren
    print("3. Initialisiere Motor-Controller...")
    try:
        motor = MotorController(
            address=0x40, 
            busnum=7,
            max_acceleration=25.0  # 25%/s Beschleunigung (sehr sanft)
        )
        print("✓ Motor-Controller bereit!\n")
    except Exception as e:
        print(f"✗ Motor-Controller Fehler: {e}")
        print("⚠ Programm läuft ohne Motorsteuerung weiter...\n")
        motor = None
    
    # Kamera starten
    print("4. Starte Kamera-Stream...")
    vs = VideoStream(CAMERA_ID).start()
    time.sleep(1.0)
    
    width, height, fps_cap = vs.get_info()
    print(f"✓ Kamera: {width}x{height} @ {fps_cap} FPS\n")
    
    print("="*80)
    print("SYSTEM BEREIT - Drücke 'q' zum Beenden")
    print("="*80)
    print()
    
    # Display Window
    window_name = "YOLOv11 Motor Control"
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
    
    # FPS Berechnung
    fps = 0
    frame_count = 0
    start_time = time.time()
    
    # Tracking-State
    selected_track_id = None
    
    # Timing-Statistiken
    inference_times = []
    postprocess_times = []
    
    try:
        while True:
            ret, frame = vs.read()
            
            if not ret:
                print("Fehler beim Lesen des Frames")
                break
            
            frame_h, frame_w = frame.shape[:2]
            
            # Inferenz
            inference_start = time.time()
            output = yolo.infer(frame)
            inference_time = (time.time() - inference_start) * 1000
            inference_times.append(inference_time)
            
            # Postprocessing
            post_start = time.time()
            boxes, scores, class_ids = yolo.postprocess(output, frame.shape)
            post_time = (time.time() - post_start) * 1000
            postprocess_times.append(post_time)
            
            # Object Tracking
            tracked_objects = tracker.update(boxes, scores, class_ids)
            
            # Finde unterstes Objekt
            lowest = find_lowest_object(tracked_objects)
            
            if lowest:
                target_id, target_box = lowest
                
                # Wenn noch kein Objekt ausgewählt oder altes nicht mehr sichtbar
                if selected_track_id is None or \
                   not any(tid == selected_track_id for tid, _, _, _ in tracked_objects):
                    selected_track_id = target_id
                    print(f"\n>>> Neues Ziel-Objekt ausgewählt: ID #{selected_track_id}")
                
                # Finde das ausgewählte Objekt
                selected_object = None
                for tid, box, score, cid in tracked_objects:
                    if tid == selected_track_id:
                        selected_object = (tid, box, score, cid)
                        break
                
                if selected_object:
                    tid, box, score, cid = selected_object
                    h_pos, v_pos = get_position_description(box, frame_w, frame_h)
                    
                    # Motor-Geschwindigkeiten berechnen
                    left_speed, right_speed = calculate_motor_speeds(h_pos)
                    
                    # Motoren steuern (nicht-blockierend!)
                    if motor:
                        motor.set_target_speeds(left_speed, right_speed)
                    
                    # Status ausgeben
                    print(f"\r>>> Ziel ID #{tid}: {h_pos:12s} / {v_pos:6s} | "
                          f"Motoren: L={left_speed:4d}% R={right_speed:4d}% | "
                          f"FPS: {fps:5.1f}", end="", flush=True)
                    
                    # Ziel-Box hervorheben (Gelb)
                    x, y, w, h = box
                    cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 255), 5)
                    
                    label = f"ZIEL #{tid}: {h_pos}/{v_pos} | L:{left_speed}% R:{right_speed}%"
                    (label_w, label_h), _ = cv2.getTextSize(label, 
                                                             cv2.FONT_HERSHEY_SIMPLEX, 
                                                             0.8, 2)
                    cv2.rectangle(frame, (x, y-label_h-20), (x+label_w+10, y), (0, 255, 255), -1)
                    cv2.putText(frame, label, (x+5, y-10), 
                               cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 0), 2)
            else:
                # Kein Objekt erkannt - Motoren stoppen
                if motor:
                    motor.set_target_speeds(0, 0)
                selected_track_id = None
                print(f"\r>>> Kein Objekt | Motoren: STOP | FPS: {fps:5.1f}     ", end="", flush=True)
            
            # Motor-Update (nicht-blockierendes Ramping)
            if motor:
                motor.update()
            
            # Alle getrackte Objekte zeichnen
            for track_id, box, score, class_id in tracked_objects:
                if track_id == selected_track_id:
                    continue  # Ziel-Objekt bereits gezeichnet
                
                x, y, w, h = box
                
                if w <= 0 or h <= 0:
                    continue
                
                # Farbe für normale Objekte (Grün)
                color = (0, 255, 0)
                
                cv2.rectangle(frame, (x, y), (x+w, y+h), color, 2)
                
                h_pos, v_pos = get_position_description(box, frame_w, frame_h)
                label = f"ID#{track_id} ({h_pos}/{v_pos})"
                (label_w, label_h), _ = cv2.getTextSize(label, 
                                                         cv2.FONT_HERSHEY_SIMPLEX, 
                                                         0.5, 1)
                cv2.rectangle(frame, (x, y-label_h-10), (x+label_w+5, y), color, -1)
                cv2.putText(frame, label, (x+2, y-5), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
            
            # FPS Info auf Frame
            fps_text = f"FPS: {fps:.1f} | Objekte: {len(tracked_objects)}"
            if selected_track_id:
                fps_text += f" | ZIEL: #{selected_track_id}"
            cv2.putText(frame, fps_text, (10, 30), 
                       cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 255), 2)
            
            # FPS berechnen
            frame_count += 1
            elapsed = time.time() - start_time
            if elapsed >= 1.0:
                fps = frame_count / elapsed
                frame_count = 0
                start_time = time.time()
            
            # Display
            cv2.imshow(window_name, frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
    
    except KeyboardInterrupt:
        print("\n\n⚠ Unterbrochen durch Benutzer")
    
    finally:
        # Aufräumen
        print("\n\n" + "="*80)
        print("SYSTEM-SHUTDOWN")
        print("="*80)
        
        if motor:
            print("Stoppe Motoren...")
            motor.emergency_stop()
            print("✓ Motoren gestoppt")
        
        print("Stoppe Kamera...")
        vs.stop()
        print("✓ Kamera gestoppt")
        
        cv2.destroyAllWindows()
        print("✓ Display geschlossen")
        
        # Finale Statistiken
        if inference_times:
            print("\n" + "="*80)
            print("PERFORMANCE-STATISTIKEN")
            print("="*80)
            print(f"Durchschnittliche Inferenz-Zeit: {np.mean(inference_times):.2f}ms")
            print(f"Durchschnittliche Postprocess-Zeit: {np.mean(postprocess_times):.2f}ms")
            total_time = np.mean(inference_times) + np.mean(postprocess_times)
            print(f"Durchschnittliche Gesamt-Zeit: {total_time:.2f}ms")
            print(f"Theoretische maximale FPS: {1000.0 / total_time:.2f}")
            print("="*80)
        
        print("\n✓ Shutdown abgeschlossen\n")


if __name__ == "__main__":
    main()
