
import tensorflow as tf
import numpy as np
import cv2
import random
import time
import tkinter as tk
from tkinter import simpledialog, messagebox
from collections import Counter

modelo_path = "C:/Users/luisa/Downloads/converted_savedmodel/model.savedmodel"
labels_path = "C:/Users/luisa/Downloads/converted_savedmodel/labels.txt"

model = tf.saved_model.load(modelo_path)
print("✅ Modelo cargado correctamente (SavedModel).")

with open(labels_path, "r") as f:
    labels = [line.strip().split(' ', 1)[1] for line in f.readlines()]

# Filtrar etiquetas válidas
labels_validas = [l for l in labels if l not in ["Fondo", "Nada"]]
if not labels_validas:
    raise ValueError("❌ No se encontraron etiquetas válidas. Revisa tu archivo labels.txt")
print("✅ Etiquetas válidas:", labels_validas)

infer = model.signatures["serving_default"]

# ==========================================
# 3️⃣ FUNCIONES DE APOYO
# ==========================================
def preprocess_image(img):
    """Redimensiona y normaliza la imagen para el modelo"""
    img_resized = cv2.resize(img, (224, 224))
    img_array = np.expand_dims(img_resized.astype(np.float32), axis=0)
    img_array = (img_array / 127.5) - 1.0
    return img_array

def obtener_resultado(jugador, computadora):
    """Determina el resultado del juego"""
    if jugador == computadora:
        return "Empate"
    elif (jugador == "Piedra" and computadora == "Tijera") or \
         (jugador == "Papel" and computadora == "Piedra") or \
         (jugador == "Tijera" and computadora == "Papel"):
        return "Ganaste"
    else:
        return "Perdiste"

# ==========================================
# 4️⃣ FUNCION PRINCIPAL DEL JUEGO (SIN CAMARA VISIBLE)
# ==========================================
def iniciar_juego(puntaje):
    cap = cv2.VideoCapture(0)
    UMBRAL = 0.5  # bajamos el umbral para no perder gestos
    jugando = True

    while jugando:
        jugador = "Nada"

        # Mensaje de inicio de ronda
        messagebox.showinfo("Nueva Ronda", "Cuando cierres este mensaje, se comenzará a detectar tu jugada.")

        predicciones_temp = []

        # Tomamos varias capturas mientras presionas la mano
        start_time = time.time()
        while time.time() - start_time < 5:
            ret, frame = cap.read()
            if not ret:
                messagebox.showerror("Error", "No se pudo acceder a la cámara.")
                jugando = False
                break

            roi = frame[100:400, 100:400]
            img_array = preprocess_image(roi)
            input_tensor = tf.convert_to_tensor(img_array, dtype=tf.float32)
            output = infer(tf.constant(input_tensor))
            predicciones = list(output.values())[0].numpy()[0]

            # Mostrar probabilidades de todas las clases
            prob_dict = dict(zip(labels, predicciones))
            print(f"Probabilidades: {prob_dict}")

            index = np.argmax(predicciones)
            prob = np.max(predicciones)
            clase = labels[index]

            if clase in labels_validas and prob >= UMBRAL:
                predicciones_temp.append(clase)

            if cv2.waitKey(1) & 0xFF == 32:  # barra espaciadora para salir
                jugando = False
                break

        if not jugando:
            break

        if predicciones_temp:
            jugador = Counter(predicciones_temp).most_common(1)[0][0]
        else:
            messagebox.showinfo("Aviso", "No se detectó un gesto válido. Intenta de nuevo.")
            continue

        # Elección de la computadora (solo entre etiquetas válidas)
        computadora = random.choice(labels_validas)
        resultado = obtener_resultado(jugador, computadora)

        mensaje = f"Tú: {jugador}\nCPU: {computadora}\nResultado: {resultado}"
        messagebox.showinfo("Resultado", mensaje)

        if resultado == "Ganaste":
            puntaje["ganadas"] += 1
        elif resultado == "Perdiste":
            puntaje["perdidas"] += 1

        salir = messagebox.askyesno("Continuar", "¿Deseas jugar otra ronda? (No = volver al menú)")
        if not salir:
            jugando = False

    cap.release()
    cv2.destroyAllWindows()
    return puntaje

# ==========================================
# 5️⃣ FUNCIÓN MENÚ CON MESSAGEBOX
# ==========================================
def mostrar_menu():
    puntaje = {"ganadas": 0, "perdidas": 0}
    root = tk.Tk()
    root.withdraw()

    while True:
        opcion = simpledialog.askstring("Menú Piedra, Papel o Tijera",
                                        "Selecciona una opción:\n1️⃣ Iniciar juego\n2️⃣ Ver puntaje\n3️⃣ Salir")
        if opcion == "1":
            puntaje = iniciar_juego(puntaje)
        elif opcion == "2":
            messagebox.showinfo("Puntaje",
                                f"Partidas ganadas: {puntaje['ganadas']}\nPartidas perdidas: {puntaje['perdidas']}")
        elif opcion == "3":
            messagebox.showinfo("Salir", "¡Gracias por jugar!")
            break
        else:
            messagebox.showwarning("Error", "Opción no válida. Intenta nuevamente.")

# ==========================================
# 6️⃣ INICIO DEL PROGRAMA
# ==========================================
if __name__ == "__main__":
    mostrar_menu()

