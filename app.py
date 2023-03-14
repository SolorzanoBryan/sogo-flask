from flask import Flask, render_template, Response
import cv2
import numpy as np
import math
import time

# PARA EL RECONOCIMEINTO DE LAS MANOS
from cvzone.HandTrackingModule import HandDetector
from cvzone.ClassificationModule import Classifier 

app = Flask(__name__)

camera = cv2.VideoCapture(0)
detector = HandDetector(maxHands=1)
Classifier = Classifier("model/keras_model.h5", "model/labels.txt")
offset = 20
imgSize = 300

folder = "data/C"
# counter = 0
labels = ["A", "I"]

def gen_frames():
    frameWhite = ''   
    counter = 0
     
    while True:
        '''Frame: viene hacer nuestra matriz de imagen principal'''
        success, frame = camera.read()  # read the camera frame
        # imgOutput = frame.copy()
        hands, frame = detector.findHands(frame)
        if success:
            '''VAMOS A COMPROBAR SI EXISTE ALGO EN LAS MANOS'''
            if hands:
                '''Vamos a obtener la información del recuadro marcado'''
                hand = hands[0]
                x, y, w, h = hand['bbox'] # Nos de los los datos del cuadro delimitador
                '''Ya con los datos es posible recortar la imagen'''
                '''Para solucionar el problema de que la imagen peude salir muy larga o ancha se creau una nueva matriz con un fondo especial (white'''
                '''Se be asignar los tipos de valores, de lo contrario no vera los colores correctos '''
                '''El np.uint8: especifica que va de 0 a 155'''
                frameWhite = np.ones((imgSize, imgSize, 3), np.uint8)*255 # El 3 para que sea de colores y se multiplica para 255 para que el fonfo sea blanco
                # cv2.imshow("ImageW", frameWhite)
                
                '''Preparamos la imagen a capturar y se debe especificar esas dimensiones'''
                '''Se usara la imagen principal para capturar las dimenciones por medio de los parametros antes recopilados (x, y, w, h)'''
                frameCapture = frame[y-offset:y + h + offset, x - offset:x + w + offset]
                # cv2.imshow("ImageCap", frameCapture)
                
                frameShape = frameCapture.shape
                
                aspectRatio = h / w
                
                if(aspectRatio > 1):
                    k = imgSize/h
                    wCal = math.ceil(k*w)
                    imgResize = cv2.resize(frameCapture, (wCal, imgSize))
                    imgResizeShape = imgResize.shape
                    wGap = math.ceil((imgSize-wCal) / 2)
                    frameWhite[: , wGap:wCal+wGap, :] = imgResize
                    # frameWhite[0: imgResizeShape[0], 0: imgResizeShape[1]] = imgResize
                    prediction, index = Classifier.getPrediction(frameWhite, draw=False)
                    print(prediction, index)
                    
                else:
                    k = imgSize / w
                    hCal = math.ceil(k*h)
                    imgResize = cv2.resize(frameCapture, (imgSize, hCal))
                    imgResizeShape = imgResize.shape
                    hGap = math.ceil((imgSize-hCal) / 2)
                    frameWhite[hGap:hCal + hGap , :] = imgResize
                    prediction, index = Classifier.getPrediction(frameWhite, draw=False)
                    
                cv2.putText(frame, labels[index], (x, y - 50), cv2.FONT_HERSHEY_COMPLEX, 2, (0, 255, 255), 2)
                
                cv2.imshow("ImageW", frameWhite)
                key = cv2.waitKey(1)
                if key == ord("s"):
                    counter += 1
                    cv2.imwrite(f'{folder}/image_{time.time()}.jpg', frameWhite)
                    print(counter)
                # cv2.imshow("ImageC", frameCapture)
            
        # print('HOLA')
        # cv2.imshow("Image", frame)
        cv2.waitKey(1)
        key = cv2.waitKey(1)
        ret, buffer = cv2.imencode('.jpg', frame) # Pra ocmprimir formatos de imagenes para facilitar la tranfercencia por red al ser allacenadas en cache
        frame = buffer.tobytes() # Convertir una imagen en cadena de caracteres
        
        yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')
        
        
        
@app.route('/')
def index():
    return render_template('index.html')

@app.route('/video_feed')
def video_feed():
    return Response(gen_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

if __name__== '__main__':
    app.run(debug=True)
