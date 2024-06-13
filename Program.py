import cv2
import numpy as np 

video_path = 'Video2.mp4'
cap = cv2.VideoCapture(video_path)#apertura de mi video 

cv2.namedWindow('Ventana Proyecto', cv2.WINDOW_NORMAL)
cv2.resizeWindow('Ventana Proyecto', 1280, 720)#Añadi el tamaño de la venta ya que con hstack lo que hace es 
#"unirme" los frames y quedaba una ventana gigante

verde1 = np.array([35, 80, 80])
verde2 = np.array([90, 255, 255])#Los rangos que utilizare para hacer mi mascara

ret, primer_frame = cap.read()
frame_anterior = cv2.cvtColor(primer_frame, cv2.COLOR_BGR2GRAY)
frame_anterior = np.zeros_like(frame_anterior)#leer mi primer frame :)

while cap.isOpened():
    ret, frame = cap.read()

    if not ret:
        break
    
    #1. Segmentacion del color:
    video_hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

    mascara = cv2.inRange(video_hsv, verde1, verde2)

    mascara_invertida = cv2.bitwise_not(mascara)

    resultado = cv2.bitwise_and(frame, frame, mask=mascara_invertida) #Aplique el mismo modelo que vimos 
    #en el notebook 3 (Segmentacion del color), ejercicio 6

    #2. Deteccion de movimiento con absdiff
    frame_gris = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    diferencia = cv2.absdiff(frame_gris, frame_anterior)
    frame_anterior = frame_gris.copy()#utilice la misma sintaxis que enseñaste en el notebook 4 en 
    #deteccion de movimiento

    diferencia_color = cv2.cvtColor(diferencia, cv2.COLOR_GRAY2BGR) #Esto lo hago con el fin de poder mostrar 
    #usando hstack ya que para que funcionen los frames deben de ser iguales 

    #3Detectar movimiento en area especifica
    movimiento = diferencia_color.copy()
    zona = movimiento[524:652, 504:664]

    if np.any(zona):
        cv2.putText(movimiento, 'Movimiento Detectado', (504, 490), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

    cv2.rectangle(movimiento, (504,524), (664,652), (0, 0, 255), 3)

    #4.Mostrar mis frames (me ayude de chatgpt para hacer esto ya que al ser video pyplot no funciona )
    final = np.hstack((frame, resultado))
    final2 = np.hstack((diferencia_color, movimiento))

    final_frame = np.vstack((final, final2))

    cv2.imshow("Ventana Proyecto", final_frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()

