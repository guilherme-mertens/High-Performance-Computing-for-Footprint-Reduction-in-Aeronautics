import cv2
import numpy as np
import pandas as pd

# Carrega a imagem
imagem = cv2.imread('final/airplane_2d/concorde_broken_nose.png')

# Converte a imagem para escala de cinza
imagem_cinza = cv2.cvtColor(imagem, cv2.COLOR_BGR2GRAY)

# Encontra os contornos na imagem
contornos, _ = cv2.findContours(imagem_cinza, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

# Desenha os contornos na imagem original
imagem_contornos = imagem.copy()
cv2.drawContours(imagem_contornos, contornos, -1, (0, 255, 0), 3)



# cv2.imshow("a", imagem_contornos)
# cv2.waitKey(0)


# Inicializa listas para armazenar os pontos de contorno
pontos_contorno_x = []
pontos_contorno_y = []

for contorno in contornos:
    # Aproxima o contorno
    epsilon = 0.0005 * cv2.arcLength(contorno, True)
    aprox_contorno = cv2.approxPolyDP(contorno, epsilon, True)
    
    # Adiciona os pontos do contorno suavizado à lista
    for ponto in aprox_contorno:
        x, y = ponto[0]
        pontos_contorno_x.append(x)
        pontos_contorno_y.append(y)

dados_contorno = pd.DataFrame({'X': pontos_contorno_x, 'Y': pontos_contorno_y})

# Salva o DataFrame em um arquivo CSV
dados_contorno.to_csv('final/airplane_2d/concorde_broken_nose.csv')