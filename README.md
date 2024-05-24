
Desenvolvi este projeto em Python utilizando bibliotecas como OpenCV, NumPy, TensorFlow e MediaPipe. Ele é capaz de reconhecer letras do alfabeto por meio da detecção de gestos de mão em imagens e vídeos. Comecei carregando imagens de referência das letras A, B e C e as pré-processando para uso no treinamento do modelo. Utilizei TensorFlow para criar e treinar uma rede neural convolucional que pudesse reconhecer essas letras.

Para a detecção de mãos, empreguei a biblioteca MediaPipe, que oferece um detector de mãos eficiente. Durante a execução do script, a câmera captura vídeo e, em cada quadro, o sistema detecta mãos, pré-processa a imagem e classifica a letra correspondente ao gesto da mão utilizando o modelo treinado.
