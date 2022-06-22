# Classificador de placas de trânsito

Para configurar o nosso projeto, você terá que entrar no site do nosso [dataset](https://www.kaggle.com/datasets/meowmeowmeowmeowmeow/gtsrb-german-traffic-sign?select=Meta.csv), 
baixá-lo e extrair tudo que na source do projeto.

Há um modelo pré-treinado que está no projeto chamado model.h5, você poderá utilizà-lo para realizar a classificação.

Caso queiras treinar seu próprio modelo, uma vez configurado, basta executar o arquivo main.py que o treinamento começará.

Para utilizar o modelo treinado para a classificação basta executar o arquivo predictionTest.py. Lembre-se apenas de modificar o path da imagem para a qual você queira classificar - nesta linha:
```py
image = cv2.imread("image.png")
```
