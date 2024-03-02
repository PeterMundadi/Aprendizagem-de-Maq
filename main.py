from function import *


#Pastas contento os espectrogramas
caminho_1 = "Spectrogramas/f1/fold1"
caminho_2 = "Spectrogramas/f2/fold2"
caminho_3 = "Spectrogramas/f3/fold3"

#Lista de imagens
Fold1=[]
Fold2=[]
Fold3=[]

#Listas para cada estilo
S1=[]#1-30
S2=[]#31-60
S3=[]#61-90
S4=[]#91-120
S5=[]#121-150
S6=[]#151-180
S7=[]#181-210
S8=[]#211-240
S9=[]#241-270
S10=[]#271-300


#Acessa os arquivos de um diretório e recuperação das imagens
Fold1 = files_access(caminho_1,"fold1")
Fold2 = files_access(caminho_2,"fold2")
Fold3 = files_access(caminho_3,"fold3")

#==== Pré-processamento ====
print("--- iniciando pré-processamento ---")
S1_N, S2_N, S3_N, S4_N, S5_N, S6_N, S7_N, S8_N, S9_N, S10_N =preprocess(Fold1, Fold2, Fold3)
print("--- fim do pré-processamento ---")


#===== Divisão dos dados em treino e teste ====
X = S1_N + S2_N + S3_N + S4_N + S5_N + S6_N + S7_N + S8_N + S9_N + S10_N
interv = 90

# # Defina os rótulos para os estilos musicais
labels_musicais = ["Jazz", "Reggae", "Salsa", "Samba", "Tango", "Merengue", "Cumbia", "Bossa Nova", "Mambo", "Bachata"]

# # Divida os dados em treino e teste
Espc_train, Espc_test, Lab_train, Lab_test = data_div(X,interv,labels_musicais)


#==== Treinamento e Teste ====
print("--- iniciando treinamento ---")

X_train_flat = Espc_train.reshape(Espc_train.shape[0], -1)
X_test_flat = Espc_test.reshape(Espc_test.shape[0], -1)

Trein_knn(X_train_flat,X_test_flat, Lab_train, Lab_test)
Trein_svm(X_train_flat,X_test_flat, Lab_train, Lab_test)
Trein_arvore(X_train_flat,X_test_flat, Lab_train, Lab_test)
Trein_bayes(X_train_flat,X_test_flat, Lab_train, Lab_test)
Trein_random_forest(X_train_flat,X_test_flat, Lab_train, Lab_test)
print("--- fim do treinamento ---")

