import cv2
import os
import numpy as np
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier


#Apagador


#Acessa os arquivos de um diretório e recuperação das imagens
def files_access(caminho,diretorio):
    lst_img = []

    for img in os.listdir(caminho):

        file = os.path.join(caminho, img)

        img_cinza = cv2.imread(file,0)
        lst_img.append(img_cinza)
        #break

    # Mostra os arquivos encontrados
    #print("Quantidade de imagens no diretório "+diretorio+":")
    #print(len(lst_img))
    #for img in lst_img:
        #print(img)
        #print(type(img))
    return lst_img

#Normalização dos espectrogramas
def normalize_spectrograms(estilo):
    norm_specs =[]

    for spectrogram in estilo:
        max_value = np.max(spectrogram)
        normalized_img = spectrogram / max_value
        norm_specs.append(normalized_img)
    return norm_specs



#Repartição das imagens por estilo
def Style_repartirion(Fold, S1, S2, S3, S4, S5, S6, S7, S8, S9, S10):
   
    for i in range (300):

        if i<=30:
            S1.append(Fold[i])

        elif i>30 and i<=60:
            S2.append(Fold[i])

        elif i>60 and i<=90:
            S3.append(Fold[i])

        elif i>90 and i<=120:
            S4.append(Fold[i])

        elif i>120 and i<=150:
            S5.append(Fold[i])

        elif i>150 and i<=180:
            S6.append(Fold[i])

        elif i>180 and i<=210:
            S7.append(Fold[i])

        elif i>210 and i<=240:
            S8.append(Fold[i])

        elif i>240 and i<=270:
            S9.append(Fold[i])

        elif i>270 and i<=300: 
            S10.append(Fold[i])

        else:
            print("Erro na repartição das imagens")
            
    return S1, S2, S3, S4, S5, S6, S7, S8, S9, S10

#=========== Caracteristicas ============
#Media Espectral (distribuição geral de energia em diferentes partes do espectro)
def spectral_mean(spectrogram):
   
    return np.mean(spectrogram, axis=1)

#Centróide Espectral(capturar a localização central das frequências dominantes em um espectrograma, o que pode ser útil para distinguir estilos com diferentes distribuições de energia em diferentes frequências)
def spectral_centroid(spectrogram, freq_bins):
    return np.sum(freq_bins * np.sum(spectrogram, axis=1)) / np.sum(np.sum(spectrogram, axis=1))

#Rollof Espectral(capturar a distribuição de energia em diferentes partes do espectro e identificar frequências de destaque em diferentes estilos musicais)
def spectral_rolloff(spectrogram, roll_percent=0.85):
    total_energy = np.sum(spectrogram)
    cumulative_energy = np.cumsum(spectrogram, axis=0)
    rolloff_freq = np.argmax(cumulative_energy >= roll_percent * total_energy, axis=0)
    return rolloff_freq

#Banda de freq dominante (as frequências mais proeminentes em um espectrograma)
def dominant_frequency_band(spectrogram):
    taxa_amost = 44100
    n_fft = 1024 #número de pontos na FFT (Transformada Rápida de Fourier

    resolucao = taxa_amost / n_fft
    freq_bins = np.arange(0, n_fft // 2 + 1) * resolucao

    max_amplitude_idx = np.argmax(np.max(spectrogram, axis=1))
    return freq_bins[max_amplitude_idx]

#Contraste Espectral (Mede a diferença de energia entre regiões adjacentes no espectrograma)
def spectral_contrast(spectrogram):
    spectral_contrast = np.diff(spectrogram, axis=0)
    return spectral_contrast

#Normalização das caracteristicas
def normalize_features(features):
       
    # Calcule o máximo e o mínimo de cada coluna
    max_values = np.max(features, axis=0)
    min_values = np.min(features, axis=0)
    
    # Normalize as características para o intervalo [0, 1]
    normalized_features = (features - min_values) / (max_values - min_values)
    
    return normalized_features

#Extrator Geral
def general_extractor(estilo):

    carac_estilo = []
   
    for img in estilo:

        #Extração das caracteristicas
        med_spc = np.array(spectral_mean(img))#media espectral
        ctr_spc = np.array(spectral_centroid(img, 1))#centroide espectral
        rol_spc = np.array(spectral_rolloff(img, 0.85))#rollof espectral
        dom_spc = np.array(dominant_frequency_band(img))#banda de freq dominante
        con_spc = np.array(spectral_contrast(img))#contraste espectral

        #Normalização das caracteristicas
        med_spc_N=normalize_features(med_spc)
        ctr_spc_N=normalize_features(ctr_spc)
        rol_spc_N=normalize_features(rol_spc)
        dom_spc_N=normalize_features(dom_spc)
        con_spc_N=normalize_features(con_spc)

        #tupla representando as caracteristicas normalizadas da imagem
        img_carac = (med_spc, ctr_spc, rol_spc, dom_spc, con_spc)
            
        carac_estilo.append(img_carac)
    
    
    return carac_estilo





#Pré-processamento
def preprocess(F1,F2,F3): #Retorna lista com as caracteristicas de cada estilo Normalizado


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

   
    #Repartição dos espectrogramas por 
    
    Style_repartirion(F1, S1, S2, S3, S4, S5, S6, S7, S8, S9, S10)
    Style_repartirion(F2, S1, S2, S3, S4, S5, S6, S7, S8, S9, S10)
    Style_repartirion(F3, S1, S2, S3, S4, S5, S6, S7, S8, S9, S10)


    #Extração das caracteristicas de cada estilo de S1 a S10
    S1_Norm = general_extractor(S1)
    S2_Norm = general_extractor(S2)
    S3_Norm = general_extractor(S3)
    S4_Norm = general_extractor(S4)
    S5_Norm = general_extractor(S5)
    S6_Norm = general_extractor(S6)
    S7_Norm = general_extractor(S7)
    S8_Norm = general_extractor(S8)
    S9_Norm = general_extractor(S9)
    S10_Norm = general_extractor(S10)

    return S1_Norm, S2_Norm, S3_Norm, S4_Norm, S5_Norm, S6_Norm, S7_Norm, S8_Norm, S9_Norm, S10_Norm


#Crição dos labels e divisão dos dados em treino e teste

def data_div(X,interv,labels_musicais):

    # Inicialize a lista de rótulos
    labels = []

    # Atribua um rótulo para cada conjunto contíguo de espectrogramas
    for i in range(len(X)):
        label = labels_musicais[i // interv]  # Use a divisão inteira para atribuir o mesmo rótulo para cada conjunto de 90 espectrogramas
        labels.append(label)
    
    #Embaralhamento dos espectrogramas e rótulos juntos para manter a correspondência entre eles
    img, labels = np.array(X), np.array(labels)
    indices_embaralhados = np.random.permutation(len(X))
    espectrogramas_embaralhados = X[indices_embaralhados]
    labels_embaralhados = labels[indices_embaralhados]


    

    # Dividir os dados em conjuntos de treinamento e teste (80% para treinamento e 20% para teste)
    X_train, X_test, y_train, y_test = train_test_split(espectrogramas_embaralhados, labels_embaralhados, test_size=0.2, random_state=42)

    # X_train: conjunto de treinamento de espectrogramas
    # X_test: conjunto de teste de espectrogramas
    # y_train: rótulos correspondentes ao conjunto de treinamento
    # y_test: rótulos correspondentes ao conjunto de teste

    
    return X_train, X_test, y_train, y_test


#======= Treinamento
#KNN
def Trein_knn(X_train, X_test, y_train, y_test):
   
    # Inicialize o classificador KNN
    knn = KNeighborsClassifier(n_neighbors=5)

    # Treine o classificador KNN
    knn.fit(X_train, y_train)

    # Faça previsões para o conjunto de teste
    y_pred = knn.predict(X_test)

    # Calcule a precisão do classificador KNN
    knn_accuracy = accuracy_score(y_test, y_pred)
    print("Precisão do classificador KNN: {:.2f}%".format(knn_accuracy * 100))

    #return knn_accuracy

#SVM
def Trein_svm(X_train, X_test, y_train, y_test):
    
    # Inicializar o classificador SVM
    svm_clf = SVC(kernel='linear', random_state=42)  # Aqui estamos usando um kernel linear para simplicidade, mas você pode experimentar com outros kernels

    # Treinar o classificador nos dados de treinamento
    svm_clf.fit(X_train, y_train)

    # Prever os rótulos para os dados de teste
    y_pred_svm = svm_clf.predict(X_test)

    # Calcular a acurácia do modelo SVM
    accuracy_svm = accuracy_score(y_test, y_pred_svm)
    print("Acurácia do modelo SVM:", accuracy_svm)

    #return accuracy_svm

#Arvore de Decisão
def Trein_arvore(X_train, X_test, y_train, y_test):
    # Inicialize o classificador de árvore de decisão
    tree_clf = DecisionTreeClassifier(random_state=42)

    # Treine o classificador nos dados de treinamento
    tree_clf.fit(X_train, y_train)

    # Prever os rótulos para os dados de teste
    y_pred_tree = tree_clf.predict(X_test)

    # Calcular a acurácia do modelo de árvore de decisão
    accuracy_tree = accuracy_score(y_test, y_pred_tree)
    print("Acurácia do modelo de árvore de decisão:", accuracy_tree)

    #return accuracy_tree



