import cv2
import os
import time
import numpy as np
from function import *
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from skimage.feature import graycomatrix, graycoprops
from sklearn.ensemble import RandomForestClassifier



#Apagador


#Acessa os arquivos de um diretório e recuperação das imagens
def files_access(caminho,diretorio):
    # check for diretorio + ".npy"
    if os.path.exists(diretorio + ".npy"):
        print("--- Retrieving spectrum from file ...")
        print("Arquivo "+diretorio+".npy"+" encontrado")
        return np.load(diretorio + ".npy", allow_pickle=True)

    print("--- Retrieving spectrum for the first time ...")
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
    
    # salva dados em um arquivo
    np.save(diretorio + ".npy", lst_img)

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

        if i<30:
            S1.append(Fold[i])

        elif i>=30 and i<60:
            S2.append(Fold[i])
            
        elif i>=60 and i<90:
            S3.append(Fold[i])

        elif i>=90 and i<120:
            S4.append(Fold[i])

        elif i>=120 and i<150:
            S5.append(Fold[i])

        elif i>=150 and i<180:
            S6.append(Fold[i])

        elif i>=180 and i<210:
            S7.append(Fold[i])

        elif i>=210 and i<240:
            S8.append(Fold[i])

        elif i>=240 and i<270:
            S9.append(Fold[i])

        elif i>=270 and i<300:
            S10.append(Fold[i])

        else:
            print("Erro na repartição dos estilos")

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

    # Normalizar as características, cuidando com a divisão por zero
    normalized_features = (features - min_values) / (max_values - min_values + 1e-6)
    return normalized_features

#Funçaõ para ajustar o tamanho de uma lista para um tamanho desejado
def tam_ajust(entrada, tamanho_desejado):
    C = []

    #Verifica se a entrada é uma lista, float ou int 
    if (type(entrada) == float) or (type(entrada) == int):
        C=[entrada]
    elif type(entrada) == list:
        C=entrada

    valor_padrao=0
    tamanho_atual = len(C)

    if tamanho_atual == tamanho_desejado:
        return C
   
    else:
        elementos_adicionais = tamanho_desejado - tamanho_atual
        return C + [valor_padrao] * elementos_adicionais


#Extrator Geral
counter = 0
path_to_preprocessed = "preprocessed/"
def general_extractor(estilo, name):
    # check if txt already exists
    #if os.path.exists(path_to_preprocessed + name + "_carac.txt"):
        #print("Caracteristicas do estilo " + name + " already extracted")
        #return
    
    # Definir a distância e o ângulo para calcular a GLCM
    distances = [1, 2, 3]  # Distâncias para considerar na matriz GLCM
    angles = [0, np.pi/4, np.pi/2, 3*np.pi/4]  # Ângulos para considerar na matriz GLCM

    
    carac_estilo = []
    

    # create an empty file to append the features
    #open(path_to_preprocessed + name + "_carac.txt", "w").close()
    
    global counter
    init_time = time.time()
    for img in estilo:

        # Calcular a GLCM para cada distância e ângulo
        glcm = graycomatrix(img, distances=distances, angles=angles, levels=256, symmetric=True, normed=True)

                
        # Calcular as propriedades da GLCM (descritores de textura)
        contrast = graycoprops(glcm, 'contrast')
        dissimilarity = graycoprops(glcm, 'dissimilarity')
        homogeneity = graycoprops(glcm, 'homogeneity')
        energy = graycoprops(glcm, 'energy')
        correlation = graycoprops(glcm, 'correlation')
        ASM=graycoprops(glcm, 'ASM')
        
        texture_features = (contrast.ravel(), dissimilarity.ravel(), homogeneity.ravel(), energy.ravel(), correlation.ravel(), ASM.ravel())


        # print("Extraindo caracteristicas da imagem "+str(i))
        #Extração e normalização das caracteristicas 
        #med_spc = normalize_features(spectral_mean(img))
        #ctr_spc = normalize_features(spectral_centroid(img, 1))
        #rol_spc = normalize_features(spectral_rolloff(img, 0.85))
        #dom_spc = normalize_features(dominant_frequency_band(img))
        #con_spc = normalize_features(spectral_contrast(img))

        #maior = max(len(med_spc), len(rol_spc),len(con_spc))

        #Ajusta o tamanho das listas para evitar o erro do np.array
        # med_spc_N = tam_ajust(med_spc, maior)
        # ctr_spc_N = tam_ajust(ctr_spc, maior)
        # rol_spc_N = tam_ajust(rol_spc, maior)
        # dom_spc_N = tam_ajust(dom_spc, maior)
        # con_spc_N = tam_ajust(con_spc, maior)

        #tupla representando as caracteristicas normalizadas da imagem
        #img_carac = (med_spc_N, ctr_spc_N, rol_spc_N, dom_spc_N, con_spc_N)
               
        #carac_estilo.append(img_carac)
        carac_estilo.append(texture_features)
        #print(type(carac_estilo))
        # append to the file the features
        # with open(path_to_preprocessed + name + "_carac.txt", "a") as file:
        #     # write a marker to identify the start of a new image
        #     file.write("img " + str(counter) + "\n")
        #     file.write(str(texture_features) + "\n")
        #     counter += 1

    print("Tempo de execução: "+str(time.time()-init_time))
    
    return carac_estilo


def repartition(F1, F2, F3):
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

    Style_repartirion(F1, S1, S2, S3, S4, S5, S6, S7, S8, S9, S10)
    Style_repartirion(F2, S1, S2, S3, S4, S5, S6, S7, S8, S9, S10)
    Style_repartirion(F3, S1, S2, S3, S4, S5, S6, S7, S8, S9, S10)

    return S1, S2, S3, S4, S5, S6, S7, S8, S9, S10

def extractor(S1, S2, S3, S4, S5, S6, S7, S8, S9, S10):
    S1_f = general_extractor(S1, "S1")
    S2_f =general_extractor(S2, "S2")
    S3_f =general_extractor(S3, "S3")
    S4_f =general_extractor(S4, "S4")
    S5_f =general_extractor(S5, "S5")
    S6_f =general_extractor(S6, "S6")
    S7_f =general_extractor(S7, "S7")
    S8_f =general_extractor(S8, "S8")
    S9_f =general_extractor(S9, "S9")
    S10_f =general_extractor(S10, "S10")
    #print(type(S1_f))

    # general_extractor(S1, "S1")
    # general_extractor(S2, "S2")
    # general_extractor(S3, "S3")
    # general_extractor(S4, "S4")
    # general_extractor(S5, "S5")
    # general_extractor(S6, "S6")
    # general_extractor(S7, "S7")
    # general_extractor(S8, "S8")
    # general_extractor(S9, "S9")
    # general_extractor(S10, "S10")

    return S1_f, S2_f, S3_f, S4_f, S5_f, S6_f, S7_f, S8_f, S9_f, S10_f

#Pré-processamento
def preprocess(F1,F2,F3): #Retorna lista com as caracteristicas de cada estilo Normalizado
    
    S1, S2, S3, S4, S5, S6, S7, S8, S9, S10 = repartition(F1, F2, F3)

    S1_f, S2_f, S3_f, S4_f, S5_f, S6_f, S7_f, S8_f, S9_f, S10_f =extractor(S1, S2, S3, S4, S5, S6, S7, S8, S9, S10)
   
    return S1_f, S2_f, S3_f, S4_f, S5_f, S6_f, S7_f, S8_f, S9_f, S10_f


#Crição dos labels e divisão dos dados em treino e teste

def data_div(X,interv,labels_musicais):

    # Inicialize a lista de rótulos
    labels = []

    # Atribua um rótulo para cada conjunto contíguo de espectrogramas
    for i in range(len(X)):
        label = labels_musicais[i // interv]  # Use a divisão inteira para atribuir o mesmo rótulo para cada conjunto de 90 espectrogramas
        labels.append(label)

    # Embaralhe os espectrogramas e os rótulos juntos para manter a correspondência
    espectrogramas, labels = np.array(X), np.array(labels)
    indices_embaralhados = np.random.permutation(len(espectrogramas))
    espectrogramas_embaralhados = espectrogramas[indices_embaralhados]
    labels_embaralhados = labels[indices_embaralhados]



    # #Adiciona label a lista de tuplas X criando outra tupla composta (labels[i], X[i]) 
    # X = [(labels[i], X[i]) for i in range(len(X))]
      
    # #Embaralhamento do X
    # np.random.shuffle(X)

    # espectrogramas_embaralhados = [x[1] for x in X]
    # labels_embaralhados = [x[0] for x in X]

    # Dividir os dados em conjuntos de treinamento e teste (80% para treinamento e 20% para teste)
    X_train, X_test, y_train, y_test = train_test_split(espectrogramas_embaralhados, labels_embaralhados, test_size=0.2, random_state=42)
    # X é uma lista de espectrogramas e y é uma lista de rótulos correspondentes
    
    


    return X_train, X_test, y_train, y_test


#======= Treinamento e Teste ========
#KNN
def Trein_knn(X_train, X_test, y_train, y_test):

    #printa os tipos
    #print(X_train)
    # Flattening the spectrograms
    


   
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
    #print("Acurácia do modelo SVM:", accuracy_svm)
    print("Precisão do classificador SVM: {:.2f}%".format(accuracy_svm * 100))


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
    #print("Acurácia do modelo de árvore de decisão:", accuracy_tree)
    print("Precisão do classificador Arvore: {:.2f}%".format(accuracy_tree * 100))

    #return accuracy_tree

#Algoritmo Bayes
def Trein_bayes(X_train, X_test, y_train, y_test):

    # Inicializar o classificador Naive Bayes
    nb = GaussianNB()

    # Treinar o classificador nos dados de treinamento
    nb.fit(X_train, y_train)

    # Prever os rótulos para os dados de teste
    y_pred_nb = nb.predict(X_test)

    # Calcular a acurácia do modelo Naive Bayes
    accuracy_nb = accuracy_score(y_test, y_pred_nb)
    #print("Acurácia do modelo Naive Bayes:", accuracy_nb)
    print("Precisão do classificador Naive Bayes: {:.2f}%".format(accuracy_nb * 100))


#Algoritmo Random Forest
def Trein_random_forest(X_train, X_test, y_train, y_test):

    # Inicializar o classificador Random Forest com 100 árvores (você pode ajustar este número)
    rf = RandomForestClassifier(n_estimators=100, random_state=42)

    # Treinar o classificador nos dados de treinamento
    rf.fit(X_train, y_train)

    # Prever os rótulos para os dados de teste
    y_pred_rf = rf.predict(X_test)

    # Calcular a acurácia do modelo Random Forest
    accuracy_rf = accuracy_score(y_test, y_pred_rf)
    #print("Acurácia do modelo Random Forest:", accuracy_rf)
    print("Precisão do classificador Random Forest: {:.2f}%".format(accuracy_rf * 100))



