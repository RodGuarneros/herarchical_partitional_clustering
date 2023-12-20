import numpy as np
#from scipy.spatial.distance import cosine
from sklearn.cluster import KMeans as skmeans
from matplotlib import pyplot as plt
from matplotlib.pyplot import cm
from sklearn.decomposition import PCA
import pandas as pd
from matplotlib import transforms

#distancia euclideana
def euclidiana(x,y):
    m=x-y
    return np.sqrt(np.sum(m*m))

#distancia coseno
def coseno(x,y):
    dist=1.0 - np.dot(x, y) / (np.linalg.norm(x) * np.linalg.norm(y))
    # si los vectores no están normalizados se podría utilizar la siguente linea
    #dist=1.0 - np.dot(x, y)
    return dist

# Función para graficar clusters en dos y tres dimensiones
def plotClusters(data,labels,centroids={},f="",centroids_txt_labels={}):
    fig=plt.figure(figsize=(6, 6))
    sbox = dict(boxstyle='round', facecolor='white', alpha=0.4)
    d=len(data[1])
    if d==3:
        ax = fig.add_subplot(111, projection='3d')
    else:
        ax = fig.add_subplot(111)
    K=np.unique(labels)
    color_map=iter(cm.viridis(np.linspace(0,1,len(K))))
    for k in K:
        D=data[np.where(labels==k)]
        x,y=D[:,0],D[:,1]
        cl=next(color_map)
        if d==3:
            z=D[:,2]
            ax.scatter(x,y,z, color=cl,s=32)
        else:
            ax.scatter(x,y, color=cl,s=32)
        if len(centroids):
            txt_label=centroids_txt_labels and str(centroids_txt_labels[k]) or str(k)
            if len(centroids[k])==3:
                xc,yc,zc=centroids[k]
                ax.text(xc,yc,zc,txt_label,bbox=sbox,fontsize=14)
            else:
                xc,yc=centroids[k]
                ax.text(xc,yc,txt_label,bbox=sbox,fontsize=14)
    if len(data[0])==3:
        ax.set_zticks([])
    ax.set_xticks([])
    ax.set_yticks([])
    if f:
        fig.savefig(f)

# Transformar datos N>3 dimensionales a 2 o tres dimensiones usando PCA                  
def plotPCA(data, labels, d=2,f="",centroids={},vectors=True):
    pca = PCA(n_components=d)
    pca.fit(data)
    X=pca.transform(data)
    origin2d=[0],[0]
    origin3d=[0],[0],[0]
    pca_centroids={}
    for k,c in centroids.items():
        pca_centroids[k]=pca.transform([centroids[k]])[0,:]
    plotClusters(X,labels,f=f,centroids=pca_centroids)
    if len(centroids)>0 and vectors:
        for k,c in pca_centroids.items():
            if d==2:
                plt.quiver(*origin2d, pca_centroids[k][0],pca_centroids[k][1],angles='xy',
                        scale_units='xy', scale=1, color='skyblue')
            else: 
                plt.quiver(*origin3d, pca_centroids[k][0],pca_centroids[k][1],pca_centroids[k][2],color='skyblue')

# Plantilla simple para implementar métodos de clustering
class Clustering:
    
    ## Calcular SSE se usa inertia igual que en la implementacion de sckit-learn 
    def _inertia(self):
        self.inertia_=0
        for j in range(len(self.data)):
            dists=[(self.distance_function(c,self.data[j]),i) for i,c in self.centroids_.items()]
            self.inertia_+=dists[0][0]
            
    # asigna los elementos en la colección a su centroide más cercano
    # genera las etiquetas de los clusters 
    def _assign_nearest_centroids(self):
         self.labels_=[-1 for x in self.data]
         for j in range(len(self.data)):
             dists=[(self.distance_function(c,self.data[j]),i,self.data[j])
                    for i,c in self.centroids_.items()]
             dists.sort()
             self.labels_[j]=dists[0][1]
       
    # Ejemplo de random Clustering, es equivalente a la primera iteración de KMeans
    def randomClustering(self):
         # seleccionamos K elmentos de forma aleatoria
         idx=np.random.randint(self.data.shape[0], size=self.n_clusters)
         self.centroids_=dict(zip(idx,self.data[idx,:])) # creamos un diccionario {id_cluster: vector} 
         self._assign_nearest_centroids() #asignamos las etiquetas
         self._inertia() # calculamos el SSE
         return self

    def KMeans(self):
         print("Su implentación de KMeans")

    def FFTraversal(self):
        print("su implementación de Farthest First Traversal") 

    # Metodo para entrenar el modelo, solo recibe un numpy.array con los dato de n x N.
    # Donde n es el número de elmentos y N la dimensión            
    def fit(self,data):
        self.data=data
        self.algorithm()
        return self
    #Metodo que asinga un clusters a los elementos en data
    def predict(self,data):
        labels=[-1 for x in data]
        for j in range(len(data)):
            dists=[(self.distance_function(c,data[j]),i,data[j])
                    for i,c in self.centroids_.items()]
            dists.sort()
            labels[j]=dists[0][1]
        return np.array(labels)
    #estructura propuesta para los algoritmos
    # La variable algorithm es un string con el nombre de su función de clustering
    def __init__(self,n_clusters=3,distance_function=euclidiana,algorithm='randomClustering'):
        self.n_clusters=n_clusters  # número de clusters K
        self.inertia_ = 0  # SSE
        self.distance_function = distance_function #Funcion de distancia, por defecto euclidiana
        self.algorithm = getattr(self, algorithm) 
    
    

    
                                
    

         
