***Human Activity Recognition using CNN in Keras***
 Antoine Zundel et Alexandre Philippon


L'objectif de ce tp est de programmer un modele de reconnaisance CNN en cuda. Nous utiliserons ainsi le GPU pour optimiser les performances d'execution de notre modele.

![image](https://user-images.githubusercontent.com/92809568/211397305-c23f75b3-b8c5-443b-b43b-4322ab9f133f.png)


**Prise en main de Cuda : Multiplication de matrices**
fichier fonctionMatrice.h pour addition et multiplication de matrice sur CPU
fichier vector_add_N_P.h et vector_multi_N_P.cu pour l'addition et multiplication de matrice sur GPU



-Addition de deux matrices

On cherche dans cette partie à identifier le gain en performance entre l'utilisation CPU et GPU pour calculer l'addition de deux matrices.

- CPU time pour une addition sur N=1000, on obtient 4.34s
- GPU time pour une addition sur N=1000, on obtient 0.13s

- Multiplication de deux matrices NxN

On cherche dans cette partie à identifier le gain en performance entre l'utilisation CPU et GPU pour calculer le produit de deux matrices.

- CPU time pour une multiplication sur N=1000, on obtient 8.650s
- GPU time pour une multiplication sur N=1000, on obtient 0.236s

On constate une amélioration conséquencte au niveau des performances en utilisant le GPU, si la matrice est trés grande. Cepandant si la matrice est de petite taille, le transfere de données du cpu au gpu n'est pas compensé par la capacité de parallélisation du GPU.

**Premières couches du réseau de neurone LeNet-5**
Fichier convolutionLayer.h contient les fonctions du layer convolution sur GPU.

On choisit de créer des tableaux à une dimention.
Méthode pour la convolution sur GPU:
- En entrée on à le vecteur data_raw de taille SxI*SxI*SzI avec SxI la taille d'une couche et SzI la pronfondeur de l'entrée.
- Le vecteur C1_kernel de dimention Sxk*Sxk*SzK
- La sortie est le vecteur C1_data de dimention (SxI-Sxk+1)*(SxI-Sxk+1)*SzI*SzK

On utilise pour se faire deux dimensions de block et deux threads
Ainsi l'idée est de définir:

-  xo = blockIdx.x * blockDim.x: correspond à la ligne de sortie
- yo = threadIdx.x : correspond à la colonne de sortie 
- zI = blockIdx.z:   profondeur du kernel à utiliser 
- zK = threadIdx.z:   proffondeur de l'image input à utilisé 


Test de la convolution pour:
![image](https://user-images.githubusercontent.com/92809568/211406291-1cbbabd3-90c0-41b6-8d7c-417fb2db5835.png)
On obtient:
![image](https://user-images.githubusercontent.com/92809568/211406565-74b1bfbe-7beb-4971-9fc8-2eae18786bb6.png)


Cela correspond bien à ce qui est attendu.


**Layer 3 - Sous-échantillonnage**
Fichier meanPoolingTanH contient les fonctions du layer meanPooling sur GPU.

Pour cette étape on utilise la meme méthode que pour la conv:

Méthode pour le mean pooling sur GPU:
- L'entré est le vecteur C1_data de dimention (SxI-Sxk+1)*(SxI-Sxk+1)*SzI*SzK
- La sortie est le vecteur S1_data de dimension (SxI-Sxk+1)*(SxI-Sxk+1)*SzI*SzK/4

On utilise pour se faire deux dimensions de block et deux dim threads
Ainsi l'idée est de définir:

xo = blockIdx.x * blockDim.x: correspond à la ligne de sortie
yo = threadIdx.x : correspond à la colonne de sortie 
zI = blockIdx.z:   profondeur du kernel à utiliser 

Test de meanPooling pour la sortie C1 précédente:
On obtient:

![image](https://user-images.githubusercontent.com/92809568/211408660-dfe0009b-5429-4f76-add9-e39482446581.png)

Cela correspond bien à ce qui est attendu.


**Couche dense**
Fichier dense.h contient les fonctions du layer dense sur GPU.

On teste la couche dense sans active function input size = 10 ;output size = 2  ;weight size = 2*10 
- input:                [1,2,3,4,5,6,7,8,9,10]
- weight:               [[1,1,0,0,0,0,0,0,0,0],[2,2,0,0,0,0,0,0,0,0]
- output Without bias:  [3,6]
- bias:                 [1,2]
- output :              [4,8]


Cela correspond bien à ce qui est attendu: output[x] = sum_i(w[x][i]*i[x])+bias(x)

**Test du modele avec les poids importé du modele.ipyb et en entré les images du mnist**
- fichier modele.h contient la fonction pour appliquer le modele à une image 32*32 pour effectuer de la classification sur 10 classe et identifier le chifre sur l'image imput.
- fichier readfileWeight.h contient les fonctions pour lire le fichier qui contient les Weights ("a.out")
- fichier affichage.h contient les fonctions pour lire l'image raw_data et reshape l'image input et la normalisé.

Résultat du modele sur l'image input : 

![image](https://user-images.githubusercontent.com/92809568/212177975-16e8de43-d0f6-4a07-84a9-a7990199d0a6.png)

On obtient :

![image](https://user-images.githubusercontent.com/92809568/212394310-b2304944-118d-4a07-9643-af5efa8d5c15.png)



On constate que l'image input est bien classé dans la classe 5 en revanche il arrive que le modele ne donne pas toujours le bon résultat il semble que cela vienne de la derinere couche dense et de la fonction softmax. En effet il arrive que le calcule de la fonction d'activation softmax donne un résultat du type infini sur infini. Dans ce cas le résultat est abérant.  


**execution du code**
fonction utilisée: nvcc -cu pour compiler
et time ./a.out pur executer



