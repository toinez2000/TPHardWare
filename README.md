**Human Activity Recognition using CNN in Keras**


L'objectif de ce tp est de programmer un modele de reconnaisance CNN en cuda. Nous utiliserons ainsi le GPU pour optimiser les performance d'execution de notre modele.

![image](https://user-images.githubusercontent.com/92809568/211397305-c23f75b3-b8c5-443b-b43b-4322ab9f133f.png)


*Prise en main de Cuda : Multiplication de matrices*

-Addition de deux matrices

On cherche dans cette partie à identifier le gain en performance entre l'utilisation CPU et GPU pour calculer l'addition de deux matrices.

- CPU time:
- GPU time:

-Multiplication de deux matrices NxN

On cherche dans cette partie à identifier le gain en performance entre l'utilisation CPU et GPU pour calculer le produit de deux matrices.

- CPU time:
- GPU time:



*Premières couches du réseau de neurone LeNet-5*

On choisie de créer des tableaux à une dimention.
Méthode pour la convolution sur GPU:
-En entrée on à le vecteur data_raw de taille SxI*SxI*SzI avec SxI la taille d'une couche et SzI la pronfondeur de l'entrée.
-Le vecteur C1_kernel de dimention Sxk*Sxk*SzK
-La sortie est le vecteur C1_data de dimention (SxI-Sxk+1)*(SxI-Sxk+1)*SzI*SzK

On utilise pour se faire deux dimensions de block et deux threads
Ainsi l'idée est de définir:

xo = blockIdx.x * blockDim.x: correspond à la ligne de sortie
yo = threadIdx.x : correspond à la colonne de sortie 
zI = blockIdx.z:   profondeur du kernel à utiliser 
zK = threadIdx.z:   proffondeur de l'image input à utilisé 


Test de la convolution pour:
![image](https://user-images.githubusercontent.com/92809568/211406291-1cbbabd3-90c0-41b6-8d7c-417fb2db5835.png)
On obtient:
![image](https://user-images.githubusercontent.com/92809568/211406565-74b1bfbe-7beb-4971-9fc8-2eae18786bb6.png)


Cela correspond bien à ce qui est attendu


*Layer 3 - Sous-échantillonnage*


Pour cette étape on utilise la meme métode que pour la conv:

Méthode pour le mean pooling sur GPU:
-L'entré est le vecteur C1_data de dimention (SxI-Sxk+1)*(SxI-Sxk+1)*SzI*SzK
-La sortie est le vecteur S1_data de dimension (SxI-Sxk+1)*(SxI-Sxk+1)*SzI*SzK/4

On utilise pour se faire deux dimensions de block et deux dim threads
Ainsi l'idée est de définir:

xo = blockIdx.x * blockDim.x: correspond à la ligne de sortie
yo = threadIdx.x : correspond à la colonne de sortie 
zI = blockIdx.z:   profondeur du kernel à utiliser 

Test de meanPooling pour la sortie C1 précédente:
On obtient:

![image](https://user-images.githubusercontent.com/92809568/211408660-dfe0009b-5429-4f76-add9-e39482446581.png)

Cela correspond bien à ce qui est attendu.


*couche dense*







TEST avec en entrée: 
