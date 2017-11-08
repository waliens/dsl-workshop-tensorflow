class: middle, center

# TensorFlow and Keras (deep learning by Google) 

[Romain Mormont](http://www.montefiore.ulg.ac.be/~rmormont/)


---
name: with_footer

#  Who am I ?

- **Doctorant en machine learning (ML)** à Montefiore sous la supervision de Pierre Geurts et Raphaël Marée  
- **Recherche**: machine learning appliqué au traitement de (très) grandes images médicales

---

# Machine learning ?

Le **machine learning** (supervisé) est un ensemble de méthodes permettant à un système informatique de construire/d'apprendre un modèle entrée(s)-sortie(s) sur base d'un ensemble de données. 

Un **modèle** peut être vu comme une relation entre une ensemble d'entrées (i.e. les variables) et une sortie. Par exemple:

.grid[
	.col-1-1[<img src="images/model_cat_dog.png" height="75%">]
	.col-2-1[<img src="images/model_mnist.png">]
]

---

# Deep learning ?

Le **deep learning** est un ensemble de méthodes de machine learning basée sur l'apprentissage de **hierarchies de descripteurs** (<i>hierarchical features</i>). 

Exemples de hierarchies naturelles:

- <i>vision par ordinateur</i>: pixel > bord > texton > motif > morceau > objet
- <i>reconnaissance vocale</i>: échantillon > bande spectrale > formant > motif > phonème > mot
- <i>traitement du langage naturel</i>: caractère > mot > groupe nominal/verbal > clause > phrase > histoire

<h2 style="text-align: center; padding-top: 40px;">
	Les méthodes de <em>deep learning</em> exploitent ces hierarchies pour construire des modèles performants.
</h2>

---
# Deep learning frameworks 

Beaucoup de frameworks disponibles: 

- **TensorFlow** (by Google)
- **Keras** (intégré dans Tensorflow depuis 01/2017)
- **Torch/PyTorch** (by Facebook)
- **Caffe2** (by Facebook)
- **CNTK** (by Microsoft)
- **MXNet** (sponsored by the Apache Incubator)
- **DL4j** ()
- **Theano** (fin du support annoncée le 28/09/2017)
- ...

Message principal: *ne pas se concentrer sur un seul framework !*

---
# TensorFlow

D'après Google: "**<i>An open-source software library for Machine Intelligence</i>** "

Quelques dates:

- 09/11/15: [open-source release](https://research.googleblog.com/2015/11/tensorflow-googles-latest-machine_9.html)
- 03/01/17: [intégration de Keras](https://research.googleblog.com/2015/11/tensorflow-googles-latest-machine_9.html) comme interface haut-niveau
- 15/02/17: [version 1.0.0](https://github.com/tensorflow/tensorflow/releases/tag/v1.0.0)
- 06/11/17: [dernière release stable (1.4.0)](https://github.com/tensorflow/tensorflow/releases/tag/v1.4.0)

Quelques chiffres (sur GitHub):

- 1134 contributeurs
- 24117 commits, ~250 par semaine
- 76199 followers
- 1162 issues ouverts (7689 fermées)

---
# TensorFlow 

Fonctionnalités importantes:

- Graphe de calcul statique
	- <span style="color: green;">(+)</span> Optimisations du graphe possibles
	- <span style="color: red;">(-)</span> Ne permet pas d'implémenter des comportements dynamiques
- Programmation déclarative
	- Le graphe de calcul est construit de manière déclarative
	- Un pas vers l'impératif avec **Eager** (pre-alpha)
- TensorBoard: outil de visualisation


---
# Keras

Une **interface python** à TensorFlow (et à d'autres frameworks: Theano, CNTK, MXNet,...)

Philosophie:

- building bricks de haut-niveau
- routines d'entraînement 
- routines de préparation et d'accès aux données 

---
# TensorFlow vs. PyTorch

PyTorch est dévelopé par Facebook. Philosopie:

- graphe dynamique
- profondément intégré avec Python (*à la numpy*)
- impératif 

---
# TensorFlow vs. Caffe2


---
# Deep **learning** ?

Travailler avec un modèle sur une tâche cible implique en général deux grandes étapes:

1. **entraînement** (<i>training</i>): optimisation du modèle à l'aide de données d'entraînement afin qu'il devienne performant sur la tâche en question
2. **inférence** (<i>prediction, inference</i>): utilisation du modèle afin de produire une prédiction sur des nouvelles données

---
# Deep learning in practice !
 
Concentrons-nous sur un problème en particulier: la **reconnaissance d'image**.

Problème: reconnaissance de chiffres écrits à la main

- Image en noir et blanc, taille 28x28 pixels
- Chiffres de 0 à 9
- Objectif: étant donnée l'image, prédire le chiffre qu'elle contient  

<img src="https://clomax.me.uk/assets/images/ocr/mnist.png" hight="50%">

---
# Building bricks: perceptron (neurone)

---
# Building bricks: single-layer perceptron

---
# Building bricks: multi-layer perceptron

---
# How to deal with images ?

---
# Building bricks: convolutional layer


---
# TensorBoard

---
# Eager


---
# Appendix: comparaison des frameworks

- [PyTorch vs TensorFlow — spotting the difference \[EN\]](https://towardsdatascience.com/pytorch-vs-tensorflow-spotting-the-difference-25c75777377b)
- [Quora - What are the pros and cons of PyTorch vs Keras? \[EN\]](https://www.quora.com/What-are-the-pros-and-cons-of-PyTorch-vs-Keras)
