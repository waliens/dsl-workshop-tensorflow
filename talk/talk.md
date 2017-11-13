class: middle, center

# TensorFlow and Keras (deep learning by Google) 

[Romain Mormont](http://www.montefiore.ulg.ac.be/~rmormont/)


---
name: with_footer

#  I, Me, Mine

- **Doctorant en machine learning (ML)** à Montefiore sous la supervision de Pierre Geurts et Raphaël Marée  
- **Recherche**: machine learning appliqué au traitement de (très) grandes images médicales

---

# Machine learning ?

Le **machine learning** (supervisé) est un ensemble de méthodes permettant à un système informatique de construire/d'apprendre un modèle entrée(s)-sortie(s) sur base d'un ensemble de données. 

Un **modèle** peut être vu comme une relation entre une ensemble d'entrées (i.e. les variables) et une sortie. Il possède des **paramètres** que l'on peut modifier afin l'adapter à un problème cible. Par exemple:

.grid[
    <img src="images/model_cat_dog.png" height="75%">
    <img src="images/model_mnist.png">
]

---

# Deep learning ?

Le **deep learning** est un ensemble de méthodes de machine learning basée sur l'apprentissage de **hierarchies de descripteurs** (<i>hierarchical features</i>). 

Un **descripteur** (<i>feature</i>) est une information extraite d'une entité dans le but de la décrire.

En <i>vision par ordinateur</i>: *pixel > bord > texton > motif > morceau > objet*

.center[<img src="images/hierarchical_data.png" height="350px">]

???

---
# Deep learning ?

Autres exemples:

- <i>reconnaissance vocale</i>: échantillon > bande spectrale > formant > motif > phonème > mot
- <i>traitement du langage naturel</i>: caractère > mot > groupe nominal/verbal > clause > phrase > histoire
<h2 style="text-align: center; padding-top: 20px; font-size: 115%;">
    Les méthodes de <em>deep learning</em> exploitent ces hierarchies à l'aide de réseaux neuronaux en couche pour obtenir des modèles performants.
</h2>

.center[<img src="images/nn_learns_hfeatures.png" height="225px">]

---
# Deep **learning** ?

Travailler avec un modèle sur une tâche cible implique en général deux grandes étapes:

- **entraînement** (<i>training, learning</i>): on optimise les paramètres du modèle afin d'améliorer ses performances. Procédure classique: 
    1. on fournit au modèle un sous-ensemble aléatoire des données d'entraînement (<i>batch</i>) pour lesquels on connait l'objectif et il retourne un ensemble de prédiction $\hat{y}$
    2. on indique au modèle à quel point ces prédictions $\hat{y}$ sont erronnées à l'aide d'une **fonction de perte/d'erreur** (<i>loss function</i>)
    3. sur base de l'erreur, les paramètres du modèle sont ajustés pour améliorer ses performances (via **backpropagation**)
    4. on répète les étapes 1 à 3 jusqu'à ce que le modèle soit suffisamment performant
- **inférence** (<i>prediction, inference</i>): utilisation du modèle afin de produire une prédiction sur des nouvelles données

---
# Deep learning frameworks 

Beaucoup de frameworks disponibles: 

- **TensorFlow** (by Google)
- **Keras** (intégré dans TensorFlow depuis 01/2017)
- **Torch/PyTorch** (by Facebook)
- **Caffe2** (by Facebook)
- **CNTK** (by Microsoft)
- **MXNet** (sponsored by the Apache Incubator)
- **DL4j** (by Skymind)
- **Theano** (fin du support annoncée le 28/09/2017)
- ...

Ne pas s'enfermer dans un framework en particulier ! 

---
# TensorFlow

D'après Google: "**<i>An open-source software library for Machine Intelligence</i>** "

Quelques dates:

- 09/11/15: [open-source release](https://research.googleblog.com/2015/11/tensorflow-googles-latest-machine_9.html)
- 03/01/17: [choix de Keras](https://research.googleblog.com/2015/11/tensorflow-googles-latest-machine_9.html) comme interface haut-niveau
- 15/02/17: [version 1.0.0](https://github.com/tensorflow/tensorflow/releases/tag/v1.0.0)
- 06/11/17: [dernière release stable (1.4.0)](https://github.com/tensorflow/tensorflow/releases/tag/v1.4.0)

Quelques chiffres (sur GitHub):

- 1134 contributeurs
- 24117 commits, ~250 par semaine
- 76199 followers
- 1162 issues ouverts (7689 fermées)

???
Librairie de calcul numérique avec des outils pour le deep learning.

---
# TensorFlow 

<span style="font-size: 90%"> 
- **Interface de programmation en Python** mais coeur en C++
- **Graphe de calcul statique**: les noeuds sont les *opérations* et les arêtes sont les *tenseurs*
    - Le graphe doit être compilé avant d'être utilisé
    - Optimisations possibles à la compilation
    - <span><img src="https://assets-cdn.github.com/images/icons/emoji/unicode/26a0.png" height="20px" style="position: relative; bottom: -5px;"/></span>&nbsp; Ne permet pas d'implémenter des comportements dynamiques
- **Support deep learning**: beaucoup de composants de base disponibles
- **Programmation déclarative**: le graphe de calcul est construit de manière déclarative
    - Un pas vers l'impératif avec **Eager** (pre-alpha)
- **Support (multi) GPU**
- **TensorBoard**: outil de visualisation, monitoring temps réel via une interface web
- ...
</span>

???
- **`Dataset` API**: construction de pipelines complexes d'accès aux données
- **Debugger dédié `tfdbg`** 

Pourquoi "TensorFlow" ?

Un **tenseur** (<i>tensor</i>) est un objet mathématique. Un scalaire est un tenseur d'ordre 0, un vecteur est un tenseur d'ordre 1, une matrice est un tensor d'ordre 2,... 

- Scalaire: 
```
5.23
``` 
- Vecteur : 
```
[5.3 26.7 33.23]
```
- Matrice : 
```
| 5.3  26.7  33.23 |
| 5.3  3.25  22.53 |
| 8.03 27.1  33.00 |
```

---
# TensorFlow
## Graphe de calcul: définition et exécution

Implémentons l'opération suivante en TensorFlow: 

$$ y = \sqrt{a + \sin(b)} $$


```python
from math import pi
import tensorflow as tf

if __name__ == "__main__":
    # graph definition
    a = tf.placeholder(tf.float32, shape=(), name="a")
    b = tf.placeholder(tf.float32, shape=(), name="b")
    radicand = a + tf.sin(b)
    y = tf.sqrt(radicand)

    # execution
    with tf.Session() as sess:
        feed = {a: 3, b: pi / 2.0}
        res, = sess.run([y], feed_dict=feed)

        print("y: {}".format(res))
```

???
- s'exécute sur GPU par défaut si un est disponible et si tensorflow est installé avec support GPU

- compilation du graphe lors de l'instanciation de la session

---
# TensorFlow
## Graphe de calcul: visualisation avec TensorBoard

<img src="images/tb_small_graph.png" height="425px">

---
# Keras

Keras, d'après son créateur:

> <i>**Keras** is a high-level neural networks API, written in Python and capable of running on top of TensorFlow, CNTK, or Theano.</i>

Philosophie:

- **User-friendliness**: dévelopé de manière à réduire la charge cognitive du développeur (i.e. API cohérente et simple, feedback d'erreor clairs et exploitables, use cases communs implémentables en un nombre réduit d'étapes...)
- **Modularité**: la librairie est structurée comme un ensemble de modules indépendants qui peuvent être combinés pour créer des modèles
- **Extensibilité**: de nouveaux modules peuvent être créés et intégrés de manière simple

---
# TensorFlow vs. PyTorch

PyTorch est dévelopé par Facebook. Philosopie:

- graphe dynamique
- profondément intégré avec Python (*à la numpy*)
- impératif 

---
# TensorFlow vs. Caffe2


---
# Deep learning with TensorFlow
 
Concentrons-nous sur un problème en particulier: la reconnaissance d'image et plus particulièrement la **reconnaissance de chiffres écrits à la main**.

- Chaque image contient un chiffre de 0 à 9
- Image en noir et blanc, taille 28x28 pixels
- **Objectif**: étant donnée l'image, prédire le chiffre qu'elle contient  

.center[<img src="images/mnist.png" height="300px">]

---
# (Deep) learning with TensorFlow
## Perceptron binaire > modèle (i)

Commençons par essayer de distinguer deux chiffres $c\_0$ et $c\_1$ avec un modèle simple (et *superficiel*): le **perceptron binaire**.

$$ \hat{y} = \sigma\left( \sum\_{i = 1}^{N} w\_{i} x\_{i} + b\right) $$

.center[<img src="images/binary-perceptron.png" height="90%">]

.center[On optimise **les paramètres** du perceptron (i.e. les poids $w_i$ et le biais $b$) afin que le modèle puisse distinguer les deux chiffres.]

???

- on transforme/aplati la matrice/image d'entrée un en vecteur (784 éléments)

---
# (Deep) learning with TensorFlow
## Perceptron binaire > modèle (ii)

- le **perceptron est aussi appelé neurone** car inspiré du fonctionnement de la cellule cérébrale
- l'opérateur $\sigma(\cdot)$ est la fonction sigmoïde: $\sigma(x) = \frac{1}{1 + e^{-x}}$

.center[<img src="images/sigmoid.png" height="90%">]

- $\sigma(\cdot)$ est la **fonction d'activation** du neurone
- $\hat{y}$ est la probabilité que le chiffre soit $c\_1$ 
- pour déterminer le chiffre, on choisit $c\_0$ si $\hat{y} \leq 0.5$, $c\_1$ sinon

???


---
# (Deep) learning with TensorFlow
## Perceptron binaire > modèle (iii)

Pour exploiter les capacités de TensorFlow, on va transformer le perceptron en un **problème vectoriel**. 

$$ \hat{y} = \sigma\left(\mathbf{x}^T\mathbf{w} + b\right) $$

- **entrée**: un vecteur $\mathbf{x}$ de 784 éléments
- **poids**: un vecteur $\mathbf{w}$ de 784 éléments
- **biais**: un scalaire $b$

En pratique, on veut que le modèle puisse **traiter plusieurs images à la fois** !

Dans le code, on va donc plutôt utiliser une matrice d'entrée $\mathbf{X}$ de taille ($B \times 784$) où $B$ est le nombre d'images.

???
Pourquoi ? pas possible d'implémenter efficacement autrement, plus élégant, GPU

---
# (Deep) learning with TensorFlow
## Perceptron binaire > modèle (iv)

Entrées et variables: 

- entrée du modèle $\mathbf{X}$ $\rightarrow$ `tf.placeholder`
```python
x = tf.placeholder(shape=[None, 784], dtype=tf.float32, name="x")
```
   
- variables du modèle $\mathbf{w}$ et $b$ $\rightarrow$ `tf.Variable` 
```python
w = tf.Variable(  # weights
        initial_value=tf.truncated_normal(shape=[784, 1]),
        trainable=True, name="w"
)
```
```python
b = tf.Variable(  # bias
        initial_value=tf.zeros(shape=[1], dtype=tf.float32),
        trainable=True, name="b"
)
```

???
- None indique une taille indéterminée
- les variables peuvent être <i>entraînables</i> ou non
- on doit définir une stratégie d'initialisation 

---
# (Deep) learning with TensorFlow
## Perceptron binaire > modèle (v)

Opérations:

- produit matriciel $\rightarrow$ `tf.matmul`
```python
mult = tf.matmul(x, w)
```
- ajouter le biais $\rightarrow$ `tf.nn.bias_add`
```python
with_bias = tf.nn.bias_add(mult, b)
```
- sigmoïde $\rightarrow$ `tf.nn.sigmoid`
```python
    out = tf.nn.sigmoid(with_bias, name="out")
```

Notons l'utilisation de `tf.nn`, le **module <i>neural networks</i> de TensorFlow**.

---
# (Deep) learning with TensorFlow
## Perceptron binaire > entraînement > loss

Pour optimiser le modèle, il faut pouvoir quantifier son erreur avec la fonction de perte.

**Entropie croisée binaire** (<i>binary cross-entropy</i>): 

$$\mathcal{L}(y, \hat{y}) = - y \log \hat{y} - (1 - y) \log (1 - \hat{y}) $$

- **$y$ identifie le véritable chiffre** pour une image d'entrée. Il vaut 0 si le chiffre est $c\_0$, 1 sinon.
```python
y = tf.placeholder(shape=[None, 1], dtype=tf.float32, name="y")
```

- implémentation de la fonction de perte
```python
ce = - y * tf.log(out) - (1 - y) * tf.log(1 - out)
loss = tf.reduce_mean(ce, name="loss")
```

---
# (Deep) learning with TensorFlow
## Perceptron binaire > entraînement > optimisation

Il faut maintenant définir la **stratégie d'optimisation** que TensorFlow va utiliser pour mettre à jour le modèle.

Nous allons utiliser une **descente de gradient**:

```python
optimizer = tf.train.GradientDescentOptimizer(learning_rate=1e-3)
minimize = optimizer.minimize(loss)
```

Beaucoup d'[autres méthodes](https://www.tensorflow.org/api_docs/python/tf/train) sont implémentées dans TensorFlow: 

- `AdamOptimizer`
- `MomentumOptimizer`
- `RMSPropOptimizer`
- ...

---
# (Deep) learning with TensorFlow
## Perceptron binaire

.center[<img src="images/perceptron_graph.png" height="450px">]

???

Le graphe est prêt !

---
# (Deep) learning with TensorFlow
## Perceptron binaire > entraînement > code

```python
initializer = tf.global_variables_initializer()

with tf.Session() as sess:
    sess.run([initializer])

    for i in range(1000):
        feed = {
            x: # random batch of images...
            y_true: # and their classes
        }
        _loss, _ = sess.run([loss, minimize], feed_dict=feed)
        print("At iter {}, loss is {}".format(i, _loss))
```
<span style="font-size: 65%">
- On exécute l'opération `minimize` un certain nombre d'itérations
- À chaque itération, on fournit un batch aléatoire
- Et .. c'est tout !

.center[**TensorFlow fait beaucoup de choses pour nous dans l'ombre !**]
</span>

???
- forward pass
- backpropagation
- mise à jour des poids

---
# Deep learning with TensorFlow
## Perceptron binaire > résultats 

TODO


---
# Deep learning with TensorFlow
## Vers un modèle plus réaliste... 

Le *perceptron binaire* n'est jamais utilisé seul dans les applications réelles car **ce modèle est trop simple**.

- il ne permet pas de *capturer la complexité* de problèmes réels
- il est limité à des *problèmes binaires* 
- il ne permet pas d'exploiter la *nature hierarchique* de certains problèmes

Nous allons étudier un *autre modèle* qui:

- combine des perceptrons
- tente de résoudre les problèmes évoqués ci-dessus

<h2 style="text-align: center; padding-top: 20px; font-size: 115%;">
   	Ce modèle est le <em>perceptron multicouche</em> <br> 
   	<span style="font-size: 80%">(<i>multi-layer perceptron</i>)</span>
</h2>

---
# Deep learning with TensorFlow
## Perceptron multicouche
Ce modèle est un ensemble de perceptrons arrangés **en couches**.

.center[<img src="images/mlp.png" height="250px">]

<span style="font-size: 0.8em;">
- les couches internes s'appellent **couches cachées** (<i>hidden layer</i>)
- chaque neurone est connecté à tous les neurones de la couche précédente
- il y a *autant de neurones que de classes* dans la dernière couche 
</span>

---
# Deep learning with TensorFlow
## Perceptron multicouche > définition (i)

- $L$ est le **nombre de couches**. La couche 0 est le vecteur d'entrée $\mathbf{x}$ 
- $n_l$ est le **nombre de neurones** à la couche $l \in [0, L]$
- $\mathbf{W}\_l$ est la **matrice des poids** de la couche $l \in [1, L]$, (dim. $n\_{l-1} \times n\_l$) 
- $\mathbf{b}\_l$ est le **vecteur de biais** pour la couche $l \in [1, L]$ (dim. $n\_l$)
- *couche d'entrée* (<i>input layer</i>, $l = 0$, $n\_l = 784$): $ $ $\mathbf{a}\_0  = \mathbf{x}$
- *couche cachée* (<i>hidden layer</i>, $l \in [1, L-1]$): 
$$ \mathbf{a}\_l  = \sigma\left(\mathbf{a}\_{l-1}^T \mathbf{W}\_l + \mathbf{b}\_l \right) $$
- *couche de sortie* (<i>output layer</i>, $l = L$, $n\_l = 10$): 
$$\mathbf{\hat{y}}  = \text{softmax}\left(\mathbf{a}\_{L-1}^T \mathbf{W}\_L + \mathbf{b}\_L \right)$$

---
# Deep learning with TensorFlow
## Perceptron multicouche > définition (ii)

La fonction **softmax** est une nouvelle *fonction d'activation*:

- elle est définie comme suit: elle prend en entrée un vecteur $\mathbf{x}$ de taille $n$

$$ \text{softmax}(\mathbf{x})\_i = \dfrac{e^{x\_i}}{\sum\_{k = 1}^{n} e^{x\_k}} $$ 

- $\text{softmax}(\mathbf{x})\_i$ peut être interprété comme une probabilité
- la *sortie du réseau* $\mathbf{\hat{y}}$ est donc un *vecteur de probabilité* 
- l'élément **$\hat{y}\_i$ est la probabilité que le chiffre soit $i$**

???
- entropie croisée multi-classe: 

---
# Deep learning with TensorFlow
## Perceptron multicouche > implémentation (i)

- fonction générique pour créer un couche dense (<i>fully connected</i> or <i> dense layer</i>):
```python
def layer(input_layer, input_size, output_size, name=""):
        w = tf.Variable(
            initial_value=tf.truncated_normal([input_size, output_size]),
            trainable=True, name="{}/weights".format(name)
        )
        bias = tf.Variable(
            initial_value=tf.zeros([output_size]),
            trainable=True, name="{}/bias".format(name)
        )

        # compute activation
        prod = tf.matmul(input_layer, w)
        with_bias = tf.nn.bias_add(prod, bias)
        return tf.nn.sigmoid(with_bias, name="{}/out".format(name))
```

---
# Deep learning with TensorFlow
## Perceptron multicouche > implémentation (ii)

- *vecteur de sortie* $\mathbf{y}$ ($y\_i$ vaut 1 si le véritable chiffre est $i$, 0 sinon):
```python
y = tf.placeholder([None, 10], dtype=tf.float32, name="y")
```

- construction des *couches cachées* (ici 3 couches de tailles 64, 32 et 16):
```python
prev_layer = x
prev_size = 784
for i, size in enumerate([64, 32, 16]):
        prev_layer = layer(
            input_layer=prev_layer,
            input_size=prev_size, 
            output_size=size,
            name="hidden_{}".format(i + 1)
        )
        prev_size = size
```


---
# Deep learning with TensorFlow
## Perceptron multicouche > implémentation (iii)

- la *couche de sortie*:
```python
    classif = layer(
        input_layer=prev_layer,
        input_size=prev_size, 
        output_size=10
        name="output_layer"
    )

    out = tf.nn.softmax(classif)
```

- la *fonction de perte*: $ $ $\mathcal{L}(\mathbf{y}, \mathbf{\hat{y}}) = - \sum\_{i = 1}^{10} y\_i \log \hat{y}\_i$
```python
    ce = - tf.reduce_sum(y * tf.log(out), axis=-1)
    loss = tf.reduce_mean(ce)
```

- on utilise le **même code d'entraînement** !


---
# Deep learning with TensorFlow
## Perceptron multicouche > graphe

.col-gallery[
    <img src="images/mlp_tb.png" height="450px">
    <img src="images/mlp_tb_h3.png" height="450px">
]

---
# Deep learning with TensorFlow
## Perceptron multicouche > résultats

TODO

---
# Deep learning with Keras
## Perceptron multicouche > implémentation (i)

**Keras** propose une interface de plus haut niveau que TensorFlow:

- le modèle est un objet de type `Model` 
- un `Model` est une succession de `Layer` (par ex.: `Dense` pour une couche <i>fully connected</i>)

Pour construire le même réseau que précédemment:
```python
input = Input(shape=(784,))

# create hidden layers
x = input
for size in [64, 32, 16]:
    x = Dense(size, activation="sigmoid", use_bias=True)(x)

# output layer
x = Dense(10, activation="softmax")(x)
model = Model(inputs=[input], outputs=[x])
```

---
# Deep learning with Keras
## Perceptron multicouche > implémentation (ii)

- c'est à la **compilation du modèle** qu'on précise la *fonction de perte* et la *stratégie d'optimisation*:
```python
model.compile(
        optimizer=sgd(lr=5e-2), 
        loss="categorical_crossentropy", 
        metrics=["accuracy"]
)
```

- *entraînement* via la méthode `fit` de `Model`:
```python
# train
model.fit(
        x=# all training images ...
        y=# all training classes ...
        batch_size=batch_size,
        epochs=epochs
)
``` 

---
# Deep learning with Keras
## Perceptron multicouche > implémentation (iii)

- `fit` écrit des *informations sur la progression de l'entraînement* sur la sortie standard: 
```logs
Epoch 1/200
55000/55000 [============] - 4s - val_loss: 2.3002 - val_acc: 0.1126
Epoch 2/200
55000/55000 [============] - 1s - val_loss: 2.2999 - val_acc: 0.1126
Epoch 3/200
55000/55000 [============] - 1s - val_loss: 2.2988 - val_acc: 0.1126
Epoch 4/200
43648/55000 [=========>..] - ETA: 0s - loss: 2.2988 - acc: 0.1172
```

- *inférence* via la méthode `predict` de `Model`:
```python
y_pred = model.predict(new_images, batch_size=batch_size)
```

---
# How to deal with images ?

---
# Pre-trained model


---
# TensorFlow
## Exécution: choix du device

<span style="font-size: 90%"> 
Forcer l'exécution sur CPU
</span>

```python
# ...
if __name__ == "__main__":
    with tf.device('/cpu:0'):
        a = tf.placeholder(tf.float32, shape=(), name="a")
        b = tf.placeholder(tf.float32, shape=(), name="b")
        y = tf.sqrt(a + tf.sin(b))
# ...
```

<span style="font-size: 90%"> 
Forcer l'exécution sur un autre GPU que le GPU par défaut (`'/device:GPU:0'`)
</span>

```python
# ...
if __name__ == "__main__":
    with tf.device('/device:GPU:1'):
        a = tf.placeholder(tf.float32, shape=(), name="a")
        b = tf.placeholder(tf.float32, shape=(), name="b")
        y = tf.sqrt(a + tf.sin(b))
# ...
```

---
# (Deep) learning with TensorFlow
## Perceptron binaire > inférence - code

```python
    # ... in the same session 
    for i in range(1000):
        feed = {
            x: # batch of images with unknown class
        }
        probas, = sess.run([out], feed_dict=feed)
```

---
# Interesting resources

- [PyTorch vs TensorFlow — spotting the difference \[EN\]](https://towardsdatascience.com/pytorch-vs-tensorflow-spotting-the-difference-25c75777377b)
- [Quora - What are the pros and cons of PyTorch vs Keras? \[EN\]](https://www.quora.com/What-are-the-pros-and-cons-of-PyTorch-vs-Keras)
- [Video: What is backpropagation and what is it actually doing?](https://www.youtube.com/watch?v=Ilg3gGewQ5U) 