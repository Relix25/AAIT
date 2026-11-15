# Méthode, à comprendre avant l'oral

## Problème

Crafter renvoie une observation RGB de 64 × 64 pixels, 17 actions discrètes, une récompense et un indicateur de fin d'épisode. Le monde est partiellement observé. Les achievements sont rares et forment un arbre : ramasser du bois ouvre la possibilité de fabriquer une table, puis une pioche, etc. Une simple action au hasard atteint surtout les premiers paliers.

## DQN de base

Le CNN produit un vecteur de 17 nombres `Q(o, a; θ)`. Chaque nombre estime le retour actualisé si l'agent choisit l'action `a` depuis l'observation `o`. À l'entraînement, la politique ε-greedy joue une action aléatoire avec probabilité ε et choisit `argmax Q` sinon. ε baisse au fil des pas.

Les transitions vont dans un **replay buffer**. Un lot aléatoire casse la corrélation temporelle. Un second réseau, la **target network**, est copié périodiquement depuis le réseau entraîné ; ses valeurs changent donc moins vite. La loss Huber et le gradient clipping limitent l'effet des écarts extrêmes.

Pour DQN à un pas, la cible est :

$$y_t = r_{t+1} + \gamma m_{t+1} \max_a Q(o_{t+1}, a; \bar\theta).$$

Ici `m=0` si le monde est terminal (mort) et `m=1` autrement. À la limite de 10 000 pas, le code conserve le bootstrap puisque cette limite coupe l'épisode sans rendre le monde terminal.

## Deux améliorations implémentées

**Double DQN.** Le réseau courant choisit l'action future ; le réseau cible l'évalue. Cela réduit le biais de surestimation créé quand le même estimateur choisit et note le maximum :

$$a^* = \arg\max_a Q(o_{t+n}, a; \theta), \qquad y_t = R_t^{(n)} + \Gamma_t Q(o_{t+n},a^*;\bar\theta).$$

**Retours n-step.** Au lieu d'attendre qu'une récompense rare influence les pas précédents un par un, on somme jusqu'à `n` récompenses :

$$R_t^{(n)} = \sum_{k=0}^{n-1}\gamma^k r_{t+k+1}, \qquad \Gamma_t = \gamma^n\prod_{k=1}^{n}m_{t+k}.$$

En fin d'épisode, les transitions restantes sont émises avec leur **horizon réel** : le code n'applique pas `γ^n` si moins de `n` pas existent. La loss optimisée est :

$$L(\theta)=\mathbb E_{(o,a,R,o',\Gamma)\sim U(D)}\left[\operatorname{Huber}\left(y-Q(o,a;\theta)\right)\right].$$

## À savoir expliquer dans le code

- `agent/network.py` transforme les pixels en 17 Q-values.
- `agent/replay.py` stocke les images en `uint8`, tire des lots uniformes et construit le retour n-step.
- `train.py` applique ε-greedy, calcule la cible sans gradient, minimise Huber et copie périodiquement le réseau cible.
- `evaluate.py` utilise des seeds séparées et, par défaut, ε = 0 : le réseau choisit toujours l'action de plus grande Q-value. La baseline aléatoire tire uniformément parmi 17 actions.
- `metrics.py` mesure des épisodes complets. Un achievement compte comme réussi s'il apparaît au moins une fois dans l'épisode.

## Ce que cette méthode ne prouve pas

Double DQN et n-step sont des hypothèses d'amélioration. Seule une comparaison à budget égal, sur plusieurs seeds, permet de dire si elles aident ici. L'image seule ne conserve pas toute l'histoire ; sans mémoire explicite, certaines décisions restent ambiguës.
