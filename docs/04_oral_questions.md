# Oral : fil conducteur et questions probables

## Introduction en 45 secondes

« Nous avons entraîné un agent à choisir parmi 17 actions dans Crafter à partir d'une image. Les récompenses importantes arrivent quand il débloque des achievements, parfois après une longue suite d'actions. Notre agent est un Double DQN avec des retours n-step, codé à partir des mécanismes DQN du cours. Double DQN réduit la surestimation des valeurs, et n-step transmet plus vite les récompenses aux actions précédentes. Nous le comparons à un agent aléatoire avec les mêmes épisodes d'évaluation. »

« Après **1 million de pas** pour chacune des deux seeds d'entraînement, notre agent obtient **4,15 ± 0,16** de reward épisodique et **4,14 ± 0,60 %** de score Crafter sur 50 épisodes neufs par seed, avec ε = 0. L'aléatoire obtient **1,32** et **1,23 %** sur ces mêmes épisodes. » Les ± sont des écarts-types **entre deux entraînements**, pas des intervalles de confiance. Ne présente pas les résultats du papier comme les nôtres.

**Répartition proposée en binôme :** Kilian présente Crafter, la fonction objectif et DQN ; Mathis explique Double DQN, n-step, le protocole et les résultats. Vous pouvez inverser ces rôles. Chacun doit néanmoins pouvoir répondre sur le replay, la métrique Crafter et les limites.

## Questions et réponses courtes

**Pourquoi du reinforcement learning ?** Il n'y a pas d'action correcte fournie pour chaque image. L'agent apprend à partir des conséquences de ses actions et de récompenses retardées.

**Que prédit Q ?** Le retour actualisé attendu si l'on choisit une action depuis une observation, puis que l'on suit la politique implicite du maximum des Q-values.

**Pourquoi γ = 0,99 ?** Il donne du poids au futur sans rendre la cible infinie. C'est un hyperparamètre à tester : un γ plus élevé peut aider les chaînes longues mais rend l'apprentissage plus difficile.

**Pourquoi un replay buffer ?** Des images successives sont très corrélées. Échantillonner des transitions anciennes et récentes diversifie les lots et réutilise l'expérience.

**Pourquoi un target network ?** Si le réseau produit à la fois la prédiction et une cible qui change à chaque mise à jour, la régression poursuit une cible mouvante. La copie périodique la stabilise.

**Différence entre DQN et Double DQN ?** DQN prend le maximum des valeurs estimées par le réseau cible. Double DQN fait choisir l'action par le réseau courant, puis la fait évaluer par le réseau cible, ce qui limite la surestimation du maximum bruité.

**Que fait n-step ?** Il additionne plusieurs récompenses réelles avant de bootstraper. Une récompense rare influence ainsi plus directement les actions qui la précèdent. Le revers est une variance potentiellement plus forte.

**Pourquoi Huber ?** Près de zéro elle se comporte comme l'erreur quadratique ; pour de grands écarts, sa croissance est linéaire et limite l'influence des extrêmes.

**Mort ou limite de temps ?** La mort termine le monde et coupe le bootstrap. À 10 000 pas, l'épisode est tronqué par une règle de mesure ; le prochain état conserve une valeur possible.

**Pourquoi le reward et le score Crafter peuvent diverger ?** Le reward récompense les gains pendant un épisode ; le score agrège les taux de succès de 22 compétences par moyenne géométrique. Répéter quelques compétences faciles peut améliorer le reward sans ouvrir de nouvelles branches.

**Pourquoi plusieurs seeds ?** L'exploration, la carte et l'initialisation du réseau varient. Une seule courbe peut donner une impression trompeuse.

**Pourquoi ε = 0 à l'évaluation ?** À l'entraînement, ε sert à explorer. Une fois le réseau figé, nous choisissons l'action de plus grande valeur Q sans ajouter d'actions aléatoires. Un essai précédent à 150k pas avait comparé ε = 0 et ε = 0,01 sur les épisodes 3001–3030 ; ses fichiers ne sont pas conservés dans `runs/`. Le protocole final à 1M utilise ε = 0 sur les 50 épisodes neufs 7001–7050.

**Pourquoi 1 million de pas ?** Le sujet présente Crafter comme conçu pour ce budget. À 300k, 500k et 1M, le score moyen de validation des deux seeds est respectivement 3,62 %, 3,81 % et 4,08 %. Le checkpoint à 1M a donc été retenu avant le test final. Ce budget ne change pas l'architecture du modèle.

**Quels hyperparamètres faut-il connaître ?** `n=3`, `γ=0,99`, replay uniforme de 50 000 transitions, batch de 32, apprentissage tous les 4 pas après 10 000 pas, copie du réseau cible tous les 2 000 pas, Adam à `10^-4`, ε de 1 à 0,05 sur 150 000 pas. Savoir expliquer le rôle de chacun est plus utile que mémoriser tous les chiffres.

**Avez-vous mesuré le gain de Double DQN ou de n-step séparément ?** Non. Le code permet un DQN simple, mais nous n'avons pas réalisé une ablation à budget égal. Le gain observé est celui de la configuration complète face à random ; nous ne l'attribuons pas à une seule modification.

**L'agent atteint-il la pierre ou les pioches ?** Il fabrique une **pioche en bois dans 10 %** des épisodes, et collecte de la pierre dans **1 %**. Il ne fabrique **aucune pioche de pierre** dans les 100 épisodes finaux réunis. Le début de l'arbre de fabrication est acquis, pas les branches profondes.

**Avez-vous dépassé la baseline « coding agent » du sujet ?** Son score et son protocole ne sont donnés ni dans l'énoncé ni dans le dépôt officiel Crafter ; nous ne pouvons pas conclure sur ce seuil. Notre comparaison vérifiable est celle avec l'aléatoire. Voir [05_recherche_baseline.md](05_recherche_baseline.md) pour les sources examinées.

**Qu'avez-vous réellement observé ?** Dans un épisode tracé du modèle seed 1, l'agent ramasse une pousse au pas 13 et la plante au pas 14. Il collecte du bois aux pas 18 et 22, place une table au pas 23 puis fabrique une pioche en bois au pas 186. Il répète aussi excessivement la collecte de boisson vers la fin. Cette trajectoire montre des étapes utiles, pas une preuve de planification intentionnelle. Sur les 100 épisodes finaux et les trois épisodes tracés, on observe aussi : un zombie vaincu dans 44 % des épisodes et une vache mangée dans 23 %, contre 0 % pour l'aléatoire ; des cycles répétés pousse → plantation ; une seconde pioche en bois fabriquée au pas 223 du même épisode tracé, sans récompense possible ; un réveil moins fréquent que l'aléatoire, 77 % contre 94 %, car la politique gloutonne ne dort presque jamais (1,7 % des pas tracés contre 5,9 % pour l'aléatoire) ; et environ un tiers des pas tracés consacrés à des fabrications ou placements impossibles avec l'inventaire, donc sans effet.

**Pourquoi 4,14 % alors qu'un étudiant rapporte 6,1 % ?** Son chiffre concerne aussi 1M de pas, mais nous n'avons ni ses seeds ni son protocole d'évaluation. Notre chiffre est la moyenne de deux entraînements évalués sur 50 épisodes neufs chacun. Il est inférieur à son chiffre rapporté ; une comparaison scientifique exigerait les mêmes épisodes et la même formule de score.

**Pourquoi votre agent aléatoire est-il plus faible que celui du papier Crafter ?** Le tableau officiel donne 2,1 de reward et 1,6 % de score pour l'aléatoire, calculés sur tous les épisodes d'un budget de 1 million de pas, soit plusieurs milliers d'épisodes. Notre référence vient de 50 épisodes sur des seeds fixées, celles du test final, avec notre propre code d'évaluation : les taux de succès ont une granularité de 2 % et une variance plus forte. Les deux chiffres ne sont pas comparables directement ; la comparaison utile est agent contre aléatoire sur les mêmes épisodes.

**Le facteur Γ vaut-il toujours γ³ ?** Non. La slide l'écrit ainsi par raccourci. Dans `NStepAccumulator`, chaque transition est émise avec son horizon réel : les deux dernières transitions d'un épisode ont un horizon de 1 ou 2 pas, donc γ¹ ou γ². À la mort, le masque met Γ à zéro ; à la troncature à 10 000 pas, Γ garde γ^h et le bootstrap reste.

**D'où viennent 0,02 → 1,57 et 0,034 ?** De `scalars.csv` des deux entraînements, où loss et Q moyen sont moyennés par fenêtre de 1 000 pas. 0,02 est la première fenêtre loggée après le warmup, au pas 11 000 ; 1,57 et 0,034 sont les moyennes des deux seeds sur les 10 000 derniers pas. La toute dernière fenêtre donne 1,59 pour Q.

**Pourquoi une machine G4 ?** Crafter génère les transitions séquentiellement sur CPU. Le CNN utilise le GPU, mais celui-ci n'est pas saturé. Le runtime Colab G4, avec 48 cœurs logiques d'après la configuration affichée à l'époque, a accéléré deux runs parallèles sans changement de méthode ; aucun relevé matériel n'est conservé dans le rendu, seul le débit moyen d'environ 540 pas d'environnement par seconde figure dans `scalars.csv`. Déplacer l'environnement sur GPU demanderait une refonte de la simulation ou plusieurs mondes parallèles.

**Quelle est la limite majeure ?** L'observation est locale et ne résume pas toute l'histoire. Le réseau actuel voit une image unique ; une mémoire ou plusieurs images successives pourraient aider.

**Pourquoi pas Munchausen ou Rainbow complet ?** Nous avons limité l'implémentation à une méthode que nous pouvons expliquer et évaluer. Une méthode plus complexe demanderait une expérimentation supplémentaire ; les chiffres actuels ne permettent pas de dire qu'elle aiderait.

## Trois choses à ne jamais confondre

1. Courbe **pendant l'entraînement** : ε change, le modèle apprend, les données ne sont pas indépendantes.
2. Performance **d'évaluation** : modèle figé, ε fixé, épisodes distincts.
3. Benchmark du **papier Crafter** : comparaison publiée, pas mesure produite par ce code.

## Avant l'oral

Pouvoir retrouver dans le code : `optimize` (cible Double DQN), `NStepAccumulator` (horizon réel), `summarize` (score Crafter), et la ligne où la target network est copiée. Pour le comportement cité, ouvrir `runs/behavior_1m_seed1/trace.jsonl` ; la trace suffit à justifier les pas annoncés.
