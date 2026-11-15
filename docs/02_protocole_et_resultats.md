# Protocole et résultats

## Résultat final

Le même agent **Double DQN + retours n-step** a été entraîné deux fois, avec les seeds 1 et 2, pendant **1 000 000 de pas par seed**. Les checkpoints à 300k, 500k et 1M ont été comparés sur 30 épisodes de validation (seeds 3001–3030) avec une politique gloutonne (ε = 0). Le checkpoint commun à **1M** a obtenu la meilleure moyenne de score. Il a ensuite été évalué sur **50 nouveaux épisodes** (seeds 7001–7050) pour chaque modèle. L'agent aléatoire a joué ces mêmes 50 épisodes.

| Agent | Reward moyen par épisode | Score Crafter | Épisodes |
|---|---:|---:|---:|
| Aléatoire | 1,32 | 1,23 % | 50 |
| Double DQN + n-step, seed 1 | 4,04 | 3,72 % | 50 |
| Double DQN + n-step, seed 2 | 4,26 | 4,56 % | 50 |
| **Moyenne des deux seeds entraînées** | **4,15 ± 0,16** | **4,14 ± 0,60 %** | **50 par seed** |

Le **±** est l'écart-type des **deux moyennes par seed d'entraînement**. Il ne constitue pas un intervalle de confiance : avec seulement deux seeds, l'incertitude est mal estimée. Le gain face à l'aléatoire est **+2,83 reward** et **+2,91 points de score** sur cet ensemble d'épisodes. Les données sont dans `runs/final_1m_g4_seed1/`, `runs/final_1m_g4_seed2/`, `runs/random_seed7001/` et l'agrégation dans `runs/final_report.json`.

## Ce que montre la courbe

| Checkpoint | Score seed 1 | Score seed 2 | Moyenne |
|---|---:|---:|---:|
| 300k | 3,68 % | 3,56 % | 3,62 % |
| 500k | 4,13 % | 3,48 % | 3,81 % |
| **1M** | **4,13 %** | **4,03 %** | **4,08 %** |

Ces chiffres viennent des **mêmes 30 épisodes de validation**, distincts des épisodes finaux. Le reward d'entraînement augmente puis plafonne vers 4,4–4,6 ; il inclut les actions d'exploration et ne remplace pas une évaluation. `figures/training_reward.png` montre la moyenne par tranches de 10k pas et l'écart-type entre seeds. `figures/evaluation_curve.png` montre reward et score des checkpoints. Le checkpoint à 1M a été choisi sur la **moyenne des deux seeds**, avant de regarder les 50 épisodes finaux.

## Compétences et comportement

Un taux de succès est le pourcentage d'épisodes où l'achievement apparaît au moins une fois. Le score Crafter agrège les **22 taux** par moyenne géométrique : `exp(moyenne_i log(1 + s_i)) − 1`, avec `s_i` en pourcentage. Les 22 compétences sont représentées dans `figures/achievements.png`.

| Achievement | Agent, moyenne des seeds | Aléatoire |
|---|---:|---:|
| Collect wood | 86 % | 12 % |
| Place table | 61 % | 0 % |
| Defeat zombie | 44 % | 0 % |
| Make wood pickaxe | 10 % | 0 % |
| Collect stone | 1 % | 0 % |
| Wake up | 77 % | 94 % |

L'agent progresse dans le début de l'arbre de fabrication, mais **ne fabrique pas de pioche de pierre** dans les 100 épisodes évalués. Il réussit moins souvent `wake_up` que l'aléatoire : les compétences ne progressent pas toutes ensemble.

Dans un épisode tracé du modèle seed 1, il ramasse une pousse au pas 13, la plante au pas 14, collecte du bois aux pas 18 et 22, place une table au pas 23 et fabrique une pioche en bois au pas 186. Cette **séquence observée** ne prouve pas une planification intentionnelle. Dans le même épisode, il répète longtemps l'action de collecte de boisson : c'est un échec de comportement à reconnaître à l'oral. Trace brute : `runs/behavior_1m_seed1/trace.jsonl`.

Autres observations, tirées des 100 épisodes finaux et des trois épisodes tracés (seeds 8001–8003) : l'agent bat un zombie dans 44 % des épisodes et mange une vache dans 23 %, contre 0 % pour l'aléatoire ; il plante des pousses bien plus souvent (84 % contre 50 %), parfois en cycles répétés pousse → plantation, cinq fois dans l'épisode tracé 2 ; il fabrique une **seconde pioche en bois** au pas 223 de l'épisode tracé 1 alors que seul le premier déblocage rapporte une récompense ; il se réveille moins souvent que l'aléatoire (77 % contre 94 %), la politique gloutonne ne choisissant l'action `sleep` que dans 1,7 % des pas tracés contre 1/17 ≈ 5,9 % pour l'aléatoire ; enfin, environ un tiers des pas tracés sont des tentatives de fabrication ou de placement impossibles avec l'inventaire du moment, donc sans effet. Il survit un peu plus longtemps que l'aléatoire : 191 pas contre 170 en moyenne.

## Limites et comparaison honnête

- Le réseau voit une image à la fois, sans mémoire explicite du monde partiellement observé.
- Il n'y a **pas d'ablation DQN seul à budget égal** : le gain mesuré est celui du système complet face à random, sans preuve de l'apport individuel de Double DQN ou de n-step.
- La baseline « coding agent » du sujet n'a pas de score ni de protocole publiés dans les sources fournies. On ne peut pas affirmer l'avoir dépassée. Voir [05_recherche_baseline.md](05_recherche_baseline.md).
- Les **6,1 %** rapportés par un autre étudiant concernent également 1M de pas, mais son protocole d'évaluation et ses seeds sont inconnus : ce n'est pas une comparaison contrôlée. Notre score final de **4,14 %** est inférieur à ce chiffre rapporté.
- Les résultats officiels du [dépôt Crafter](https://github.com/danijar/crafter) utilisent leurs propres protocoles ; ils donnent un contexte, pas une mesure directement comparable à notre expérience.

L'ancien essai à 150k pas avait donné **2,65 ± 0,07 %** sur d'autres épisodes finaux (5001–5050), avec ε = 0. Il documente la progression du projet, mais le gain numérique entre 150k et 1M ne doit pas être traité comme une comparaison parfaitement appariée : les runs et les épisodes finaux diffèrent. Les fichiers de cet essai ne sont pas conservés dans `runs/` ; le chiffre vient des notes de l'époque et n'est pas reproductible depuis ce dossier.
