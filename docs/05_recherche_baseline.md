# Recherche de la baseline « coding agent »

## Conclusion

**Aucun score ni artefact identifiable pour cette baseline n'est fourni dans les documents du projet.** On ne peut donc pas vérifier si notre agent la dépasse. La comparaison expérimentale vérifiable reste celle avec l'agent aléatoire exécuté sur les mêmes 50 graines d'évaluation.

## Éléments vérifiés

| Source | Ce qu'elle dit |
|---|---|
| `assignment_v0.pdf`, p. 1 | L'objectif évoque une « baseline obtained by a coding agent », sans score, modèle, lien, protocole ou fichier de résultats. |
| `assignment_v0.pdf`, p. 2 | La règle opérationnelle demande une performance supérieure à l'agent aléatoire ; les slides doivent montrer cette comparaison. |
| `assignment_v0.pdf`, p. 3–4 | Les recommandations, liens et références ne renvoient pas à une baseline de coding agent. |
| `01_value_based.pdf`, p. 81–84 | Présente Crafter et son arbre technologique, sans résultat de coding agent. OCR OXR-1.0 effectué sur les **152 pages** : 152 résultats, zéro erreur ; aucune occurrence de « baseline » ou « coding agent ». Le « Rainbow » de la p. 148 concerne un tableau Atari-57, pas Crafter. |
| Notes de travail antérieures du projet (`01_cours_value_based_condense.md`, `02_plan_projet.md`), non incluses dans l'archive | Le plan constatait déjà que cette référence est non chiffrée ; ses nombres Rainbow/Random viennent du benchmark Crafter, pas du coding agent du cours. |
| [Dépôt officiel Crafter](https://github.com/danijar/crafter) et [répertoire `scores`](https://github.com/danijar/crafter/tree/main/scores) | Publient Random, RND, Plan2Explore, PPO, Rainbow, DreamerV2 et Human. Aucune entrée « coding agent ». |
| [Dépôt des implémentations de référence](https://github.com/danijar/crafter-baselines) | Contient les mêmes familles d'agents ; aucune baseline de coding agent. |

## Chiffres publics à ne pas confondre

Le [tableau officiel](https://github.com/danijar/crafter#scoreboards) donne **Random : score 1,6 %, reward 2,1**, **PPO : 4,6 %, reward 4,2**, **Rainbow : 4,3 %, reward 6,0** avec un budget de 1 million de pas d'environnement. Ces nombres servent de contexte, mais ne sont pas la baseline du coding agent. Notre agent a lui aussi été entraîné 1 million de pas par seed, puis évalué sur 50 épisodes frais ; même à budget d'entraînement égal, une comparaison directe de ses **4,14 %** avec les chiffres publics reste limitée par les différences de protocole d'évaluation.

**Phrase pour l'oral :** « L'énoncé mentionne une baseline produite par un coding agent, mais n'en communique ni la valeur ni le protocole. Nous ne prétendons donc pas l'avoir battue. Notre comparaison reproductible montre 4,15 de reward moyen contre 1,32 pour l'aléatoire sur les mêmes graines d'évaluation. »

Si l'enseignant fournit le score et son protocole, comparer la même métrique, le même budget d'interactions et le même protocole d'évaluation avant toute conclusion.

La transcription OCR intégrale a été produite dans un dossier de travail `tmp/ocr_output/` (un fichier par page et `manifest.json`), qui n'est inclus ni dans l'archive ni dans toutes les copies du dossier ; le PDF original reste la référence. Le texte natif du PDF a également été vérifié, car l'OCR peut manquer des éléments de diagrammes (par exemple une partie de l'arbre technologique de la page 84).
