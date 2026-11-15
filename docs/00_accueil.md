# Crafter — guide du projet

## En une minute

Un agent reçoit une image du monde Crafter, choisit une des 17 actions et reçoit une récompense. Le réseau apprend la valeur attendue de chaque action. L'objectif expérimental est de dépasser l'agent aléatoire et, si la référence est disponible, la baseline du coding agent mentionnée dans le sujet.

## Parcours de lecture

0. [00_GUIDE_COMPLET_ORAL.md](00_GUIDE_COMPLET_ORAL.md) : guide complet de révision pour l'oral (script des 11 slides, repères code, chiffres clés et Q&A).
1. [01_methode.md](01_methode.md) : comprendre le code et la fonction objectif.
2. [02_protocole_et_resultats.md](02_protocole_et_resultats.md) : voir ce qui est mesuré et ce qui manque encore.
3. [03_rendu_et_slides.md](03_rendu_et_slides.md) : texte prêt à placer dans les slides, en anglais.
4. [04_oral_questions.md](04_oral_questions.md) : explication orale et questions probables.
5. [05_recherche_baseline.md](05_recherche_baseline.md) : vérification documentaire autour de la baseline « coding agent ».

Le code source est dans le dossier parent et l'ensemble des résultats légers dans `runs/`. L'archive Moodle (`Rendu_final_Calmets_Pezet/Calmets_Kilian.zip`) contient le code, un README anglais, les figures, la configuration et les courbes scalaires des deux entraînements, les petits fichiers d'évaluation finale et la trace de comportement ; elle exclut les notes d'oral françaises (`docs/`), les épisodes d'entraînement complets et les checkpoints volumineux.
