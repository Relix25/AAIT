# Guide complet pour préparer l'oral — Projet Crafter (AAIT)

**Binôme :** Kilian Calmets · Mathis Pezet  
**Support de présentation :** `Calmets_Kilian.pdf` (**11 slides**, format 16:9, en anglais)  
**Algorithme implémenté :** Double DQN + retours $n$-step ($n = 3$), écrit à la main en PyTorch

---

## 1. Calendrier officiel et répartition en binôme

- **Dépôt Moodle (date limite stricte / *hard cut-off*) :** **dimanche 15 novembre 2026 à 23h59** (les deux fichiers du dossier [`Rendu_final_Calmets_Pezet/`](../Rendu_final_Calmets_Pezet/) : [`Calmets_Kilian.pdf`](../Rendu_final_Calmets_Pezet/Calmets_Kilian.pdf) et [`Calmets_Kilian.zip`](../Rendu_final_Calmets_Pezet/Calmets_Kilian.zip)). Les résultats présentés à l'oral doivent correspondre exactement à ceux déposés sur Moodle.
- **Soutenances orales (reportées suite à l'annonce Teams du 9 novembre) :** **mardi 24 novembre 2026** (première moitié des binômes) et **mardi 8 décembre 2026** (seconde moitié). *(Dans le sujet initial `assignment_v0.pdf`, l'horaire prévu était 18h00 ; l'heure exacte et l'ordre de passage n'ont pas été précisés dans l'annonce du 9 novembre.)*
- **Répartition conseillée :**
  - **Kilian :** présentation de Crafter (Slides 1–2), architecture DQN de base et fonction objectif (Slides 3–4).
  - **Mathis :** Double DQN / $n$-step, protocole expérimental, diagnostics d'entraînement, résultats et limites (Slides 5–11).
  - *Les rôles peuvent être inversés, mais chacun doit maîtriser le replay buffer, le calcul du score Crafter et les limites.*

---

## 2. Introduction en 45 secondes (à maîtriser par cœur)

> « Nous avons entraîné un agent à choisir parmi 17 actions dans Crafter à partir d'une seule image RGB $64 \times 64$. Les récompenses importantes arrivent quand il débloque un achievement pour la première fois dans l'épisode, parfois après une longue suite d'actions. Notre agent est un Double DQN avec retours $n$-step codé à partir des mécanismes du cours : Double DQN réduit la surestimation des valeurs Q, et le retour à 3 pas propage plus vite les récompenses rares aux actions précédentes.
>
> Après **1 million de pas** sur chacune de **deux seeds d'entraînement**, notre agent obtient **4,15 ± 0,16** de reward épisodique et **4,14 ± 0,60 %** de score Crafter sur 50 nouveaux épisodes par seed en politique gloutonne ($\varepsilon = 0$), contre **1,32** et **1,23 %** pour l'agent aléatoire sur ces mêmes 50 épisodes. »

**Règle critique :** préciser dès qu'on cite un `±` qu'il s'agit de l'**écart-type entre les deux moyennes par seed d'entraînement** ($N = 2$), **pas** d'un intervalle de confiance.

---

## 3. Script détaillé slide par slide (11 slides)

| # | Titre de la slide | Idée forte & chiffres à l'écran | Ce qu'il faut dire à l'oral |
|---:|---|---|---|
| **1** | **Learning to survive and craft in Crafter** | Double DQN with $n$-step returns · Kilian Calmets · Mathis Pezet | Accroche courte : agent value-based compact évalué face à une référence aléatoire appariée. |
| **2** | **The task** | $64 \times 64$ RGB · 17 actions · 1M steps/run · 22 achievements · `+1` au 1er déblocage de chaque achievement par épisode, `±0.1` par point de vie | Crafter est un monde 2D procédural et partiellement observé. Pour passer de `wood` $\rightarrow$ `table` $\rightarrow$ `wood_pickaxe` $\rightarrow$ `stone`, il faut propager le crédit sur plusieurs pas. Le score Crafter est une moyenne géométrique sur les 22 compétences : $\exp(\frac{1}{22}\sum_{i=1}^{22}\ln(1 + s_i)) - 1$ (avec $s_i$ en %). |
| **3** | **DQN foundation** | CNN (`32→64→64` + FC `512`) $\rightarrow$ 17 Q-values $\rightarrow$ $\varepsilon$-greedy (`1.0→0.05` sur 150k pas) · Replay `50 000` `uint8` · batch `32` tous les `4` pas · target copy tous les `2 000` pas · $\gamma = 0{,}99$ · Huber · Adam `1e-4` (`eps=1.5e-4`), warmup `10 000` | Le replay buffer casse la corrélation temporelle entre images successives et stocke en `uint8` pour tenir en RAM ; le réseau cible stabilisé par copie dure tous les 2 000 pas évite de poursuivre une cible mouvante ; la perte Huber + clipping de norme à `10` borne les gradients extrêmes. |
| **4** | **Our two extensions and objective** | Double DQN + 3-step return · Équations + note : $\Gamma_t = \gamma^h \cdot m_t$, $h=3$ sauf les 2 dernières transitions ($h=1$ ou $2$), $m_t=0$ si mort, $m_t=1$ si troncature 10k | **Double DQN** dissocie sélection (`argmax` par `online`) et évaluation (`Q_target(o', a*)`), réduisant le biais de surestimation. **3-step** somme 3 récompenses avant de bootstraper. Insister sur l'horizon réel $h \le 3$ en fin d'épisode (`NStepAccumulator`) et la distinction mort ($m=0$) vs troncature 10k ($m=1$). Reconnaître honnêtement qu'aucune ablation à budget égal n'isole chaque extension séparément. |
| **5** | **Experimental protocol** | 01 Training : 2 seeds × 1M · 02 Model Selection : 30 ép. (`3001–3030`) à 300k/500k/1M · 03 Final Test : 50 nouveaux ép. (`7001–7050`) par seed ($\varepsilon=0$) · 04 Random : mêmes 50 seeds | Protocole strict sans fuite de données : les 50 épisodes de test final (`7001–7050`) n'ont servi ni à régler les hyperparamètres ni à choisir le checkpoint. |
| **6** | **Training reward** | Courbe 10k bins (2 seeds, ±1 SD) vs ligne pointillée Random (`1.32`) · Vignettes : `MEAN Q-VALUE 0.02 → 1.57`, `HUBER TD LOSS 0.034 (stable)` | Le reward d'entraînement monte puis plafonne autour de `4,4–4,6`. Rappel : cette courbe inclut l'exploration ($\varepsilon \ge 0{,}05$), ce n'est pas le score final. Préciser les agrégats des vignettes : `0,02` = 1re fenêtre loggée après warmup (pas `11k`), `1,57` et `0,034` = moyennes des 2 seeds sur les `10 000` derniers pas (`scalars.csv`). |
| **7** | **Training diagnostics** | *(Nouvelle slide · point (g))* 2 panneaux `scalars.csv` (fenêtres de 1k pas, 2 seeds, ±1 SD) : Mean Q-value (`0.02` $\rightarrow$ ~`1.6`) et Huber TD loss (`0.02–0.05`) | Montre la stabilité numérique : les Q-values croissent régulièrement avec le reward sans diverger, et la perte Huber reste bornée (`0,03–0,04` après `300k`). **Observation clé :** les deux courbes plongent vers `150k–250k` juste après que $\varepsilon$ atteint son plancher `0,05` à `150k` (lecture plausible mais non prouvée : changement de distribution du replay buffer quand la politique devient gloutonne). Ce sont des diagnostics de régression, pas des métriques de jeu. |
| **8** | **Checkpoint validation** | `300k`: **3,62 %** (R 4,28) $\rightarrow$ `500k`: **3,81 %** (R 4,53) $\rightarrow$ **`1M`**: **4,08 %** (R 4,48) ✓ | Le choix du checkpoint à **1M** a été fait sur la moyenne des deux seeds sur les 30 épisodes de validation (`3001–3030`) **avant** le test final. |
| **9** | **Final test** | Reward **4,15 ± 0,16** vs **1,32** · Score **4,14 ± 0,60 %** vs **1,23 %** · Longueur **191,0 ± 1,2** vs **170,2** | Gain net et reproductible face à l'aléatoire (+2,83 reward, +2,91 points de score). Seed 1 : reward `4,04`, score `3,72 %` ; Seed 2 : reward `4,26`, score `4,56 %` (la variance entre seeds justifie d'en entraîner au moins deux). |
| **10** | **Achievement success rates** | Les 22 compétences : `collect_wood` **86 %** (vs 12 %), `place_table` **61 %** (vs 0 %), `defeat_zombie` **44 %** (vs 0 %), `make_wood_pickaxe` **10 %** (vs 0 %), `collect_stone` **1 %** (vs 0 %) | L'agent maîtrise la collecte initiale, la plantation (`84 %`), la fabrication de table et combat même des zombies, mais bloque sur les branches profondes : **0 %** de `make_stone_pickaxe` sur les 100 épisodes finaux. |
| **11** | **Behavior, limits, conclusion** | Séquence tracée + 44 % zombie / 23 % vache · Modes d'échec (`collect_drink` en boucle, 2e pioche inutile, ~1/3 de pas craft/place impossibles) · Conclusion & prochaines étapes | Être transparent sur ce qui est démontré et ce qui ne l'est pas : la séquence (`sapling` 13 $\rightarrow$ `plant` 14 $\rightarrow$ `wood` 18, 22 $\rightarrow$ `table` 23 $\rightarrow$ `wood_pickaxe` 186) montre une compétence utile mais pas une planification explicite. Une image $64 \times 64$ unique manque de mémoire dès que la table sort du champ de vision. |

---

## 4. Repères exacts dans le code (si le jury demande à le lire en direct)

- **`agent/network.py` — `QNetwork` (lignes 5–22) :**
  - Entrée `(B, 64, 64, 3)` `uint8` $\rightarrow$ `obs.permute(0, 3, 1, 2).float().div(255)` (ligne 21).
  - Encodeur CNN à 3 convolutions : `Conv2d(3, 32, 8, stride=4)`, `Conv2d(32, 64, 4, stride=2)`, `Conv2d(64, 64, 3)` + `ReLU` + `Flatten` (lignes 10–15).
  - Tête MLP : `Linear(width, 512) -> ReLU -> Linear(512, 17)` (ligne 18).
- **`agent/replay.py` :**
  - `ReplayBuffer` (lignes 6–31) : préalloue `uint8` pour `obs`/`next_obs` (`capacity = 50_000`) afin d'économiser la RAM ; `sample()` tire `batch_size = 32` indices uniformément.
  - `NStepAccumulator` (lignes 34–60) : `_pop()` (lignes 41–50) calcule $R_t^{(h)} = \sum_{k=0}^{h-1} \gamma^k r_{t+k+1}$ et $\Gamma_t = \gamma^h \prod_{k=1}^h m_{t+k}$ avec **`horizon = min(self.n, len(self.queue))`**, et `add()` (lignes 57–59) vide la file à la fin de l'épisode avec l'horizon réel ($h = 1$ ou $2$ pour les deux dernières transitions).
- **`train.py` :**
  - `parse_args()` (lignes 18–37) : sans argument, lance exactement la config retenue (`steps=1_000_000`, `algorithm="double"`, `n_step=3`, `gamma=0.99`, `learning_rate=1e-4`, `buffer_size=50_000`, `batch_size=32`, `learning_starts=10_000`, `train_frequency=4`, `target_frequency=2_000`, `epsilon_end=0.05`, `epsilon_decay_steps=150_000`).
  - `optimize()` (lignes 40–60) : lignes 49–51 pour la cible **Double DQN** (`online(next_obs).argmax(...)` puis `target(next_obs).gather(...)`), ligne 55 pour `smooth_l1_loss`, ligne 58 pour `clip_grad_norm_(online.parameters(), 10)`.
  - Masque de bootstrap (ligne 100) : `bootstrap_mask = float(info.get("discount", 0.0 if done else 1.0))` (coupe à `0` en cas de mort, conserve `1` à la troncature à 10k pas).
  - Copie dure du réseau cible (lignes 119–120) : `if step % args.target_frequency == 0: target.load_state_dict(online.state_dict())`.
- **`evaluate.py` & `metrics.py` & `report.py` :**
  - `evaluate.py` : évalue un checkpoint à $\varepsilon = 0{,}0$ par défaut (ou random si `--checkpoint` omis, ligne 48) sur des seeds dédiées (`args.seed + episode`) ; en `--trace`, enregistre `unlocked` dès que `count > previous_achievements.get(key, 0)` (lignes 56–57).
  - `metrics.py` (`summarize`, lignes 10–21) : calcule les stats uniquement sur des épisodes complets ; taux de succès = `%` d'épisodes avec `count > 0` ; score Crafter ligne 17 (`math.exp(sum(math.log1p(rate) for rate in success.values()) / 22) - 1`).
  - `report.py` (lignes 24–33) : agrège les dossiers par seed d'entraînement (`reward_mean`, `score_mean`, écarts-types entre seeds).

---

## 5. Tableau récapitulatif des chiffres à retenir

| Métrique | Seed 1 (50 ép.) | Seed 2 (50 ép.) | Moyenne Agent (2 seeds) | Aléatoire (50 ép.) |
|---|---:|---:|---:|---:|
| **Reward épisodique** | 4,04 | 4,26 | **4,15 ± 0,16** | **1,32** |
| **Score Crafter (%)** | 3,72 % | 4,56 % | **4,14 ± 0,60 %** | **1,23 %** |
| **Longueur moyenne (pas)** | 190,16 | 191,84 | **191,0 ± 1,2** | **170,2** |

- **Achievements clés (moyenne des 2 seeds vs aléatoire) :** `collect_sapling` **87 %**, `collect_wood` **86 %** (vs 12 %), `place_plant` **84 %** (vs 50 %), `wake_up` **77 %** (vs 94 %), `place_table` **61 %** (vs 0 %), `defeat_zombie` **44 %** (vs 0 %), `collect_drink` **31 %**, `eat_cow` **23 %** (vs 0 %), `make_wood_pickaxe` **10 %** (vs 0 %), `make_wood_sword` **2 %**, `collect_stone` **1 %** (vs 0 %), `make_stone_pickaxe` **0 %**.
- **Chiffres d'entraînement (`scalars.csv`) :** moyenne sur les 2 seeds du Q moyen = `0,02` à `11k` pas $\rightarrow$ `1,57` sur `990k–1M` (`1,59` dernière fenêtre) ; perte Huber = `0,034` sur `990k–1M`.

---

## 6. Questions pièges probables & réponses prêtes à l'emploi

1. **Pourquoi le facteur $\Gamma_t$ n'est-il pas toujours égal à $\gamma^3$ ?**  
   Dans `NStepAccumulator`, `horizon = min(self.n, len(self.queue))` utilise l'horizon réel $h$ en fin d'épisode : les 2 dernières transitions d'un épisode ont $h=2$ et $h=1$, donc $\gamma^2$ et $\gamma^1$. En cas de mort, `m_t = 0` annule le bootstrap ; en cas de troncature à 10 000 pas, `m_t = 1` le conserve.
2. **D'où viennent exactement `0,02 → 1,57` et `0,034` sur la Slide 6 ?**  
   De `scalars.csv` des deux runs 1M (`runs/double_nstep_1m_g4_seed*/scalars.csv`), où `q_mean` et `loss` sont loggés par fenêtres de `1 000` pas. `0,02` est la 1re fenêtre après le warmup (`step = 11 000`), `1,57` et `0,034` sont les moyennes des 2 seeds sur les `10 000` derniers pas (la toute dernière fenêtre à `1M` donne `1,59`).
3. **Pourquoi les courbes de Q et de loss plongent-elles entre 150k et 250k pas (Slide 7) ?**  
   Ce creux intervient juste après que $\varepsilon$ atteint son plancher `0,05` à `150 000` pas : la distribution des transitions entrant dans le replay buffer de `50 000` pas change brutalement quand la politique devient quasi gloutonne (lecture plausible, non démontrée isolément).
4. **Pourquoi votre agent aléatoire obtient-il 1,32 reward / 1,23 % score contre 2,1 / 1,6 % dans le papier Crafter ?**  
   Le tableau officiel évalue l'agent aléatoire sur tous les épisodes d'un budget de 1M de pas (plusieurs milliers d'épisodes). Notre référence vient de **50 épisodes** fixés (`7001–7050`) — les mêmes que notre agent pour garantir une comparaison strictement appariée (granularité de 2 % par épisode et variance plus élevée).
5. **Pourquoi `collect_drink` apparaît-il répété dans `unlocked` de `trace.jsonl` ?**  
   Crafter n'accorde la récompense `+1` qu'au **premier** déblocage d'un achievement dans l'épisode, mais son compteur interne continue d'augmenter à chaque réussite de l'action. `evaluate.py` enregistre chaque hausse du compteur (`count > previous`), ce qui rend visible cette boucle comportementale en fin d'épisode.
6. **Quels autres comportements marquants avez-vous observés dans les traces ?**  
   Au-delà de la séquence jusqu'à la pioche en bois (`186`), l'agent combat (`defeat_zombie` 44 %), chasse (`eat_cow` 23 %), fabrique parfois une **2e pioche en bois** au pas `223` sans récompense, dort rarement (`1,7 %` des pas tracés contre ~`5,9 %` aléatoire, expliquant `wake_up` 77 % vs 94 %), et environ **1/3 des pas tracés** sont des tentatives de craft/place impossibles avec l'inventaire courant (donc sans effet).
7. **Avez-vous isolé l'effet de Double DQN vs 3-step par une ablation ?**  
   Non : le gain est mesuré pour le système complet face à l'agent aléatoire ; l'ablation à budget égal fait partie des suites immédiates identifiées (avec le *frame stacking* / mémoire pour ne pas oublier la table hors champ de vision, ou le *prioritized replay*).
8. **Avez-vous battu la baseline « coding agent » du sujet ?**  
   Ni son score ni son protocole ne figurent dans `assignment_v0.pdf`, les supports de cours ou le dépôt Crafter officiel (`docs/05_recherche_baseline.md`). Nous ne prétendons pas l'avoir battue et nous nous comparons rigoureusement à la référence aléatoire appariée.
9. **Pourquoi ne pas comparer directement à un camarade à 6,1 % ou aux scores Rainbow/PPO du papier ?**  
   Le protocole d'évaluation et les seeds du camarade à 6,1 % sont inconnus, et les scores du papier Crafter (Random `1,6 %`, Rainbow `4,3 %`, PPO `4,6 %`) utilisent leur propre protocole sur 1M de pas : ils servent de **contexte** (montrant que notre `4,14 %` est cohérent), pas de comparaison contrôlée.
10. **Pourquoi avoir utilisé une instance G4 ?**  
    Crafter simule les pas séquentiellement sur CPU ; le runtime G4 a accéléré l'exécution parallèle des deux seeds (~`540` env steps/s enregistrés dans `scalars.csv`) sans rien modifier à l'algorithme ni aux hyperparamètres.

---

## 7. Fichiers complémentaires dans `docs/` et le dépôt

- [00_accueil.md](00_accueil.md) — vue d'ensemble du parcours de lecture
- [01_methode.md](01_methode.md) — équations mathématiques exactes et lien avec chaque fichier Python
- [02_protocole_et_resultats.md](02_protocole_et_resultats.md) — tableaux complets de résultats, validation et limites
- [03_rendu_et_slides.md](03_rendu_et_slides.md) — plan détaillé des slides et texte *Say:* associé
- [04_oral_questions.md](04_oral_questions.md) — banque complète de questions/réponses courtes
- [05_recherche_baseline.md](05_recherche_baseline.md) — preuves documentaires concernant l'absence de score « coding agent »
- [Calmets_Kilian.pdf](../Rendu_final_Calmets_Pezet/Calmets_Kilian.pdf) — le diaporama final à 11 slides (identique au dépôt Moodle)
- [`figures/`](../figures/) — les 6 graphiques haute résolution utilisés dans les slides
