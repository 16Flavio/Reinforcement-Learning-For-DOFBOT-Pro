# Identification des servos du DOFBOT Pro

Ce dossier sert à régler le modèle MuJoCo (`3DModels/dofbot_pro.xml`) pour que ses servos se comportent
comme ceux du vrai bras.

- `record_robot.py` : tourne **sur le robot**, envoie des consignes connues et enregistre la réponse des servos en CSV.
- `fit_xml.py` : tourne **sur le PC**, rejoue ces consignes dans MuJoCo et ajuste les paramètres du XML.

> Sur le vrai bras, seul `--gentle` a tourné, sur le servo 1, avant l'ajout de la relecture décrite dans
> « Cadence du bus » : la version actuelle n'a été testée qu'avec des servos simulés (voir « Ce qui a été
> vérifié sans robot »).

## Pourquoi

La politique RL est entraînée en simulation puis transférée sur le bras. Les valeurs actuelles du XML
(`kp=20`, `kv=0.3`, `damping=0.05`, `armature=0.001`) sont des suppositions. Si le servo simulé est plus
rapide, moins amorti ou sans retard par rapport au vrai, la politique apprend à exploiter un robot qui
n'existe pas. On mesure donc la réponse des vrais servos, puis on règle la simulation pour qu'elle réponde
de la même façon. Les valeurs trouvées servent ensuite de centre pour la randomisation de domaine.

## Principe

1. **Exciter** : on envoie à chaque servo des consignes connues (échelons, sinusoïde de fréquence croissante).
2. **Mesurer** : on relit l'angle du servo aussi vite que la carte le permet (environ toutes les 13 ms) avec l'heure de chaque lecture.
3. **Ajuster** : on rejoue exactement les mêmes consignes dans MuJoCo et on cherche les paramètres qui
   minimisent l'écart entre l'angle simulé et l'angle mesuré.

## Le modèle ajusté

Chaque servo est un actionneur `<position>` de MuJoCo sur une articulation :

```
couple = borne(kp * (consigne - q) - kv * vitesse, ± forcerange)
         - damping * vitesse - frictionloss * signe(vitesse)
inertie = inertie des segments + armature
```

| Paramètre | Sens physique | Où il est écrit dans le XML |
|---|---|---|
| `kp` | raideur de l'asservissement en position | `<position kp=…>` |
| `kv` | amortissement de l'asservissement | `<position kv=…>` |
| `damping` | frottement visqueux ; fixe la vitesse maximale quand le couple sature (`forcerange / damping`) | `<joint damping=…>` |
| `armature` | inertie du moteur et du réducteur vue de l'articulation | `<joint armature=…>` |
| `frictionloss` | frottement sec | `<joint frictionloss=…>` |
| retard | temps mort entre l'envoi de la consigne et le début du mouvement | pas d'équivalent XML : `data/fit_results.csv`, à reproduire dans l'environnement RL |
| butées | course réelle (test `limits`) | `<joint range=…>` et `<position ctrlrange=…>` |

`forcerange` (couple maximal) n'est **pas** ajusté : avec des positions seules, un servo fort et un servo
faible ayant moins de frottement et d'inertie donnent la même courbe. Le couple maximal fixe donc l'échelle
de toutes les autres valeurs. Mettez dans le XML la valeur mesurée ou celle de la fiche technique **avant**
de lancer l'ajustement.

Valeurs annoncées par Yahboom pour le DOFBOT Pro (couple de blocage, 1 kgf·cm = 0,098 N·m) :

| Servos | Couple annoncé | `forcerange` |
|---|---|---|
| 1 à 5 | ≥ 15 kgf·cm à 7,4 V | `-1.47 1.47` |
| 6 (pince) | ≥ 6 kgf·cm à 6 V | `-0.59 0.59` |

## Les tests de `record_robot.py`

Position de départ (`HOME`) : bras à la verticale (90° partout), pince à 120°. Un seul servo bouge à la
fois, les autres restent en `HOME`.

| Test | Ce que fait le bras | Ce qu'on en tire |
|---|---|---|
| `timing` | rien (lectures et écritures sans mouvement) | durée d'un cycle écriture + lecture des six servos, pauses comprises, donc fréquence de contrôle maximale ; lectures ratées |
| `steps` | sauts de ±10° et ±30° autour du centre, puis 60° d'un bout à l'autre, à vitesse maximale | retard, vitesse maximale, dépassement, erreur finale |
| `chirp` | sinusoïde de ±15° qui accélère de 0,2 à 1,5 Hz en 10 s | suivi d'une consigne qui bouge en continu, comme celle d'une politique |
| `limits` (sur demande) | amène lentement les servos 1 à 5 aux deux bouts de leur course | course réellement atteinte |

Avant tout mouvement, le script relit les six servos et s'arrête si l'un ne répond pas. Après chaque test,
il affiche un bilan `OK` / `PROBLEM` : le servo finit-il à moins de 3° de sa consigne, et moins de 5 % des
lectures ont-elles échoué ? `Ctrl+C` arrête le test et ramène le bras en `HOME`.

Fichiers écrits (angles en degrés Yahboom, tels qu'envoyés à `Arm_Lib` et relus) :

| Fichier | Colonnes |
|---|---|
| `timing.csv` | `operation, n, mean_ms, median_ms, max_ms, failed_reads, reads` |
| `steps_servo<N>.csv`, `chirp_servo<N>.csv` | `run, t, cmd, t_cmd, pos, hold1…hold6` : nom de l'essai, instant de la lecture [s] (0 = début de la consigne), consigne en cours, instant où elle a été envoyée, angle relu (vide si la lecture a échoué), consignes maintenues sur les six servos |

## Cadence du bus

Deux particularités de la carte du bras, mesurées sur le servo 1 (aucune ne produit d'erreur) :

- **Une opération trop rapprochée de la précédente est ignorée.** Sans pause, les trois consignes envoyées
  juste après une lecture réussie ont été perdues ; avec 10 ms de pause, toutes sont passées.
- **Relire le servo qu'on vient de lire échoue toujours** (`None`), que la pause soit de 0 ou de 5 ms. La
  demande suivante réussit, de même qu'une lecture qui suit une écriture ou la lecture d'un autre servo.

`record_robot.py` laisse donc une pause avant chaque opération (`READ_GAP` et `WRITE_GAP`, en haut du
fichier) : 5 ms entre deux lectures, 10 ms avant et après une écriture (la valeur des exemples Yahboom).
Après une lecture ratée, il redemande aussitôt, une fois. Conséquences :

- un servo est relu environ toutes les 13 ms pendant un échelon, toutes les 25 ms pendant le chirp (une
  consigne par lecture) ;
- un cycle complet « écrire les six consignes, relire les six angles » dure environ 75 ms, soit 13 Hz au
  plus (mesuré sur le bras) : c'est la borne à respecter pour la fréquence de contrôle de la politique,
  sauf à relire moins de servos.

L'environnement RL qui pilotera le vrai bras devra respecter les mêmes règles.

## Le mode `--gentle`

À lancer en premier sur le robot. Il fait les mêmes tests, mais en douceur :

| | Normal | `--gentle` |
|---|---|---|
| Échelons | 10°, 30° et 60°, vitesse maximale | 10° et 20°, exécutés en 0,5 s |
| Chirp | ±15°, jusqu'à 1,5 Hz, 10 s | ±5°, jusqu'à 0,5 Hz, 5 s |
| Déplacements entre les tests | 2 s | 4 s |
| Test des butées | sur demande | refusé |
| Dossier de sortie | `data/` | `data_gentle/` |
| Durée pour 6 servos | environ 5 min | environ 3 min 30 |

Il permet de vérifier, sans brusquer le bras : que la librairie et le bus répondent, que chaque servo
bouge dans le bon sens et arrive à sa consigne, que les CSV s'écrivent, et que `Ctrl+C` fonctionne.

Les mesures de `--gentle` **ne servent pas à ajuster le modèle** : les servos n'y vont pas à vitesse
maximale, alors que le modèle MuJoCo suppose une consigne appliquée immédiatement. On peut quand même les
passer à `fit_xml.py --data sysid/data_gentle --dry-run --plot` pour vérifier que toute la chaîne
fonctionne, sans tenir compte des valeurs obtenues.

## Procédure

Sur le robot :

1. Arrêter l'application d'usine, qui utilise le même bus : `arm-stop`.
2. Vérifier la librairie : `python3 -c "from Arm_Lib import Arm_Device"`.
3. Dégager l'espace autour du bras, tenir ou fixer le socle, garder l'interrupteur d'alimentation à portée de main.
4. `python3 record_robot.py --gentle` et lire les bilans. Pendant ce test, regarder :
   - que le servo 2 penche du même côté que dans MuJoCo quand la consigne passe de 90° à 100° ;
   - que la pince se ferme quand la consigne augmente.
5. `python3 record_robot.py --servos 6`, puis `--servos 1` (ces deux servos déplacent très peu de masse).
6. `python3 record_robot.py` pour le test complet.
7. Recopier `data/` dans `sysid/` sur le PC.

Sur le PC :

8. `uv run python sysid/fit_xml.py --plot --dry-run` : ajuste sans toucher au XML et écrit
   `data/fit_servo<N>.png` (robot contre MuJoCo, avant et après l'ajustement). Regarder les courbes.
9. `uv run python sysid/fit_xml.py --plot` : écrit les valeurs dans le XML. L'original est gardé en
   `dofbot_pro.xml.bak`.

Options utiles : `--servos 2 3` (les deux scripts), `--tests steps chirp`, `--out`, `--data`, `--xml`,
`--dry-run` (sur `record_robot.py` : faux servos, sans robot).

## Comment `fit_xml.py` ajuste

- Les servos sont traités un par un. Pour chacun, tous ses essais (échelons et chirp) sont rejoués avec
  la consigne enregistrée, les autres servos tenant leur pose.
- Écart minimisé : somme des carrés de `angle simulé(t − retard) − angle mesuré(t)`, en degrés, sur
  toutes les lectures.
- Optimiseur : moindres carrés non linéaires (`scipy.optimize.least_squares`), sur le logarithme des
  cinq gains et sur le retard, depuis deux points de départ (les valeurs actuelles du XML et un jeu
  « servo raide »). Le meilleur des deux est gardé.
- Contacts et butées sont désactivés pendant l'ajustement : les tests sont des mouvements libres.
- Durée : environ 1 à 1,5 min par servo.
- Le script affiche l'erreur (RMSE en degrés) avant et après, et signale un paramètre collé à une borne,
  signe que les données ne permettent pas de le déterminer.

## Hypothèses à vérifier sur le vrai bras

| Hypothèse | Origine | Comment vérifier |
|---|---|---|
| Servos 1 à 5 : `q = (deg − 90) · π/180` | pilote MoveIt de Yahboom | **vérifié sur les cinq servos**. À 120°, les servos 2, 3 et 4 penchent le bras au-dessus du plateau du socle, le servo 1 le fait tourner dans le sens inverse des aiguilles d'une montre vu de dessus, le servo 5 fait tourner la pince dans ce même sens. Le servo 5 tournait à l'envers dans le modèle : l'axe de `Arm5_Joint` a été inversé dans le XML (`0 1 0` au lieu du `0 -1 0` de l'URDF Yahboom) |
| Pince : 30° = ouverte (`q = 0`), 180° = fermée (`q = −π/2`) | rapport de 0,6 du pilote Yahboom ; sens d'après le modèle et les exemples Yahboom (135° serre un cube, 60° relâche). Le pilote MoveIt de Yahboom associe au contraire `q = 0` à 180° | **sens vérifié** : la pince est plus fermée à 135° qu'à 60° |
| Durée `0` = vitesse maximale | tutoriel Yahboom | comparer un échelon normal et un échelon `--gentle` |
| `Arm_Lib` du Pro identique à celle du DOFBOT classique | **vérifié** : code source copié depuis le robot (`Arm_Lib/Arm_Lib.py`, V1.0.1, bus I2C 7) | — |

Ces conversions sont dans la table `SERVOS` de `fit_xml.py`, les appels à la librairie dans la classe
`Arm` de `record_robot.py`.

## Limites

- **Lecture en degrés entiers, tronqués** : une consigne de 100° peut se relire 99°. Cela limite la
  précision sur `kp` et surtout sur `frictionloss`.
- **Le modèle obtenu vaut pour des consignes à vitesse maximale** (durée 0). Il faut piloter le bras de la
  même façon au déploiement.
- **Masses** : celles de l'URDF semblent sous-estimées (environ 0,5 kg au total). Elles ne sont pas
  ajustées ici ; une pesée du bras permettrait de les corriger.
- **Un seul servo bouge à la fois** : les effets de couplage entre articulations ne sont pas mesurés.

## Ce qui a été vérifié sans robot

- `record_robot.py` tourne de bout en bout avec `--dry-run` et avec un substitut d'`Arm_Lib` reproduisant
  les particularités lues dans son code source (durée entière obligatoire, `except:` nu qui avale
  `Ctrl+C`, angles tronqués). L'arrêt par `Ctrl+C` a été testé dans les deux cas.
- Les faux servos de `--dry-run` ont les deux particularités de la vraie carte (« Cadence du bus »). Sans
  les pauses ni la relecture, ils reproduisent les bilans des deux essais faits sur le robot (premier
  essai : une lecture sur deux ratée et des échelons non exécutés ; deuxième, avec les pauses : 558 lectures
  ratées sur 1118 contre 565 sur 1138 sur le bras). Avec le script actuel, toutes les lectures réussissent.
- `fit_xml.py` a été testé sur des enregistrements synthétiques produits par MuJoCo avec des paramètres
  connus, échantillonnés et tronqués comme sur le robot : `damping`, `armature` et le retard sont retrouvés
  à quelques pourcents près, `kp` et `kv` à environ 15 % près, `frictionloss` est surestimé (de 15 à 90 %).
  L'erreur finale (0,5 à 0,8°) correspond à la troncature des lectures.

## Sécurité

- Le script n'envoie que des consignes de position dans les plages documentées. Il ne modifie ni la
  calibration, ni les identifiants des servos, ni le firmware.
- Le risque est mécanique. Par ordre décroissant : le test `limits` sur le servo 2 (bras tendu à
  l'horizontale, au-dessus du socle), les sauts de 60° à vitesse maximale sur les servos 2 à 4, le premier
  déplacement vers `HOME` depuis une pose quelconque.
- Pour réduire les amplitudes du test normal, modifier `STEPS` en haut de `record_robot.py`
  (par exemple `[10]` au lieu de `[10, 30]`).
- L'arrêt par `Ctrl+C` est logiciel : en cas de blocage du Jetson, seule l'alimentation arrête le bras.

## Sources

- [Présentation du DOFBOT Pro par Yahboom](https://www.yahboom.store/blogs/news/unboxing-reviewing-dofbot-pro) :
  couple des servos.
- [Dépôt Yahboom DOFBOT-Pro (tutoriels PDF)](https://github.com/YahboomTechnology/DOFBOT-Pro) :
  « Control Single Servo », « Read Servo Angle », « Control All Servos », « Drive the real machine »,
  « Manage APP control services ».
