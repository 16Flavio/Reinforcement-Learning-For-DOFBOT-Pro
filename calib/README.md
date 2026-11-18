# Gain et zéro des servos, mesurés avec la caméra du bras

Ce dossier sert à vérifier qu'un angle de servo donne bien la même pose sur le vrai bras et dans le modèle
MuJoCo, et à mesurer l'écart quand ce n'est pas le cas. Deux choses sont mesurées :

- le **gain** des servos 1 à 4 : de combien de degrés l'articulation tourne quand le servo relit un degré de plus ;
- le **zéro** des servos 2, 3 et 4 : l'angle de servo auquel l'articulation est au zéro du modèle (90 en théorie).

La conversion est `q [rad] = gain · (angle du servo − zéro) · π/180`.

- `capture_robot.py` : tourne **sur le robot**, amène le bras dans une centaine de poses et photographie une mire.
- `fit_offsets.py` : tourne **sur le PC**, retrouve gains et zéros dans ces images.
- `charuco_A4.pdf` : la mire à imprimer. `poses.csv` : les poses de prise de vue.

> État : deux captures ont été faites sur le vrai bras. Les résultats et ce qu'il faut en retenir sont dans
> « Résultats sur le vrai bras ».

## Principe

Un servo ne voit pas son propre décalage : s'il est monté 3° de travers, il relit quand même 90°. Il faut un
capteur extérieur, et la caméra fixée sur le dernier segment en est un.

1. La caméra photographie une mire posée à plat devant le bras, depuis des poses variées.
2. Pour chaque image, le modèle MuJoCo prédit où se trouve la caméra d'après les angles relus sur les servos,
   donc où les coins de la mire devraient apparaître.
3. On cherche les gains et les zéros qui font coïncider les coins prédits et les coins détectés. La position
   de la mire sur la table et celle de la caméra sur le bras sont ajustées en même temps.

Les poses de `poses.csv` sont de deux sortes :

- autour de deux poses de base, chaque servo est **balayé seul** par pas de 3°, aussi loin que la mire reste
  dans l'image (25 à 35° : c'est le champ de la caméra qui limite). La rotation de la caméra, comparée aux
  angles relus, donne le gain du servo sans passer par le modèle du bras. Entre deux balayages, le bras
  repasse par la pose de base, ce qui montre s'il revient bien au même endroit ;
- 24 poses variées, qui couvrent une plage plus large de chaque servo.

**Lectures fines.** `Arm_Lib` arrondit l'angle relu au degré. C'est ce qui limitait la première capture : un
servo qui s'affaisse d'un demi-degré pendant qu'un autre bouge ne se voit pas. `capture_robot.py` demande
donc à la carte la position brute des servos, par la même requête qu'`Arm_Lib` mais sans l'arrondi (un pas
vaut 1/12°). Il vérifie au démarrage que ces lectures concordent avec celles d'`Arm_Lib` et affiche
`fine readings: on`. Si ce n'est pas le cas, il continue au degré près et le dit.

## Procédure

Sur le PC :

1. Imprimer `charuco_A4.pdf` à 100 % (taille réelle, sans « ajuster à la page »), puis mesurer le côté d'une
   case à la règle. Il doit faire 38 mm ; noter la valeur mesurée.

Sur le robot, avec les dossiers `calib/` et `sysid/` côte à côte (`capture_robot.py` réutilise `sysid/record_robot.py`) :

2. Poser la feuille à plat sur la table, en paysage, devant le bras : du côté où le plateau est court, celui
   vers lequel le bras se penche quand les servos 2 à 4 sont sous 90°. Elle est centrée gauche-droite sur le
   bras. Les poses sont calculées pour l'emplacement de la première capture, où le centre de la mire était à
   39 cm de l'axe vertical du servo 1 : **laisser la feuille où elle était**, ou la remettre à 3 cm près. Le
   sens de la mire n'a pas d'importance. La scotcher.
3. Arrêter l'application d'usine (`arm-stop`) et lancer la caméra dans un autre terminal :
   `roslaunch orbbec_camera dabai_dcw2.launch`. C'est la commande de ROS 1 (Noetic), celui de notre robot ;
   avec ROS 2 ce serait `ros2 launch orbbec_camera dabai_dcw2.launch.py`, et le script accepte les deux.
4. `python3 capture_robot.py --check` : le bras va dans la première pose et enregistre `data/check.png`.
   Regarder l'image : la mire doit en occuper la plus grande partie. Regarder aussi la ligne `fine readings`.
5. `python3 capture_robot.py` : les 108 poses, en 7 minutes environ, pince ouverte. Ne pas toucher la feuille
   ni le socle pendant la capture. Après un `Ctrl+C`, les poses déjà prises sont enregistrées quand même.
6. Recopier `data/` dans `calib/` sur le PC.

Sur le PC :

7. `uv run python calib/fit_offsets.py --square 38.0`, avec le côté mesuré à l'étape 1.

Pour essayer toute la chaîne sans robot : `uv run python calib/capture_robot.py --dry-run --out /tmp/essai`, puis
`uv run python calib/fit_offsets.py --data /tmp/essai`. Le bras simulé a des zéros de 92, 88,5 et 91° et des
gains de 0,97, 1,05, 1,00 et 1,03, que l'ajustement doit retrouver.

Pour une feuille placée ailleurs : `uv run python calib/fit_offsets.py --plan --board-at X Y` recalcule
`poses.csv` (centre de la mire dans le repère du socle, en mètres ; +y est l'avant du bras).

## Lire le résultat

Sortie de l'essai `--dry-run` :

```
105 images with the board; servo readings in 1/12 deg steps
servo   gain    +/-    angle at the model's zero [deg]   +/-     (q [rad] = gain * (deg - zero) * pi / 180)
  1    0.971   0.001    90 (not measurable, see README.md)
  2    1.051   0.001     91.97                           0.20
  3    1.001   0.001     88.40                           0.24
  4    1.030   0.002     91.25                           0.22
gains from the camera's rotation alone, without the arm model: servo 1 0.972 +/- 0.000, servo 2 1.051 +/- 0.001, ...
board corners predicted within 0.21 px; servo readings corrected by 0.02, 0.01, 0.01, 0.01 deg rms (servos 1 to 4)
board: tilted by 0.0 deg from the base plane, 14 mm above the model's table
camera: lens -10 mm along the housing, +2 mm forward, -1 mm away from the arm; image turned by 181 deg, yaw -0.8 deg
repeatability: back at a base pose, the camera is within 0.2 mm of where it was
cross-check, even and odd images fitted apart: zeros [91.93 88.39 91.25] and [91.99 88.41 91.27] deg, gains ...
```

- Dans la table `SERVOS` de `sysid/fit_xml.py`, chaque servo a un `deg0` et une `pente` : le zéro remplace le
  `90`, et la pente devient `gain * DEG`. La même conversion doit servir à l'environnement qui pilotera le
  vrai bras. `data/offsets.csv` contient les mêmes valeurs. Rien n'est modifié automatiquement.
- Un zéro à moins d'un `+/-` de 90, ou un gain à moins d'un `+/-` de 1, n'est pas significatif : laisser la
  valeur théorique.
- `gains from the camera's rotation alone` : les mêmes gains, obtenus sans le modèle du bras. Les deux
  estimations doivent s'accorder. La ligne n'apparaît qu'avec les lectures fines.
- `cross-check` : une image sur deux, ajustées séparément. Les deux moitiés doivent donner les mêmes zéros et
  les mêmes gains ; sinon le script avertit, et le `+/-` de la table est trop optimiste.
- `repeatability` : écart entre les passages par une même pose de base. Plus de 2 mm veut dire que le socle a
  bougé ou qu'un servo a du jeu ; les zéros ne peuvent alors pas être plus précis que cela.
- Les autres lignes sont des contrôles : coins prédits à moins d'un pixel environ, mire à plat, objectif à
  quelques millimètres de sa place dans le modèle 3D. Le script avertit quand l'un d'eux échoue.

## Précision attendue

Sur huit bras simulés avec les défauts vus ou soupçonnés sur le vrai (gains faux de ±5 %, affaissement,
calibration de la caméra fausse de 0,5 %, segments faux de 0,5 %, caméra déplacée de 3 mm) :

| Lectures | Flexion cachée | Erreur sur les zéros (moyenne / pire) | Erreur sur les gains |
|---|---|---|---|
| fines | non | 0,3° / 0,5° | 0,2 % |
| fines | oui | 0,4° / 1,0° | 0,4 %, sauf servo 2 : 2 % |
| au degré | non | 0,2 à 0,9° / 1,5° | 0,3 à 0,9 % |
| au degré | oui | 0,3 à 1,2° / 2,3° | 0,5 à 1,9 % |

« Flexion cachée » : le bras plie sous son poids sans que les servos le voient (jusqu'à 2° au servo 2). Elle
ressemble à un gain supérieur à 1 sur le servo 2, et la caméra ne peut pas distinguer les deux.

## Résultats sur le vrai bras

### Deuxième capture (`data/`) : balayages, lectures fines

108 images sur 108 utilisables, lectures fines actives et conformes à `Arm_Lib`, coins prédits à 0,5 px.

| Servo | Gain | Zéro | Zéro sur chaque moitié des images |
|---|---|---|---|
| 1 | 0,978 (0,965 d'après la rotation de la caméra seule) | non mesurable | |
| 2 | 1,010 | 90,2 | 88,9 et 91,4 |
| 3 | 0,984 | 89,2 | 90,5 et 87,8 |
| 4 | 1,010 | 91,9 | 91,9 et 91,6 |

Ce qu'on peut en retenir :

- **Servos 2 et 3 : pas de décalage mesurable.** Leur zéro est à 90° à ±1,5° près ; on ne peut pas faire mieux
  (voir plus bas). Garder 90.
- **Servo 4 : 92° au lieu de 90**, stable d'une moitié à l'autre, mais seulement si la caméra est montée comme
  dans le modèle 3D. Sinon c'est la caméra qui penche de 2°.
- **Servo 1 : il tourne 2 à 3,5 % de moins que ce qu'il relit.** Les deux captures et les deux méthodes
  donnent ce résultat (0,967 à la première capture ; 0,967 et 0,962 sur les deux balayages).
- **Servos 2 à 4 : gain de 1 à 2 % près.** Les 5 % du servo 2 et les 3 % du servo 4 de la première capture ne
  sont pas confirmés : c'était un effet des lectures au degré.

Pourquoi les zéros ne sont pas plus précis : **le bras ne revient pas au même endroit**. Renvoyé cinq fois
dans la même pose de base, il place la caméra jusqu'à 5 mm et 1° plus loin. Deux causes, vues sans aucun modèle :

- les servos s'arrêtent avant leur consigne, à un endroit qui dépend d'où ils viennent : en moyenne 1,5° avant
  pour le servo 2 (jusqu'à 3,2°), 1,2° pour le servo 3 (jusqu'à 2,4°), 0,5° pour le servo 4. La même consigne
  envoyée deux fois donne des lectures qui diffèrent jusqu'à 1,7°. Les lectures le voient ;
- les articulations 2 à 4 ont du jeu que les servos ne voient pas : pendant un balayage, la caméra tourne de
  0,3° (jusqu'à 0,6°) de plus ou de moins que ce que disent les lectures, et le premier pas après un
  changement de sens perd jusqu'à 1,3°. Le servo 1, lui, suit à 0,1° près.

Pour la simulation, cela veut dire qu'un décalage aléatoire d'environ ±1° par articulation et une erreur de
1 à 3° entre consigne et position atteinte décrivent mieux le vrai bras qu'un zéro réglé au dixième de degré.

### Première capture (`data_1/`) : 24 poses variées, lectures au degré

Elle donnait des gains de 0,967, 1,050, 0,999 et 1,032 et des zéros incohérents d'une moitié des images à
l'autre (4 à 5° d'écart sur les servos 2 et 3). Seul le gain du servo 1 a été confirmé par la deuxième.

- Un décalage du servo 1 n'explique rien : imposer +5° au servo 1 ne change rien, la rotation de la feuille
  l'absorbe. La focale de la caméra non plus (ajustée librement, elle reste à 0,1 % de celle du pilote).
- La hauteur de la table dans le repère du socle n'est pas déterminée : entre −13 et +3 mm selon la capture
  et les images retenues, pour −15 mm dans le modèle.

## Hypothèses et limites

- **Servo 4** : la caméra est fixée juste après lui. Une caméra montée un peu penchée et un servo 4 décalé
  donnent exactement les mêmes images. Le script suppose donc que la caméra est inclinée comme dans le modèle
  3D de Yahboom ; un support faussé d'un degré se retrouve tel quel dans le zéro du servo 4. Son gain, lui,
  n'en dépend pas.
- **Servo 1** : son zéro n'est pas mesurable, il revient à tourner la feuille sur la table. Son gain l'est.
- **Zéro extrapolé** : pendant la capture, le servo 4 reste entre 10 et 65°, loin de 90°. Son zéro est donc
  extrapolé avec le gain : 1 % d'erreur sur le gain le déplace d'un demi-degré. La conversion reste juste
  dans la plage balayée, celle où le bras travaille devant lui.
- **Calibration de la caméra** : avec ROS, `capture_robot.py` enregistre celle que publie le pilote
  (`intrinsics.json`). Sans elle (`--camera 0`), `fit_offsets.py` calibre la caméra sur les images de la mire,
  ce qui est nettement moins bon.
- **Longueurs des segments** : ce sont celles du modèle.
- **Mire et socle sur la même table** : l'ajustement suppose qu'ils sont parallèles à 0,5° près.
- **Sens de montage de la caméra** : le script essaie les quatre orientations de l'image.

## Ce qui a été vérifié sans robot

- Toute la chaîne tourne avec `--dry-run` : faux servos qui s'affaissent, arrondissent leurs lectures comme
  les vrais et donnent leur position brute comme la carte, images de la mire dessinées d'après le modèle.
- Les essais simulés du tableau « Précision attendue ».
- Le nombre de coins que le planificateur prévoit dans chaque image tombe à un ou deux coins près sur ce que
  la première capture a réellement détecté.
- Dans les poses prévues et pendant les trajets, la pince ouverte du modèle reste à 7 cm ou plus de la table,
  et à 3,9 cm si les servos 2 à 4 penchent chacun de 4° de plus vers l'avant.
- Les lectures fines ont depuis été vérifiées sur la vraie carte (elles concordent avec `Arm_Lib` à son
  arrondi près sur les 108 poses). **Non vérifié** : la lecture directe des images par OpenCV (`--camera 0`).

## Si quelque chose ne marche pas

| Symptôme | Cause probable |
|---|---|
| `roslaunch: command not found` | ROS n'est pas chargé dans ce terminal : `source /opt/ros/noetic/setup.bash` |
| `[dabai_dcw2.launch] is neither a launch file in package [orbbec_camera]…` | l'espace de travail du robot n'est pas chargé : `source ~/dofbot_pro_ws/devel/setup.bash` (chemin à vérifier sur le robot) |
| `ROS is not set up in this terminal` ou `No ROS master` | même chose dans le terminal de `capture_robot.py`, ou pilote de la caméra non lancé |
| `No image on /camera/color/image_raw` | pilote de la caméra non lancé, ou sujet d'un autre nom : `rostopic list` le donne, à passer avec `--topic` |
| `fine readings: OFF` | la carte ne répond pas comme prévu à la requête brute ; la capture reste utilisable, au degré près |
| peu de marqueurs vus dans presque toutes les poses | feuille déplacée ; refaire `--check`, ou recalculer les poses avec `fit_offsets.py --plan --board-at X Y` |
| `not enough images to fit` | même cause, ou image trop sombre ou floue |
| coins prédits à plusieurs pixels | mauvaise valeur de `--square`, feuille gondolée ou déplacée pendant la capture |

## Sécurité

Le bras reste penché vers l'avant au-dessus de la feuille. La pince passe à 7 cm de la table dans le modèle ;
sur le vrai bras, dont on ne connaît justement pas les zéros, elle peut passer plus bas. Ne rien laisser sur
la feuille, dégager l'espace autour du bras, tenir le socle comme pour `sysid/record_robot.py`, et garder la
main près de `Ctrl+C`, qui arrête la capture et ramène le bras à la verticale.

## Sources

- [Tutoriel Yahboom « Dabai_DCW2 camera »](https://www.yahboom.net/study/DOFBOT-Pro/12547) : commande de
  lancement du pilote avec ROS 2 et nom du sujet de l'image couleur.
- [Guide Forlinx sur la DCW2 avec ROS 1 Noetic](https://www.forlinx.net/industrial-news/orbbec-dcw2-depth-camera-rk3588-ros1-guide-816.html) :
  commande `roslaunch orbbec_camera dabai_dcw2.launch`, même nom de sujet.
- [Tutoriel Yahboom « Calibration offset »](https://www.yahboom.net/study/DOFBOT-Pro/12502) : la calibration
  d'usine, qui corrige la position de la pince en x, y, z et non les angles des servos.
