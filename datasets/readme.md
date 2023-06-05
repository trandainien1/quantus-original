# Fichiers CSV
**2000idx_ILSVRC2012** sont les indices des 2000 images sélectionnées aléatoirement dans la base de données de test d'ImageNet avec :
- **1445idx_ILSVRC2012_top1** : les images où le modèle vit_b16_224 ne se trompe pas pour la top1 classe.
- **398idx_ILSVRC2012_top5** : les images où le modèle vit_b16_224 ne se trompe pas pour la top5 classe.
- **157idx_ILSVRC2012_else** : les images où le modèle vit_b16_224 se trompe.

# Fichiers JSON
Référence chaque image par l'indice de la ligne dans 2000idx_ILSVRC2012.csv. Pour chaque image, donne les informations suivante : 
- classe cible
- top1 classe
- ...
- top5 classe

