import os
from pathlib import Path

resultat = []

for d in os.listdir("../data/bbc/"):
    repertoire = os.path.join('../data/bbc', d)
    if os.path.isdir(repertoire):
        for f in os.listdir(repertoire):
            try:
                full_path = os.path.join(repertoire, f)

                contenu = Path(full_path).read_text()

                element = {
                    'fichier': full_path,
                    'contenu': contenu,
                }
                
                resultat.append(element)
            except:
                print("probleme avec {}".format(full_path))
