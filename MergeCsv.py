import os
import csv

def merge_csv_files(directory_path, output_file):
    # Vérifier si le répertoire existe
    if not os.path.exists(directory_path):
        print(f"Erreur : Le répertoire '{directory_path}' n'existe pas.")
        return

    # Liste pour stocker les noms de fichiers CSV
    csv_files = [f for f in os.listdir(directory_path) if f.endswith('.csv')]
    
    if not csv_files:
        print("Aucun fichier CSV trouvé dans le répertoire.")
        return

    with open(output_file, mode='w', newline='', encoding='utf-8') as merged_file:
        writer = csv.writer(merged_file)
        header_written = False

        for csv_file in csv_files:
            file_path = os.path.join(directory_path, csv_file)
            with open(file_path, mode='r', newline='', encoding='utf-8') as input_file:
                reader = csv.reader(input_file)
                header = next(reader)  # Lire l'en-tête

                # Écrire l'en-tête une seule fois
                if not header_written:
                    writer.writerow(header)
                    header_written = True

                # Écrire les lignes du fichier CSV
                for row in reader:
                    writer.writerow(row)

    print(f"Tous les fichiers CSV du répertoire '{directory_path}' ont été fusionnés dans '{output_file}'.")

# Exemple d'utilisation
merge_csv_files("c:/Users/rtahar/JobSpy", "fichier_fusionne.csv")  # Fusionner tous les CSV dans le dossier spécifié