from PIL import Image
import os

# Pfad zum Ordner mit den 16-Bit-Bildern
input_folder = "/Users/carstenschmotz/Library/CloudStorage/OneDrive-Persönlich/02_Studium/04_Master/03_Fächer/01_WS2223/SS24/Projekt/train"

# Pfad zum Ausgabeordner für die 8-Bit-Bilder
output_folder = "/Users/carstenschmotz/Desktop/8bit"
# Überprüfen und erstellen des Ausgabeordners, wenn er nicht existiert
if not os.path.exists(output_folder):
    os.makedirs(output_folder)

# Durchlaufen aller Dateien im Eingabeordner
for filename in os.listdir(input_folder):
    # Überprüfen, ob die Datei eine Bilddatei ist
    if filename.endswith(".png") or filename.endswith(".jpg") or filename.endswith(".bmp"):
        # Öffnen des Bildes
        img = Image.open(os.path.join(input_folder, filename))
        # Konvertieren des Bildes in 8-Bit
        img_8bit = img.convert("L")
        # Speichern des 8-Bit-Bildes im Ausgabeordner mit gleichem Dateinamen
        img_8bit.save(os.path.join(output_folder, filename))
        print(f"{filename} konvertiert.")
