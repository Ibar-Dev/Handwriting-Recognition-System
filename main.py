import argparse
from data_preprocessing.dataset_loader import load_emnist
from model.train import train_model
from app.gui import create_gui

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--train', action='store_true', help='Entrenar modelo')
    parser.add_argument('--gui', action='store_true', help='Lanzar interfaz gráfica')
    args = parser.parse_args()
    
    if args.train:
        print("Cargando dataset...")
        train, test, _ = load_emnist()
        print("Entrenando modelo...")
        train_model(train, test)
        print("Modelo entrenado y guardado.")
    
    if args.gui:
        print("Iniciando interfaz gráfica...")
        gui = create_gui("saved_models/cnn_model.h5")
        gui.launch()

if __name__ == "__main__":
    main()