import mnist_loader
import network

def main():
    print("Trwa ładowanie danych MNIST...")
    # 1. Wczytanie danych za pomocą load_data_wrapper
    training_data, validation_data, test_data = mnist_loader.load_data_wrapper()
    
    # KRYTYCZNA POPRAWKA DLA PYTHONA 3: 
    # Rzutowanie wyników funkcji zip() na listy, aby działało len() i random.shuffle()
    training_data = list(training_data)
    test_data = list(test_data)
    
    print(f"Pomyślnie załadowano {len(training_data)} próbek treningowych i {len(test_data)} testowych.")
    print("Inicjalizacja sieci neuronowej...")
    
    # 2. Tworzenie architektury sieci
    # Pierwsza warstwa: 784 neurony (bo obrazki MNIST to 28x28 pikseli = 784)
    # Druga warstwa (ukryta): np. 30 neuronów (możesz eksperymentować z tą liczbą)
    # Trzecia warstwa (wyjściowa): 10 neuronów (bo rozpoznajemy cyfry od 0 do 9)
    net = network.Network([784, 30, 10])
    
    print("Rozpoczęcie trenowania (SGD)...")
    
    # 3. Odpalenie treningu (Stochastic Gradient Descent)
    # Parametry: 
    # - 30 epok (przejść przez cały zbiór)
    # - mini_batch_size = 10 (aktualizacja wag co 10 obrazków)
    # - eta (learning rate) = 3.0
    net.SGD(training_data, epochs=30, mini_batch_size=10, eta=3.0, test_data=test_data)

if __name__ == "__main__":
    main()