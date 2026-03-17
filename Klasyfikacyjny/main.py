import mnist_loader
import network
import network2

def main():
    print("Trwa ładowanie danych MNIST...")
    # 1. Wczytanie danych
    training_data, validation_data, test_data = mnist_loader.load_data_wrapper()
    
    # Rzutowanie na listy dla Pythona 3
    training_data = list(training_data)
    validation_data = list(validation_data)
    test_data = list(test_data)
    
    print(f"Pomyślnie załadowano dane.")
    print("Inicjalizacja ulepszonej sieci neuronowej (Cross-Entropy, L2 Regularization)...")
    
    # 2. Tworzenie architektury sieci (korzystamy z domyślnej CrossEntropyCost)
    # Rozmiar sieci: 784 neurony wejściowe, 30 ukrytych, 10 wyjściowych
    net = network2.Network([784, 30, 10], cost=network2.CrossEntropyCost)
    
    print("Rozpoczęcie trenowania (SGD)...")
    
    # 3. Odpalenie treningu z nowymi parametrami
    # eta = 0.5 (lepsza inicjalizacja wag i inna funkcja kosztu pozwala na inny learning rate)
    # lmbda = 5.0 (parametr regularyzacji L2, który zapobiega overfittingowi)
    evaluation_cost, evaluation_accuracy, training_cost, training_accuracy = net.SGD(
        training_data=training_data, 
        epochs=30, 
        mini_batch_size=10, 
        eta=0.5, 
        lmbda=5.0, 
        evaluation_data=validation_data, # Używamy zbioru walidacyjnego do oceny
        monitor_evaluation_accuracy=True,
        monitor_training_accuracy=True,
        monitor_training_cost=True,
        monitor_evaluation_cost=True
    )
    
    # ==== PIERWSZA SIEĆ ====
    # net.SGD(training_data, epochs=30, mini_batch_size=10, eta=3.0, test_data=test_data)

if __name__ == "__main__":
    main()