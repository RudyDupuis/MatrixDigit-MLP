import numpy as np
from dataset import get_dataset, display_digit, add_noise
from model import MLP


def predict_digit(model: MLP, image: np.ndarray, label: int):
    prediction = model.forward(image)
    predicted_digit = np.argmax(prediction)

    print(f"Image testée (Chiffre {label}) :")
    display_digit(image)
    print(f"\nLe réseau a prédit : {predicted_digit}")
    print(f"Confiance : {prediction[0][predicted_digit]*100:.2f}%")

    if predicted_digit == label:
        print("✅")
    else:
        print("❌")


def train_model():
    images, labels = get_dataset()
    targets = np.eye(10)[labels]

    model = MLP(input_size=25, hidden_size=16, output_size=10)

    print("Début de l'entraînement...")

    epochs = 2000
    for epoch in range(epochs):
        loss = model.train(images, targets, learning_rate=0.2)

        if epoch % 200 == 0:
            print(f"Époque {epoch:4} | Erreur (Loss): {loss:.6f}")

    print("\nEntraînement terminé !")

    print("\n--- TEST DE PRÉDICTION ---")

    for i in range(10):
        predict_digit(model, images[i], labels[i])

    print("\n--- TEST DE PRÉDICTION AVEC BRUIT ---")

    for i in range(10):
        original_image = images[i]
        noisy_img = add_noise(original_image, num_pixels=4)
        predict_digit(model, noisy_img, labels[i])

    print("\n--- TEST DE PRÉDICTION ALEATOIRE ---")

    label = 4
    image = [
        [0, 0, 1, 1, 0],
        [0, 1, 0, 1, 0],
        [1, 1, 1, 1, 1],
        [0, 0, 0, 1, 0],
        [0, 0, 0, 1, 0],
    ]

    predict_digit(model, np.array(image).flatten().reshape(1, -1), label)


train_model()
