from Task_1.subtask_1 import Regularized_autoencoder, Undercomplete_autoencoder, Variational_autoencoder
import data_preprocess
import torch
from sklearn.model_selection import train_test_split
import routine


def main():
    print("using torch version:", torch.__version__)
    print("will use cuda:", torch.cuda.is_available())
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    data = data_preprocess.DataLoader("data_identity.csv", "./data_transaction.csv")
    X_train, X_test = train_test_split(data.X_data, test_size=0.2, random_state=42)
    print("data has been loaded")

    # Undercomplete autoencoder
    UA = Undercomplete_autoencoder.UnderCompleteAutoencoder(866, 10).to(device)
    print("Undercomplete autoencoder has been created")

    print("Validating before train")
    UA.validate(X_test, 128, routine.loss_function, device)
    UA.fit(X_train, 128, 20, routine.loss_function, torch.optim.Adam(UA.parameters(), lr=0.001), device)
    UA.validate(X_test, 128, routine.loss_function, device)

    torch.save(UA.state_dict(), "autoencoders/UnderComplete.pth")

    # Regularized autoencoder
    RA = Regularized_autoencoder.RegularizedAutoencoder(866, 10).to(device)
    print("Regularized autoencoder has been created")

    print("Validating before train")
    RA.validate(X_test, 128, routine.loss_function, device)
    RA.fit(X_train, 128, 20, routine.loss_function, torch.optim.Adam(RA.parameters(), lr=0.001), device)
    RA.validate(X_test, 128, routine.loss_function, device)

    torch.save(RA.state_dict(), "autoencoders/Regularized.pth")

    # Variational autoencoder
    VA = Variational_autoencoder.VariationalAutoencoder(866, 10).to(device)
    print("Variational autoencoder has been created")

    print("Validating before train")
    VA.validate(X_test, 128, routine.loss_function, device)
    VA.fit(X_train, 128, 20, routine.loss_function, torch.optim.Adam(VA.parameters(), lr=0.001), device)
    VA.validate(X_test, 128, routine.loss_function, device)

    torch.save(VA.state_dict(), "autoencoders/Variational.pth")


if __name__ == "__main__":
    main()
