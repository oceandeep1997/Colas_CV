from model_building.model_creation import *
import torch
import torch.nn as nn
import config
import ipdb


class multi_output_model_colas:
    def __init__(self, number_outputs: int) -> None:
        """
        class for colas cv challenge : it coordinates colas_model_single_output models for each target.
        args:
            - number_outputs: int>0, number of different labels/outputs to predict
        """
        self.number_outputs = number_outputs
        models = []
        for i in range(self.number_outputs):
            model_cnn = single_output_model_vgg()
            models.append(colas_model_single_output(model_cnn))
        self.models = models

    def train(
        self,
        df_train: pd.DataFrame,
        df_val: pd.DataFrame,
        learning_rate: float = config.learning_rate,
        batch_size: int = config.batch_size,
        dataset_path: str = config.dataset_path,
        num_epochs=config.number_epochs,
    ) -> None:
        """
        sequentially trains colas cnn model for each target/objective/label

        inputs:
            - df_train: pd.DataFrame containing the training data
            - df_val : pd.DataFrame containing the validation data
        """
        train_columns = df_train.columns
        for i in range(self.number_outputs):
            print(f"We're starting training for {train_columns[i+1]}")
            colonnes = [train_columns[0], train_columns[i + 1]]

            transform = create_image_transform()
            train_data = df_train[colonnes]
            train_data = Colas_Dataset(
                train_data, os.path.join(dataset_path, "train"), transform
            )
            val_data = df_val[colonnes]
            val_data = Colas_Dataset(
                val_data, os.path.join(dataset_path, "train"), transform
            )
            f1_score_val = self.models[i].train(
                train_data, val_data, learning_rate, batch_size, num_epochs
            )
            print(
                f"for {train_columns[i+1]}, the f1_score of val set is {f1_score_val:.3f}"
            )
            print("\n")
            print("\n")

    def predict_proba(self, df_test, batch_size):
        test_columns = df_test.columns
        predictions = np.array((len(df_test, self.number_outputs)))
        for i in range(self.number_outputs):
            colonnes = [test_columns[0], test_columns[i + 1]]
            test_data = df_test[colonnes]
            test_data = Colas_Dataset(test_data)
            y_pred_proba = self.models[i].predict_proba(test_data, batch_size)
            predictions[:, i] = np.array(y_pred_proba)
        return predictions
