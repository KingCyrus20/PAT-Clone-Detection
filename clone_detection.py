from data_loader import load_big_clone_bench
from data_preprocess import big_clone_bench_preprocess

import click
from sklearn.metrics import accuracy_score
import pandas as pd
from joblib import load
import matplotlib.pyplot as plt


@click.command()
@click.option('--classifier', type=click.Choice(['svm', 'random_forest']), default='random_forest', help='Set the trained model')
@click.option('--val_output', type=click.Path(), default='val_prediction.csv', help="Path to save the val prediction")
@click.option('--test_output', type=click.Path(), default='test_prediction.csv', help="Path to save the test prediction")
def evaluate_clone_detection(classifier, val_output, test_output):
    """
    evaluate the clone detection model on the given validation, and test datasets.
    """
    # Load validation datasets
    val_bcb = load_big_clone_bench("validation")
    val_df = big_clone_bench_preprocess(val_bcb)

    # load test dataset
    test_bcb = load_big_clone_bench("test")
    test_df = big_clone_bench_preprocess(test_bcb)

    # load the trained model
    if classifier == 'svm':
        # Get the SVM model
        trained_model = load('svm.pkl');
        pass
    elif classifier == 'random_forest':
        # get the Random Forest model
        trained_model = load('random_forest.joblib');
        pass

    # Evaluate the model on the validation set
    val_predictions = trained_model.predict(val_df.drop("target", axis=1))
    # Save val predictions to a CSV file
    pd.DataFrame({"validation_prediction": val_predictions}).to_csv(val_output, index=False)

    # get the validation accuracy
    val_accuracy = accuracy_score(val_df["target"], val_predictions)


    # Evaluate the model on the test set
    test_predictions = trained_model.predict( test_df.drop("target", axis=1))
    # Save val predictions to a CSV file
    pd.DataFrame({"test_prediction": val_predictions}).to_csv(test_output, index=False)

    #get the test accuracy
    test_accuracy = accuracy_score(test_df["target"], test_predictions)

    # Print results
    click.echo(f'Validation accuracy: {val_accuracy}')
    click.echo(f'Test accuracy: {test_accuracy}')



if __name__ == '__main__':
    evaluate_clone_detection()
