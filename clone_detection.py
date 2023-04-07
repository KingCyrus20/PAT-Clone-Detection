from data_loader import load_big_clone_bench
from data_preprocess import big_clone_bench_preprocess

import click
from sklearn.metrics import accuracy_score
import pandas as pd
from joblib import load
import matplotlib.pyplot as plt

def load_model(classifier):
    if classifier == 'svm':
        # Get the SVM model
        trained_model = load('svm.pkl')
        pass
    elif classifier == 'random_forest':
        # get the Random Forest model
        trained_model = load('random_forest.joblib')
        pass
    return trained_model

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
    trained_model = load_model(classifier)    

    # Evaluate the model on the validation set
    val_predictions = trained_model.predict(val_df.drop("target", axis=1))
    # Save val predictions to a CSV file
    pd.DataFrame({"validation_prediction": val_predictions}).to_csv(val_output, index=False)

    # get the validation accuracy
    val_accuracy = accuracy_score(val_df["target"], val_predictions)

    # Evaluate the model on the test set
    test_predictions = trained_model.predict( test_df.drop("target", axis=1))
    # Save test predictions to a CSV file
    pd.DataFrame({"test_prediction": val_predictions}).to_csv(test_output, index=False)

    #get the test accuracy
    test_accuracy = accuracy_score(test_df["target"], test_predictions)

    # Print results
    click.echo(f'Validation accuracy: {val_accuracy}')
    click.echo(f'Test accuracy: {test_accuracy}')

def predict_custom_data(custom_data, classifier, prediction_output):
    bcb_format_data = pd.read_csv(custom_data, names=["id", "id1", "id2", "func1", "func2", "label"])
    data_dicts = bcb_format_data.to_dict('records')
    data_df = big_clone_bench_preprocess(data_dicts)

    trained_model = load_model(classifier)

    predictions = trained_model.predict(data_df.drop("target", axis=1))
    pd.DataFrame({"prediction": predictions}).to_csv(prediction_output, index=False)
    click.echo(f'Predictions saved to {prediction_output}')

@click.command()
@click.option('--custom_data', type=click.Path(), default="", help="csv file containing snippets in the same schema as https://huggingface.co/datasets/code_x_glue_cc_clone_detection_big_clone_bench")
@click.option('--prediction_output', type=click.Path(), default="prediction.csv", help="Path to save custom dataset prediction")
@click.option('--classifier', type=click.Choice(['svm', 'random_forest']), default='random_forest', help='Set the trained model')
@click.option('--val_output', type=click.Path(), default='val_prediction.csv', help="Path to save the val prediction")
@click.option('--test_output', type=click.Path(), default='test_prediction.csv', help="Path to save the test prediction")
def main(custom_data, prediction_output, classifier, val_output, test_output):
    if custom_data == "":
        evaluate_clone_detection(classifier, val_output, test_output)
    else:
        predict_custom_data(custom_data, classifier, prediction_output)

if __name__ == '__main__':
    main()
