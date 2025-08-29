"""
loads pre-trained model
"""
import os
from pycaret.classification import load_model, predict_model

def get_model(model_path):
	"""
	model_path: str, path to model

	loads and returns pipeline using pycaret

	returns
		pipeline: sklearn.pipeline.Pipeline
	"""
	# ensure file exists
	assert os.path.exists("{}.pkl".format(model_path)), "Path to saved wheat pipeline {}.pkl not found".format(model_path)

	# load model
	model = load_model(model_path)
	return model


def make_prediction(model, df):
	"""
	model: sklearn.pipeline.Pipeline
	df: pd.DataFrame

	runs classification on supplied dataframe

	returns
		predictions: pd.DataFrame, cols=["Predicted", "Confidence"]
	"""
	# run prediction
	predictions = predict_model(model, data=df) # cols=["prediction_label", "prediction_score"]

	# obtain columns
	return predictions.rename(columns={"prediction_label": "Predicted", "prediction_score": "Confidence"})