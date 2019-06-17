# Grab AI SEA: Safety Challenge

This repository is the submission for the AI challenge for S.E.A hosted by Grab. The selected challenge is __Safety__. The challenge can be found via this [website](https://www.aiforsea.com/challenges).

#### Author

__by__: Satsawat Natakarnkitkul (Net)

__Email__: n.satsawat@gmail.com

__Country__: Thailand

__Motivation__: This challenge is very interesting in so many ways, but as I use Grab nearly everyday. Hence this challenge is the most impact to the Grab users.

<br>

# Repository structure

#### Notebook

- The notebook `Grab AI Challenge_Safety_Data Exploration.ipynb` is mainly used as part of data understanding and EDA for sensor data provided by Grab. You may not run this notebook, but it will provide some understandings and explanation onto telemetry data of the sensor world.
- The notebook `Grab AI Challenge_ML model comparison.ipynb` is purposely created to train and test ML techniques to produce the final model as well as try on feature engineering and other data transformation.

#### Model

- This folder contains the final model object to be used for prediction.

#### Code

- This folder contains the final python source code for manipulating, creating new features and predicting the data set.

#### Img

- This folder contains the image embedded onto EDA and other notebooks.

<br>

# Run Instruction

The model is used to predict the safety of the trip as such the assumption is that this is __not__ the real time prediction (online), but rather an offline (data for each booking ID is available). The transformation is the aggregation of each booking ID onto single observations and feed into the model for prediction.

To run the prediction, please use __Safety_Prediction.py__ in the `code` directory.

1. The script in `code` folder will read in the __feature__ data file within `data/safety/features` folder.
  - If there's any change in the data path, please adjust the `DATA_DIR` onto the correct folder respectively.
2. The script will automatically run the feature transformation and engineering.
3. The script will load the XGBoost model object from `model` directory to make a prediction.
4. The script will save the prediction with bookingID onto `../output/all_prediction.csv` file.
5. If `LABEL_IND = True` in the script, it will attempt to run evaluation between the prediction with true label.
  - The true label file should be in the `data/safety/labels` folder with the proper bookingID and label columns.
  - If there's any change to `labels` folder, please adjust this to `LABEL_DIR` in the script respectively.
  - If the evaluation is not neeeded, you can turn this off by setting `LABEL_IND = False` in the script.
