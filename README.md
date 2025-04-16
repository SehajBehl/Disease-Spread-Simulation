# COVID-19 Simulation Using SIR and SEIR Models

This project is part of the SOFE 4820U: Modelling and Simulation course at Ontario Tech University. It focuses on simulating and analyzing the spread of COVID-19 in Canada during 2020 using epidemiological models (SIR and SEIR) combined with Monte Carlo simulations.

## Objectives

- Model and simulate the COVID-19 outbreak using classical compartmental models: SIR and SEIR.
- Compare baseline and intervention scenarios.
- Evaluate the accuracy of simulations using real-world COVID-19 case data.
- Analyze uncertainty using Monte Carlo methods.

## Techniques Used

- **SIR & SEIR differential equation models**
- **Monte Carlo simulation**
- **Parameter tuning with random sampling**
- **ODE solving with `scipy.integrate.odeint`**
- **Performance evaluation using Mean Squared Error (MSE)**

## Dataset

- Source: COVID-19 Canada dataset (`Covid_Data_Canada.csv`) from Kaggle
- Contains: Daily active case data for 2020

## Files

- `SEIR_SIR.ipynb` – Main Jupyter Notebook containing simulation logic and analysis
- `Covid_Data_Canada.csv` – COVID-19 data used for simulation
- `README.md` – Project description and setup instructions

## How to Run

1. Install required libraries:
    ```bash
    pip install numpy pandas matplotlib scipy scikit-learn
    ```
2. Open the notebook:
    ```bash
    jupyter notebook SEIR_SIR.ipynb
    ```
3. Run all cells to view simulation outputs and graphs.

## Results Summary

- The SEIR model more accurately fits real COVID-19 trends, especially during the initial phase.
- Intervention scenarios (reduced beta) show clear impact in flattening the curve.
- Monte Carlo simulations help understand the uncertainty range in prediction.

## Future Work

- Extend models with vaccination and death compartments.
- Use machine learning to estimate parameters more accurately.
- Integrate real-time data for dynamic forecasting.

## Authors

- [Parasjeet Marwah]
- [Sehaj Behl]
- [Soumil Thete]
- [Kris Biswa]
