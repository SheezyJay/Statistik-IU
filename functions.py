from sklearn.linear_model import LinearRegression

from utilities import ExcelUtilities
import pandas as pd
import numpy as np
import scipy.stats as stats
from sklearn.metrics import r2_score, mean_squared_error
df = ExcelUtilities('excel-Statistik.xlsx').clean_inv_data()
df_male = df[df['Geschlecht'] == 1]
df_female = df[df['Geschlecht'] == 2]


def count_consolidated(df):
    print("Konsolidierte Werte:")

    print(df['Geschlecht'].value_counts())
    print(df['Geschlecht'].value_counts(normalize=True))

    print(df['Durchschnittsnote im Abitur'].value_counts())
    print(df['Durchschnittsnote im Abitur'].value_counts(normalize=True))

    print(df['Gewissenhaftigkeit1'].value_counts())
    print(df['Gewissenhaftigkeit1'].value_counts(normalize=True))


def contingency_tables(df):
    print("Kontingenztabellen:")

    print(pd.crosstab(
        df['Geschlecht'], df['Durchschnittsnote im Abitur']))

    print(pd.crosstab(df['Geschlecht'], df['Gewissenhaftigkeit1']))

    print(pd.crosstab(df['Durchschnittsnote im Abitur'],
          df['Gewissenhaftigkeit1']))


def analyse_abinote(df):
    print("Analyse Abinote:")

    print(f"Median: {df['Durchschnittsnote im Abitur'].median()}")
    print(f"Durchschnitt: {df['Durchschnittsnote im Abitur'].mean()}")
    print(f"Standardabweichung: {df['Durchschnittsnote im Abitur'].std()}")
    print(f"Varianz: {df['Durchschnittsnote im Abitur'].var()}")
    print(
        f"Interquartilsabstand: {df['Durchschnittsnote im Abitur'].quantile(0.75) - df['Durchschnittsnote im Abitur'].quantile(0.25)}")
    print(f"Modus: {df['Durchschnittsnote im Abitur'].mode()}")


def correlation(df):
    print("Berechnung Korrelation:")

    # calculate
    corr_male, p_value_male = stats.pearsonr(
        df_male['Durchschnittsnote im Abitur'], df_male['Gewissenhaftigkeit1'])

    corr_female, p_value_female = stats.pearsonr(
        df_female['Durchschnittsnote im Abitur'], df_female['Gewissenhaftigkeit1'])

    # comparison
    print("korrelation")
    print(f" Männlich: {corr_male}, Weiblich: {corr_female}")


def regression(df):
    print("Berechnung Regression:")

    reg_male = LinearRegression().fit(
        df_male[['Gewissenhaftigkeit1']], df_male['Durchschnittsnote im Abitur'])
    reg_female = LinearRegression().fit(
        df_female[['Gewissenhaftigkeit1']], df_female['Durchschnittsnote im Abitur'])

    # coefficients
    print("Regressionsgleichung")
    print(f"Männlich: y = {reg_male.intercept_} + {reg_male.coef_[0]} * x")
    print(f"Weiblich: y = {reg_female.intercept_} + {reg_female.coef_[0]} * x")


def model_quality(df):
    print("Berechnung Modellqualität")
    p = 1

    reg_male = LinearRegression().fit(
        df_male[['Gewissenhaftigkeit1']], df_male['Durchschnittsnote im Abitur'])
    reg_female = LinearRegression().fit(
        df_female[['Gewissenhaftigkeit1']], df_female['Durchschnittsnote im Abitur'])

    y_male_pred = reg_male.predict(df_male[['Gewissenhaftigkeit1']])
    y_female_pred = reg_female.predict(df_female[['Gewissenhaftigkeit1']])

    r2_male = r2_score(df_male['Durchschnittsnote im Abitur'], y_male_pred)
    r2_female = r2_score(
        df_female['Durchschnittsnote im Abitur'], y_female_pred)

    n_male = df_male.shape[0]
    adj_r2_male = 1 - (1 - r2_male) * (n_male - 1) / (n_male - p - 1)

    n_female = df_female.shape[0]
    adj_r2_female = 1 - (1 - r2_female) * (n_female - 1) / (n_female - p - 1)

    print(f"R-Squared Männlich: {r2_male}")
    print(f"Adjusted R-Squared Männlich: {adj_r2_male}")
    print(f"R-Squared Weiblich: {r2_female}")
    print(f"Adjusted R-Squared Weiblich: {adj_r2_female}")


def select_and_execute_function(df):
    options = {

        1: count_consolidated,
        2: contingency_tables,
        3: analyse_abinote,
        4: correlation,
        5: regression,
        6: model_quality
    }

    while True:

        print("Bitte wählen Sie eine Funktion aus (0=Alle Funktionen ausführen):")
        for key, value in options.items():
            print(f"{key}: {value.__name__}")

        selection = input("Auswahl (exit zum Beenden): \n")

        if selection == "exit":
            break

        try:
            selection = int(selection)
            if selection in options:
                options[selection](df)

                print("\n")
            elif selection == 0:
                for key in options:

                    options[key](df)
                    print("\n")

            else:
                print("Ungültige Auswahl")
        except ValueError:
            print("Ungültige Auswahl")
