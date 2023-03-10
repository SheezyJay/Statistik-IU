import pandas as pd


class ExcelUtilities:
    def __init__(self, file_path):
        self.df = pd.read_excel(file_path)

    def clean_inv_data(self):
        df = self.df.iloc[1:].loc[~((self.df['Geschlecht'] == -99) | (
            self.df['Durchschnittsnote im Abitur'] == -99) | (self.df['Gewissenhaftigkeit1'] == -99))]

        invalid_rows = df[(df['Geschlecht'] == -99) | (
            df['Durchschnittsnote im Abitur'] == -99) | (df['Gewissenhaftigkeit1'] == -99)]

        if invalid_rows.empty:
            self.df = df
            return self.df
        else:
            return None
