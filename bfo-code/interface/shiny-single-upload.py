#how to run: command line: shiny run --reload shiny-single-upload.py
from shiny import App, ui, render, reactive
import pandas as pd
import re

# --- UI ---
app_ui = ui.page_fluid(
    ui.h2("Upload Feature and Targets"),
    ui.input_file("file", "Choose a CSV file", accept=[".csv"]),
    ui.output_table("table")
)

# --- Server ---
def server(input, output, session):
    @reactive.Calc
    def csv_data():
        file_info = input.file()
        if file_info is None:
            return None
        try:
            # Read CSV from uploaded file path
            df = pd.read_csv(file_info[0]["datapath"],usecols = [1,2,3,4])
            return df
        except Exception as e:
            print(f"Error reading CSV: {e}")
            return None

    @output
    @render.table
    def table():
        df_x = csv_data()
        if df_x is None:
            return pd.DataFrame({"Message": ["Please upload a CSV file."]})
        df_x.columns = [re.sub('[^A-Za-z0-9Δ]+', '_', element) for element in df_x.columns]
        return df_x


# --- App ---
app = App(app_ui, server)

"""
if __name__ == "__main__":
    app.run()
"""