# shiny_app.py
from shiny import App, ui, render, reactive
import pandas as pd

# --- UI ---
app_ui = ui.page_fluid(
    ui.h2("Upload CSV File (One at a Time)"),
    ui.input_file(
        "file_upload",
        "Choose a CSV file",
        accept=[".csv"],  # Restrict to CSV files
        multiple=False    # Only one file at a time
    ),
    ui.output_text_verbatim("file_info"),
    ui.output_table("data_preview")
)

# --- Server ---
def server(input, output, session):

    # Reactive value to store the DataFrame
    @reactive.Calc
    def uploaded_df():
        file = input.file_upload()
        if not file:
            return None
        try:
            # Read CSV into DataFrame
            df = pd.read_csv(file[0]["datapath"])
            return df
        except Exception as e:
            return pd.DataFrame({"Error": [str(e)]})

    # Show file info
    @output
    @render.text
    def file_info():
        file = input.file_upload()
        if not file:
            return "No file uploaded yet."
        return f"Uploaded file: {file[0]['name']} ({file[0]['size']} bytes)"

    # Show data preview
    @output
    @render.table
    def data_preview():
        df = uploaded_df()
        if df is None:
            return pd.DataFrame()
        return df.head(10)  # Show first 10 rows

# --- Run App ---
app = App(app_ui, server)

if __name__ == "__main__":
    app.run()