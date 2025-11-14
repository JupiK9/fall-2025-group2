import os, sys
from pathlib import Path

CUR_DIR = Path(__file__).resolve().parent
SRC_DIR = CUR_DIR.parent
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from component.pipeline_main import (html_csv_pipeline, pdf_pipeline, popularity_pipeline, optimization_pipeline)

if __name__ == "__main__":
    """
    Main function to run the entire pipeline.
    """
    
    html_csv_pipeline()
    pdf_pipeline()
    popularity_pipeline()
    optimization_pipeline()
    print("\nData Analyzing Complete!"
    "\nPlease navigate to the Streamlit app to view the results."
    "\nSet demo / pages as the working directory."
    "\nThen run the following command in your terminal:" 
    "\nstreamlit run streamlit_app.py"
    )