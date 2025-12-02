# Smart School Food Service Analytics: AI-Driven Demand Forecasting and Waste Reduction Using Fairfax County Public Schools Data

## Table of Contents
- Project Overview
- Dataset Overview
- Data Pipeline Diagram
- Initial Steps
- Processes and Scripts
- Folder Structure

## Project Overview

Food waste in U.S. public schools presents a significant economic and environmental challenge, with approximately 530,000 tons of food discarded annually, resulting in $1.2 billion in losses. Fairfax County Public Schools (FCPS) in Virginia seeks to address this issue by leveraging data-driven insights to optimize their food service operations. Despite widespread availability of digital data from Point-of-Sale (POS) systems and kitchen production records, most districts—including FCPS—lack the analytical tools necessary to effectively utilize this information. This project analyzes FCPS data from March to May 2025 to uncover patterns in food consumption, identify inefficiencies, and recommend actionable strategies for reducing waste. Key research objectives include evaluating a la carte item popularity at the school level, assessing participation trends through year-over-year comparisons, and segmenting data by FCPS regions and High School Pyramids. Through advanced data analytics, this study aims to support FCPS in making informed, efficient, and sustainable decisions for school meal planning and delivery.

## Dataset Overview

- `data_breakfast.csv`: Details daily breakfast meal metrics for various schools.
- `data_lunch.csv`: Details daily lunch meal metrics for various schools.
- `sales.csv`: Transactional sales data for both breakfast and lunch across various schools.
- `2022-2025 Fairfax County School Student Count.csv`: Provides student enrollment numbers for various schools in Fairfax County.
- `fcps_nutrition_values.csv`: Provides a nutritional database for food items offered in schools.
- `School_Regions.geojson`: Geospatial data that defines the boundaries of the different regions for the Fairfax County Public School system.
- `unit_costs.csv`: Provides the individual unit cost of food items.

## Data Pipeline Diagram
![Capstone Data Pipeline](./demo/images/Capstone%20Data%20Pipeline.png)

## Initial Steps

### Module Installation
You can install the following modules through pip

```bash
pip install -r src/requirements.txt
```

The script will install the following modules required for the program to run:

- `pandas`
- `numpy`
- `matplotlib`
- `seaborn`
- `statsmodels`
- `scipy`
- `beautifulsoup4`
- `tqdm`
- `plotly`
- `PyPDF2`
- `pdfplumber`
- `nbformat`
- `geopandas`
- `shapely`
- `folium`
- `fpdf`
- `fpdf2`
- `streamlit`
- `streamlit-folium`
- `geojson`

### Download Required Data
The raw data for this project (HTML, PDF, GEOJSON) is hosted on Box and can be downloaded using the provided script.

**Prerequisites**
- **Windows**: You **must** use **Git Bash** to run the script. The standard Windows Command Prompt (cmd) or Powershell will not work.
- **macOS/Linux**: You can use the standard built-in terminal.
- **Software**: The script requires `wget` and `unzip`, which are typically included with Git Bash, macOS, and Linux distributions.

**Download Instructions**
Navigate to the `src/shellscripts` directory where the script is located:

```bash
cd src/shellscripts
```

Run the downloader script:

```bash
bash data-downloader.sh
```

### Run the Data Pipeline (Backend)

Before opening the dashboard, you must generate the data:

```bash
python src/main/main.py
```

### Launch the Dashboard (Frontend)

Start the local web server:

```bash
streamlit run demo/streamlit_app.py
```

## Processes and Scripts

### 1. Downloading the required data
- **Objective**: Download the required data files from Box
- **Script**: `data-downloader.sh`
- **Inputs**: None
- **Outputs**:
    - May 2025 Breakfast and Lunch Records (42 HTML Files)
    - Breakfast and Lunch Sales Transactions from March to May 2025 (6 PDF files)
    - `2022-2025 Fairfax County School Student count.csv`
    - `unit_costs.csv`
    - `fcps_nutrition_values.csv`
    - `School_Regions.geopjson`

### 2.1. Tranforming HTML record files into CSV files
- **Objective**: Transform the HTML breakfast and lunch records into individual CSV record files
- **Script**: `data-transformer_html.py`
- **Inputs**:
    - May 2025 Breakfast and Lunch Records (42 HTML Files)
- **Outputs**:
    - Individual Production Records for Individual Dates in May for Breakfast and Lunch (42 CSV files)

### 2.2. Combining CSV Production Records into simpler CSV files
- **Objective**: Combine the individual production records into two CSV files; Breakfast and Lunch.
- **Script**: `csv_combiner.py`
- **Inputs**:
    - Individual Production Records for Individual Dates in May for Breakfast and Lunch (42 CSV files)
- **Outputs**:
    - `data_breakfast.csv`
    - `data_lunch.csv`

### 3. Processing PDF Monthly Daily Sales
- **Objective**: Extract from the PDF monthly daily sales into a combined transactional sales file.
- **Script**: `pdf_processor.py`
- **Inputs**:
    - Breakfast and Lunch Sales Transactions from March to May 2025 (6 PDF files)
- **Outputs**:
    - `sales.csv`

### 4. The Main Program
- **Objective**: Analyze the input files, perform exploratory data analysis, popularity analysis, optimization, and regression analysis.
- **Script**: `main.py`
- **Input**:
    - `data_breakfast.csv`
    - `data_lunch.csv`
    - `2022-2025 Fairfax County School Student Count.csv`
    - `unit_costs.csv`
    - `fcps_nutrition_values.csv`
    - `School_Regions.geojson`
    - `sales.csv`
- **Outputs**:
    - Folder with EDA result files
    - Folder with leftover result files
    - Folder with optimization result files
    - Folder with popularity result files
    - Folder with regression result files
    - Folder with graphs and map result files

### 5. Streamlit UI Application
- **Objective**: Provide users a user interface of data analysis performed by the main program, and allow users to produce recommendation forms
- **Script**: `streamlit_app.py`
- **Inputs**:
    - Folder with EDA result files
    - Folder with leftover result files
    - Folder with optimization result files
    - Folder with popularity result files
    - Folder with regression result files
    - Folder with graphs and map result files
- **Outputs**:
    - `recommendation.pdf`

## Folder Structure
```
.
├── cookbooks
│   └── Capstone.ipynb
│
├── demo
│   ├── fig
│   ├── images
│   └── pages
│           ├── 0⚡️Quick Overview.py
│           ├── 1🏭Production EDA.py
│           ├── 2📊Sales EDA.py
│           ├── 3🔮Future Ideas.py
│           ├── 4📊Results.py
│           ├── coates_app.py
│           └── streamlit_app.py
│
├── presentation
│   └── fig
│
├── reports
│   ├── Latex_report
│   │   
│   ├── Markdown_Report
│   │
│   ├── Progress_Report
│   │                 ├── Markdown_CheatSheet
│   │                 └── Progress_Report.md
│   │
│   └── Word_Report
│                 ├── Final Report Script.docx
│                 ├── GLM Research.docx
│                 ├── Problem Statement.docx
│                 ├── Summary of UI Findings.docx
│                 └── Summary of Food Waste Research.docx
├── research_paper
│    ├── Latex
│    │   └── Fig
│    └── Word
│
└── src
    ├── component
    │           ├── csv_combiner.py
    │           ├── data_transformer.py
    │           ├── optimization.py
    │           ├── pdf_generator.py
    │           ├── pdf_processor.py
    │           ├── pipeline_main.py
    │           ├── popularity.py
    │           └── regression_analysis.py
    ├── data
    │      ├── clean-data
    │      │            ├── data_breakfast.csv
    │      │            ├── data_lunch.csv
    │      │            └── sales.csv
    │      ├── FairfaxCounty
    │      │               ├── Item Sales Reports - Mar May 2025
    │      │               ├── May 2025 Breakfast production records
    │      │               ├── May 2025 Lunch production records
    │      │               └── Menus
    │      ├── leftover-data
    │      │               ├── breakfast_leftover_rate_by_school.csv
    │      │               └── lunch_leftover_rate_by_school.csv
    │      ├── optimization-data
    │      │                   ├── annual_school_breakdown_baseline.csv
    │      │                   ├── annual_school_breakdown_lower_bound.csv
    │      │                   ├── annual_school_breakdown_upper_bound.csv
    │      │                   ├── monthly_items_baseline.csv
    │      │                   ├── monthly_items_lower_bound.csv
    │      │                   ├── monthly_items_upper_bound.csv
    │      │                   └── school_food_item_optimization_ilp.csv
    │      ├── popularity-data
    │      │                 ├── breakfast_net_consumption_by_school.csv
    │      │                 └── lunch_net_consumption_by_school.csv
    │      ├── preprocessed-data
    │      │                   ├── Breakfast production
    │      │                   ├── Lunch production
    │      │                   ├── 2022-2025 Fairfax County School Student Count.csv
    │      │                   ├── data_breakfast_with_coordinates.csv
    │      │                   ├── data_lunch_with_coordinates.csv
    │      │                   ├── fcps_nutrition_values.csv
    │      │                   └── School_Regions.geojson
    │      ├── results
    │      │         ├── Baseline Budget
    │      │         ├── EDA
    │      │         ├── Lower Budget Bounds
    │      │         └── Upper Budget Bounds
    │      │
    │      └── data-downloader.sh
    ├── docs
    ├── main_code
    │           └── main.py
    ├── preprocess
    │            ├── html-processing
    │            └── pdf-processing
    ├── tests
    │       ├── test_html_csv_combiner.py
    │       ├── test_optimization.py
    │       ├── test_pdf_processing.py
    │       └── test_popularity.py
    │
    └── requirements.txt

```