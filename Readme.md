# Smart School Food Service Analytics: AI-Driven Demand Forecasting and Waste Reduction Using Fairfax County Public Schools Data

## Table of Contents
- Project Overview
- Dataset Overview
- Data Pipeline Diagram
- Folder Structure
- Installation Steps
- Processes and Scripts

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
│           └── 5_Final.py
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
    │      │         ├── Lower Budget Bounds
    │      │         └── Upper Budget Bounds
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

## Installation Steps

You can install the following modules through pip

```bash
pip install -r src/requirements.txt
```

## Processes and Scripts
