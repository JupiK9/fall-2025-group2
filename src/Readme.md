## Src Folder

```
└── src
      ├── component
      │           ├── EDA.py
      │           ├── __init__.py
      │           ├── csv_combiner.py
      │           ├── data_transformer_html.py
      │           ├── optimization.py
      │           ├── pdf_generator.py
      │           ├── pdf_processor.py
      │           ├── pipeline_main.py
      │           ├── popularity.py
      │           ├── regression_analysis.py
      │           └── utils.py
      │
      ├── data
      │      ├── FairfaxCounty
      │      │               ├── Item Sales Reports - Mar May 2025
      │      │               ├── May 2025 Breakfast production records
      │      │               ├── May 2025 Lunch production records
      │      │               └── Menus
      │      ├── clean-data
      │      │            ├── data_breakfast.csv
      │      │            ├── data_lunch.csv
      │      │            └── sales.csv
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
      │      └── results
      │                ├── Baseline Budget
      │                ├── EDA
      │                ├── Lower Budget Bounds
      │                └── Upper Budget Bounds
      │
      ├── docs
      │
      ├── main_code
      │           └── main.py
      │
      ├── preprocess
      │            ├── html-processing
      │            └── pdf-processing
      │
      ├── shellscript
      │             └── data-downloader.sh
      │
      ├── tests
      │       ├── test_html_csv_combiner.py
      │       ├── test_optimization.py
      │       ├── test_pdf_processing
      │       └── test_popularity.py
      │
      └── requirements.txt

```