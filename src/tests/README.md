# School Meal Cost Optimization
## Fairfax County Public Schools Case Study

### Overview
This project demonstrates how to use data analysis, time-series, and linear programming to optimize meal costs and minimize food waste in school food service operations for Fairfax County Public Schools in Northern Virginia. The optimization model helps balance budget constraints, nutritional requirements, and demand forecasting.

### Problem Definition
---

**Objective:** Minimizing total meal production costs while reducing food waste and meeting nututional standards.

**Key Decision Variables**
- Determine the quantity of each meal item to produce for each school
- Resource allocation across the different schools and meal programs across Fairfax County

**Constraints:**
- Budget limitations (total and per-school)
- Minimum nutritional requirements
- Demand forecasting limits

---

## Data Preparation

### Data Transformer (HTML)
>
> This pipeline processes structured HTML reports of school lunch and breakfast production data and converts them into clean csv files for analysis.
> - Reads all `.html` files from `../data/FairfaxCounty/May 2025 Lunch Production records/May 2025 Breakfast production records` and `../data/FairfaxCounty/May 2025 Lunch Production records/May 2025 Lunch production records`.
> - Extracts tables for each school using `BeautifulSoup`.
> - Parses key metrics: planned, offered, served, discarded, and cost-related values.
> - Automatically detects the reporting date from the content or file name.
> - Saves parsed results as individual `.csv` files in the target directory.
>

### CSV Combiner (HTML)
>
> Gathers all the csv files created from the Data Transformer and combines them together to their respective period they represent.
> - Collects all `.csv` files in an output folder.
> - Merges them into one combined dataset.
> - Optionally sorts by columns like `School_Name`, `Date`, and `Identifier`.
> - Saves final merged CSVs for both **Breakfast** and **Lunch** datasets.
>

### PDF Processing
>
> Reads all PDF files from a specified folder and gathers all the relevant values from each individual PDF files and generate a combined `sales.csv` file.
>

---

## Data Cleaning and Exploratory Data Analysis
>
>
>

### Food Popularity
>
> This section in the analysis provides an analysis of menu items that students and adults are purchasing, eating, and not leaving as much left over when the food period concludes.
>
> `food_popularity`: function that provides the user the individual menu items with their total amount served to students and adults across every school in the Fairfax County Public School system.
>
> `net_consumption`: function that provides an analysis of menu items that were served to students and adults subtracted by the amount of each menu item was discarded across all the schools in the system.
>
> `leftover_rate`: 

---

## Optimization Approaches
---

#### 1. Linear Programming (LP)
Best for continuous variables like portion sizes and ingredient quantities

**Advantages**
- Efficient computation
- Guaranteed global optimum
- Well-established algorithms

**Limitations**
- Assumes linear relationships
- May produce fractional solutions

**Usage**
> #### Method 1: Optimization with Uniform Production Bounds
> 
> **Method Objective:** To minimize the total cost of producing meals across all schools, while also adding a small penalty for potential waste (defined as producing more than the historical average demand).
>
> **How it Works:** The model decides the optimal number of breakfasts and lunches to produce for each of the schools. These decisions are limited by two main constraints:
>   1. **Budget Constraint:** The total cost of meals produced for each school cannot exceed that school's specific budget, and the grand total cost cannot exceed the overall budget for the entire school system.
>
>   2. **Production Bounds:** Set a uniform production window for every school, regardless of its student population size. The model is forced to produce a quantity that is between 90% and 110% of that school's average historical demand for each meal.
>
> ---
>
> #### Method 2: Optimization with Size-Based Production Bounds
>
> **Model Objective:** Minimize the combined cost of production and waste.
>
> **How it Works:** This method uses the exact same linear programming model but changes one critial rule: the production bounds. Instead of a uniform ±10% window, the bounds are now variable and based on the student population size of the school. We have categorized the schools into sizes from "xxs" to "xxxl" and assigned them different production windows:
> 
>   - **Very small populated schools (xxs):** Have a wider production window (e.g., 80% to 120% of demand).
>
>   - **Very large populated schools (xxxl):** Have a very tight production window (e.g., 99% to 101% of demand)
>
> ***Note:*** *School size categorizing is based on the student population size for the academic 2024-2025 school year from the [School Profiles database](https://schoolprofiles.fcps.edu/schlprfl/f?p=108%3A8)*
>
> ---
>
> #### Method 3. Optimization with Monthly Aggregation
>
> **Model Objective:** Produce a monthly aggregate of recommended food item quantities for each school to produce.
> 
> **How it Works:** This method incorporates a monthly cycle, in this case 20 days, in which it aggregates together to produce the recommendation of food quantities per school based on demand for the school.
>

#### 2. Integer Linear Programming (ILP)
Used when decisions involve whole units, in our case number of meals.

**Usage**
> **Model Objective:** To minimize the total cost of producing meals across all schools. Unlike the Linear Programming Optimization model, this model ensures that the value ends with a whole number, which is ideal for meal production as we cannot serve a fraction of an item.
>
> **How it Works:** The model decides the optimal number of breakfasts and lunches to produce for each of the schools. These decisions are limited by two main constraints:
>   1. **Budget Constraint:** The total cost of meals produced for each school cannot exceed that school's specific budget, and the grand total cost cannot exceed the overall budget for the entire school system.
>
>   2. **Production Bounds:** Set a uniform production window for every school, regardless of its student population size. The model is forced to produce a quantity that is between 90% and 110% of that school's average historical demand for each meal.
>

#### 3. Multi-Objective Optimization
Balances multiple competing goals simultaneously.

**Objectives:**

1. **Minimize Cost:** Produce fewer meals in order to save money.

2. **Minimize Waste:** Produce enough meals to meet demand and avoid leftovers.

**Usage**
> **Model Objective:** To minimize the total cost of producing meals and reducing food waste for each school in Fairfax County in proportion to their school size.
>
> **How it Works:** The model recommends the optimal number of food items for breakfast and lunch for each of the schools. The recommendations are based by two constraints:
>   1. **Budget Constraint:** The total cost of meals produced for each school cannot exceed the school's specific budget, which the model has proportionally allocated funds based on student population size.
>
>   2. **Production Bounds:** This model does not use a uniform production bound, but rather the size-based production bounds used in an earlier model, where smaller schools have a wider production window and larger schools have a narrower production window.
>