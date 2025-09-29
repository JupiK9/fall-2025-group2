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

### Optimization Approaches
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