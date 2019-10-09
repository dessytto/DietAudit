# DietAudit (Insight Data Science project): Nutritional adviser for people following restrictive diets.

* [Slide Deck](https://docs.google.com/presentation/d/1hOtW9uzhZfo8JJnF5rh6y9LbC6lZDA7wYaOp-Ui0rSg/edit#slide=id.g5cdacf5423_2_0)

## Motivation:
In the US, millions of people follow restrictive diets which exclude entire food groups. In addition, people's personal preferences further limit their food choices. Long-term adherence to such diets often results in vitamin and mineral deficiencies.

I developed a nutritional adviser which generates shopping suggestions for groceries based on user input. The suggestions are optimized for nutritional density given the user's personal dietary constraints and calorie restrictions.

### Primary Data sources:
* [USDA SR28](https://www.ars.usda.gov/northeast-area/beltsville-md-bhnrc/beltsville-human-nutrition-research-center/nutrient-data-laboratory/docs/usda-national-nutrient-database-for-standard-reference/) - USDA National Nutrient Database for Standard Reference
* [NIH Nutrient Recommendations](https://ods.od.nih.gov/Health_Information/Dietary_Reference_Intakes.aspx) - USDA National Nutrient Database for Standard Reference

### Main machine learning algorithms used:
PCA, t-SNE, K-means clustering, Gaussian Mixture Models for clustering, non-negative matrix factorization, greedy optimizer, cosine similarity  
