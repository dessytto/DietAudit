from flask import render_template
from flaskexample import app
from flask import request
from flask import Response


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.decomposition import NMF
import random
import io # to show the bar plot
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas
from matplotlib.figure import Figure


#     
# 
@app.route('/')
@app.route('/input')
def nut_user_input():
    return render_template("input.html")
# 
    

@app.route('/output_list', methods=['GET', 'POST'])
def call_sort_and_pick():
    # Read the input from the input page

    group = request.form['group']
    age = request.form['age']
    life_group = group+age
    calories = request.form['calories']
    if calories == '': # if blank
        calories = 2000

    #return(str(calories))

    diet = request.form['diet']

    bad_foods = str(request.form['bad_foods']).strip()
    if not bad_foods:
    	bad_foods = 'duck, natto'   
       
    #return bad_foods

    
###########################  Import and wrangle the data ###########################

    bad_foods = [x.strip() for x in bad_foods.split(',')]
    try:
        #data_all = pd.read_csv("data_cluster_Ni_noNaN.csv")
        if diet == 'diet_none':
            data = pd.read_csv("none_no_cereal.csv")
        if diet == 'diet_vegan':
            data = pd.read_csv("vegan_no_cereal.csv")
        if diet == 'diet_vegetarian':
            data = pd.read_csv("vegetarian_no_cereal.csv")
        if diet == 'diet_pescatarian':
            data = pd.read_csv("pescatarian_no_cereal.csv")
        if diet == 'diet_lowfat':
            data = pd.read_csv("lowfat.csv")
        if diet == 'diet_lowcarb':
            data = pd.read_csv("lowcarb.csv")
        data_RDA_all = pd.read_csv("RDA_micros_all.csv")
        data_RDA = data_RDA_all.loc[data_RDA_all.Life_group == life_group].drop(['Life_group'], axis=1)
        #data_RDA = pd.read_csv("RDA_micros.csv")
        print("Nutrition dataset has {} samples with {} features each.".format(*data.shape))
    except:
        print("Dataset could not be loaded. Is the dataset missing?")
     
    data2 = data.replace((',ALL COMM VAR',',RAW', ', RAW', ' RAW',',ALL AREAS', ',MATURE SEEDS', 
                          ',MICROWAVED', 'OR FILBERTS', 'HOUSE FOODS PREMIUM ',
                          ',COMMERCIAL', ',SCOTCH', ',ZESPRI', ',AS PURCHASED',
                          'MORI-NU,', ',CRUDE', ',UNENR', ',KRNLS UNSPEC',
                          'SILK ', ',UNPREP', ' UNPREP', ',STORED', ',FRESHLY HARVEST',
                          ',SLICED', ',BENGAL GM', ',LIFEWAY', ',FLUID', 
                          ',PROT FORT', ',EX SWT VAR', ',REDUCED/LOW NA',',ALL GRD', ', ALL GRD',
                          ',ALL GRDS', ',ALL G', ',DENCUT', ',FLAKED,CHOPD,FORMED & THINLY SLICED',
                          ',BROOK,NEW YORK STATE', ',BRLD', ',KIPPERED', ',NZ,IMP',
                          ' - FULLY FRENCHED', ',NORTHERN', ',KING', ',SPANISH', 
                          ',NATIVE', 'SUNFISH,', ',MXD SP', ',STEWING', ',MEAT ONLY', ',LO MOIST',
                          ',RED FAT', ',WNTR', ',MIXED SPECIES', ',FLORIDA'), 
                         '', regex=True)
	
    data2 = data2.replace(',',', ', regex = True)
    data2 = data2.replace(',LN',',LEAN', regex = True)
    data2 = data2.replace(' LN',' LEAN', regex = True)
    data2 = data2.replace('LOFAT','LOW FAT', regex = True)
    data2 = data2.replace('LOWFAT','LOW FAT', regex = True)
    data2 = data2.replace('DK MEAT','DARK MEAT', regex = True)
    data2 = data2.replace('LT MEAT','LIGHT MEAT', regex = True)
    data2 = data2.replace('FRSH','FRESH', regex = True)
    data2 = data2.replace('VAR','VARIETY', regex = False)
    data2 = data2.replace('PLN','PLAIN', regex = True)
    data2 = data2.replace('BKD','BAKED', regex = True)
    data2 = data2.replace('CNTR','CENTER', regex = True)
    data2 = data2.replace('BNLESS','BONELESS', regex = True)
    data2 = data2.replace('WHL','WHOLE', regex = True)
    data2 = data2.replace('SHLDR','SHOULDER', regex = True)
    data2 = data2.replace('RST,','ROAST,', regex = True)
    data2 = data2.replace(' RST ',' ROAST ', regex = True)
    data2 = data2.replace('RSTD','ROASTED', regex = True)
    data2 = data2.replace('TSTD','TOASTED', regex = True)
    data2 = data2.replace('SKN','SKIN', regex = True)
    data2 = data2.replace('EX FIRM','EXTRA FIRM', regex = True)
    data2 = data2.replace('W/','WITH ', regex = True)
    data2 = data2.replace('GRN','GREEN', regex = True)
    data2 = data2.replace(',FORT ',',FORTIFIED ', regex = True)
    data2 = data2.replace('BTTRMLK','BUTTERMILK', regex = True)
    data2 = data2.replace('KRNLS','KERNELS', regex = True)
    data2 = data2.replace('BTTRMLK','BUTTERMILK', regex = True)
    data2 = data2.replace(' BNS',' BEANS', regex = True)
    data2 = data2.replace(',SML',', SMALL', regex = True)
    data2 = data2.replace(', SML',', SMALL', regex = True)
    data2 = data2.replace(',SWT',',SWEET', regex = True)


     
	# Keep foods with specified serving sizes only
    data2 = data2[np.isfinite(data['GmWt_1'])]
    # We dropped some rows, want to re-index to get rid of nan rows?
    data2 = data2.reset_index(drop=True) # reindex() # Gah!
    # Drop non-numeric columns
    all_nutr_data = data2.drop(['NDB_No', 'Shrt_Desc', 'GmWt_1', 'GmWt_Desc1', 'GmWt_2', 'GmWt_Desc2', 'Refuse_Pct'], axis = 1)
    # Fill NaN-s with zeros
    all_nutr_data = all_nutr_data.fillna(0)


	# Normalize nutrients per weight of serving size (currently given per 100g)
    data_scaled_Wt = all_nutr_data.multiply(data2['GmWt_1']/100, axis = 0)
	# Halving large-calorie servings
    row_indices = (data_scaled_Wt['Energ_Kcal'] > 400)
    data_scaled_Wt.loc[row_indices] = data_scaled_Wt.loc[row_indices]/2.0
    indices = [i for i, val in enumerate(row_indices) if val]  #
    for irow in indices:
        oldstring = data2['GmWt_Desc1'][irow]
        data2.at[irow, 'GmWt_Desc1']= oldstring.replace(('1'), '1/2')
    
	# Create subsets of the data for micronutrients and macronutrients
    micro_data_0 = data_scaled_Wt[['Calcium_mg','Iron_mg','Magnesium_mg','Phosphorus_mg','Zinc_mg','Copper_mg','Manganese_mg','Selenium_mcg','Vit_C_mg',
                         'Thiamin_mg','Riboflavin_mg','Niacin_mg','Panto_Acid_mg','Vit_B6_mg','Folate_Tot_mcg',
                         'Choline_Tot_mg','Vit_B12_mcg','Vit_A_IU',
                         'Vit_E_mg','Vit_K_mcg']]
    macro_data_0 = data_scaled_Wt[['Energ_Kcal', 'Protein_g', 'Lipid_Tot_g', 'Carbohydrt_g','Fiber_TD_g']]
    micro_macro_data_0 = pd.concat((macro_data_0, micro_data_0), axis = 1)
    # Rescale micronutrient data by RDA
    micro_data_scaled_RDA = micro_data_0.div(data_RDA.iloc[0])
    micro_macro_data_scaled_RDA = pd.concat((micro_data_scaled_RDA, macro_data_0), axis = 1)
    Nutri_index = micro_data_scaled_RDA.sum(axis = 1)
    Nutri_index = pd.DataFrame(Nutri_index)
    Nutri_index.columns = ['Nutri_index']
    
    
    ########################### NMF to obtain food categories ##############################

    micro_data_RDA = micro_data_scaled_RDA #.fillna(0)
    micro_data_RDA_T = micro_data_RDA.transpose()
    model = NMF(n_components=12, init='random', random_state=0)
    W = model.fit_transform(micro_data_RDA_T)
    H = model.components_
    H_df = pd.DataFrame(H)
    W_df = pd.DataFrame(W)
    nuts = pd.DataFrame(micro_data_RDA_T.index.tolist(), columns = ['Nutrients'])
    foods = data2['Shrt_Desc']
    foods = pd.DataFrame(foods)
    W_df_nuts = pd.concat((nuts, W_df), axis = 1)
    H_df_T = pd.DataFrame(H).transpose()
    H_df_T_foods = pd.concat((foods, H_df_T), axis = 1)

    # Finding the nutrient barcode of each cluster
    W_df_nuts_T = pd.DataFrame(W_df_nuts).transpose()
    W_df_nuts_T.columns = W_df_nuts_T.iloc[0]
    W_df_nuts_T = W_df_nuts_T.drop(['Nutrients'])
    cluster = pd.DataFrame(H_df_T.idxmax(axis=1))
    Segments = ['Cluster {}'.format(i) for i in range(0,len(W_df_nuts_T))]
    food_cluster = pd.concat((foods,cluster), axis = 1) 
    food_cluster.columns = ['Shrt_Desc', 'Cluster_ID']
    W_df_nuts_T.index = Segments

    # Prep for optimization
    # Make a large dataframe with food name, dominant cluster id, Ni, serving size, all nutrients
    # Sort by cluster, then by Ni
    data_everything = pd.concat((food_cluster, Nutri_index, data2['GmWt_Desc1'], micro_macro_data_scaled_RDA), axis = 1)
    data_everything = data_everything.sort_values(['Cluster_ID','Nutri_index'], ascending=False)
    # ! the barcode lookup table is the dataframe W_df_nuts_T !

     ########################### Optimizer  ###########################

    # choose 1 great food to start with
    food1_options = data_everything.sort_values(['Nutri_index'], ascending=False)
    food1_options = food1_options.head(n=20)
    foodN_options = data_everything
    food_list = []
    serving_list = []
    calories_list = []
    totalcal = 0
    topchoices_nuts = pd.DataFrame()
    depletedclusters = [] # keep track if no foods in the cluster are good for us anymore
    
    # Greedy algorithm
    while totalcal < (int(calories)-200):
        if len(food_list) == 0: # if first food
           food1 = food1_options.sample(n=1)
           this_food_is_bad = any(word.lower() in str(food1['Shrt_Desc']).lower() for word in bad_foods)
           while this_food_is_bad and len(food1_options) > 1:
               food1_options = food1_options.drop(index = food1.index) # Drop the bad one
               food1 = food1_options.sample(n=1)
               this_food_is_bad = any(word.lower() in str(food1['Shrt_Desc']).lower() for word in bad_foods)
           # We only exit this loop if we found a good food1
           if (not this_food_is_bad):
            
                for value in food1.Shrt_Desc.values:
                    food_list += [value]
        
                for value in food1.GmWt_Desc1.values:
                    serving_list += [value]
                
                for value in food1.Energ_Kcal.values:
                    calories_list += [value]
                    totalcal = int(value) + totalcal
                
                topchoices_nuts  = pd.concat([topchoices_nuts , food1.iloc[:,4:24]] , axis = 0)
                # define deficit after the first food is chosen
                total = topchoices_nuts.sum()

                deficit = 1 - total
                deficit[deficit < 0] = 0
        else: # not first food
            bestcluster = -1
            bestsimilarity = -1        
            for i in range(0,len(W_df_nuts_T)): # For each cluster...            
                # choose a cluster based on similarity to deficit
                similarity = cosine_similarity([W_df_nuts_T.iloc[i]], [deficit])
                #print(i, similarity)
                if (similarity > bestsimilarity and 
                    len(foodN_options.loc[foodN_options['Cluster_ID'] == i]) > 0 and
                    i not in depletedclusters):
                    bestcluster = i
                    bestsimilarity = similarity
            # Now we have best cluster

            # choose a food from the best foods in this cluster
            data_sorted_Cluster = foodN_options.loc[foodN_options['Cluster_ID'] == bestcluster]
            data_sorted_Cluster = data_sorted_Cluster.head(n=15)
            foodN = data_sorted_Cluster.sample(n=1)
            
            this_food_is_bad = any(word.lower() in str(foodN['Shrt_Desc']).lower() for word in bad_foods)
            while this_food_is_bad and len(data_sorted_Cluster) > 1:
                data_sorted_Cluster = data_sorted_Cluster.drop(index = foodN.index) # Drop the bad one
                foodN = data_sorted_Cluster.sample(n=1)
                this_food_is_bad = any(word.lower() in str(foodN['Shrt_Desc']).lower() for word in bad_foods)
            # We only exit this loop if we found a good foodN

            if (this_food_is_bad):
                depletedclusters += [bestcluster]
            else:                      
            
                for value in foodN.Shrt_Desc.values:
                    food_list += [value]
                    fullstring = [x.strip() for x in value.split(',')]
                    bad_foods += [fullstring[0]] # Dont pick it again: add it to the bad_foods list
        

                for value in foodN.GmWt_Desc1.values:
                    serving_list += [value]
                
                for value in foodN.Energ_Kcal.values:
                    calories_list += [value]
                    totalcal = int(value) + totalcal
                
                topchoices_nuts  = pd.concat([topchoices_nuts , foodN.iloc[:,4:24]] , axis = 0)
                # define deficit after the first food is chosen
                total = topchoices_nuts.sum()

                deficit = 1 - total
                deficit[deficit < 0] = 0

    final_list = []
    for i in range(0,len(food_list)):
        food = food_list[i] # pd.DataFrame([   foods[i]   ]) # columns=')
        serving = serving_list[i] # pd.DataFrame([  servings[i]  ])
        calories_per_serving = calories_list[i]
        final_list.append(dict(food=food.lower(), serving=serving, calories_per_serving=calories_per_serving))
        #print(food, serving, calories_per_serving)


    # Calculating the nutrients in our food list
    final_list_df = pd.DataFrame(final_list, columns = ['calories_per_serving', 'food','serving'])
    final_list_with_nuts = data_everything[data_everything['Shrt_Desc'].str.lower().isin(final_list_df['food'])]
    goals = final_list_with_nuts.iloc[:, 4:29].sum(axis = 0)
    goals = pd.DataFrame(goals)
    goals_micronuts = goals.drop(['Energ_Kcal', 'Lipid_Tot_g', 'Protein_g', 'Fiber_TD_g', 'Carbohydrt_g'], axis = 0)
    goals_micronuts = goals_micronuts.sort_values([0], ascending = False)
    goals_micronuts[0] = goals_micronuts[0].apply(lambda x: round(x*100))

	 ########################### Chart  ###########################
	 
    # Cap them at max value (percentage) to plot
    temporary_plotting_threshold = 100 
    goals_micronuts[0] = goals_micronuts[0].apply(lambda x: min(x,temporary_plotting_threshold))

    final_nutrients_list = goals_micronuts.index.tolist()
    final_nutrients_values_list = goals_micronuts.values.tolist()

    final_nutrients_values_list = [item for sublist in final_nutrients_values_list for item in sublist]    
    final_nutrients_values_list = [str(i) for i in final_nutrients_values_list] # turn floats into strings
	
    for i in range(len(final_nutrients_list)):
        final_nutrients_list[i] = final_nutrients_list[i].replace('_',' ')
        final_nutrients_list[i] = final_nutrients_list[i].replace(' mg',', mg ')
        final_nutrients_list[i] = final_nutrients_list[i].replace(' mcg',', mcg ')
        final_nutrients_list[i] = final_nutrients_list[i].replace('Tot','')
        final_nutrients_list[i] = final_nutrients_list[i].replace(' ,',',')
        final_nutrients_list[i] = final_nutrients_list[i].replace('IU',', IU ')


    return render_template('output_list.html', food_table=final_list, title='OptimalNutrion', max=150, labels=final_nutrients_list, values=final_nutrients_values_list)
           


@app.route('/input_page')
def nut_user_input2():
    return render_template("input_page.html")