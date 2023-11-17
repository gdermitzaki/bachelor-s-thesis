import pandas as pd

# Load the Excel file into a DataFrame
file_path = "G:\\Το Drive μου\\AI\\Georgia Thesis\\Code\\EXCEL4.xlsx"
try:
    xls = pd.ExcelFile(file_path)
    sheet_names = xls.sheet_names  # Get the names of all sheets in the Excel file
except Exception as e:
    sheet_names = str(e)

#sheet_names

# Read the relevant sheets into DataFrames
try:
    df_sheet1 = pd.read_excel(file_path, sheet_name='sheet1')
    df_sheet2 = pd.read_excel(file_path, sheet_name='Sheet2')
except Exception as e:
    error_message = str(e)

# Show a preview of the data from both sheets
#df_sheet1.head(), df_sheet2.head()
#print(df_sheet1)
try:
    df_sheet1 = pd.read_excel(file_path, sheet_name='sheet1')
    df_sheet2 = pd.read_excel(file_path, sheet_name='Sheet2')
except Exception as e:
    error_message = str(e)
#print(df_sheet1)
#print(df_sheet2)


import random
import numpy as np

# Initialize parameters
x_building = int(df_sheet2['X_Building'].iloc[0])
y_building = int(df_sheet2['Y_Building'].iloc[0])

population_size = 50
num_generations = 100

# Initialize spaces from Sheet1
spaces = df_sheet1.to_dict(orient='records')

# Function to generate a random layout
def generate_random_layout(spaces, x_building, y_building):
    layout = {}
    for space in spaces:
        x_dim = space.get('X_DIM', 0)
        y_dim = space['Y_DIM']
        space_id = space['SPACE_ID']
        
        # Find a suitable random position for the space within the building
        x_pos = random.randint(0, x_building - x_dim)
        y_pos = random.randint(0, y_building - y_dim)
        
        layout[space_id] = {'x': x_pos, 'y': y_pos, 'x_dim': x_dim, 'y_dim': y_dim}
    
    return layout

# Initialize the first generation with random layouts
population = [generate_random_layout(spaces, x_building, y_building) for _ in range(population_size)]

# Show an example layout from the initial population
population[0]
print(population)
      

# Function to evaluate a layout based on the defined criteria
def evaluate_layout(layout, spaces):
    total_distance = 0
    
    # Calculate the distance between adjacent spaces
    for space in spaces:
        space_id = space['SPACE_ID']
        adjacents = str(space['ADJACENT_SPACES']).split(",")
        x1, y1 = layout[space_id]['x'], layout[space_id]['y']
        
        for adj in adjacents:
            adj = adj.strip()  # Remove any leading/trailing whitespaces
            if adj in layout:
                x2, y2 = layout[adj]['x'], layout[adj]['y']
                distance = np.sqrt((x2 - x1)**2 + (y2 - y1)**2)
                total_distance += distance
    
    return total_distance

# Evaluate the initial population
fitness_scores = [evaluate_layout(layout, spaces) for layout in population]

# Show fitness scores of the initial population
fitness_scores[:10]


# Genetic Algorithm Operations

# Crossover operation
def crossover(parent1, parent2):
    keys = list(parent1.keys())
    crossover_point = random.randint(1, len(keys) - 1)
    child1 = {**dict(list(parent1.items())[:crossover_point]), **dict(list(parent2.items())[crossover_point:])}
    child2 = {**dict(list(parent2.items())[:crossover_point]), **dict(list(parent1.items())[crossover_point:])}
    return child1, child2

# Mutation operation
def mutate(layout, x_building, y_building):
    keys = list(layout.keys())
    mutate_key = random.choice(keys)
    
    x_dim = layout[mutate_key]['x_dim']
    y_dim = layout[mutate_key]['y_dim']
    
    x_pos = random.randint(0, x_building - x_dim)
    y_pos = random.randint(0, y_building - y_dim)
    
    layout[mutate_key]['x'] = x_pos
    layout[mutate_key]['y'] = y_pos

# Selection and generation of the new population
def new_generation(population, fitness_scores, x_building, y_building):
    new_pop = []
    sorted_indices = np.argsort(fitness_scores)
    
    # Elitism: Select the top 10% layouts
    elite_count = int(0.1 * len(population))
    for i in range(elite_count):
        new_pop.append(population[sorted_indices[i]])
    
    # Crossover and Mutation
    while len(new_pop) < len(population):
        parent1 = population[random.choice(sorted_indices[:elite_count])]
        parent2 = population[random.choice(sorted_indices[:elite_count])]
        child1, child2 = crossover(parent1, parent2)
        
        # Apply mutation with a 10% chance
        if random.random() < 0.1:
            mutate(child1, x_building, y_building)
        if random.random() < 0.1:
            mutate(child2, x_building, y_building)
        
        new_pop.extend([child1, child2])
    
    return new_pop[:len(population)]

# Initialize for Genetic Algorithm loop
best_layouts = []
best_scores = []

# Main Genetic Algorithm loop
for generation in range(num_generations):
    # Evaluate the current population
    fitness_scores = [evaluate_layout(layout, spaces) for layout in population]
    
    # Store the best layout and score for this generation
    best_layout = population[np.argmin(fitness_scores)]
    best_score = min(fitness_scores)
    best_layouts.append(best_layout)
    best_scores.append(best_score)
    
    # Generate a new population
    population = new_generation(population, fitness_scores, x_building, y_building)
    
    if generation % 10 == 0:
        print(f"Generation {generation}, Best Score: {best_score}")

# Show the best score and layout at the end of the optimization
best_score, best_layouts[-1]

#---------------------------------------------------------------------------------
import matplotlib.pyplot as plt

# Function to visualize a layout
def visualize_layout(layout, x_building, y_building):
    fig, ax = plt.subplots()
    
    # Draw the building outline
    ax.set_xlim([0, x_building])
    ax.set_ylim([0, y_building])
    ax.set_aspect('equal', 'box')
    ax.set_title('Optimal Office Layout')
    
    for space_id, attributes in layout.items():
        x = attributes['x']
        y = attributes['y']
        x_dim = attributes['x_dim']
        y_dim = attributes['y_dim']
        
        rect = plt.Rectangle((x, y), x_dim, y_dim, linewidth=1, edgecolor='r', facecolor='none')
        ax.add_patch(rect)
        plt.text(x + x_dim / 2, y + y_dim / 2, str(space_id), fontsize=12, ha='center', va='center')
    
    plt.show()

# Visualize the best layout
visualize_layout(best_layouts[-1], x_building, y_building)


