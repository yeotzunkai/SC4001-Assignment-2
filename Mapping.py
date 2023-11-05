# This Python code will create a mapping of lexicographic order to numeric order
# for the numbers 1 through 102.

# Create a list of numbers as strings
numbers_as_strings = [str(i) for i in range(1, 103)]

# Sort this list in lexicographic order
lexicographically_sorted = sorted(numbers_as_strings)

# Create a dictionary to map lexicographic order to numeric order
lexicographic_to_numeric = {number: index  for index, number in enumerate(lexicographically_sorted)}

# Now you can access lexicographic_to_numeric to see the mapping
# For example, to print out the mapping:
for lexicographic, numeric in lexicographic_to_numeric.items():
    print(f"Lexicographic: {lexicographic} -> Numeric: {numeric}")

# If you want the dictionary itself, just use the variable lexicographic_to_numeric
import json

# Convert the mapping dictionary to a JSON string for a structured format
json_mapping = json.dumps(lexicographic_to_numeric, indent=4)

# Specify the filename
filename = 'lexicographic_to_numeric_mapping.json'

# Write the JSON string to a file
with open(filename, 'w') as f:
    f.write(json_mapping)

