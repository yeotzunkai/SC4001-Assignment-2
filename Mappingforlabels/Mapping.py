import json

# Create a list of numbers as strings
numbers_as_strings = [str(i) for i in range(0, 102)]

# Sort this list in lexicographic order
lexicographically_sorted = sorted(numbers_as_strings)

# Create a dictionary to map numeric order to lexicographic order
numeric_to_lexicographic = {index + 1: number for index, number in enumerate(lexicographically_sorted)}

# Convert the dictionary to a JSON string
json_mapping_flipped = json.dumps(numeric_to_lexicographic, indent=4)

# Specify the filename for the flipped mapping
flipped_filename = 'numeric_to_lexicographic_mapping.json'

# Write the JSON string to a file
with open(flipped_filename, 'w') as f:
    f.write(json_mapping_flipped)

print(f"Flipped mapping saved to {flipped_filename}")
