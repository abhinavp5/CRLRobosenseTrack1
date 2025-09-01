import json
import re
import argparse
import os

def classify_question(question, category):
    """
    Classifies a question into a subcategory based on its content and main category.
    
    Args:
        question (str): The question text.
        category (str): The main category (e.g., "Perception", "Planning").

    Returns:
        str: The determined subcategory, or None if no match is found.
    """
    # Normalize category to handle variations like "perception" vs "Perception"
    category = category.lower()

    if category == "perception":
        if re.search(r"moving status of object.*options:", question, re.IGNORECASE):
            return "Perception-MCQ"
        elif re.search(r"visual description of <c[0-9]+,CAM_.*,.*,.*>", question, re.IGNORECASE):
            return "Perception-VQA-Object"
        elif re.search(r"important objects in the current scene", question, re.IGNORECASE):
            return "Perception-VQA-Scene"
            
    elif category == "prediction":
        # Only 1 prediction category
        return "Prediction-MCQs"

    elif category == "planning":
        if re.search(r"What actions could the ego vehicle take based on <c[0-9]+,CAM_.*,.*,.*>", question, re.IGNORECASE):
            return "Planning-VQA-Object"
        elif re.search(r"(safe actions|dangerous actions|comment on this scene)", question, re.IGNORECASE):
            return "Planning-VQA-Scene"
        else: 
            return "Planning-VQA-Object"

            
    # Return None if no subcategory matches
    

def process_data(file_path="perceptionTop.json", output_path="track_1_with_subcategory.json"):
    """
    Loads data from a JSON file, adds a 'subcategory' to each item, 
    and writes the result to a new JSON file.

    Args:
        file_path (str): The path to the input JSON file.
        output_path (str): The path where the output JSON file will be saved.
    """
    # Load the JSON data
    try:
        with open(file_path, 'r', encoding='utf-8') as file:
            data = json.load(file)
    except FileNotFoundError:
        print(f"Error: The file was not found at {file_path}")
        return
    except json.JSONDecodeError:
        print(f"Error: Could not decode JSON from the file {file_path}")
        return

    print(f"Loaded {len(data)} items from {file_path}")
    
    # Process each item
    processed_items = []
    counter = 1
    for i, item in enumerate(data):
        try:
            # Ensure question and category keys exist
            if "question" in item and "category" in item:
                subcategory = classify_question(item["question"], item["category"])
                
                # Add the subcategory to the json item
                item['subcategory'] = subcategory
                processed_items.append(item)

                #adding unique id to object
                item['id'] = counter
                counter+=1
            else:
                print(f"Skipping item {i} due to missing 'question' or 'category' key.")
                continue

            # Progress indicator for large files
            if (i + 1) % 100 == 0:
                print(f"Processed {i + 1}/{len(data)} items...")
                
        except Exception as e:
            print(f"Error processing item {i}: {e}")
            continue

    # Write the processed data to the output file
    try:
        with open(output_path, 'w', encoding='utf-8') as outfile:
            json.dump(processed_items, outfile, indent=4, ensure_ascii=False)
        print(f"\nSuccessfully processed and saved {len(processed_items)} items to {output_path}")
    except Exception as e:
        print(f"Error writing to output file {output_path}: {e}")


def split_data(path="track_1_with_subcategory.json"):
    # Load the JSON data
    try:
        with open(path, 'r', encoding='utf-8') as file:
            data = json.load(file)
    except FileNotFoundError:
        print(f"Error: The file was not found at {path}")
        return
    except json.JSONDecodeError:
        print(f"Error: Could not decode JSON from the file {path}")
        return

    print(f"Loaded {len(data)} items from {path}")

    # Custom order for sorting
    order = {
        "Perception-MCQ": 0,
        "Perception-VQA-Object": 1,
        "Perception-VQA-Scene": 2,
        "Prediction-MCQs": 3,
        "Planning-VQA-Object": 4,
        "Planning-VQA-Scene": 5
    }

    # Sort by subcategory using custom order
    sorted_data = sorted(
        data,
        key=lambda x: order.get(x.get("subcategory", ""), float('inf'))
    )

    # Group by subcategory
    categories = {}
    for item in sorted_data:
        subcat = item.get("subcategory", "Unknown")
        categories.setdefault(subcat, []).append(item)

    # Create output directory based on input filename
    base_name = os.path.splitext(os.path.basename(path))[0]
    output_dir = f"split_output_{base_name}"
    os.makedirs(output_dir, exist_ok=True)

    # Save each group into its own file
    for subcat, items in categories.items():
        filename = os.path.join(output_dir, f"{subcat}.json")
        with open(filename, "w", encoding="utf-8") as f:
            json.dump(items, f, indent=4, ensure_ascii=False)
        print(f"Saved {len(items)} items to {filename}")

    print("Split complete.")

def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Classify and split JSON data by subcategories")
    parser.add_argument("filepath", help="Path to the JSON file to be processed and separated")
    parser.add_argument("--output", default="track_1_with_subcategory.json", 
                       help="Output file path for processed data (default: track_1_with_subcategory.json)")
    return parser.parse_args()

# Example of how to run the function
if __name__ == '__main__':
    args = parse_args()
    
    # Process the data first
    print(f"Processing data from: {args.filepath}")
    process_data(args.filepath, args.output)
    
    # Then split the processed data
    print(f"Splitting data from: {args.output}")
    split_data(args.output)
