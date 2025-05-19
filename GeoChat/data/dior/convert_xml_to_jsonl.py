import os
import json
import xml.etree.ElementTree as ET

def convert_xml_to_jsonl(input_folder, output_file, start_index=1, end_index=1000):
    question_id_counter = 1  # Initialize question_id counter

    with open(output_file, 'w') as outfile:
        # Iterate over each file in the input folder
        for filename in os.listdir(input_folder):
            if filename.endswith('.xml'):
                # Extract image_id from the XML filename (without the extension)
                image_id = filename.split('.')[0]

                # Convert image_id to an integer to filter by range
                image_number = int(image_id)

                # Process only files in the specified range
                if start_index <= image_number <= end_index:
                    # Parse the XML file
                    tree = ET.parse(os.path.join(input_folder, filename))
                    root = tree.getroot()

                    # Get width and height of the image (if needed for normalization)
                    size = root.find('size')
                    width = int(size.find('width').text)
                    height = int(size.find('height').text)

                    # Iterate over each object in the XML
                    for obj in root.findall('object'):
                        class_name = obj.find('name').text  # Get the class name

                        # Extract bounding box coordinates
                        bndbox = obj.find('bndbox')
                        xmin = int(bndbox.find('xmin').text)
                        ymin = int(bndbox.find('ymin').text)
                        xmax = int(bndbox.find('xmax').text)
                        ymax = int(bndbox.find('ymax').text)

                        # Format question string with bounding box coordinates
                        question = f"{{<{xmin}><{ymin}><{xmax}><{ymax}>}}"

                        # Create output dictionary for each object
                        output_dict = {
                            "image_id": image_id,
                            "question": question,
                            "dataset": "dior",
                            "question_id": f"dior_{question_id_counter:06d}",
                            "ground_truth": class_name
                        }

                        # Write the output dictionary to the JSONL file
                        json.dump(output_dict, outfile)
                        outfile.write('\n')

                        # Increment the question_id counter
                        question_id_counter += 1

# Set the input folder and output file
input_folder = '/workspace/Dataset/OpenDataLab___DIOR/raw/DIOR/Annotations/Horizontal Bounding Boxes'  # Replace with your XML folder path
output_file = '/workspace/GeoChat/data/dior/dior_tranval_horizontal.jsonl'

# Call the function to convert XML to JSONL, specifying the range 00001 to 01000
convert_xml_to_jsonl(input_folder, output_file, start_index=1, end_index=1000)
