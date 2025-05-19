import os
import cv2
import xml.etree.ElementTree as ET

def crop_and_save_images(xml_folder, image_folder, output_folder, start_index=1, end_index=1000):
    # Ensure the output folder exists
    os.makedirs(output_folder, exist_ok=True)
    
    # Iterate through all XML files in the xml_folder
    for xml_file in os.listdir(xml_folder):
        if xml_file.endswith('.xml'):
            # Extract file index from xml_file (e.g., 00001 from 00001.xml)
            file_index = int(os.path.splitext(xml_file)[0])

            # Check if the file index is within the specified range
            if start_index <= file_index <= end_index:
                xml_path = os.path.join(xml_folder, xml_file)
                
                # Parse the XML file
                tree = ET.parse(xml_path)
                root = tree.getroot()
                
                # Extract the image filename from the XML
                filename = root.find('filename').text
                image_path = os.path.join(image_folder, filename)
                
                # Load the image
                image = cv2.imread(image_path)
                if image is None:
                    print(f"Could not load image {filename}. Skipping.")
                    continue

                # Iterate over all objects (bounding boxes) in the XML
                for obj in root.findall('object'):
                    # Extract the class name and bounding box coordinates
                    class_name = obj.find('name').text
                    bbox = obj.find('bndbox')
                    xmin = int(bbox.find('xmin').text)
                    ymin = int(bbox.find('ymin').text)
                    xmax = int(bbox.find('xmax').text)
                    ymax = int(bbox.find('ymax').text)
                    
                    # Crop the image based on bounding box
                    cropped_image = image[ymin:ymax, xmin:xmax]
                    
                    # Save the cropped image to the output folder
                    output_filename = f"{os.path.splitext(filename)[0]}_{class_name}_{xmin}_{ymin}_{xmax}_{ymax}.jpg"
                    output_path = os.path.join(output_folder, output_filename)
                    cv2.imwrite(output_path, cropped_image)
                    print(f"Saved cropped image: {output_path}")

# Define input folders and output folder
xml_folder = '/workspace/Dataset/OpenDataLab___DIOR/raw/DIOR/Annotations/Horizontal Bounding Boxes'  # Replace with your XML folder path
image_folder = '/workspace/Dataset/OpenDataLab___DIOR/raw/DIOR/JPEGImages-trainval'  # Replace with your image folder path
output_folder = '/workspace/GeoChat/data/dior/crop_with_gt_from1to1000'  # Replace with your output folder path

# Run the function to crop and save images
crop_and_save_images(xml_folder, image_folder, output_folder, start_index=1, end_index=1000)
