##########################################################################

# Analyze directory (and/or subdirectories) of sequentially-numbered infant
# images (jpg or png) one by one using same position prompt to
# Google Gemini via API

# copyright 2025, University of Delaware
# Christopher Rasmussen cer@cis.udel.edu

# Prerequisites:
# * Get a Google API key: https://ai.google.dev/gemini-api/docs/api-key
# * Install latest Google Gen AI SDK: pip install --upgrade google-genai
# * Put prompt text in prompt_text.txt

# How to run:
# python3 gemini_analysis.py --image_dir <path to images> --model <"name of Gemini model"> --filter_re <regexp describing filenames of images to process>

# Note that the line "Answer only, no explanation" was not in the original prompt for this paper.
# However, gemini-2.5-pro-preview-06-05 is no longer available and gemini-3-pro-preview
# seems to need it to discourage extra explanations from being generated.

##########################################################################

import time
import os
import sys
import re
import argparse
import PIL.Image as Image
from google import genai

##########################################################################

def parse_args():
    
    parser = argparse.ArgumentParser(description="Gemini image analyzer")
    
    parser.add_argument(
        "--image_directory",
        type=str,
        default="images/",
        help="Directory containing input images",
    )

    parser.add_argument(
        "--filter_re",
        type=str,
        default="[0-9].(jpg|png)$", 
        help="Regular expression for image filename match criterion",
    )

    parser.add_argument(
        "--model",
        type=str,
        default="gemini-3-pro-preview",
        help="Choose the model to use",
    )

    return parser.parse_args()

##########################################################################

# send same prompt for every image in directory

def send_filtered_image_prompts_for_directory(img_dir, filter_re, client, model_string, prompt_string, responses_fp, already_written_set):

    max_retries = 4
    delay_seconds = 1

    # iterate over images in img_dir

    for img_filename in sorted(os.listdir(img_dir)):

        # skip images with filenames that don't match filter_re or that have already been processed

        if re.search(filter_re, img_filename) == None:
            continue
            
        if img_filename in already_written_set:
            print("ALREADY WRITTEN SO SKIPPING ", img_filename)
            continue

        # read current image
        
        file_path = os.path.join(img_dir, img_filename)
        if os.path.isfile(file_path):
            
            img = Image.open(file_path)

            retry_count = 0

            # send image and prompt for processing.  retry if necessary
            
            while retry_count < max_retries:
                try:
                    response = client.models.generate_content(
                        model=model_string,
                        contents=[
                            prompt_string,
                            img
                        ]
                    )

                    # success: write response to file
                    
                    response_csv_line = img_filename + ", " + response.text + "\n"
                    print(response_csv_line, end="")

                    responses_fp.write(response_csv_line)
                    responses_fp.flush()

                    already_written_set.add(img_filename)

                    break  # get out of this try loop

                # something went wrong -- wait a little bit before trying again
                
                except Exception as e:
                    print("Error on image " + file_path + ", try " + str(retry_count + 1) + "/" + str(max_retries))
                    print(e)
                    retry_count += 1
                    if retry_count < max_retries:
                        wait_time = delay_seconds * retry_count
                        print(f"Retrying in {wait_time} seconds...")
                        time.sleep(wait_time)
                    else:
                        print("Max retries reached. Could not generate content")
                        
##########################################################################

def main():

    args = parse_args()

    # echo API key
    
    GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
    print(f"GOOGLE_API_KEY: {GOOGLE_API_KEY}")
    client = genai.Client(api_key=GOOGLE_API_KEY)

    # list possible models

    print("Available models:")
    
    for m in client.models.list():
        for action in m.supported_actions:
            if action == "generateContent":
                print(m.name)
                
    # file where responses will be written
    
    responses_filename = "responses.txt" 

    # read prompt text from file

    with open("prompt_text.txt", 'r', encoding='utf-8') as file:
        prompt_string = file.read() # The read() method reads the entire file content

    # check to see the directory has already been partially processed (i.e., previous job was interrupted)

    already_written_set = set()

    if os.path.isfile(responses_filename):
        with open(responses_filename, "r") as file:
            for line in file:
                entry_list = line.strip().split(",")
                already_written_set.add(entry_list[0])
        print(str(len(already_written_set)) + " existing responses detected")
    else:
        print("Starting new responses file")
        
    # start processing
    
    responses_fp = open(responses_filename, "a")

    for dirpath, dirnames, filenames in os.walk(args.image_directory):

        # no subdirectories, so assume images are at top level
        
        if not dirnames:
            send_filtered_image_prompts_for_directory(args.image_directory, args.filter_re, client, args.model, prompt_string, responses_fp, already_written_set)
 
        # iterate over subdirectories and process separately
        
        else: 
            for subdirname in sorted(dirnames):
                print(subdirname)
                full_subdirname = os.path.join(args.image_directory, subdirname)
                
                send_filtered_image_prompts_for_directory(full_subdirname, args.filter_re, client, args.model, prompt_string, responses_fp, already_written_set)
                
##########################################################################

if __name__ == "__main__":
    main()

##########################################################################
##########################################################################
