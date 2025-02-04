#!/usr/bin/env python3

import sys
import pyperclip
import os

def read_file_content(file_path):
    try:
        with open(file_path, 'r', encoding='utf-8') as file:
            return file.read()
    except Exception as e:
        return f"Error reading file: {str(e)}"

def main():
    if len(sys.argv) < 2:
        print("Usage: script.py file1 file2 ...")
        sys.exit(1)

    # Initialize an empty string to store all content
    combined_content = []

    # Process each file
    for file_path in sys.argv[1:]:
        if os.path.exists(file_path):
            # Get just the filename from the path
            filename = os.path.basename(file_path)
            
            # Read the file content
            content = read_file_content(file_path)
            
            # Format with filename and markdown code block
            formatted_content = f"{filename}\n```\n{content}\n```\n"
            
            # Add to our collection
            combined_content.append(formatted_content)
        else:
            print(f"File not found: {file_path}")

    # Join all content with newlines
    final_content = "\n".join(combined_content)

    # Copy to clipboard
    try:
        pyperclip.copy(final_content)
        print("Content copied to clipboard successfully!")
    except Exception as e:
        print(f"Error copying to clipboard: {str(e)}")

if __name__ == "__main__":
    main()
