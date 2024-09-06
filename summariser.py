import io
import base64
import fitz  
from PIL import Image
from openai import OpenAI
import requests
from dotenv import load_dotenv
import os
from pdf2image import convert_from_path

load_dotenv()
api_key= os.getenv('OPENAI_API_KEY')

def summarize_image(image_bytes):
    image_base64 = base64.b64encode(image_bytes).decode('utf-8')

    headers = {
    "Content-Type": "application/json",
    "Authorization": f"Bearer {api_key}"
    }

    payload = {
    "model": "gpt-4o",
    "messages": [
        {
        "role": "user",
        "content": [
            {
            "type": "text",
            "text": "See the diagram on the page and summarise it. The page may contain more than one images, summarise them one by one"
            },
            {
            "type": "image_url",
            "image_url": {
                "url": f"data:image/jpeg;base64,{image_base64}"
            }
            }
        ]
        }
    ],
    "max_tokens": 1000
    }

    response = requests.post("https://api.openai.com/v1/chat/completions", headers=headers, json=payload)

    response = response.json()
    summary = response['choices'][0]['message']['content']
    return summary

def add_wrapped_text(page, text, rect, fontsize=8, fontname="helv"):
    text_area = page.insert_textbox(rect, text, fontsize=fontsize, fontname=fontname)
    return text_area

def generate_pdf(pdf_path):
    try:
        pdf_file = fitz.open(pdf_path)
        file_name = os.path.basename(pdf_path)

        # Iterate over each page in the PDF
        for page_index in range(len(pdf_file)):
            page = pdf_file[page_index]
            image_list = page.get_images(full=True)

            if image_list:
                print(f"[+] Found a total of {len(image_list)} images on page {page_index + 1}")

                # Convert this page to image
                images = convert_from_path(pdf_path, first_page=page_index+1, last_page=page_index+1)
                image = images[0]
                image_bytes = io.BytesIO()
                image.save(image_bytes, format='PNG')
                image_bytes = image_bytes.getvalue()

                # Summarize the page
                summary = summarize_image(image_bytes)

                existing_text = page.get_text("text")

                new_text = existing_text + summary

                # print("NEW TEXT: ", new_text)
                new_doc = fitz.open()

                remaining_text = new_text
                while remaining_text:
                    new_page = new_doc.new_page()
                    rect = fitz.Rect(50, 50, new_page.rect.width - 50, new_page.rect.height - 50)
                    text_area = int(add_wrapped_text(new_page, remaining_text, rect))+1
                    
                    # print(text_area)
                    if text_area< 0:  
                        remaining_text = remaining_text[:text_area]  
                    else:
                        remaining_text = ""  

                pdf_file.delete_page(page_index)
                pdf_file.insert_pdf(new_doc, from_page=0, to_page=len(new_doc), start_at=page_index)
                
                # Close the new document
                new_doc.close()
            
        # Save the modified PDF to a new file
        updated_pdf_path = f"Updated_{file_name}"
        pdf_file.save(updated_pdf_path)
        pdf_file.close()
        print(f"Updated PDF saved as: {updated_pdf_path}")

    except Exception as e:
        print(f"An error occurred: {e}")

generate_pdf("sample2-1-6.pdf")
