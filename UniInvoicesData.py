import ocifs
import numpy as np
import pyarrow.dataset as pyarrowDs
import pyarrow as pa
import pyarrow.parquet as pq
import pandas as pd
import oci
import os
import pandas as pd
import json
from IPython.display import display
import tabula
from pypdf import PdfReader
import pytesseract
from pdf2image import convert_from_path
from PIL import Image
import re
from io import BytesIO
from pdf2image import convert_from_bytes


def ocr_pdf_bytes(pdf_bytes):
    """
    Performs OCR on PDF bytes and returns the text.
    """
    # 1. Convert PDF bytes to a list of PIL Images
    # dpi=300 is recommended for OCR accuracy
    pages = convert_from_bytes(pdf_bytes, dpi=300)
    final_text = ""
    # 2. Iterate through pages and perform OCR
    for page in pages:
        # Pytesseract image_to_string can take a PIL image directly
        text = pytesseract.image_to_string(page, lang='eng')
        final_text += text + "\n"
    return final_text

def read_pdf_with_ocr(pdf_path):
    """
    Reads text from a potentially scanned PDF using OCR.
    """
    # 1. Convert PDF pages to a list of images
    # Ensure Poppler is installed and accessible in your system's PATH
    images = convert_from_path(pdf_path)
    extracted_text = ""
    # 2. Perform OCR on each image
    for i, image in enumerate(images):
        # Optional: Save images to a temporary file for debugging
        # image_file = f"page_{i}.jpg"
        # image.save(image_file, "JPEG")
        # Extract text from the image using Tesseract
        text = pytesseract.image_to_string(image)
        extracted_text += text + "\n"
        # Optional: Remove temporary image files
        # os.remove(image_file)
    return extracted_text

def extractDataDF ( pdfContent ):
    # text_content = read_pdf_with_ocr(pdfContent)
    text_content = ocr_pdf_bytes(pdfContent.getvalue())
    customerpattern = r"(?<=Customer #: — ).*?(?=\s)"
    paymentPattern = r"(?<=Payment Terms: ).*?(?=\s)"
    routePattern = r"(?<=Route #: ).*?(?=\s)"
    InvoiceDtePattern = r"(?<=Invoice Date: ).*?(?=\s)"
    InvoiceNumPattern = r"(?<=Invoice #: ).*?(?=\s)"
    addressPattern = r"\b([A-Z]{2})\b[,\s]+(\d{5}(-\d{4})?)"
    
    try: 
        custMatch = re.search(customerpattern, text_content).group(0) if re.search(customerpattern, text_content) else np.nan
        payMatch = re.search(paymentPattern, text_content).group(0) if re.search(paymentPattern, text_content) else np.nan
        routeMatch = re.search(routePattern, text_content).group(0) if re.search(routePattern, text_content) else np.nan
        InvoiceDteMatch = re.search(InvoiceDtePattern, text_content).group(0) if re.search(InvoiceDtePattern, text_content) else np.nan
        InvoiceNumMatch = re.search(InvoiceNumPattern, text_content).group(0) if re.search(InvoiceNumPattern, text_content) else np.nan
        addressMatch = re.findall(addressPattern, text_content) if re.findall(addressPattern, text_content) else [(np.nan, np.nan)]

        df = pd.DataFrame({
            "CUSTOMER": custMatch,
            "PAYMENT_TERMS": payMatch,
            "ROUTE": routeMatch,
            "INVOICE_DATE": InvoiceDteMatch,
            "INVOICE_NUMBER": InvoiceNumMatch,
            "STATE": addressMatch[0][0],
            "ZIP_CODE": addressMatch[0][1]
        }, index=[0])
        return ( df )
    except Exception as e:
        print(f"Error processing PDF: {e}")
        print ( text_content )
    

global_df = pd.DataFrame()
ocifs.OCIFileSystem(config="~/.oci/config")
fs = ocifs.OCIFileSystem(config="~/.oci/config")
for file in fs.ls('PROD_ABSINV_BUCKET@idwhr2jotzyc/'):
    try:
        print ( file )
        with fs.open("oci://{}".format ( file ), "rb") as f:
            pdf_content = BytesIO(f.read())
            # df2 = extractDataDF(pdf_content)
            df_list = tabula.read_pdf(pdf_content, pages='all')[0]
            df_list.fillna(0, inplace=True)
            if len(df_list.columns) == 5:
                df_list = df_list.iloc[3:]
                df_list.columns = ['DEPT', 'QTY', 'SERVICE_DESC', 'TAXABLE', 'TOTAL']
            elif len(df_list.columns)== 6 :
                df_list.columns = ['DEPT', 'QTY', 'SERVICE_DESC', 'UNIT' , 'TAXABLE', 'TOTAL']
            elif len(df_list.columns) == 7 :
                df_list.columns = ['DEPT', 'QTY', 'ITEM', 'SERVICE_DESC', 'RATE' , 'AMT' , 'TAXABLE']
            elif len(df_list.columns) == 8 :
                df_list = df_list.iloc[3:]
                df_list.columns = ['DEPT', 'QTY', 'ITEM', 'SERVICE_DESC', 'RATE' , 'AMT' , 'TAXABLE', 'TOTAL' ]

            # df_list['FILE_NAME'] = file
            # df_list['CUSTOMER'] = df2['CUSTOMER'][0] 
            # df_list['PAYMENT_TERMS'] = df2['PAYMENT_TERMS'][0]
            # df_list['ROUTE'] = df2['ROUTE'][0]
            # df_list['INVOICE_DATE'] = df2['INVOICE_DATE'][0]
            # df_list['INVOICE_NUMBER'] = df2['INVOICE_NUMBER'][0]
            # df_list['STATE'] = df2['STATE'][0]
            # df_list['ZIP_CODE'] = df2['ZIP_CODE'][0]
            
            
            global_df = pd.concat([global_df, df_list], ignore_index=True , axis=0)
    except Exception as e:
        print(f"Error processing file {file}: {e}")

global_df.fillna(0, inplace=True)
global_df.to_csv("output.csv", index=False)
