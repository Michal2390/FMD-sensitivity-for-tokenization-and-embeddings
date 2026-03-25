#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import pdfplumber
import sys

sys.stdout.reconfigure(encoding='utf-8')

with pdfplumber.open('PDFs/DesignProposal_z8.pdf') as pdf:
    print(f'Total pages: {len(pdf.pages)}')
    for page_num in range(1, len(pdf.pages) + 1):
        print(f'\n--- Page {page_num} ---')
        try:
            text = pdf.pages[page_num-1].extract_text()
            if text:
                print(text)
        except Exception as e:
            print(f'Error on page {page_num}: {e}')

