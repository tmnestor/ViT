import os
from pathlib import Path
import random
import argparse
import numpy as np
from PIL import Image, ImageDraw, ImageFont
from tqdm import tqdm
from datetime import datetime, timedelta
import string

def anonymize_name(name):
    """
    Replace random characters in a name with 'X' for privacy.
    Always replace at least one character.
    """
    if not name:
        return name
        
    # Convert to list to modify individual characters
    name_chars = list(name)
    
    # Ensure we replace at least one character, up to 30% of the name
    num_to_replace = max(1, int(len(name) * random.uniform(0.1, 0.3)))
    
    # Choose random positions to replace
    positions = random.sample(range(len(name)), min(num_to_replace, len(name)))
    
    # Replace with 'X'
    for pos in positions:
        name_chars[pos] = 'X'
    
    return ''.join(name_chars)

def anonymize_tfn(tfn):
    """
    Replace some digits in a Tax File Number with 'X' for privacy.
    """
    # Convert to list to modify individual characters
    tfn_chars = list(tfn)
    
    # Replace 30-50% of digits with 'X'
    digit_positions = [i for i, char in enumerate(tfn_chars) if char.isdigit()]
    num_to_replace = max(2, int(len(digit_positions) * random.uniform(0.3, 0.5)))
    
    # Choose random digit positions to replace
    if digit_positions:
        replace_positions = random.sample(digit_positions, min(num_to_replace, len(digit_positions)))
        
        # Replace with 'X'
        for pos in replace_positions:
            tfn_chars[pos] = 'X'
    
    return ''.join(tfn_chars)

def generate_tax_document(width=595, height=842, doc_type=None):
    """
    Generate a synthetic Australian tax-related document (not a receipt) that might appear
    in tax submissions.
    
    Args:
        width: Width of the document (default A4 portrait in pixels at 72 DPI)
        height: Height of the document (default A4 portrait in pixels at 72 DPI)
        doc_type: Type of document to generate. If None, a random type is selected.
        
    Returns:
        document_img: PIL Image of the synthetic tax document
    """
    # List of possible document types for Australian context
    document_types = [
        "Tax Return Summary", "Income Statement", "PAYG Payment Summary", 
        "ATO Notice of Assessment", "Private Health Insurance Statement", 
        "Medicare Levy Exemption Certificate", "Superannuation Statement",
        "Tax Invoice", "Capital Gains Tax Statement", "Investment Income Summary",
        "Rental Property Statement", "Business Activity Statement (BAS)",
        "Vehicle Registration", "Interest Income Statement", "Dividend Statement"
    ]
    
    if doc_type is None:
        doc_type = random.choice(document_types)
    
    # Create a white document (A4 portrait)
    document = Image.new('RGB', (width, height), color=(255, 255, 255))
    draw = ImageDraw.Draw(document)
    
    # Add ATO-style header
    # Draw blue ATO banner at top
    ato_blue = (0, 51, 160)  # ATO blue color
    draw.rectangle([(0, 0), (width, 60)], fill=ato_blue)
    
    # Try to draw ATO text in white
    try:
        header_font = None
        for font_path in ['/Library/Fonts/Arial Bold.ttf', '/usr/share/fonts/truetype/freefont/FreeSansBold.ttf', 'C:\\Windows\\Fonts\\arialbd.ttf']:
            if os.path.exists(font_path):
                header_font = ImageFont.truetype(font_path, 30)
                break
                
        if header_font is None:
            header_font = ImageFont.load_default()
            
        # Draw "Australian Taxation Office" in white
        draw.text((20, 15), "Australian Taxation Office", fill=(255, 255, 255), font=header_font)
    except Exception:
        # Fallback if font loading fails
        draw.text((20, 15), "Australian Taxation Office", fill=(255, 255, 255), font=ImageFont.load_default())
    
    # Try to load a font, fall back to default if not available
    try:
        # Try different common font paths
        font_paths = [
            '/Library/Fonts/Arial.ttf',  # macOS
            '/System/Library/Fonts/Supplemental/Arial.ttf',  # macOS alternative
            '/usr/share/fonts/truetype/msttcorefonts/Arial.ttf',  # Linux
            'C:\\Windows\\Fonts\\arial.ttf',  # Windows
            '/usr/share/fonts/truetype/freefont/FreeMono.ttf',  # Linux alternative
        ]
        
        # Try to find a usable font
        header_font = None
        title_font = None
        regular_font = None
        small_font = None
        
        for font_path in font_paths:
            if os.path.exists(font_path):
                try:
                    header_font = ImageFont.truetype(font_path, 28)
                    title_font = ImageFont.truetype(font_path, 22)
                    regular_font = ImageFont.truetype(font_path, 16)
                    small_font = ImageFont.truetype(font_path, 12)
                    break
                except IOError:
                    continue
        
        # Fall back to default if no font found
        if header_font is None:
            header_font = ImageFont.load_default()
            title_font = ImageFont.load_default()
            regular_font = ImageFont.load_default()
            small_font = ImageFont.load_default()
            
    except Exception:
        # Fallback to default font
        header_font = ImageFont.load_default()
        title_font = ImageFont.load_default()
        regular_font = ImageFont.load_default()
        small_font = ImageFont.load_default()
    
    # Generate document content based on the type - Australia specific documents
    if "Income Statement" in doc_type or "PAYG" in doc_type:
        create_income_statement(document, draw, header_font, title_font, regular_font, small_font)
    elif "Assessment" in doc_type:
        create_notice_of_assessment(document, draw, header_font, title_font, regular_font, small_font)
    elif "Tax Return" in doc_type:
        create_tax_return_summary(document, draw, header_font, title_font, regular_font, small_font)
    elif "Superannuation" in doc_type:
        create_superannuation_statement(document, draw, header_font, title_font, regular_font, small_font)
    elif "Private Health" in doc_type:
        create_private_health_statement(document, draw, header_font, title_font, regular_font, small_font)
    elif "Medicare" in doc_type:
        create_medicare_statement(document, draw, header_font, title_font, regular_font, small_font)
    elif "Invoice" in doc_type:
        create_tax_invoice(document, draw, header_font, title_font, regular_font, small_font)
    elif "BAS" in doc_type:
        create_business_activity_statement(document, draw, header_font, title_font, regular_font, small_font)
    else:
        # Generic Australian tax document
        create_generic_document(document, draw, header_font, title_font, regular_font, small_font, doc_type)
    
    return document

def create_w2_form(document, draw, header_font, title_font, regular_font, small_font):
    """Create a W-2 form"""
    # Document header
    draw.text((document.width//2, 40), "Form W-2 Wage and Tax Statement", fill=(0, 0, 0), font=header_font, anchor="mt")
    draw.text((document.width//2, 80), f"Tax Year: {random.randint(2020, 2023)}", fill=(0, 0, 0), font=title_font, anchor="mt")
    
    # Employer details box
    draw_box(draw, 50, 120, document.width//2 - 70, 200, "Employer Information")
    employer_name = f"{random.choice(['ABC', 'XYZ', 'Global', 'National', 'American'])} {random.choice(['Corp', 'Inc', 'LLC', 'Enterprises', 'Company'])}"
    draw.text((60, 150), f"Employer: {employer_name}", fill=(0, 0, 0), font=regular_font)
    draw.text((60, 170), f"EIN: {random.randint(10, 99)}-{random.randint(1000000, 9999999)}", fill=(0, 0, 0), font=regular_font)
    draw.text((60, 190), f"{random.randint(100, 9999)} Main St, City, State {random.randint(10000, 99999)}", fill=(0, 0, 0), font=regular_font)
    
    # Employee details box
    draw_box(draw, document.width//2 + 20, 120, document.width - 50, 200, "Employee Information")
    employee_first = random.choice(["John", "Mary", "Robert", "Patricia", "Michael", "Jennifer"])
    employee_last = random.choice(["Smith", "Johnson", "Williams", "Brown", "Jones", "Garcia"])
    # Anonymize employee name
    anon_first = anonymize_name(employee_first)
    anon_last = anonymize_name(employee_last)
    draw.text((document.width//2 + 30, 150), f"Employee: {anon_first} {anon_last}", fill=(0, 0, 0), font=regular_font)
    draw.text((document.width//2 + 30, 170), f"SSN: XXX-XX-{random.randint(1000, 9999)}", fill=(0, 0, 0), font=regular_font)
    draw.text((document.width//2 + 30, 190), f"{random.randint(100, 9999)} Oak St, City, State {random.randint(10000, 99999)}", fill=(0, 0, 0), font=regular_font)
    
    # Income details
    y_pos = 240
    draw_box(draw, 50, y_pos, document.width - 50, y_pos + 300, "Income and Tax Information")
    
    wages = random.randint(30000, 120000) + random.random()
    fed_tax = round(wages * random.uniform(0.12, 0.25), 2)
    ss_tax = round(wages * 0.062, 2)
    medicare = round(wages * 0.0145, 2)
    state_tax = round(wages * random.uniform(0.03, 0.09), 2)
    
    columns = [
        ("Box 1: Wages, tips, other comp.", f"${wages:.2f}"),
        ("Box 2: Federal income tax withheld", f"${fed_tax:.2f}"),
        ("Box 3: Social security wages", f"${wages:.2f}"),
        ("Box 4: Social security tax withheld", f"${ss_tax:.2f}"),
        ("Box 5: Medicare wages and tips", f"${wages:.2f}"),
        ("Box 6: Medicare tax withheld", f"${medicare:.2f}"),
        ("Box 16: State wages, tips, etc.", f"${wages:.2f}"),
        ("Box 17: State income tax", f"${state_tax:.2f}")
    ]
    
    # Draw columns of data
    y_offset = y_pos + 30
    for i, (label, value) in enumerate(columns):
        # Two columns layout
        x_pos = 60 if i % 2 == 0 else document.width//2 + 30
        y = y_offset + (i // 2) * 30
        draw.text((x_pos, y), label, fill=(0, 0, 0), font=regular_font)
        # Align values to the right
        value_width = regular_font.getsize(value)[0] if hasattr(regular_font, "getsize") else 100
        draw.text((x_pos + 250 - value_width, y), value, fill=(0, 0, 0), font=regular_font)
    
    # Footer with official-looking text
    draw.text((document.width//2, document.height - 50), "This is an important tax document. Please retain for your records.", 
              fill=(0, 0, 0), font=small_font, anchor="mt")

def create_income_statement(document, draw, header_font, title_font, regular_font, small_font):
    """Create an Australian Income Statement (formerly PAYG Summary)"""
    # Document header is already added with ATO banner
    
    # Title under the ATO header
    draw.text((document.width//2, 80), "Income Statement", fill=(0, 0, 0), font=header_font, anchor="mt")
    
    # Financial year (Australia uses July 1 to June 30 fiscal year)
    year_end = random.randint(2020, 2023)
    year_start = year_end - 1
    draw.text((document.width//2, 120), f"For the period 1 July {year_start} to 30 June {year_end}", 
              fill=(0, 0, 0), font=title_font, anchor="mt")
    
    # Employer details box
    draw_box(draw, 50, 160, document.width//2 - 20, 260, "Payer Information")
    employer_name = f"{random.choice(['Westpac', 'Commonwealth', 'National', 'ANZ', 'Telstra', 'Woolworths', 'Coles'])} {random.choice(['Group', 'Corporation', 'Pty Ltd', 'Ltd', 'Australia'])}"
    draw.text((60, 190), f"Employer: {employer_name}", fill=(0, 0, 0), font=regular_font)
    draw.text((60, 210), f"ABN: {random.randint(10, 99)} {random.randint(100, 999)} {random.randint(100, 999)} {random.randint(100, 999)}", fill=(0, 0, 0), font=regular_font)
    
    # Australian address
    states = ["NSW", "VIC", "QLD", "SA", "WA", "TAS", "ACT", "NT"]
    state = random.choice(states)
    postcode = random.randint(2000, 2999) if state == "NSW" else random.randint(3000, 8999)
    draw.text((60, 230), f"{random.randint(1, 999)} {random.choice(['George', 'Pitt', 'Collins', 'Elizabeth', 'Bourke'])} Street", fill=(0, 0, 0), font=regular_font)
    draw.text((60, 250), f"{random.choice(['Sydney', 'Melbourne', 'Brisbane', 'Perth', 'Adelaide'])}, {state} {postcode}", fill=(0, 0, 0), font=regular_font)
    
    # Employee details box
    draw_box(draw, document.width//2 + 20, 160, document.width - 50, 260, "Payee Information")
    employee_first = random.choice(["John", "Mary", "Robert", "Patricia", "Michael", "Jennifer", "David", "Sarah"])
    employee_last = random.choice(["Smith", "Johnson", "Williams", "Brown", "Jones", "Wilson", "Taylor", "Anderson"])
    # Anonymize employee name and TFN
    anon_first = anonymize_name(employee_first)
    anon_last = anonymize_name(employee_last)
    draw.text((document.width//2 + 30, 190), f"Employee: {anon_first} {anon_last}", fill=(0, 0, 0), font=regular_font)
    
    # Generate and anonymize TFN
    tfn = f"{random.randint(100, 999)} {random.randint(100, 999)} {random.randint(100, 999)}"
    anon_tfn = anonymize_tfn(tfn)
    draw.text((document.width//2 + 30, 210), f"TFN: {anon_tfn}", fill=(0, 0, 0), font=regular_font)
    draw.text((document.width//2 + 30, 230), f"{random.randint(1, 99)}/{random.randint(1, 99)} {random.choice(['King', 'Queen', 'William', 'Market'])} Street", fill=(0, 0, 0), font=regular_font)
    draw.text((document.width//2 + 30, 250), f"{random.choice(['Sydney', 'Melbourne', 'Brisbane', 'Perth', 'Adelaide'])}, {random.choice(states)} {random.randint(2000, 9999)}", fill=(0, 0, 0), font=regular_font)
    
    # Income details
    y_pos = 280
    draw_box(draw, 50, y_pos, document.width - 50, y_pos + 300, "Payment Summary")
    
    # Generate income details
    base_salary = random.randint(50000, 120000)
    allowances = round(base_salary * random.uniform(0.01, 0.05), 2)
    tax_withheld = round(base_salary * random.uniform(0.19, 0.37), 2)  # Australian tax rates
    super_contribution = round(base_salary * 0.10, 2)  # 10% superannuation in Australia
    
    items = [
        ("Gross Payments", f"${base_salary:.2f}"),
        ("Allowances", f"${allowances:.2f}"),
        ("Total Gross Payments", f"${base_salary + allowances:.2f}"),
        ("Tax Withheld", f"${tax_withheld:.2f}"),
        ("Superannuation", f"${super_contribution:.2f}")
    ]
    
    # Draw data
    y_offset = y_pos + 30
    for i, (label, value) in enumerate(items):
        draw.text((60, y_offset + i * 40), label, fill=(0, 0, 0), font=regular_font)
        
        # Highlight the important values (bold if possible)
        font_to_use = title_font if i in [2, 3] else regular_font
        
        # Align values to the right
        value_width = font_to_use.getsize(value)[0] if hasattr(font_to_use, "getsize") else 100
        draw.text((document.width - 100 - value_width, y_offset + i * 40), value, fill=(0, 0, 0), font=font_to_use)
    
    # Footer with official-looking text
    draw.text((document.width//2, document.height - 80), 
              "This is a copy of the information reported to the Australian Taxation Office.", 
              fill=(0, 0, 0), font=small_font, anchor="mt")
    draw.text((document.width//2, document.height - 50), 
              "Please retain this statement for your taxation records.", 
              fill=(0, 0, 0), font=small_font, anchor="mt")

def create_property_tax(document, draw, header_font, title_font, regular_font, small_font):
    """Create a property tax statement"""
    # Document header
    draw.text((document.width//2, 40), "Property Tax Statement", fill=(0, 0, 0), font=header_font, anchor="mt")
    year = random.randint(2020, 2023)
    draw.text((document.width//2, 80), f"Tax Year: {year}", fill=(0, 0, 0), font=title_font, anchor="mt")
    
    # County information
    county_name = f"{random.choice(['Washington', 'Franklin', 'Jefferson', 'Lincoln', 'Madison'])} County"
    draw.text((document.width//2, 120), county_name, fill=(0, 0, 0), font=title_font, anchor="mt")
    draw.text((document.width//2, 150), f"County Assessor's Office", fill=(0, 0, 0), font=regular_font, anchor="mt")
    
    # Property owner info
    draw_box(draw, 50, 190, document.width - 50, 290, "Property & Owner Information")
    owner_first = random.choice(["John", "Mary", "Robert", "Patricia", "Michael", "Jennifer"])
    owner_last = random.choice(["Smith", "Johnson", "Williams", "Brown", "Jones", "Garcia"])
    
    street_name = random.choice(["Oak", "Maple", "Pine", "Cedar", "Elm", "Walnut", "Main"])
    street_type = random.choice(["St", "Ave", "Blvd", "Dr", "Way", "Ln", "Rd"])
    
    draw.text((60, 220), f"Owner: {owner_first} {owner_last}", fill=(0, 0, 0), font=regular_font)
    draw.text((60, 240), f"Property Address: {random.randint(100, 9999)} {street_name} {street_type}", fill=(0, 0, 0), font=regular_font)
    draw.text((60, 260), f"Parcel ID: {random.randint(10, 99)}-{random.randint(1000, 9999)}-{random.randint(100, 999)}", fill=(0, 0, 0), font=regular_font)
    
    # Assessment and tax details
    draw_box(draw, 50, 310, document.width - 50, 500, "Assessment & Tax Details")
    
    assessed_value = random.randint(150000, 800000)
    tax_rate = random.uniform(0.01, 0.025)
    annual_tax = round(assessed_value * tax_rate, 2)
    
    draw.text((60, 340), f"Assessed Value: ${assessed_value:,}", fill=(0, 0, 0), font=regular_font)
    draw.text((60, 370), f"Tax Rate: {tax_rate:.3f}", fill=(0, 0, 0), font=regular_font)
    draw.text((60, 400), f"Annual Tax Amount: ${annual_tax:,.2f}", fill=(0, 0, 0), font=regular_font)
    draw.text((60, 430), f"Due Date: {random.choice(['January', 'February', 'April', 'October'])} {random.randint(1, 28)}, {year}", fill=(0, 0, 0), font=regular_font)
    
    # Payment status
    status = random.choice(["PAID", "DUE", "PARTIALLY PAID"])
    status_color = (0, 100, 0) if status == "PAID" else ((200, 0, 0) if status == "DUE" else (200, 100, 0))
    draw.text((document.width - 200, 470), f"Status: {status}", fill=status_color, font=title_font)
    
    # Footer
    draw.text((document.width//2, document.height - 100), 
              f"This statement is for informational purposes only.", 
              fill=(0, 0, 0), font=small_font, anchor="mt")
    draw.text((document.width//2, document.height - 70), 
              f"Please retain this document for your tax records.", 
              fill=(0, 0, 0), font=small_font, anchor="mt")

def create_mortgage_statement(document, draw, header_font, title_font, regular_font, small_font):
    """Create a mortgage interest statement"""
    # Bank header
    bank_name = f"{random.choice(['First', 'United', 'National', 'Premier', 'Liberty'])} {random.choice(['Bank', 'Mortgage', 'Financial', 'Home Loans'])}"
    draw.text((document.width//2, 40), bank_name, fill=(0, 0, 0), font=header_font, anchor="mt")
    draw.text((document.width//2, 80), "Form 1098: Mortgage Interest Statement", fill=(0, 0, 0), font=title_font, anchor="mt")
    draw.text((document.width//2, 120), f"Tax Year: {random.randint(2020, 2023)}", fill=(0, 0, 0), font=regular_font, anchor="mt")
    
    # Lender information
    draw_box(draw, 50, 160, document.width//2 - 70, 260, "Lender Information")
    draw.text((60, 190), f"Lender: {bank_name}", fill=(0, 0, 0), font=regular_font)
    draw.text((60, 210), f"EIN: {random.randint(10, 99)}-{random.randint(1000000, 9999999)}", fill=(0, 0, 0), font=regular_font)
    draw.text((60, 230), f"{random.randint(100, 9999)} Financial Ave, City, State {random.randint(10000, 99999)}", fill=(0, 0, 0), font=regular_font)
    
    # Borrower information
    draw_box(draw, document.width//2 + 20, 160, document.width - 50, 260, "Borrower Information")
    borrower_first = random.choice(["John", "Mary", "Robert", "Patricia", "Michael", "Jennifer"])
    borrower_last = random.choice(["Smith", "Johnson", "Williams", "Brown", "Jones", "Garcia"])
    draw.text((document.width//2 + 30, 190), f"Borrower: {borrower_first} {borrower_last}", fill=(0, 0, 0), font=regular_font)
    draw.text((document.width//2 + 30, 210), f"SSN: XXX-XX-{random.randint(1000, 9999)}", fill=(0, 0, 0), font=regular_font)
    
    street_name = random.choice(["Oak", "Maple", "Pine", "Cedar", "Elm", "Walnut", "Main"])
    street_type = random.choice(["St", "Ave", "Blvd", "Dr", "Way", "Ln", "Rd"])
    draw.text((document.width//2 + 30, 230), f"Property: {random.randint(100, 9999)} {street_name} {street_type}", fill=(0, 0, 0), font=regular_font)
    
    # Mortgage details
    y_pos = 280
    draw_box(draw, 50, y_pos, document.width - 50, y_pos + 250, "Mortgage Information")
    
    principal = random.randint(100000, 800000)
    interest_rate = random.uniform(0.025, 0.06)
    interest_paid = round(principal * interest_rate, 2)
    points_paid = round(principal * random.uniform(0.005, 0.02), 2)
    property_tax = round(principal * random.uniform(0.01, 0.025), 2)
    
    items = [
        ("1. Mortgage interest received from borrower", f"${interest_paid:,.2f}"),
        ("2. Outstanding mortgage principal", f"${principal:,}"),
        ("4. Mortgage insurance premiums", f"${round(principal * 0.005, 2):,.2f}"),
        ("5. Points paid on purchase of residence", f"${points_paid:,.2f}"),
        ("10. Property taxes", f"${property_tax:,.2f}")
    ]
    
    # Draw data
    y_offset = y_pos + 30
    for i, (label, value) in enumerate(items):
        draw.text((60, y_offset + i * 40), label, fill=(0, 0, 0), font=regular_font)
        # Align values to the right
        value_width = regular_font.getsize(value)[0] if hasattr(regular_font, "getsize") else 150
        draw.text((document.width - 100 - value_width, y_offset + i * 40), value, fill=(0, 0, 0), font=regular_font)
    
    # Footer
    draw.text((document.width//2, document.height - 50), 
              "This information is being furnished to the Internal Revenue Service.", 
              fill=(0, 0, 0), font=small_font, anchor="mt")

def create_donation_receipt(document, draw, header_font, title_font, regular_font, small_font):
    """Create a charitable donation receipt"""
    # Organization header
    org_name = f"{random.choice(['Community', 'Global', 'Hope', 'United', 'National'])} {random.choice(['Foundation', 'Charity', 'Relief Fund', 'Alliance', 'Society'])}"
    draw.text((document.width//2, 40), org_name, fill=(0, 0, 0), font=header_font, anchor="mt")
    draw.text((document.width//2, 80), "Charitable Donation Receipt", fill=(0, 0, 0), font=title_font, anchor="mt")
    draw.text((document.width//2, 120), "For Tax Purposes", fill=(0, 0, 0), font=regular_font, anchor="mt")
    
    # Organization details
    draw_box(draw, 50, 160, document.width - 50, 260, "Organization Information")
    draw.text((60, 190), f"Organization: {org_name}", fill=(0, 0, 0), font=regular_font)
    draw.text((60, 210), f"Tax ID: {random.randint(10, 99)}-{random.randint(1000000, 9999999)}", fill=(0, 0, 0), font=regular_font)
    draw.text((60, 230), f"{random.randint(100, 9999)} Charity Ave, City, State {random.randint(10000, 99999)}", fill=(0, 0, 0), font=regular_font)
    
    # Donor information
    draw_box(draw, 50, 280, document.width - 50, 380, "Donor Information")
    donor_first = random.choice(["John", "Mary", "Robert", "Patricia", "Michael", "Jennifer"])
    donor_last = random.choice(["Smith", "Johnson", "Williams", "Brown", "Jones", "Garcia"])
    draw.text((60, 310), f"Donor: {donor_first} {donor_last}", fill=(0, 0, 0), font=regular_font)
    
    street_name = random.choice(["Oak", "Maple", "Pine", "Cedar", "Elm", "Walnut", "Main"])
    street_type = random.choice(["St", "Ave", "Blvd", "Dr", "Way", "Ln", "Rd"])
    draw.text((60, 330), f"Address: {random.randint(100, 9999)} {street_name} {street_type}, City, State {random.randint(10000, 99999)}", fill=(0, 0, 0), font=regular_font)
    
    # Donation details
    y_pos = 400
    draw_box(draw, 50, y_pos, document.width - 50, y_pos + 250, "Donation Details")
    
    if random.random() < 0.7:
        # Cash donation
        donation_type = "Cash Donation"
        donation_amount = random.randint(50, 5000)
        donation_description = f"${donation_amount:,}.00"
    else:
        # Non-cash donation
        donation_type = "Non-Cash Donation"
        items = random.choice([
            "Clothing and household items",
            "Furniture and appliances",
            "Electronics and computer equipment",
            "Books and educational materials",
            "Artwork and collectibles"
        ])
        donation_amount = random.randint(100, 3000)
        donation_description = f"{items} - Estimated Fair Market Value: ${donation_amount:,}.00"
    
    # Date of donation
    year = random.randint(2020, 2023)
    month = random.randint(1, 12)
    day = random.randint(1, 28)
    donation_date = f"{month:02d}/{day:02d}/{year}"
    
    draw.text((60, y_pos + 30), f"Date of Donation: {donation_date}", fill=(0, 0, 0), font=regular_font)
    draw.text((60, y_pos + 70), f"Type: {donation_type}", fill=(0, 0, 0), font=regular_font)
    draw.text((60, y_pos + 110), f"Description: {donation_description}", fill=(0, 0, 0), font=regular_font)
    
    # Receipt ID and signature
    receipt_id = ''.join(random.choices(string.ascii_uppercase + string.digits, k=8))
    draw.text((60, y_pos + 170), f"Receipt ID: {receipt_id}", fill=(0, 0, 0), font=regular_font)
    
    official_name = random.choice(["James Wilson", "Sarah Johnson", "Robert Davis", "Elizabeth Taylor", "Michael Brown"])
    draw.text((document.width - 200, y_pos + 170), f"{official_name}", fill=(0, 0, 0), font=regular_font)
    draw.text((document.width - 200, y_pos + 190), "Authorized Signature", fill=(0, 0, 0), font=small_font)
    
    # Tax deductibility statement
    draw.text((document.width//2, document.height - 80), 
              f"This receipt confirms that {org_name} did not provide any goods or services", 
              fill=(0, 0, 0), font=small_font, anchor="mt")
    draw.text((document.width//2, document.height - 60), 
              "in return for this donation. This donation is tax-deductible to the extent allowed by law.", 
              fill=(0, 0, 0), font=small_font, anchor="mt")

def create_medical_expense(document, draw, header_font, title_font, regular_font, small_font):
    """Create a medical expense summary"""
    # Header
    provider_name = f"{random.choice(['Summit', 'Valley', 'Central', 'Community', 'Regional'])} {random.choice(['Medical Center', 'Hospital', 'Health Partners', 'Care Clinic'])}"
    draw.text((document.width//2, 40), provider_name, fill=(0, 0, 0), font=header_font, anchor="mt")
    draw.text((document.width//2, 80), "Medical Expense Summary", fill=(0, 0, 0), font=title_font, anchor="mt")
    draw.text((document.width//2, 120), "For Tax Year Documentation", fill=(0, 0, 0), font=regular_font, anchor="mt")
    
    # Patient information
    draw_box(draw, 50, 160, document.width - 50, 260, "Patient Information")
    patient_first = random.choice(["John", "Mary", "Robert", "Patricia", "Michael", "Jennifer"])
    patient_last = random.choice(["Smith", "Johnson", "Williams", "Brown", "Jones", "Garcia"])
    
    year = random.randint(2020, 2023)
    month = random.randint(1, 12)
    day = random.randint(1, 28)
    date = f"{month:02d}/{day:02d}/{year}"
    
    draw.text((60, 190), f"Patient: {patient_first} {patient_last}", fill=(0, 0, 0), font=regular_font)
    draw.text((60, 210), f"Date of Birth: {random.randint(1, 12)}/{random.randint(1, 28)}/{random.randint(1950, 2000)}", fill=(0, 0, 0), font=regular_font)
    draw.text((60, 230), f"Statement Date: {date}", fill=(0, 0, 0), font=regular_font)
    
    # Service summary
    y_pos = 280
    draw_box(draw, 50, y_pos, document.width - 50, y_pos + 300, "Medical Services & Expenses")
    
    # Generate random medical services
    services = [
        "Office visit - Primary Care",
        "Office visit - Specialist",
        "Emergency Room Visit",
        "Hospital Stay",
        "Laboratory Services",
        "X-Ray / Imaging",
        "MRI Scan",
        "Physical Therapy",
        "Prescription Medication",
        "Surgical Procedure",
        "Dental Services",
        "Vision Services",
        "Mental Health Services"
    ]
    
    # Column headers
    draw.text((60, y_pos + 30), "Service Description", fill=(0, 0, 0), font=regular_font)
    draw.text((350, y_pos + 30), "Date of Service", fill=(0, 0, 0), font=regular_font)
    draw.text((500, y_pos + 30), "Amount", fill=(0, 0, 0), font=regular_font)
    draw.text((620, y_pos + 30), "Insurance Paid", fill=(0, 0, 0), font=regular_font)
    draw.line([(60, y_pos + 50), (document.width - 60, y_pos + 50)], fill=(0, 0, 0), width=1)
    
    # Generate service entries
    num_services = random.randint(3, 8)
    y_offset = y_pos + 70
    total_billed = 0
    total_insurance = 0
    
    for i in range(num_services):
        service = random.choice(services)
        services.remove(service)  # Don't reuse services
        if len(services) < 3:  # Replenish if running low
            services = [
                "Office visit - Primary Care",
                "Office visit - Specialist",
                "Emergency Room Visit",
                "Laboratory Services",
                "X-Ray / Imaging",
                "Physical Therapy",
                "Prescription Medication"
            ]
        
        # Random service date in the tax year
        service_month = random.randint(1, 12)
        service_day = random.randint(1, 28)
        service_date = f"{service_month:02d}/{service_day:02d}/{year}"
        
        # Costs
        if "Emergency" in service or "Hospital" in service or "MRI" in service or "Surgical" in service:
            amount = random.randint(1000, 10000)
        else:
            amount = random.randint(100, 800)
        
        insurance_paid = round(amount * random.uniform(0.6, 0.9), 2)
        
        total_billed += amount
        total_insurance += insurance_paid
        
        draw.text((60, y_offset), service, fill=(0, 0, 0), font=regular_font)
        draw.text((350, y_offset), service_date, fill=(0, 0, 0), font=regular_font)
        draw.text((500, y_offset), f"${amount:,.2f}", fill=(0, 0, 0), font=regular_font)
        draw.text((620, y_offset), f"${insurance_paid:,.2f}", fill=(0, 0, 0), font=regular_font)
        
        y_offset += 30
    
    # Total
    patient_responsibility = total_billed - total_insurance
    
    draw.line([(60, y_offset + 10), (document.width - 60, y_offset + 10)], fill=(0, 0, 0), width=1)
    draw.text((350, y_offset + 30), "Total:", fill=(0, 0, 0), font=title_font)
    draw.text((500, y_offset + 30), f"${total_billed:,.2f}", fill=(0, 0, 0), font=title_font)
    draw.text((620, y_offset + 30), f"${total_insurance:,.2f}", fill=(0, 0, 0), font=title_font)
    
    # Patient responsibility
    draw.text((350, y_offset + 70), "Patient Responsibility:", fill=(0, 0, 0), font=title_font)
    draw.text((620, y_offset + 70), f"${patient_responsibility:,.2f}", fill=(0, 0, 0), font=title_font)
    
    # Footer
    draw.text((document.width//2, document.height - 50), 
              "This summary is provided for tax purposes and does not replace official medical records.", 
              fill=(0, 0, 0), font=small_font, anchor="mt")

def create_investment_statement(document, draw, header_font, title_font, regular_font, small_font):
    """Create an investment statement"""
    # Financial institution header
    institution = f"{random.choice(['Fidelity', 'Vanguard', 'Charles Schwab', 'Morgan Stanley', 'Merrill Lynch'])} Investments"
    draw.text((document.width//2, 40), institution, fill=(0, 0, 0), font=header_font, anchor="mt")
    
    year = random.randint(2020, 2023)
    draw.text((document.width//2, 80), f"Annual Investment Statement - {year}", fill=(0, 0, 0), font=title_font, anchor="mt")
    
    # Account holder information
    draw_box(draw, 50, 120, document.width - 50, 220, "Account Information")
    holder_first = random.choice(["John", "Mary", "Robert", "Patricia", "Michael", "Jennifer"])
    holder_last = random.choice(["Smith", "Johnson", "Williams", "Brown", "Jones", "Garcia"])
    
    account_type = random.choice(["Individual Investment Account", "IRA", "Roth IRA", "401(k)", "Joint Investment Account"])
    account_number = ''.join(random.choices(string.digits, k=8))
    
    draw.text((60, 150), f"Account Holder: {holder_first} {holder_last}", fill=(0, 0, 0), font=regular_font)
    draw.text((60, 170), f"Account Type: {account_type}", fill=(0, 0, 0), font=regular_font)
    draw.text((60, 190), f"Account Number: XXXX-XX{account_number[-4:]}", fill=(0, 0, 0), font=regular_font)
    
    # Investment summary
    y_pos = 240
    draw_box(draw, 50, y_pos, document.width - 50, y_pos + 200, "Annual Summary")
    
    # Generate values
    starting_balance = random.randint(10000, 500000)
    contributions = random.randint(0, 20000)
    withdrawals = random.randint(0, 10000) if random.random() < 0.4 else 0
    
    # Market performance (could be negative)
    if random.random() < 0.7:  # More likely positive
        market_change_pct = random.uniform(0.02, 0.2)
    else:  # Negative return
        market_change_pct = random.uniform(-0.15, -0.01)
    
    market_change = round(starting_balance * market_change_pct, 2)
    ending_balance = starting_balance + contributions - withdrawals + market_change
    
    # Format as currency strings
    starting_balance_str = f"${starting_balance:,.2f}"
    contributions_str = f"${contributions:,.2f}"
    withdrawals_str = f"${withdrawals:,.2f}"
    market_change_str = f"${market_change:,.2f}" if market_change >= 0 else f"-${abs(market_change):,.2f}"
    ending_balance_str = f"${ending_balance:,.2f}"
    
    # Draw summary
    items = [
        ("Starting Balance (January 1):", starting_balance_str),
        ("Contributions:", contributions_str),
        ("Withdrawals:", withdrawals_str),
        ("Investment Gains/Losses:", market_change_str),
        ("Ending Balance (December 31):", ending_balance_str)
    ]
    
    y_offset = y_pos + 30
    for i, (label, value) in enumerate(items):
        draw.text((60, y_offset + i * 30), label, fill=(0, 0, 0), font=regular_font)
        
        # Use red for losses
        text_color = (200, 0, 0) if "Losses" in label and market_change < 0 else (0, 0, 0)
        
        # Align values to the right
        value_width = regular_font.getsize(value)[0] if hasattr(regular_font, "getsize") else 150
        draw.text((document.width - 100 - value_width, y_offset + i * 30), value, fill=text_color, font=regular_font)
    
    # Tax information
    y_pos = 460
    draw_box(draw, 50, y_pos, document.width - 50, y_pos + 200, "Tax Information")
    
    # Generate tax values
    if account_type in ["IRA", "401(k)"]:
        tax_items = [
            ("Tax-Deferred Growth:", "Not taxable until withdrawal"),
            ("Contributions:", f"${contributions:,.2f} (May be tax-deductible)"),
            ("Early Withdrawals Subject to Penalty:", "Yes (if under age 59½)")
        ]
    elif account_type == "Roth IRA":
        tax_items = [
            ("Tax-Free Growth:", "Yes"),
            ("Contributions:", f"${contributions:,.2f} (Not tax-deductible)"),
            ("Qualified Withdrawals:", "Tax-Free")
        ]
    else:  # Taxable account
        dividends = round(starting_balance * random.uniform(0.01, 0.04), 2)
        capital_gains = round(market_change * 0.8, 2) if market_change > 0 else 0
        
        tax_items = [
            ("Dividends:", f"${dividends:,.2f}"),
            ("Capital Gains:", f"${capital_gains:,.2f}"),
            ("Tax Forms:", "Form 1099-DIV and 1099-B will be issued")
        ]
    
    y_offset = y_pos + 30
    for i, (label, value) in enumerate(tax_items):
        draw.text((60, y_offset + i * 30), label, fill=(0, 0, 0), font=regular_font)
        
        # Align values to the right
        value_width = regular_font.getsize(value)[0] if hasattr(regular_font, "getsize") else 200
        draw.text((document.width - 100 - value_width, y_offset + i * 30), value, fill=(0, 0, 0), font=regular_font)
    
    # Footer
    draw.text((document.width//2, document.height - 70), 
              "This statement summarizes your account activity for the calendar year.", 
              fill=(0, 0, 0), font=small_font, anchor="mt")
    draw.text((document.width//2, document.height - 40), 
              "Please consult your tax advisor regarding the tax implications of your investments.", 
              fill=(0, 0, 0), font=small_font, anchor="mt")

def create_invoice(document, draw, header_font, title_font, regular_font, small_font):
    """Create a business invoice"""
    # Company header
    company_name = f"{random.choice(['Acme', 'Summit', 'Apex', 'Elite', 'Premier'])} {random.choice(['Services', 'Solutions', 'Consulting', 'Enterprises', 'Partners'])}"
    draw.text((document.width//2, 40), company_name, fill=(0, 0, 0), font=header_font, anchor="mt")
    draw.text((document.width//2, 80), "INVOICE", fill=(0, 0, 0), font=title_font, anchor="mt")
    
    # Invoice details
    invoice_number = ''.join(random.choices(string.ascii_uppercase, k=2)) + ''.join(random.choices(string.digits, k=6))
    
    year = random.randint(2020, 2023)
    month = random.randint(1, 12)
    day = random.randint(1, 28)
    invoice_date = f"{month:02d}/{day:02d}/{year}"
    
    due_month = month + 1 if month < 12 else 1
    due_year = year if month < 12 else year + 1
    due_date = f"{due_month:02d}/{day:02d}/{due_year}"
    
    draw.text((60, 120), f"Invoice #: {invoice_number}", fill=(0, 0, 0), font=regular_font)
    draw.text((60, 140), f"Date: {invoice_date}", fill=(0, 0, 0), font=regular_font)
    draw.text((60, 160), f"Due Date: {due_date}", fill=(0, 0, 0), font=regular_font)
    
    # Company and client details
    draw_box(draw, 50, 190, document.width//2 - 20, 320, "From")
    draw.text((60, 220), company_name, fill=(0, 0, 0), font=regular_font)
    draw.text((60, 240), f"{random.randint(100, 9999)} Business Ave, Suite {random.randint(100, 999)}", fill=(0, 0, 0), font=regular_font)
    draw.text((60, 260), f"City, State {random.randint(10000, 99999)}", fill=(0, 0, 0), font=regular_font)
    draw.text((60, 280), f"Phone: (555) {random.randint(100, 999)}-{random.randint(1000, 9999)}", fill=(0, 0, 0), font=regular_font)
    draw.text((60, 300), f"Tax ID: {random.randint(10, 99)}-{random.randint(1000000, 9999999)}", fill=(0, 0, 0), font=regular_font)
    
    # Client information
    draw_box(draw, document.width//2 + 20, 190, document.width - 50, 320, "To")
    client_name = f"{random.choice(['John', 'Mary', 'Robert', 'Patricia', 'Michael', 'Jennifer', 'ABC', 'XYZ'])} {random.choice(['Smith', 'Johnson', 'Williams', 'Brown', 'Jones', 'Garcia', 'Company', 'Corp'])}"
    draw.text((document.width//2 + 30, 220), client_name, fill=(0, 0, 0), font=regular_font)
    draw.text((document.width//2 + 30, 240), f"{random.randint(100, 9999)} Client St", fill=(0, 0, 0), font=regular_font)
    draw.text((document.width//2 + 30, 260), f"City, State {random.randint(10000, 99999)}", fill=(0, 0, 0), font=regular_font)
    
    # Invoice items
    y_pos = 340
    draw_box(draw, 50, y_pos, document.width - 50, y_pos + 300, "")
    
    # Headers
    draw.text((60, y_pos + 20), "Description", fill=(0, 0, 0), font=regular_font)
    draw.text((450, y_pos + 20), "Quantity", fill=(0, 0, 0), font=regular_font)
    draw.text((550, y_pos + 20), "Unit Price", fill=(0, 0, 0), font=regular_font)
    draw.text((650, y_pos + 20), "Amount", fill=(0, 0, 0), font=regular_font)
    
    # Line
    draw.line([(60, y_pos + 40), (document.width - 60, y_pos + 40)], fill=(0, 0, 0), width=1)
    
    # Generate items
    service_items = [
        "Professional Services",
        "Consulting Fee",
        "Web Development",
        "Software License",
        "Design Services",
        "Technical Support",
        "Project Management",
        "Marketing Services",
        "Training Services",
        "Research Fee",
        "Equipment Rental",
        "Installation Services"
    ]
    
    num_items = random.randint(3, 6)
    y_offset = y_pos + 60
    subtotal = 0
    
    for i in range(num_items):
        if service_items:
            service = random.choice(service_items)
            service_items.remove(service)
        else:
            service = f"Service Item {i+1}"
        
        quantity = random.randint(1, 10)
        unit_price = random.randint(75, 500) if "Professional" in service or "Consulting" in service else random.randint(20, 200)
        amount = quantity * unit_price
        subtotal += amount
        
        draw.text((60, y_offset), service, fill=(0, 0, 0), font=regular_font)
        draw.text((450, y_offset), str(quantity), fill=(0, 0, 0), font=regular_font)
        draw.text((550, y_offset), f"${unit_price:,.2f}", fill=(0, 0, 0), font=regular_font)
        draw.text((650, y_offset), f"${amount:,.2f}", fill=(0, 0, 0), font=regular_font)
        
        y_offset += 30
    
    # Bottom line
    draw.line([(60, y_offset + 10), (document.width - 60, y_offset + 10)], fill=(0, 0, 0), width=1)
    
    # Totals
    tax_rate = random.uniform(0.05, 0.09)
    tax_amount = round(subtotal * tax_rate, 2)
    total = subtotal + tax_amount
    
    draw.text((450, y_offset + 30), "Subtotal:", fill=(0, 0, 0), font=regular_font)
    draw.text((650, y_offset + 30), f"${subtotal:,.2f}", fill=(0, 0, 0), font=regular_font)
    
    draw.text((450, y_offset + 60), f"Tax ({tax_rate:.1%}):", fill=(0, 0, 0), font=regular_font)
    draw.text((650, y_offset + 60), f"${tax_amount:,.2f}", fill=(0, 0, 0), font=regular_font)
    
    draw.text((450, y_offset + 90), "Total:", fill=(0, 0, 0), font=title_font)
    draw.text((650, y_offset + 90), f"${total:,.2f}", fill=(0, 0, 0), font=title_font)
    
    # Payment terms
    draw.text((60, y_offset + 130), "Payment Terms: Net 30", fill=(0, 0, 0), font=regular_font)
    
    # Payment methods 
    draw.text((60, y_offset + 160), "Payment Methods: Check, Credit Card, Bank Transfer", fill=(0, 0, 0), font=regular_font)
    
    # Thank you note
    draw.text((document.width//2, document.height - 60), 
              f"Thank you for your business!", 
              fill=(0, 0, 0), font=small_font, anchor="mt")

def create_notice_of_assessment(document, draw, header_font, title_font, regular_font, small_font):
    """Create an Australian ATO Notice of Assessment"""
    # Title under the ATO header
    draw.text((document.width//2, 80), "Notice of Assessment", fill=(0, 0, 0), font=header_font, anchor="mt")
    
    # Financial year (Australia uses July 1 to June 30 fiscal year)
    year_end = random.randint(2020, 2023)
    year_start = year_end - 1
    draw.text((document.width//2, 120), f"Income year: 1 July {year_start} to 30 June {year_end}", 
              fill=(0, 0, 0), font=title_font, anchor="mt")
    
    # Notice info
    draw.text((document.width//2, 160), f"Date of issue: {random.randint(1, 28)} {random.choice(['July', 'August', 'September', 'October'])} {year_end}", 
              fill=(0, 0, 0), font=regular_font, anchor="mt")
    draw.text((document.width//2, 190), f"Notice reference: {random.randint(10000000, 99999999)}", 
              fill=(0, 0, 0), font=regular_font, anchor="mt")
    
    # Taxpayer details
    draw_box(draw, 50, 220, document.width - 50, 320, "Your details")
    
    taxpayer_first = random.choice(["John", "Mary", "Robert", "Patricia", "Michael", "Jennifer", "David", "Sarah"])
    taxpayer_last = random.choice(["Smith", "Johnson", "Williams", "Brown", "Jones", "Wilson", "Taylor", "Anderson"])
    
    # Anonymize taxpayer name and TFN
    anon_first = anonymize_name(taxpayer_first)
    anon_last = anonymize_name(taxpayer_last)
    draw.text((60, 250), f"Name: {anon_first} {anon_last}", fill=(0, 0, 0), font=regular_font)
    
    # Generate and anonymize TFN
    tfn = f"{random.randint(100, 999)} {random.randint(100, 999)} {random.randint(100, 999)}"
    anon_tfn = anonymize_tfn(tfn)
    draw.text((60, 280), f"TFN: {anon_tfn}", fill=(0, 0, 0), font=regular_font)
    
    # Assessment summary
    y_pos = 340
    draw_box(draw, 50, y_pos, document.width - 50, y_pos + 300, "Assessment summary")
    
    # Generate income details
    taxable_income = random.randint(50000, 120000)
    tax_on_income = round(taxable_income * random.uniform(0.19, 0.37), 2)  # Australian tax rates
    medicare_levy = round(taxable_income * 0.02, 2)  # 2% Medicare levy
    tax_withheld = round(tax_on_income * random.uniform(0.9, 1.1), 2)  # Approximate tax withheld
    
    # Calculate refund or amount owing
    tax_payable = tax_on_income + medicare_levy
    difference = tax_payable - tax_withheld
    is_refund = difference < 0
    
    items = [
        ("Taxable income", f"${taxable_income:.2f}"),
        ("Tax on income", f"${tax_on_income:.2f}"),
        ("Medicare levy", f"${medicare_levy:.2f}"),
        ("Total tax payable", f"${tax_payable:.2f}"),
        ("PAYG tax withheld", f"${tax_withheld:.2f}"),
        ("Refund amount" if is_refund else "Amount owing", f"${abs(difference):.2f}")
    ]
    
    # Draw data
    y_offset = y_pos + 30
    for i, (label, value) in enumerate(items):
        draw.text((60, y_offset + i * 40), label, fill=(0, 0, 0), font=regular_font)
        
        # Highlight the important values (bold if possible)
        font_to_use = title_font if i == 5 else regular_font
        color = (0, 100, 0) if i == 5 and is_refund else ((200, 0, 0) if i == 5 else (0, 0, 0))
        
        # Align values to the right
        value_width = font_to_use.getsize(value)[0] if hasattr(font_to_use, "getsize") else 100
        draw.text((document.width - 100 - value_width, y_offset + i * 40), value, fill=color, font=font_to_use)
    
    # Footer
    draw.text((document.width//2, document.height - 80), 
              "This notice shows the Australian Taxation Office's assessment of your income tax return.", 
              fill=(0, 0, 0), font=small_font, anchor="mt")
    draw.text((document.width//2, document.height - 50), 
              "Please retain this for your records.", 
              fill=(0, 0, 0), font=small_font, anchor="mt")

def create_tax_return_summary(document, draw, header_font, title_font, regular_font, small_font):
    """Create an Australian Tax Return Summary"""
    # Title under the ATO header
    draw.text((document.width//2, 80), "Tax Return Summary", fill=(0, 0, 0), font=header_font, anchor="mt")
    
    # Financial year (Australia uses July 1 to June 30 fiscal year)
    year_end = random.randint(2020, 2023)
    year_start = year_end - 1
    draw.text((document.width//2, 120), f"For the financial year: 1 July {year_start} to 30 June {year_end}", 
              fill=(0, 0, 0), font=title_font, anchor="mt")
    
    # Taxpayer details
    draw_box(draw, 50, 160, document.width - 50, 260, "Taxpayer Information")
    
    taxpayer_first = random.choice(["John", "Mary", "Robert", "Patricia", "Michael", "Jennifer", "David", "Sarah"])
    taxpayer_last = random.choice(["Smith", "Johnson", "Williams", "Brown", "Jones", "Wilson", "Taylor", "Anderson"])
    
    # Anonymize taxpayer name
    anon_first = anonymize_name(taxpayer_first)
    anon_last = anonymize_name(taxpayer_last)
    draw.text((60, 190), f"Name: {anon_first} {anon_last}", fill=(0, 0, 0), font=regular_font)
    # Generate and anonymize TFN
    tfn = f"{random.randint(100, 999)} {random.randint(100, 999)} {random.randint(100, 999)}"
    anon_tfn = anonymize_tfn(tfn)
    draw.text((60, 220), f"TFN: {anon_tfn}", fill=(0, 0, 0), font=regular_font)
    
    # Summary of income and deductions
    y_pos = 280
    draw_box(draw, 50, y_pos, document.width - 50, y_pos + 400, "Income and Deductions")
    
    # Generate random income and deduction items
    salary = random.randint(50000, 120000)
    interest = random.randint(100, 3000)
    dividends = random.randint(0, 5000)
    other_income = random.randint(0, 10000)
    total_income = salary + interest + dividends + other_income
    
    work_deductions = random.randint(500, 3000)
    donations = random.randint(0, 1000)
    other_deductions = random.randint(0, 2000)
    total_deductions = work_deductions + donations + other_deductions
    
    taxable_income = total_income - total_deductions
    
    # Income items
    income_items = [
        ("Salary or wages", f"${salary:,}"),
        ("Interest", f"${interest:,}"),
        ("Dividends", f"${dividends:,}"),
        ("Other income", f"${other_income:,}"),
        ("Total income", f"${total_income:,}")
    ]
    
    # Draw income data
    y_offset = y_pos + 30
    draw.text((60, y_offset), "Income", fill=(0, 0, 0), font=title_font)
    y_offset += 30
    
    for i, (label, value) in enumerate(income_items):
        draw.text((90, y_offset + i * 30), label, fill=(0, 0, 0), font=regular_font)
        
        # Highlight the total
        font_to_use = title_font if "Total" in label else regular_font
        
        # Align values to the right
        value_width = font_to_use.getsize(value)[0] if hasattr(font_to_use, "getsize") else 100
        draw.text((document.width - 100 - value_width, y_offset + i * 30), value, fill=(0, 0, 0), font=font_to_use)
    
    # Deduction items
    deduction_items = [
        ("Work-related expenses", f"${work_deductions:,}"),
        ("Gifts and donations", f"${donations:,}"),
        ("Other deductions", f"${other_deductions:,}"),
        ("Total deductions", f"${total_deductions:,}")
    ]
    
    # Draw deduction data
    y_offset += 180
    draw.text((60, y_offset), "Deductions", fill=(0, 0, 0), font=title_font)
    y_offset += 30
    
    for i, (label, value) in enumerate(deduction_items):
        draw.text((90, y_offset + i * 30), label, fill=(0, 0, 0), font=regular_font)
        
        # Highlight the total
        font_to_use = title_font if "Total" in label else regular_font
        
        # Align values to the right
        value_width = font_to_use.getsize(value)[0] if hasattr(font_to_use, "getsize") else 100
        draw.text((document.width - 100 - value_width, y_offset + i * 30), value, fill=(0, 0, 0), font=font_to_use)
    
    # Taxable income
    y_offset += 120
    draw.text((60, y_offset), "Taxable income", fill=(0, 0, 0), font=title_font)
    
    # Align value to the right
    value = f"${taxable_income:,}"
    value_width = title_font.getsize(value)[0] if hasattr(title_font, "getsize") else 100
    draw.text((document.width - 100 - value_width, y_offset), value, fill=(0, 0, 0), font=title_font)
    
    # Footer
    draw.text((document.width//2, document.height - 50), 
              "This is a summary of your tax return information. For a complete assessment, refer to your Notice of Assessment.", 
              fill=(0, 0, 0), font=small_font, anchor="mt")

def create_private_health_statement(document, draw, header_font, title_font, regular_font, small_font):
    """Create an Australian Private Health Insurance Statement"""
    # Title under the ATO header
    draw.text((document.width//2, 80), "Private Health Insurance Statement", fill=(0, 0, 0), font=header_font, anchor="mt")
    
    # Financial year (Australia uses July 1 to June 30 fiscal year)
    year_end = random.randint(2020, 2023)
    year_start = year_end - 1
    draw.text((document.width//2, 120), f"For the period 1 July {year_start} to 30 June {year_end}", 
              fill=(0, 0, 0), font=title_font, anchor="mt")
    
    # Health fund details
    draw_box(draw, 50, 160, document.width//2 - 20, 260, "Health Fund Details")
    
    health_fund = f"{random.choice(['Medibank', 'Bupa', 'HCF', 'NIB', 'HBF'])} {random.choice(['Private', 'Health', 'Insurance'])}"
    membership_number = ''.join(random.choices(string.digits, k=10))
    
    draw.text((60, 190), f"Health Fund: {health_fund}", fill=(0, 0, 0), font=regular_font)
    draw.text((60, 220), f"Membership No: {membership_number}", fill=(0, 0, 0), font=regular_font)
    
    # Policyholder details
    draw_box(draw, document.width//2 + 20, 160, document.width - 50, 260, "Policyholder Details")
    
    holder_first = random.choice(["John", "Mary", "Robert", "Patricia", "Michael", "Jennifer", "David", "Sarah"])
    holder_last = random.choice(["Smith", "Johnson", "Williams", "Brown", "Jones", "Wilson", "Taylor", "Anderson"])
    
    # Anonymize policyholder name
    anon_first = anonymize_name(holder_first)
    anon_last = anonymize_name(holder_last)
    draw.text((document.width//2 + 30, 190), f"Name: {anon_first} {anon_last}", fill=(0, 0, 0), font=regular_font)
    draw.text((document.width//2 + 30, 220), f"Date of Birth: {random.randint(1, 28)}/{random.randint(1, 12)}/{random.randint(1950, 2000)}", fill=(0, 0, 0), font=regular_font)
    
    # Insurance details
    y_pos = 280
    draw_box(draw, 50, y_pos, document.width - 50, y_pos + 350, "Private Health Insurance Details")
    
    # Generate values
    premium_paid = random.randint(2000, 6000)
    australian_govt_rebate = round(premium_paid * random.uniform(0.25, 0.33), 2)
    benefit_code = random.choice(["30", "35", "40"])  # Different rebate tiers
    
    y_offset = y_pos + 30
    
    draw.text((60, y_offset), "Type of cover:", fill=(0, 0, 0), font=regular_font)
    draw.text((250, y_offset), random.choice(["Hospital", "Hospital and Extras", "Comprehensive"]), fill=(0, 0, 0), font=regular_font)
    y_offset += 40
    
    draw.text((60, y_offset), "Period of cover:", fill=(0, 0, 0), font=regular_font)
    draw.text((250, y_offset), f"01/07/{year_start} to 30/06/{year_end}", fill=(0, 0, 0), font=regular_font)
    y_offset += 40
    
    draw.text((60, y_offset), "Benefit Code:", fill=(0, 0, 0), font=regular_font)
    draw.text((250, y_offset), benefit_code, fill=(0, 0, 0), font=regular_font)
    y_offset += 40
    
    draw.text((60, y_offset), "Your premiums paid:", fill=(0, 0, 0), font=regular_font)
    draw.text((250, y_offset), f"${premium_paid:.2f}", fill=(0, 0, 0), font=regular_font)
    y_offset += 40
    
    draw.text((60, y_offset), "Australian Government Rebate received:", fill=(0, 0, 0), font=regular_font)
    draw.text((350, y_offset), f"${australian_govt_rebate:.2f}", fill=(0, 0, 0), font=regular_font)
    y_offset += 40
    
    draw.text((60, y_offset), "Number of adults covered:", fill=(0, 0, 0), font=regular_font)
    draw.text((250, y_offset), str(random.choice([1, 2])), fill=(0, 0, 0), font=regular_font)
    y_offset += 40
    
    num_dependents = random.randint(0, 3)
    draw.text((60, y_offset), "Number of dependents covered:", fill=(0, 0, 0), font=regular_font)
    draw.text((250, y_offset), str(num_dependents), fill=(0, 0, 0), font=regular_font)
    
    # Footer
    draw.text((document.width//2, document.height - 80), 
              "This statement is required by the Australian Taxation Office for tax return purposes.", 
              fill=(0, 0, 0), font=small_font, anchor="mt")
    draw.text((document.width//2, document.height - 50), 
              "Please retain this statement with your taxation records.", 
              fill=(0, 0, 0), font=small_font, anchor="mt")

def create_medicare_statement(document, draw, header_font, title_font, regular_font, small_font):
    """Create an Australian Medicare Statement"""
    # Title under the ATO header
    draw.text((document.width//2, 80), "Medicare Levy Exemption Certificate", fill=(0, 0, 0), font=header_font, anchor="mt")
    
    # Financial year
    year_end = random.randint(2020, 2023)
    year_start = year_end - 1
    draw.text((document.width//2, 120), f"For the financial year ending 30 June {year_end}", 
              fill=(0, 0, 0), font=title_font, anchor="mt")
    
    # Taxpayer details
    draw_box(draw, 50, 160, document.width - 50, 260, "Taxpayer Details")
    
    taxpayer_first = random.choice(["John", "Mary", "Robert", "Patricia", "Michael", "Jennifer", "David", "Sarah"])
    taxpayer_last = random.choice(["Smith", "Johnson", "Williams", "Brown", "Jones", "Wilson", "Taylor", "Anderson"])
    
    # Anonymize taxpayer name
    anon_first = anonymize_name(taxpayer_first)
    anon_last = anonymize_name(taxpayer_last)
    draw.text((60, 190), f"Name: {anon_first} {anon_last}", fill=(0, 0, 0), font=regular_font)
    # Generate and anonymize TFN
    tfn = f"{random.randint(100, 999)} {random.randint(100, 999)} {random.randint(100, 999)}"
    anon_tfn = anonymize_tfn(tfn)
    draw.text((60, 220), f"TFN: {anon_tfn}", fill=(0, 0, 0), font=regular_font)
    
    # Exemption details
    y_pos = 280
    draw_box(draw, 50, y_pos, document.width - 50, y_pos + 300, "Exemption Details")
    
    # Generate exemption details
    exemption_types = [
        "Full Medicare Levy Exemption - Category 1",
        "Full Medicare Levy Exemption - Category 2",
        "Half Medicare Levy Exemption"
    ]
    
    exemption_type = random.choice(exemption_types)
    certificate_number = f"M{random.randint(100000, 999999)}"
    
    y_offset = y_pos + 30
    
    draw.text((60, y_offset), "Exemption Type:", fill=(0, 0, 0), font=regular_font)
    draw.text((200, y_offset), exemption_type, fill=(0, 0, 0), font=regular_font)
    y_offset += 40
    
    draw.text((60, y_offset), "Certificate Number:", fill=(0, 0, 0), font=regular_font)
    draw.text((200, y_offset), certificate_number, fill=(0, 0, 0), font=regular_font)
    y_offset += 40
    
    draw.text((60, y_offset), "Valid Period:", fill=(0, 0, 0), font=regular_font)
    draw.text((200, y_offset), f"01/07/{year_start} to 30/06/{year_end}", fill=(0, 0, 0), font=regular_font)
    y_offset += 70
    
    # Certification
    draw.text((60, y_offset), "This certificate confirms that the taxpayer named above is:", fill=(0, 0, 0), font=regular_font)
    y_offset += 30
    
    if "Category 1" in exemption_type:
        reason = "Not entitled to Medicare benefits under the Health Insurance Act 1973"
    elif "Category 2" in exemption_type:
        reason = "A member of a diplomatic mission or consular post in Australia"
    else:
        reason = "Entitled to a reduction of Medicare levy under special circumstances"
    
    draw.text((80, y_offset), reason, fill=(0, 0, 0), font=regular_font)
    
    # Signature section
    y_offset += 80
    
    draw.text((60, y_offset), "Authorized by:", fill=(0, 0, 0), font=regular_font)
    draw.text((200, y_offset), "Services Australia", fill=(0, 0, 0), font=regular_font)
    
    # Footer
    draw.text((document.width//2, document.height - 50), 
              "This certificate must be retained for taxation purposes.", 
              fill=(0, 0, 0), font=small_font, anchor="mt")

def create_superannuation_statement(document, draw, header_font, title_font, regular_font, small_font):
    """Create an Australian Superannuation Statement"""
    # Title under the ATO header
    draw.text((document.width//2, 80), "Superannuation Statement", fill=(0, 0, 0), font=header_font, anchor="mt")
    
    # Financial year
    year_end = random.randint(2020, 2023)
    year_start = year_end - 1
    draw.text((document.width//2, 120), f"For the period 1 July {year_start} to 30 June {year_end}", 
              fill=(0, 0, 0), font=title_font, anchor="mt")
    
    # Fund details
    draw_box(draw, 50, 160, document.width//2 - 20, 260, "Superannuation Fund Details")
    
    super_fund = f"{random.choice(['Australian Super', 'Aware Super', 'UniSuper', 'QSuper', 'Hostplus', 'Cbus'])}"
    
    draw.text((60, 190), f"Fund: {super_fund}", fill=(0, 0, 0), font=regular_font)
    draw.text((60, 220), f"ABN: {random.randint(10, 99)} {random.randint(100, 999)} {random.randint(100, 999)} {random.randint(100, 999)}", fill=(0, 0, 0), font=regular_font)
    
    # Member details
    draw_box(draw, document.width//2 + 20, 160, document.width - 50, 260, "Member Details")
    
    member_first = random.choice(["John", "Mary", "Robert", "Patricia", "Michael", "Jennifer", "David", "Sarah"])
    member_last = random.choice(["Smith", "Johnson", "Williams", "Brown", "Jones", "Wilson", "Taylor", "Anderson"])
    
    member_number = ''.join(random.choices(string.digits, k=8))
    
    # Anonymize member name
    anon_first = anonymize_name(member_first)
    anon_last = anonymize_name(member_last)
    draw.text((document.width//2 + 30, 190), f"Name: {anon_first} {anon_last}", fill=(0, 0, 0), font=regular_font)
    draw.text((document.width//2 + 30, 220), f"Member No: {member_number}", fill=(0, 0, 0), font=regular_font)
    
    # Account summary
    y_pos = 280
    draw_box(draw, 50, y_pos, document.width - 50, y_pos + 350, "Account Summary")
    
    # Generate values
    opening_balance = random.randint(50000, 300000)
    employer_contributions = round(opening_balance * random.uniform(0.08, 0.12), 2)
    investment_earnings = round(opening_balance * random.uniform(-0.05, 0.15), 2)
    fees = round(opening_balance * random.uniform(0.005, 0.015), 2)
    closing_balance = opening_balance + employer_contributions + investment_earnings - fees
    
    y_offset = y_pos + 30
    
    items = [
        ("Opening Balance:", f"${opening_balance:,.2f}"),
        ("Employer Contributions:", f"${employer_contributions:,.2f}"),
        ("Investment Earnings:", f"${investment_earnings:,.2f}"),
        ("Fees and Charges:", f"-${fees:,.2f}"),
        ("Closing Balance:", f"${closing_balance:,.2f}")
    ]
    
    for i, (label, value) in enumerate(items):
        draw.text((60, y_offset + i * 40), label, fill=(0, 0, 0), font=regular_font)
        
        # Determine color - red for negative values
        color = (200, 0, 0) if "Investment" in label and investment_earnings < 0 else (0, 0, 0)
        font_to_use = title_font if i == 4 else regular_font  # Bold for closing balance
        
        # Align values to the right
        value_width = font_to_use.getsize(value)[0] if hasattr(font_to_use, "getsize") else 100
        draw.text((document.width - 100 - value_width, y_offset + i * 40), value, fill=color, font=font_to_use)
    
    # Investment details
    y_offset += 220
    draw.text((60, y_offset), "Investment Option:", fill=(0, 0, 0), font=regular_font)
    draw.text((220, y_offset), random.choice(["Balanced", "Growth", "Conservative", "High Growth"]), fill=(0, 0, 0), font=regular_font)
    
    # Footer
    draw.text((document.width//2, document.height - 50), 
              "This information is provided for your personal records and tax purposes.", 
              fill=(0, 0, 0), font=small_font, anchor="mt")

def create_tax_invoice(document, draw, header_font, title_font, regular_font, small_font):
    """Create an Australian Tax Invoice"""
    # Title under the ATO header
    draw.text((document.width//2, 80), "Tax Invoice", fill=(0, 0, 0), font=header_font, anchor="mt")
    
    # Invoice details
    invoice_number = f"INV-{random.randint(10000, 99999)}"
    
    # Random date from the last year
    year = random.randint(2020, 2023)
    month = random.randint(1, 12)
    day = random.randint(1, 28)
    invoice_date = f"{day:02d}/{month:02d}/{year}"
    
    draw.text((document.width//2, 120), f"Invoice: {invoice_number}", fill=(0, 0, 0), font=title_font, anchor="mt")
    draw.text((document.width//2, 150), f"Date: {invoice_date}", fill=(0, 0, 0), font=regular_font, anchor="mt")
    
    # Business details
    draw_box(draw, 50, 180, document.width//2 - 20, 280, "Business Details")
    
    business_name = f"{random.choice(['Australian', 'Sydney', 'Melbourne', 'Brisbane', 'Perth'])} {random.choice(['Services', 'Solutions', 'Consulting', 'Technologies'])}"
    
    draw.text((60, 210), business_name, fill=(0, 0, 0), font=regular_font)
    draw.text((60, 240), f"ABN: {random.randint(10, 99)} {random.randint(100, 999)} {random.randint(100, 999)} {random.randint(100, 999)}", fill=(0, 0, 0), font=regular_font)
    
    # Customer details
    draw_box(draw, document.width//2 + 20, 180, document.width - 50, 280, "Customer Details")
    
    customer_first = random.choice(["John", "Mary", "Robert", "Patricia", "Michael", "Jennifer", "David", "Sarah"])
    customer_last = random.choice(["Smith", "Johnson", "Williams", "Brown", "Jones", "Wilson", "Taylor", "Anderson"])
    
    draw.text((document.width//2 + 30, 210), f"{customer_first} {customer_last}", fill=(0, 0, 0), font=regular_font)
    draw.text((document.width//2 + 30, 240), f"Customer ID: {random.randint(10000, 99999)}", fill=(0, 0, 0), font=regular_font)
    
    # Invoice items
    y_pos = 300
    draw_box(draw, 50, y_pos, document.width - 50, y_pos + 350, "")
    
    # Table headers
    draw.text((60, y_pos + 20), "Description", fill=(0, 0, 0), font=regular_font)
    draw.text((380, y_pos + 20), "Quantity", fill=(0, 0, 0), font=regular_font)
    draw.text((450, y_pos + 20), "Unit Price", fill=(0, 0, 0), font=regular_font)
    draw.text((520, y_pos + 20), "Amount", fill=(0, 0, 0), font=regular_font)
    
    # Horizontal line
    draw.line([(60, y_pos + 40), (document.width - 60, y_pos + 40)], fill=(0, 0, 0), width=1)
    
    # Generate random items
    items = []
    subtotal = 0
    
    y_offset = y_pos + 60
    for i in range(random.randint(3, 6)):
        description = random.choice([
            "Professional Services", "Consulting Fee", "Software License",
            "Technical Support", "Equipment Rental", "Training Services",
            "Maintenance Fee", "Project Management", "Installation Services"
        ])
        
        quantity = random.randint(1, 5)
        unit_price = round(random.uniform(100, 500), 2)
        amount = quantity * unit_price
        subtotal += amount
        
        draw.text((60, y_offset), description, fill=(0, 0, 0), font=regular_font)
        draw.text((380, y_offset), str(quantity), fill=(0, 0, 0), font=regular_font)
        draw.text((450, y_offset), f"${unit_price:.2f}", fill=(0, 0, 0), font=regular_font)
        draw.text((520, y_offset), f"${amount:.2f}", fill=(0, 0, 0), font=regular_font)
        
        y_offset += 40
    
    # Horizontal line
    draw.line([(60, y_offset + 10), (document.width - 60, y_offset + 10)], fill=(0, 0, 0), width=1)
    
    # Calculate GST and total
    gst = round(subtotal * 0.1, 2)  # 10% GST in Australia
    total = subtotal + gst
    
    # Totals
    y_offset += 30
    draw.text((380, y_offset), "Subtotal:", fill=(0, 0, 0), font=regular_font)
    draw.text((520, y_offset), f"${subtotal:.2f}", fill=(0, 0, 0), font=regular_font)
    
    y_offset += 30
    draw.text((380, y_offset), "GST (10%):", fill=(0, 0, 0), font=regular_font)
    draw.text((520, y_offset), f"${gst:.2f}", fill=(0, 0, 0), font=regular_font)
    
    y_offset += 30
    draw.text((380, y_offset), "Total:", fill=(0, 0, 0), font=title_font)
    draw.text((520, y_offset), f"${total:.2f}", fill=(0, 0, 0), font=title_font)
    
    # Payment details
    y_offset += 60
    draw.text((60, y_offset), "Payment Details:", fill=(0, 0, 0), font=regular_font)
    draw.text((60, y_offset + 30), f"BSB: {random.randint(100, 999)}-{random.randint(100, 999)}", fill=(0, 0, 0), font=regular_font)
    draw.text((60, y_offset + 60), f"Account: {random.randint(10000000, 99999999)}", fill=(0, 0, 0), font=regular_font)
    
    # Footer
    draw.text((document.width//2, document.height - 50), 
              "This is a tax invoice for GST purposes.", 
              fill=(0, 0, 0), font=small_font, anchor="mt")

def create_business_activity_statement(document, draw, header_font, title_font, regular_font, small_font):
    """Create an Australian Business Activity Statement (BAS)"""
    # Title under the ATO header
    draw.text((document.width//2, 80), "Business Activity Statement", fill=(0, 0, 0), font=header_font, anchor="mt")
    
    # Period
    quarters = ["January - March", "April - June", "July - September", "October - December"]
    quarter = random.choice(quarters)
    year = random.randint(2020, 2023)
    
    draw.text((document.width//2, 120), f"For the period: {quarter} {year}", fill=(0, 0, 0), font=title_font, anchor="mt")
    
    # Business details
    draw_box(draw, 50, 160, document.width - 50, 260, "Business Details")
    
    business_name = f"{random.choice(['Australian', 'National', 'Pacific', 'Western', 'Eastern'])} {random.choice(['Trading', 'Enterprises', 'Holdings', 'Services'])} Pty Ltd"
    
    draw.text((60, 190), f"Business: {business_name}", fill=(0, 0, 0), font=regular_font)
    draw.text((60, 220), f"ABN: {random.randint(10, 99)} {random.randint(100, 999)} {random.randint(100, 999)} {random.randint(100, 999)}", fill=(0, 0, 0), font=regular_font)
    
    # BAS details
    y_pos = 280
    draw_box(draw, 50, y_pos, document.width - 50, y_pos + 400, "GST and PAYG Summary")
    
    # Generate random values
    sales = random.randint(20000, 500000)
    gst_collected = round(sales * 0.1, 2)  # 10% GST
    purchases = random.randint(int(sales * 0.3), int(sales * 0.7))
    gst_credits = round(purchases * 0.1, 2)
    net_gst = gst_collected - gst_credits
    
    payg_withholding = random.randint(5000, 50000)
    payg_installments = random.randint(1000, 20000)
    
    total_obligation = net_gst + payg_withholding + payg_installments
    
    # Draw GST section
    y_offset = y_pos + 30
    draw.text((60, y_offset), "G1. Total sales (including GST):", fill=(0, 0, 0), font=regular_font)
    draw.text((400, y_offset), f"${sales:,}", fill=(0, 0, 0), font=regular_font)
    
    y_offset += 40
    draw.text((60, y_offset), "G3. GST on sales:", fill=(0, 0, 0), font=regular_font)
    draw.text((400, y_offset), f"${gst_collected:,.2f}", fill=(0, 0, 0), font=regular_font)
    
    y_offset += 40
    draw.text((60, y_offset), "G10. Purchases (including GST):", fill=(0, 0, 0), font=regular_font)
    draw.text((400, y_offset), f"${purchases:,}", fill=(0, 0, 0), font=regular_font)
    
    y_offset += 40
    draw.text((60, y_offset), "G11. GST on purchases:", fill=(0, 0, 0), font=regular_font)
    draw.text((400, y_offset), f"${gst_credits:,.2f}", fill=(0, 0, 0), font=regular_font)
    
    # Draw a divider
    y_offset += 50
    draw.line([(60, y_offset), (document.width - 60, y_offset)], fill=(0, 0, 0), width=1)
    
    # Net GST
    y_offset += 20
    draw.text((60, y_offset), "GST Summary:", fill=(0, 0, 0), font=title_font)
    y_offset += 40
    
    draw.text((60, y_offset), "Net GST:", fill=(0, 0, 0), font=regular_font)
    
    # Highlight the net GST amount
    net_gst_str = f"${net_gst:,.2f}"
    value_width = regular_font.getsize(net_gst_str)[0] if hasattr(regular_font, "getsize") else 100
    draw.text((400, y_offset), net_gst_str, fill=(0, 0, 0), font=regular_font)
    
    # PAYG section
    y_offset += 50
    draw.text((60, y_offset), "PAYG Summary:", fill=(0, 0, 0), font=title_font)
    
    y_offset += 40
    draw.text((60, y_offset), "PAYG withholding:", fill=(0, 0, 0), font=regular_font)
    draw.text((400, y_offset), f"${payg_withholding:,.2f}", fill=(0, 0, 0), font=regular_font)
    
    y_offset += 40
    draw.text((60, y_offset), "PAYG installments:", fill=(0, 0, 0), font=regular_font)
    draw.text((400, y_offset), f"${payg_installments:,.2f}", fill=(0, 0, 0), font=regular_font)
    
    # Total amount payable/refundable
    y_offset += 50
    is_refund = total_obligation < 0
    status = "refundable" if is_refund else "payable"
    
    draw.text((60, y_offset), f"Total amount {status}:", fill=(0, 0, 0), font=title_font)
    
    # Highlight the total amount
    total_str = f"${abs(total_obligation):,.2f}"
    value_width = title_font.getsize(total_str)[0] if hasattr(title_font, "getsize") else 100
    color = (0, 100, 0) if is_refund else ((200, 0, 0) if total_obligation > 0 else (0, 0, 0))
    draw.text((400, y_offset), total_str, fill=color, font=title_font)
    
    # Footer
    draw.text((document.width//2, document.height - 50), 
              "This statement is for taxation purposes only.", 
              fill=(0, 0, 0), font=small_font, anchor="mt")

def create_generic_document(document, draw, header_font, title_font, regular_font, small_font, doc_type):
    """Create a generic Australian tax-related document"""
    # Document header is already added with ATO banner
    
    # Title under the ATO header
    draw.text((document.width//2, 80), doc_type, fill=(0, 0, 0), font=header_font, anchor="mt")
    
    # Financial year (Australia uses July 1 to June 30 fiscal year)
    year_end = random.randint(2020, 2023)
    year_start = year_end - 1
    draw.text((document.width//2, 120), f"Financial year: {year_start}-{year_end}", 
              fill=(0, 0, 0), font=title_font, anchor="mt")
    
    # Document details
    y_pos = 160
    draw_box(draw, 50, y_pos, document.width - 50, y_pos + 200, "Document Information")
    
    document_id = ''.join(random.choices(string.ascii_uppercase, k=2)) + ''.join(random.choices(string.digits, k=6))
    
    month = random.randint(1, 12)
    day = random.randint(1, 28)
    document_date = f"{day:02d}/{month:02d}/{year_end}"
    
    person_first = random.choice(["John", "Mary", "Robert", "Patricia", "Michael", "Jennifer"])
    person_last = random.choice(["Smith", "Johnson", "Williams", "Brown", "Jones", "Garcia"])
    
    draw.text((60, y_pos + 30), f"Document ID: {document_id}", fill=(0, 0, 0), font=regular_font)
    draw.text((60, y_pos + 60), f"Date: {document_date}", fill=(0, 0, 0), font=regular_font)
    draw.text((60, y_pos + 90), f"Tax Identification: XXX-XX-{random.randint(1000, 9999)}", fill=(0, 0, 0), font=regular_font)
    draw.text((60, y_pos + 120), f"Recipient: {person_first} {person_last}", fill=(0, 0, 0), font=regular_font)
    
    # Summary section
    y_pos = 380
    draw_box(draw, 50, y_pos, document.width - 50, y_pos + 300, "Summary")
    
    # Generate a few random items with dollar amounts
    items = []
    total = 0
    
    for i in range(random.randint(3, 6)):
        item_name = f"Item {i+1}: {random.choice(['Payment', 'Contribution', 'Distribution', 'Expense', 'Fee', 'Credit', 'Deduction'])}"
        amount = random.randint(100, 5000) if i < 2 else random.randint(10, 500)
        total += amount
        items.append((item_name, f"${amount:,.2f}"))
    
    items.append(("Total", f"${total:,.2f}"))
    
    # Draw items
    y_offset = y_pos + 30
    for i, (label, value) in enumerate(items):
        font = title_font if "Total" in label else regular_font
        draw.text((60, y_offset + i * 40), label, fill=(0, 0, 0), font=font)
        
        # Align values to the right
        value_width = font.getsize(value)[0] if hasattr(font, "getsize") else 150
        draw.text((document.width - 100 - value_width, y_offset + i * 40), value, fill=(0, 0, 0), font=font)
    
    # Footer
    draw.text((document.width//2, document.height - 80), 
              "This document contains important tax information and is being", 
              fill=(0, 0, 0), font=small_font, anchor="mt")
    draw.text((document.width//2, document.height - 60), 
              "furnished to the Internal Revenue Service.", 
              fill=(0, 0, 0), font=small_font, anchor="mt")

def draw_box(draw, x1, y1, x2, y2, title=None):
    """Helper function to draw a box with an optional title"""
    # Draw the box
    draw.rectangle([(x1, y1), (x2, y2)], outline=(0, 0, 0), width=1)
    
    # Draw title if provided
    if title:
        draw.text((x1 + 10, y1 + 10), title, fill=(0, 0, 0), font=ImageFont.load_default())

def create_tax_document_samples(num_samples=10, output_dir="synthetic_receipts/tax_docs"):
    """
    Create a variety of Australian tax-related document samples (not receipts).
    
    Args:
        num_samples: Number of document samples to generate
        output_dir: Directory to save the samples
    """
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Get existing samples
    existing_files = [f for f in os.listdir(output_dir) if f.startswith("taxdoc_")]
    start_num = len(existing_files) + 1
    
    print(f"Generating {num_samples} Australian tax document samples...")
    
    # Document types for variety - Australian context
    doc_types = [
        "PAYG Payment Summary", "ATO Notice of Assessment", 
        "Tax Return Summary", "Income Statement",
        "Private Health Insurance Statement", "Medicare Levy Exemption Certificate", 
        "Superannuation Statement", "Tax Invoice",
        "Capital Gains Tax Statement", "Investment Income Summary",
        "Rental Property Statement", "Business Activity Statement (BAS)"
    ]
    
    # Create varied document samples - all in portrait A4 format
    for i in range(num_samples):
        sample_num = start_num + i
        
        # A4 portrait dimensions (595x842 pixels at 72dpi)
        # Slight variations for realism
        width = random.randint(590, 600)  # Around A4 width
        height = random.randint(837, 847)  # Around A4 height
        
        # Select document type
        doc_type = random.choice(doc_types)
        
        # Generate a document
        document = generate_tax_document(width=width, height=height, doc_type=doc_type)
        
        # Save the document
        output_path = os.path.join(output_dir, f"taxdoc_{sample_num}.jpg")
        document.save(output_path, "JPEG", quality=95)
        print(f"Generated {output_path}")
    
    print(f"Created {num_samples} Australian tax document samples in {output_dir}")

def replace_zero_receipt_images(source_dir="synthetic_receipts/tax_docs", 
                             target_dir="synthetic_receipts"):
    """
    Replace 0_receipts images with tax documents
    
    Args:
        source_dir: Directory containing tax documents
        target_dir: Directory containing receipt images to replace
    """
    # Find all 0_receipts images
    zero_receipt_files = [f for f in os.listdir(target_dir) 
                        if f.endswith("_0_receipts.jpg") and os.path.isfile(os.path.join(target_dir, f))]
    
    print(f"Found {len(zero_receipt_files)} zero receipt images to replace")
    
    # Find all tax documents
    tax_docs = [f for f in os.listdir(source_dir) if f.startswith("taxdoc_")]
    
    if not tax_docs:
        print("No tax documents found. Please generate them first.")
        return
    
    if len(tax_docs) < len(zero_receipt_files):
        print(f"Warning: Only {len(tax_docs)} tax documents available for {len(zero_receipt_files)} zero receipt images.")
    
    # Replace zero receipt images with tax documents
    for i, zero_file in enumerate(zero_receipt_files):
        if i >= len(tax_docs):
            print(f"Not enough tax documents to replace all zero receipt images.")
            break
            
        tax_doc = tax_docs[i]
        
        # Copy tax document to replace zero receipt image
        tax_doc_path = os.path.join(source_dir, tax_doc)
        zero_file_path = os.path.join(target_dir, zero_file)
        
        # Read tax document and resize to match destination
        tax_img = Image.open(tax_doc_path)
        
        # Get size of zero receipt image
        zero_img = Image.open(zero_file_path)
        zero_size = zero_img.size
        
        # Create a white background with the target size
        background = Image.new('RGB', zero_size, (255, 255, 255))
        
        # For tax documents, we need to preserve portrait orientation
        # Calculate how to fit tax_img into target size while keeping portrait orientation
        tax_width, tax_height = tax_img.size
        zero_width, zero_height = zero_size
        
        # Ensure portrait orientation (height > width)
        if tax_height < tax_width:
            # If tax document is in landscape, rotate it to portrait
            tax_img = tax_img.transpose(Image.ROTATE_90)
            tax_width, tax_height = tax_height, tax_width
            
        # Calculate scale to fit while maintaining portrait orientation
        scale = min(zero_height / tax_height, zero_width / tax_width)
        new_height = int(tax_height * scale)
        new_width = int(tax_width * scale)
        tax_img_resized = tax_img.resize((new_width, new_height), Image.LANCZOS)
        
        # Center the resized image on the background
        paste_x = (zero_width - new_width) // 2
        paste_y = (zero_height - new_height) // 2
        background.paste(tax_img_resized, (paste_x, paste_y))
        
        # Save the properly oriented tax document over zero receipt image
        background.save(zero_file_path, "JPEG", quality=95)
        print(f"Replaced {zero_file} with {tax_doc}")
    
    print(f"Replaced {min(len(zero_receipt_files), len(tax_docs))} zero receipt images with tax documents")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate tax document samples")
    parser.add_argument("--num_samples", type=int, default=20,
                      help="Number of tax document samples to generate")
    parser.add_argument("--output_dir", default="synthetic_receipts/tax_docs",
                      help="Directory to save tax document samples")
    parser.add_argument("--replace", action="store_true",
                      help="Replace zero receipt images with tax documents")
    
    args = parser.parse_args()
    
    # Create tax document samples
    create_tax_document_samples(args.num_samples, args.output_dir)
    
    # Replace zero receipt images if requested
    if args.replace:
        replace_zero_receipt_images(args.output_dir, "synthetic_receipts")