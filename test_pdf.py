import os
import subprocess

def test():
    try:
        from docx2pdf import convert
        print("docx2pdf imported")
    except Exception as e:
        print(f"docx2pdf import failed: {e}")

    try:
        res = subprocess.run(["libreoffice", "--version"], capture_output=True, text=True)
        print("libreoffice:", res.stdout)
    except Exception as e:
        print(f"libreoffice run failed: {e}")
        
    try:
        res = subprocess.run(["/Applications/LibreOffice.app/Contents/MacOS/soffice", "--version"], capture_output=True, text=True)
        print("soffice:", res.stdout)
    except Exception as e:
        print(f"soffice run failed: {e}")

test()
