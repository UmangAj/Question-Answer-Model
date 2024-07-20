# pdfReader.py
from pdfreader import SimplePDFViewer, PageDoesNotExist

def read_pdf(file):
    text = ""
    viewer = SimplePDFViewer(file)
    try:
        while True:
            viewer.render()
            text += "".join(viewer.canvas.strings)
            viewer.next()
    except PageDoesNotExist:
        pass
    return text
