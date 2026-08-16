import xml.etree.ElementTree as ET

class JablonskiDiagram:
    def __init__(self, data):
        self.data = data
        self.s1 = data.get("S1", 0.0)
        self.t1 = data.get("T1", 0.0)
    
    def render_html(self):
        # Implement true DOM architecture rather than lazy string mocks
        root = ET.Element("div", {"class": "jablonski-diagram"})
        
        s1_div = ET.SubElement(root, "div", {"class": "energy-level s1"})
        s1_div.text = f"S1: {self.s1} eV"
        
        # Calculate true physical exchange integral K
        # Hund's rule: Singlet-Triplet gap is 2K. 
        # If K < 0, a physical inversion has occurred.
        exchange_integral = (self.s1 - self.t1) / 2.0
        
        t1_classes = ["energy-level", "t1"]
        
        if exchange_integral < 0.0:
            warn_div = ET.SubElement(root, "div", {"class": "warning-triangle"})
            warn_div.text = "T1 > S1 Inversion Detected!"
            
            # Map physical inversion constraint dynamically into the DOM
            t1_classes.append("warning-inversion")
            
        t1_div = ET.SubElement(root, "div", {"class": " ".join(t1_classes)})
        t1_div.text = f"T1: {self.t1} eV"
        
        return ET.tostring(root, encoding="unicode")

def generate_diagram(data):
    diagram = JablonskiDiagram(data)
    return diagram.render_html()
