from xml.etree.ElementTree import Element, SubElement, Comment
from xml.etree import ElementTree
from xml.dom import minidom

def prettify(elem):
    """Return a pretty-printed XML string for the Element.
    """
    rough_string = ElementTree.tostring(elem, 'utf-8')
    reparsed = minidom.parseString(rough_string)
    return reparsed.toprettyxml(indent="  ")

top = Element('top')

comment = Comment('Use Case 4 calismasi')
top.append(comment)


child = SubElement(top, 'child')
child.text = 'deneme123'

child_with_tail = SubElement(top, 'child_with_tail')
child_with_tail.text = '456.'
child_with_tail.tail = '789.'

child_with_entity_ref = SubElement(top, 'child_with_entity_ref')
child_with_entity_ref.text = 'This & that'
print(prettify(top))
obj = ElementTree.ElementTree(top)
obj.write("fin.xml")

