import uno
from com.sun.star.beans import PropertyValue

def update_cells(file_path, sheet_name, cell_updates):
    local_context = uno.getComponentContext()
    resolver = local_context.ServiceManager.createInstanceWithContext(
        "com.sun.star.bridge.UnoUrlResolver", local_context)
    context = resolver.resolve("uno:socket,host=localhost,port=2002;urp;StarOffice.ComponentContext")

    desktop = context.ServiceManager.createInstanceWithContext(
        "com.sun.star.frame.Desktop", context)

    file_url = uno.systemPathToFileUrl(file_path)
    doc = desktop.loadComponentFromURL(file_url, "_blank", 0, ())

    if not doc:
        raise Exception("Failed to open the document")

    sheet = doc.getSheets().getByName(sheet_name)

    for cell_name, new_value in cell_updates.items():
        cell = sheet.getCellRangeByName(cell_name)
        cell.String = new_value

    # Save the changes
    prop_values = (PropertyValue("Overwrite", 0, True, 0),)
    doc.storeToURL(file_url, prop_values)

    doc.close(True)

if __name__ == "__main__":
    # Define the path to your LibreOffice Calc file
    file_path = "/path/to/your/spreadsheet.ods"

    # Define the name of the sheet you want to update
    sheet_name = "Sheet1"

    # Define the cell updates as a dictionary, where keys are cell names (e.g., "A1") and values are the new content
    cell_updates = {
        "A1": "New Value 1",
        "B2": "New Value 2",
        # Add more cell updates as needed
    }

    # Update the cells
    update_cells(file_path, sheet_name, cell_updates)
