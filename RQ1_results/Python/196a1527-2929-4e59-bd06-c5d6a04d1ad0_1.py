import json

# Your JSON data goes here
json_data = '[{"_descriptorVersion": "0.0.1", "datePublished": "2023-06-14T11:50:53.000Z", ... }]'

# Parse the JSON data
data_list = json.loads(json_data)

# Create instances of the Python model
wizard_models = []
for data in data_list:
    author_data = data["author"]
    author = Author(author_data["name"], author_data["url"], author_data["blurb"])

    resources_data = data["resources"]
    resources = Resources(resources_data["canonicalUrl"], resources_data["downloadUrl"], resources_data["paperUrl"])

    highlighted_data = data["files"]["highlighted"]
    highlighted = Highlighted(highlighted_data["economical"], highlighted_data["most_capable"])

    all_files_data = data["files"]["all"]
    all_files = [AllFile(
        file_data["name"], file_data["url"], file_data["sizeBytes"],
        file_data["quantization"], file_data["format"],
        file_data["sha256checksum"],
        Publisher(file_data["publisher"]["name"], file_data["publisher"]["socialUrl"]),
        file_data["respository"], file_data["repositoryUrl"]
    ) for file_data in all_files_data]

    wizard_model = WizardModel(
        data["_descriptorVersion"], data["datePublished"], data["name"],
        data["description"], author, data["numParameters"],
        resources, data["trainedFor"], data["arch"], Files(highlighted, all_files)
    )

    wizard_models.append(wizard_model)

# Now you have a list of WizardModel instances representing the parsed JSON data
for wizard_model in wizard_models:
    print(wizard_model.name)
    print(wizard_model.description)
    print(wizard_model.resources.download_url)
    print(wizard_model.author.url)
    print(wizard_model.files.highlighted.economical.name)
    print(wizard_model.files.all[0].name)
    print(wizard_model.files.all[1].name)
