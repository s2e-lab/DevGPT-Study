class WizardModel:
    def __init__(self, descriptor_version, date_published, name, description, author, num_parameters, resources, trained_for, arch, files):
        self.descriptor_version = descriptor_version
        self.date_published = date_published
        self.name = name
        self.description = description
        self.author = author
        self.num_parameters = num_parameters
        self.resources = resources
        self.trained_for = trained_for
        self.arch = arch
        self.files = files

class Author:
    def __init__(self, name, url, blurb):
        self.name = name
        self.url = url
        self.blurb = blurb

class Resources:
    def __init__(self, canonical_url, download_url, paper_url):
        self.canonical_url = canonical_url
        self.download_url = download_url
        self.paper_url = paper_url

class Files:
    def __init__(self, highlighted, all_files):
        self.highlighted = highlighted
        self.all = all_files

class Highlighted:
    def __init__(self, economical, most_capable):
        self.economical = economical
        self.most_capable = most_capable

class Publisher:
    def __init__(self, name, social_url):
        self.name = name
        self.social_url = social_url

class AllFile:
    def __init__(self, name, url, size_bytes, quantization, format, sha256checksum, publisher, repository, repository_url):
        self.name = name
        self.url = url
        self.size_bytes = size_bytes
        self.quantization = quantization
        self.format = format
        self.sha256checksum = sha256checksum
        self.publisher = publisher
        self.repository = repository
        self.repository_url = repository_url
