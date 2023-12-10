from urllib.parse import urlparse

@classmethod
def from_text(cls, text: str, openapi_url: str) -> "OpenAPISpec":
    """Get an OpenAPI spec from a text."""
    try:
        spec_dict = json.loads(text)
    except json.JSONDecodeError:
        spec_dict = yaml.safe_load(text)

    if "servers" not in spec_dict:
        parsed_url = urlparse(openapi_url)
        root_url = f"{parsed_url.scheme}://{parsed_url.netloc}"
        print("root_url: ", root_url)
        spec_dict["servers"] = [{"url": root_url}]

    print("spec_dict: ", json.dumps(spec_dict, indent=2))
    return cls.from_spec_dict(spec_dict)
