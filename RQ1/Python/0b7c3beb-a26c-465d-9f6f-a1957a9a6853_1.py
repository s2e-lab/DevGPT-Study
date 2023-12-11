dataset = fiftyone.zoo.load_zoo_dataset(
    "open-images-v6",
    split="validation",
    label_types=["detections", "segmentations"],
    classes=["Cat", "Dog"],
    max_samples=100,
)
