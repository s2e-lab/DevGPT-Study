search_fields = [
    ("liden_number", "liden_number_filter", "Lidén Number", [
        ("equals", "Equals"),
        ("less_than", "Is less than"),
        ("greater_than", "Is greater than"),
    ]),
    ("respondent", "respondent_filter", "Respondent", [
        ("contains", "Contains"),
        ("contains_any", "Contains any"),
        ("contains_all", "Contains all"),
        ("contains_not", "Does not contain"),
    ]),
    ("title", "title_filter", "Title", [
        ("contains", "Contains"),
        ("contains_any", "Contains any"),
        ("contains_all", "Contains all"),
        ("contains_not", "Does not contain"),
    ]),
    ("real_date", "real_date_filter", "Date of Defense", [
        ("equals", "Equals"),
        ("less_than", "Is before"),
        ("greater_than", "Is after"),
    ]),
]

# Pass the search_fields to the template context
context = {
    "search_fields": search_fields,
}
