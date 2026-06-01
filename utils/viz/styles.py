def color_by_domain(row):

    colors = {
        "hla_alleles": "background-color: #A0E5E5",
        "sex": "background-color: #E5F5FF",
        "lifestyle": "background-color: #B0B0E5",
        "diseases": "background-color: #FFF5E5",
        "death": "background-color: #F5E5FF",
        "padding": "background-color: #F0F0F0",
    }
    return [colors.get(row["domain_name"], "")] * len(row)
