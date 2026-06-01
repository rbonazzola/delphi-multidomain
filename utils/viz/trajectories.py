import pandas as pd

from .styles import color_by_domain


def create_friendly_view(
    x_tensor,
    ages_tensor,
    domains_tensor,
    uniq_subjs,
    *,
    model,
    dataset,
):
    """
    Human-readable dataframe for inspecting trajectories.
    """

    rows = []
    B, L = x_tensor.shape

    for b in range(B):
        subject_id = int(uniq_subjs[b])
        for pos in range(L):
            token_id = int(x_tensor[b, pos])
            age_days = float(ages_tensor[b, pos])
            domain_id = int(domains_tensor[b, pos])

            domain_name = model.int_to_domain_name.get(domain_id, f"unknown_{domain_id}")
            tokenizer = dataset.domains.get(domain_name, {"tokenizer": {}})["tokenizer"]

            rows.append(
                {
                    "subject_id": subject_id,
                    "age": round(age_days / 365.25, 2),
                    "domain_id": domain_id,
                    "domain_name": domain_name,
                    "token_id": token_id,
                    "token_name": tokenizer.get(token_id, f"unknown_{token_id}"),
                }
            )

    df = pd.DataFrame(rows).sort_values(["subject_id", "age", "domain_id", "token_id"]).reset_index(drop=True)

    return df, df.style.apply(color_by_domain, axis=1)
