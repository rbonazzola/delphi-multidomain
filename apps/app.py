import glob
import os

import matplotlib.pyplot as plt
import mlflow
import pandas as pd
import plotly.express as px
import seaborn as sns
import streamlit as st
from mlflow.entities import ViewType
from mlflow.tracking import MlflowClient
from scipy.stats import norm
from tqdm import tqdm

st.set_page_config(layout="wide")
st.title("MLflow Run Explorer")

# Sidebar controls
st.sidebar.title("Settings")
mlflow_tracking_uri = st.sidebar.text_input("Tracking URI", value="./mlruns")
mlflow.set_tracking_uri(mlflow_tracking_uri)
client = MlflowClient()

HLA_SCORE_CSV = "hla_score_per_icd10_with_justification_COMPLETE.csv"

experiments = client.search_experiments()
exp_name_to_id = {e.name: e.experiment_id for e in experiments}
selected_exp_name = st.sidebar.selectbox("Select Experiment", list(exp_name_to_id.keys()))
experiment_id = exp_name_to_id[selected_exp_name]

# max_results = st.sidebar.slider("Max number of runs", 10, 1000, 200)
val_loss_threshold = st.sidebar.number_input("Filter runs with val_loss less than:", value=12.5, step=0.01)

if "runs_loaded" not in st.session_state:
    st.session_state.runs_loaded = False

if st.sidebar.button("Load runs"):
    st.session_state.runs_loaded = True


def compute_consensus_auc_by_sex(df):
    results = []

    for (disease, sex, age), group in df.groupby(["name", "sex", "age"]):
        # Con HLA
        aucs_hla = group["auc_delong_hla"].values
        vars_hla = group["auc_variance_delong_hla"].values
        weights_hla = 1 / vars_hla

        auc_consensus_hla = (aucs_hla * weights_hla).sum() / weights_hla.sum()
        var_consensus_hla = 1 / weights_hla.sum()

        # Sin HLA
        aucs_nohla = group["auc_delong_nohla"].values
        vars_nohla = group["auc_variance_delong_nohla"].values
        weights_nohla = 1 / vars_nohla

        auc_consensus_nohla = (aucs_nohla * weights_nohla).sum() / weights_nohla.sum()
        var_consensus_nohla = 1 / weights_nohla.sum()

        results.append(
            {
                "name": disease,
                "sex": sex,
                "age": age,
                "AUC_consensus_hla": auc_consensus_hla,
                "SE_consensus_hla": var_consensus_hla**0.5,
                "AUC_consensus_nohla": auc_consensus_nohla,
                "SE_consensus_nohla": var_consensus_nohla**0.5,
            }
        )

    return pd.DataFrame(results)


def compare_sex_differences(df):
    results = []

    for disease, group in df.groupby("name"):
        print(set(group["sex"]))
        if set(group["sex"]) != {"female", "male"}:
            continue  # skip if not both sexes present

        male = group[group["sex"] == "male"].iloc[0]
        female = group[group["sex"] == "female"].iloc[0]

        # Con HLA
        delta_hla = male["AUC_consensus_hla"] - female["AUC_consensus_hla"]
        se_hla = (male["SE_consensus_hla"] ** 2 + female["SE_consensus_hla"] ** 2) ** 0.5
        z_hla = delta_hla / se_hla
        p_hla = 2 * norm.sf(abs(z_hla))

        # Sin HLA
        delta_nohla = male["AUC_consensus_nohla"] - female["AUC_consensus_nohla"]
        se_nohla = (male["SE_consensus_nohla"] ** 2 + female["SE_consensus_nohla"] ** 2) ** 0.5
        z_nohla = delta_nohla / se_nohla
        p_nohla = 2 * norm.sf(abs(z_nohla))

        results.append(
            {
                "name": disease,
                "delta_auc_male_vs_female_hla": delta_hla,
                "pvalue_male_vs_female_hla": p_hla,
                "delta_auc_male_vs_female_nohla": delta_nohla,
                "pvalue_male_vs_female_nohla": p_nohla,
            }
        )

    return pd.DataFrame(results).sort_values("pvalue_interaction")


def test_interaction_by_age(df):
    results = []

    for (disease, age_group), group in df.groupby(["name", "age"]):
        if set(group["sex"]) != {"male", "female"}:
            continue  # need both sexes to compare

        male = group[group["sex"] == "male"].iloc[0]
        female = group[group["sex"] == "female"].iloc[0]

        # Delta AUC por sexo
        delta_male = male["AUC_consensus_hla"] - male["AUC_consensus_nohla"]
        delta_female = female["AUC_consensus_hla"] - female["AUC_consensus_nohla"]

        # Error estándar total
        se_total = (
            male["SE_consensus_hla"] ** 2
            + male["SE_consensus_nohla"] ** 2
            + female["SE_consensus_hla"] ** 2
            + female["SE_consensus_nohla"] ** 2
        ) ** 0.5

        z = (delta_male - delta_female) / se_total
        p_value = 2 * norm.sf(abs(z))

        results.append(
            {
                "name": disease,
                "age": age_group,
                "delta_auc_hla_male": delta_male,
                "delta_auc_hla_female": delta_female,
                "diff_of_deltas": delta_male - delta_female,
                "pvalue_interaction": p_value,
            }
        )

    return pd.DataFrame(results).sort_values("pvalue_interaction").head(20)


@st.cache_data
def load_runs(experiment_ids: str | list[str], val_loss_threshold: float = 1.0):

    client = MlflowClient()
    filter_str = f"metrics.val_loss < {val_loss_threshold} and metrics.val_loss > 11.8"
    print(f"{experiment_ids=}")

    try:
        runs = client.search_runs(
            experiment_ids if isinstance(experiment_ids, list) else [experiment_ids],
            run_view_type=ViewType.ACTIVE_ONLY,
            filter_string=filter_str,
        )
    except Exception as e:
        print(f"Error searching for runs: {e}")
        return pd.DataFrame(), 0

    records = []
    skipped = 0

    for run in tqdm(runs):
        try:
            # if run.info.status != "FINISHED":
            #    continue
            row = {"run_id": run.info.run_id, **run.data.params, **run.data.metrics}
            records.append(row)
        except Exception as e:
            skipped += 1
            print(f"Skipping run {run.info.run_id}: {e}")

    if not records:
        return pd.DataFrame(), 0

    df = pd.DataFrame(records)

    for col in df.columns:
        try:
            df[col] = pd.to_numeric(df[col])
        except ValueError:
            continue

    df = df.loc[:, df.nunique(dropna=False) > 1]

    return df, skipped


@st.cache_data
def get_loss_curve(run_id, metric="val_loss"):
    try:
        history = client.get_metric_history(run_id, metric)
        return pd.DataFrame({"step": [m.step for m in history], "value": [m.value for m in history], "run_id": run_id})
    except Exception:
        return None


# Visualization if runs were loaded
if st.session_state.runs_loaded:
    tab1, tab2, tab3, tab4, tab5 = st.tabs(
        ["\U0001f4cb Runs", "\U0001f4c8 Correlations", "\U0001f4c9 Loss Curves", "AUC", "5-fold CV"]
    )

    df, skipped = load_runs(experiment_id, val_loss_threshold)

    if df.empty:
        st.warning("No valid runs could be loaded.")
        st.stop()

    st.success(f"{len(df)} runs loaded. {skipped} discarded.")

    with tab1:
        st.subheader("Runs Table")
        st.dataframe(df)

        num_cols = df.select_dtypes(include=["float", "int"]).columns
        if len(num_cols) >= 2:
            x_col = st.selectbox("X", num_cols, index=0)
            y_col = st.selectbox("Y", num_cols, index=1)
            st.plotly_chart(px.scatter(df, x=x_col, y=y_col, hover_data=["run_id"]), use_container_width=True)

    with tab2:
        st.subheader("Correlation between hyperparameters and metrics")
        if df.select_dtypes(include="number").shape[1] >= 2:
            corr = df.select_dtypes(include="number").corr()
            fig, ax = plt.subplots(figsize=(12, 8))
            sns.heatmap(corr, annot=True, fmt=".2f", cmap="coolwarm", ax=ax)
            st.pyplot(fig)

            target = st.selectbox("Target metric", corr.columns)
            top_corr = corr[target].drop(target).sort_values(key=abs, ascending=False).head(3)
            for param in top_corr.index:
                fig = px.scatter(df, x=param, y=target, trendline="ols")
                st.plotly_chart(fig, use_container_width=True)

    with tab3:
        st.subheader("`val_loss` curves per run")
        selected_runs = st.multiselect("Select runs to display loss curves", df["run_id"].tolist(), default=[])
        # df["run_id"].tolist()

        all_curves = []
        for run_id in selected_runs:
            curve = get_loss_curve(run_id)
            if curve is not None:
                all_curves.append(curve)

        if all_curves:
            full_df = pd.concat(all_curves)
            fig = px.line(full_df, x="step", y="value", color="run_id", title="val_loss by step")
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("No loss curves found for the selected runs.")

    with tab4:
        st.subheader("AUC values")

        experiments = client.search_experiments()
        only_white = st.checkbox("Only white")

        digit_option = st.radio("Select HLA specification", ["4-digit", "2-digit"])
        if digit_option == "4-digit":
            experiments_hla = {
                e.name: e.experiment_id for e in experiments if "no" not in e.name and "2digit" not in e.name
            }
        else:
            experiments_hla = {
                e.name: e.experiment_id for e in experiments if "no" not in e.name and "2digit" in e.name
            }

        experiments_nohla = {e.name: e.experiment_id for e in experiments if "no" in e.name}
        exp_name_to_id = {e.name: e.experiment_id for e in experiments}

        runs_hla, _ = load_runs([exp_id for exp_name, exp_id in experiments_hla.items()], val_loss_threshold=11.95)
        runs_nohla, _ = load_runs([exp_id for exp_name, exp_id in experiments_nohla.items()], val_loss_threshold=11.95)

        runs_hla = sorted(
            [f"{k} {v:.4f}" for k, v in zip(runs_hla["run_id"].tolist(), runs_hla["val_loss"].tolist())],
            key=lambda x: float(x.split()[1]),
        )
        runs_nohla = sorted(
            [f"{k} {v:.4f}" for k, v in zip(runs_nohla["run_id"].tolist(), runs_nohla["val_loss"].tolist())],
            key=lambda x: float(x.split()[1]),
        )

        parquet_files = glob.glob("mlruns/*/*/artifacts/auc/*parquet")

        runs_hla = [k for k in runs_hla if any([k.split(" ")[0] in f for f in parquet_files])]
        runs_nohla = [k for k in runs_nohla if any([k.split(" ")[0] in f for f in parquet_files])]

        # runs_hla_dict = { k: v for k, v in runs_hla_dict.items() if any([k in parquet_files]) }
        # runs_nohla_dict = { k: v for k, v in runs_nohla_dict.items() if any([k in parquet_files]) }
        # selected_run_hla = st.selectbox("Select run w/HLA", runs_hla["run_id"].tolist())

        def fix_artifact_uri(artifact_dir, on_codon=False):
            if not on_codon:
                artifact_dir = artifact_dir.replace("/homes", "/home")
                artifact_dir = artifact_dir.replace("/nfs/research/birney/users", "/home")
            return artifact_dir

        def get_auc_dfs(run, suffix=""):
            artifact_dir = fix_artifact_uri(run.info.artifact_uri, on_codon="codon" in os.environ["HOSTNAME"])
            AUCDIR = f"{artifact_dir}/auc/"
            try:
                unpooled_auc = pd.read_parquet(f"{AUCDIR}/df_auc_unpooled{suffix}.parquet").query("n_diseased > 20")
            except FileNotFoundError:
                unpooled_auc = pd.read_parquet(f"{AUCDIR}/df_auc_unpooled.parquet").query("n_diseased > 20")

            try:
                both_auc = pd.read_parquet(f"{AUCDIR}/df_both{suffix}.parquet")
            except FileNotFoundError:
                both_auc = pd.read_parquet(f"{AUCDIR}/df_both.parquet")

            unpooled_auc = unpooled_auc.drop(["auc"], axis=1)
            unpooled_auc = unpooled_auc[~unpooled_auc.duplicated()]
            unpooled_auc = unpooled_auc.drop(["ICD-10 Chapter (short)", "color"], axis=1)
            # unpooled_auc = unpooled_auc.drop(['auc_variance_delong', 'count'], axis=1)
            return unpooled_auc, both_auc

        col1, col2, col3 = st.columns([1, 0.3, 1])
        with col1:
            # selected_run_hla = st.select_slider("### Select run w/HLA", runs_hla)
            if not only_white:
                selected_run_hla = st.selectbox("### Select run w/HLA", runs_hla)
            else:
                selected_run_hla = "52ee455f05e34d25812eedb4c87dd71c"
                st.write("Run 52ee55... (with 2-digit) has been chosen")

        if only_white:
            print(f"{only_white=}, {selected_run_hla=}")
        else:
            selected_run_hla = selected_run_hla.split(" ")[0]
            print(f"{only_white=}, {selected_run_hla=}")

        run_hla = client.get_run(selected_run_hla)

        # unpooled_auc_hla, both_auc_hla = get_auc_dfs(run_hla, suffix="_1y")
        if only_white:
            unpooled_auc_hla, both_auc_hla = get_auc_dfs(run_hla, suffix="only_white")
        else:
            unpooled_auc_hla, both_auc_hla = get_auc_dfs(run_hla, suffix="_1y")
        run_hla_params = run_hla.data.params
        run_hla_params.pop("ignore_tokens")

        with col3:
            # selected_run_nohla = st.select_slider("Select run wo/HLA", runs_nohla)
            selected_run_nohla = st.selectbox("Select run wo/HLA", runs_nohla)

        selected_run_nohla = selected_run_nohla.split(" ")[0]
        run_nohla = client.get_run(selected_run_nohla)
        unpooled_auc_nohla, both_auc_nohla = get_auc_dfs(run_nohla, suffix="_1y")
        run_nohla_params = run_nohla.data.params
        run_nohla_params.pop("ignore_tokens")
        with st.expander("See hyperparameters"):
            st.table(
                pd.concat([pd.Series(run_hla_params).to_frame().T, pd.Series(run_nohla_params).to_frame().T]).set_axis(
                    ["HLA", "no HLA"]
                )
            )
            # st.table()

        # print(unpooled_auc_hla)
        # print(unpooled_auc_nohla)
        # print(unpooled_auc_hla.token.unique())
        # print(unpooled_auc_nohla.token.unique())

        # st.text(f"HLA loss: {run_hla.data.metrics['val_loss']:.4f}")
        # st.text(f"NO HLA loss: {run_nohla.data.metrics['val_loss']:.4f}")

        unpooled_auc_merged = (
            pd.merge(unpooled_auc_hla, unpooled_auc_nohla, on=["age", "name", "sex"], suffixes=["_hla", "_nohla"])
            .drop(["n_healthy_hla", "n_healthy_nohla"], axis=1)
            .assign(diff=lambda x: x.auc_delong_hla - x.auc_delong_nohla)
            .sort_values("diff", ascending=False)
            .merge(pd.read_csv(HLA_SCORE_CSV), left_on="index_nohla", right_on="index")
        )  # .\
        # drop(["index_hla", "index_nohla", "token_hla", "token_nohla", "n_diseased_hla", "n_diseased_nohla"], axis=1).\
        # loc[:, ["age", "sex", "name", "auc_delong_hla", "auc_delong_nohla", "diff", 'score', 'genes', 'justification']]

        cols_to_discard = [
            "color",
            "ICD-10 Chapter",
            "index",
            # 'auc_variance_delong_nohla', 'auc_variance_delong_hla',
            "n_samples_hla",
            "n_diseased_hla",
            "n_healthy_hla",
            "index_hla",
            "count_hla",
            "token_hla",
            "n_samples_nohla",
            "n_diseased_nohla",
            "n_healthy_nohla",
            "index_nohla",
            "count_nohla",
            "token_nohla",
        ]

        both_auc_merged = (
            pd.merge(
                both_auc_hla,
                both_auc_nohla,
                on=["name", "ICD-10 Chapter", "ICD-10 Chapter (short)", "color"],
                suffixes=["_hla", "_nohla"],
            )
            .assign(diff=lambda x: x.auc_hla - x.auc_nohla)
            .sort_values("diff", ascending=False)
            .merge(pd.read_csv(HLA_SCORE_CSV), left_on="index_nohla", right_on="index")
            .drop(cols_to_discard, axis=1)
        )

        diseases = st.multiselect("Choose diseases", options=sorted(unpooled_auc_merged.name.unique()))
        # st.dataframe(both_auc_merged)
        col1, col2 = st.columns([2, 1])

        if diseases == []:
            with col1:
                st.dataframe(unpooled_auc_merged)
            with col2:
                average = st.checkbox("Average over sex and age bins")
                if average:
                    unpooled_auc_merged_averaged = (
                        unpooled_auc_merged.groupby(["name"])
                        .agg({"auc_delong_hla": "mean", "auc_delong_nohla": "mean", "score": "first", "diff": "mean"})
                        .reset_index()
                    )
                    fig = px.scatter(
                        unpooled_auc_merged_averaged,
                        x="auc_delong_nohla",
                        y="auc_delong_hla",
                        color="score",
                        hover_data=["name", "diff"],
                        labels={
                            "auc_delong_nohla": "AUC without HLA",
                            "auc_delong_hla": "AUC with HLA",
                            "name": "Disease",
                        },
                        title="Comparison with and without HLA",
                    )
                else:
                    fig = px.scatter(
                        unpooled_auc_merged,
                        x="auc_delong_nohla",
                        y="auc_delong_hla",
                        color="score",
                        # symbol="age",
                        hover_data=["name", "sex", "age", "diff"],
                        labels={
                            "auc_delong_nohla": "AUC without HLA",
                            "auc_delong_hla": "AUC with HLA",
                            "sex": "Sex",
                            "age": "Age",
                            "name": "Disease",
                        },
                        title="Comparison with and without HLA",
                    )

                fig.add_shape(type="line", x0=0, x1=1, y0=0, y1=1, line=dict(color="gray", dash="dash"))

                fig.update_layout(
                    xaxis=dict(range=[0.5, 1]), yaxis=dict(range=[0.5, 1]), autosize=False, width=600, height=475
                )
                st.plotly_chart(fig, use_container_width=True)
        else:
            # auc_for_disease = unpooled_auc_merged[unpooled_auc_merged.name.apply(lambda x: [bool(re.match(f".*{disease.lower()}.*", x.lower())))] ]
            auc_for_disease = unpooled_auc_merged[unpooled_auc_merged.name.isin(diseases)]
            with col1:
                st.dataframe(auc_for_disease)
            with col2:
                fig2, ax2 = plt.subplots(figsize=(4, 4))
                sns.lineplot(
                    # data=auc_for_disease.melt(id_vars=["age", "sex"], value_vars=["auc_delong_hla", "auc_delong_nohla"], var_name="HLA", value_name="AUC"),
                    data=auc_for_disease,
                    x="age",
                    y="diff",
                    hue="sex",
                )
                ax2.set_ylim(-0.1, 0.1)

                ax2.axhline(0, color="gray", linestyle="--", linewidth=1)
                st.pyplot(fig2)

    with tab5:

        @st.cache_data
        def load_5fold_cv_auc() -> tuple[pd.DataFrame, pd.DataFrame]:

            EXP_NOHLA = "437945341875567335"
            EXP_HLA = "278131880607437980"

            runs_hla_df = mlflow.search_runs(experiment_ids=[EXP_HLA])
            runs_nohla_df = mlflow.search_runs(experiment_ids=[EXP_NOHLA])
            assert isinstance(runs_hla_df, pd.DataFrame)
            assert isinstance(runs_nohla_df, pd.DataFrame)

            unpooled_auc_list: list[pd.DataFrame] = []
            both_auc_list: list[pd.DataFrame] = []

            for _fold_i in range(1, 6):
                fold_i = str(_fold_i)
                run_nohla_id = runs_nohla_df.query("`params.fold` == @fold_i").run_id.iloc[0]
                run_hla_id = runs_hla_df.query("`params.fold` == @fold_i").run_id.iloc[0]
                run_nohla = client.get_run(run_nohla_id)
                run_hla = client.get_run(run_hla_id)
                unpooled_auc_nohla, both_auc_nohla = get_auc_dfs(run_nohla, suffix="_onlywhite")
                unpooled_auc_hla, both_auc_hla = get_auc_dfs(run_hla, suffix="_onlywhite")

                unpooled_auc_merged = (
                    pd.merge(
                        unpooled_auc_hla, unpooled_auc_nohla, on=["age", "name", "sex"], suffixes=("_hla", "_nohla")
                    )
                    .drop(["n_healthy_hla", "n_healthy_nohla"], axis=1)
                    .assign(diff=lambda x: x.auc_delong_hla - x.auc_delong_nohla)
                    .sort_values("diff", ascending=False)
                    .merge(pd.read_csv(HLA_SCORE_CSV), left_on="index_nohla", right_on="index")
                )  # .\
                # drop(["index_hla", "index_nohla", "token_hla", "token_nohla", "n_diseased_hla", "n_diseased_nohla"], axis=1).\
                # loc[:, ["age", "sex", "name", "auc_delong_hla", "auc_delong_nohla", "diff", 'score', 'genes', 'justification']]

                cols_to_discard = [
                    "color",
                    "ICD-10 Chapter",
                    "index",
                    #'auc_variance_delong_nohla', 'auc_variance_delong_hla',
                    # 'n_samples_hla', 'n_diseased_hla', 'n_healthy_hla', 'index_hla', 'count_hla', 'token_hla',
                    # 'n_samples_nohla', 'n_diseased_nohla', 'n_healthy_nohla', 'index_nohla', 'count_nohla', 'token_nohla'
                ]

                both_auc_merged = (
                    pd.merge(
                        both_auc_hla,
                        both_auc_nohla,
                        on=["name", "ICD-10 Chapter", "ICD-10 Chapter (short)", "color"],
                        suffixes=("_hla", "_nohla"),
                    )
                    .assign(diff=lambda x: x.auc_hla - x.auc_nohla)
                    .sort_values("diff", ascending=False)
                    .merge(pd.read_csv(HLA_SCORE_CSV), left_on="index_nohla", right_on="index")
                    .drop(cols_to_discard, axis=1)
                )

                unpooled_auc_list.append(unpooled_auc_merged.assign(fold=fold_i))
                both_auc_list.append(both_auc_merged.assign(fold=fold_i))

            return pd.concat(unpooled_auc_list), pd.concat(both_auc_list)

        diseases = st.multiselect("Choose diseases", options=sorted(unpooled_auc_merged.name.unique()), key=3)

        unpooled_auc_mergeds, both_auc_mergeds = load_5fold_cv_auc()

        _fold_index = st.slider(label="fold", min_value=1, max_value=5)
        fold_index = str(_fold_index)

        unpooled_auc_merged = unpooled_auc_mergeds.query("fold == @fold_index")

        print(unpooled_auc_mergeds.columns)
        import numpy as np

        kk = compute_consensus_auc_by_sex(unpooled_auc_mergeds)
        print(kk.head())
        kk = kk.assign(
            zscore=lambda df: (
                (df.AUC_consensus_hla - df.AUC_consensus_nohla)
                / np.sqrt(df.SE_consensus_hla**2 + df.SE_consensus_nohla**2)
            )
        )
        kk = kk.sort_values("zscore", ascending=False)
        st.dataframe(kk)

        # print(kk.loc[((kk.AUC_consensus_hla-kk.AUC_consensus_nohla)/np.sqrt(kk.SE_consensus_hla**2+kk.SE_consensus_nohla**2)).sort_values().index])
        print(kk)

        kk2 = test_interaction_by_age(kk)
        # print(kk2)

        col1, col2 = st.columns([2, 1])

        if diseases == []:
            with col1:
                st.header("ΔAUC (HLA - noHLA)")
                st.dataframe(unpooled_auc_merged)

                st.header("ΔAUC(male) vs. ΔAUC(female)")
                st.dataframe(kk2)
            with col2:
                average = st.checkbox("Average over sex and age bins", key=2)
                if average:
                    unpooled_auc_merged_averaged = (
                        unpooled_auc_merged.groupby(["name"])
                        .agg({"auc_delong_hla": "mean", "auc_delong_nohla": "mean", "score": "first", "diff": "mean"})
                        .reset_index()
                    )
                    fig = px.scatter(
                        unpooled_auc_merged_averaged,
                        x="auc_delong_nohla",
                        y="auc_delong_hla",
                        color="score",
                        hover_data=["name", "diff"],
                        labels={
                            "auc_delong_nohla": "AUC without HLA",
                            "auc_delong_hla": "AUC with HLA",
                            "name": "Disease",
                        },
                        title="Comparison with and without HLA",
                    )
                else:
                    fig = px.scatter(
                        unpooled_auc_merged,
                        x="auc_delong_nohla",
                        y="auc_delong_hla",
                        color="score",
                        # symbol="age",
                        hover_data=["name", "sex", "age", "diff"],
                        labels={
                            "auc_delong_nohla": "AUC without HLA",
                            "auc_delong_hla": "AUC with HLA",
                            "sex": "Sex",
                            "age": "Age",
                            "name": "Disease",
                        },
                        title="Comparison with and without HLA",
                    )

                fig.add_shape(type="line", x0=0, x1=1, y0=0, y1=1, line=dict(color="gray", dash="dash"))

                fig.update_layout(
                    xaxis=dict(range=[0.5, 1]), yaxis=dict(range=[0.5, 1]), autosize=False, width=600, height=475
                )
                st.plotly_chart(fig, use_container_width=True, key=4)
        else:
            # auc_for_disease = unpooled_auc_merged[unpooled_auc_merged.name.apply(lambda x: [bool(re.match(f".*{disease.lower()}.*", x.lower())))] ]
            auc_for_disease = unpooled_auc_mergeds[unpooled_auc_mergeds.name.isin(diseases)]
            with col1:
                st.dataframe(auc_for_disease)
            with col2:
                fig2, ax2 = plt.subplots(figsize=(8, 4))
                print(auc_for_disease.head())
                sns.scatterplot(
                    # data=auc_for_disease.melt(id_vars=["age", "sex"], value_vars=["auc_delong_hla", "auc_delong_nohla"], var_name="HLA", value_name="AUC"),
                    data=auc_for_disease.assign(
                        age_mod=lambda x: (
                            x.age + 2.5 - 0.5 * (x.sex == "male").astype(int) + 0.5 * (x.sex == "female").astype(int)
                        )
                    ),
                    x="age",
                    y="diff",
                    hue="sex",
                    hue_order=["female", "male"],
                )
                ax2.set_ylim(-0.2, 0.4)
                ax2.set_xlabel("Age")
                ax2.set_ylabel("ΔAUC (HLA - noHLA)")
                ax2.set_title(f"{diseases[0]}")

                ax2.axhline(0, color="gray", linestyle="--", linewidth=1)
                st.pyplot(fig2)
