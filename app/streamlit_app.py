
import numpy as np
import pandas as pd
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import os, sys
import json
from datetime import datetime
from sklearn.metrics import (
    roc_auc_score,
    f1_score,
    precision_score,
    recall_score,
    matthews_corrcoef,
    balanced_accuracy_score,
)
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import RobustScaler, FunctionTransformer, PolynomialFeatures

# Add project root to sys.path
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if ROOT not in sys.path: sys.path.insert(0, ROOT)

# Core imports
from core.methods.registry import REGISTRY, DEFAULT_KWARGS, ENSEMBLE_FACTORY
from core.methods.preproc_wrapper import PreprocWrapper, AutoPreprocessor
from core.methods.sparse_auto import SparseAutoPreprocessor
from core.evaluate.cv import cv_average_precision

from core.metrics.ranking import average_precision, precision_recall_points, precision_at_k

# App Utils imports
import app.utils.ui as ui
import app.utils.data as app_data
import app.utils.metrics as app_metrics
import shap
import matplotlib.pyplot as plt

# -----------------------------------------------------------------------------
# Classification/Model Helpers
# -----------------------------------------------------------------------------

SUPERVISED_METHODS = {"XGBoost", "CatBoost", "LGBM"}

def _method_requires_y(method_name: str) -> bool:
    return method_name in SUPERVISED_METHODS

def _fit_model(model, X_data, y_data, method_name: str):
    if _method_requires_y(method_name) and y_data is not None:
        return model.fit(X_data, y_data)
    else:
        return model.fit(X_data)

# -----------------------------------------------------------------------------
# Main App
# -----------------------------------------------------------------------------

st.set_page_config(page_title="TFG Anomaly MVP", layout="wide", initial_sidebar_state="expanded")

ui.inject_custom_css()
ui.render_header()

# Sidebar
with st.sidebar:
    st.header("⚙️ Configuration")
    st.divider()
    st.header("📊 Quick Stats")
    stats_placeholder = st.empty()

# Main Container using Tabs
tab_data, tab_analysis, tab_deep_dive = st.tabs(["📂 Data & Config", "🚀 Analysis", "🔍 Deep Dives"])

# -----------------------------------------------------------------------------
# TAB 1: Data & Config
# -----------------------------------------------------------------------------
with tab_data:
    col_up, col_map = st.columns([1, 2])
    
    with col_up:
        uploaded = st.file_uploader("Upload Dataset (CSV, Parquet, MAT)", type=["csv", "parquet", "mat"])
        if not uploaded:
            st.info("👆 Upload a dataset to begin.")
            ui.show_quick_start()
            st.stop()

    try:
        file_bytes = uploaded.read()
        uploaded.seek(0)
        
        # We need a quick read for column selection
        if uploaded.name.endswith(".parquet"):
            df_preview = pd.read_parquet(uploaded)
        elif uploaded.name.endswith(".mat"):
            df_preview = app_data._read_mat_to_df(file_bytes)
        else:
            df_preview = pd.read_csv(uploaded)
            
    except Exception as e:
        st.error(f"❌ Failed to read file: {e}")
        st.stop()
        
    with col_map:
        st.subheader("🗂️ Map Schema")
        cols = list(df_preview.columns)
        c1, c2 = st.columns(2)
        with c1:
            label_options = ["(none)"] + cols
            # heuristic default
            label_default_idx = 0
            for idx, name in enumerate(label_options[1:], start=1):
                if name.lower() in ["label", "class", "target", "y"]:
                    label_default_idx = idx
                    break
            label_col = st.selectbox("Label column (optional)", label_options, index=label_default_idx)
            
        with c2:
            timestamp_col = st.selectbox("Timestamp column (optional)", ["(none)"] + cols, index=0)
            
        default_feats = [c for c in cols if c not in {label_col, timestamp_col, "(none)"}]
        feature_cols = st.multiselect("Feature columns", cols, default=default_feats)
        
        # Guardrails
        if label_col != "(none)" and label_col in feature_cols:
            feature_cols = [c for c in feature_cols if c != label_col]
            st.info("ℹ️ Removed label from features.")
        if timestamp_col != "(none)" and timestamp_col in feature_cols:
            feature_cols = [c for c in feature_cols if c != timestamp_col]

    if not feature_cols:
        st.warning("⚠️ Select at least one feature column.")
        st.stop()

    # Load Full Data
    df, X, y, meta = app_data.load_and_parse_data(
        file_bytes, uploaded.name, feature_cols,
        label_col, timestamp_col
    )
    X_df = df[feature_cols].copy()

    # NEW: Data Subsampling for Performance
    with st.expander("⚡ Performance Options"):
        st.caption(f"📊 Current dataset size: **{len(X_df):,}** rows")
        
        if len(X_df) > 1000:
            use_subsample = st.checkbox(f"Enable downsampling for faster training", value=False)
            if use_subsample:
                default_max = min(10000, len(X_df))
                n_max = st.slider("Max rows to use", 500, len(X_df), default_max, step=500)
                if n_max < len(X_df):
                    df_sub = app_data.downsample_dataframe(df, n_max)
                    X, y, meta = app_data.parse_dataframe(df_sub, feature_cols,
                                                         label_col=(None if label_col == "(none)" else label_col),
                                                         timestamp_col=(None if timestamp_col == "(none)" else timestamp_col))
                    X_df = df_sub[feature_cols].copy()
                    st.info(f"⚠️ Using {len(X):,} rows for training (downsampled from {len(df):,}).")
        else:
            st.success("✅ Dataset is small enough for efficient processing. No subsampling needed.")


    # Update stats
    with stats_placeholder.container():
        st.metric("Rows", f"{len(X):,}")
        st.metric("Features", f"{X.shape[1]}")
        if y is not None:
            anomaly_rate = (np.sum(y) / len(y) * 100)
            st.metric("Anomaly Rate", f"{anomaly_rate:.2f}%")
    
    # Data Exploration Expanders
    with st.expander("📋 Data Preview & Statistics", expanded=False):
        t1, t2, t3 = st.tabs(["Sample Data", "Statistics", "Missing Values"])
        with t1:
            st.dataframe(df_preview.head(10), width="stretch")
        with t2:
            st.dataframe(df_preview.describe(), width="stretch")
        with t3:
            missing = df_preview.isnull().sum()
            if missing.sum() > 0:
                st.bar_chart(missing[missing > 0])
            else:
                st.success("No missing values.")


        
    # Preprocessing Configuration
    st.subheader("⚙️ Pipeline Configuration")
    
    with st.expander("🔧 Feature Engineering", expanded=False):
        use_feature_eng = st.checkbox("Enable feature engineering", value=False)
        eng_options = []
        pca_components = None
        if use_feature_eng:
            eng_options = st.multiselect("Transformations", ["Polynomial Features (degree 2)", "PCA Dimensionality Reduction"])
            if "PCA Dimensionality Reduction" in eng_options:
                pca_components = st.slider("PCA components", 5, min(50, X.shape[1]), min(X.shape[1],10))

    c_mode, c_opts = st.columns([1, 2])
    with c_mode:
        mode = st.radio("Pipeline mode", ["Dense (AutoPreprocessor)", "Sparse (OHE→SVD)", "Minimal (for binary/tree methods)"], index=0)
        use_sparse_mode = (mode.startswith("Sparse"))
        use_minimal_mode = (mode.startswith("Minimal"))

    with c_opts:
        if use_minimal_mode:
            c1, c2 = st.columns(2)
            use_impute = c1.checkbox("Impute missing (median)", value=False)
            cast32 = c2.checkbox("Use float32", value=False)
            svd_components, max_ohe_card, qt_opt, use_power, use_scale = 10, 10, None, False, False
        elif use_sparse_mode:
            c1, c2, c3 = st.columns(3)
            svd_components = c1.slider("SVD components", 10, 300, 100)
            max_ohe_card = c2.number_input("Max OHE cardinality", 10, 500, 50)
            qt_opt = c3.selectbox("Quantile transform", ["none", "normal", "uniform"], index=0)
            qt_opt = None if qt_opt == "none" else qt_opt
            use_power = False
            use_impute, use_scale, cast32 = True, True, False
        else:
            c1, c2, c3 = st.columns(3)
            use_impute = c1.checkbox("Impute missing", value=True)
            use_scale = c2.checkbox("Robust scale", value=True)
            cast32 = c3.checkbox("Use float32", value=False)
            svd_components, max_ohe_card, qt_opt, use_power = 100, 50, None, False

    # Methods Selection
    st.subheader("🤖 Algorithms")
    method_names = list(REGISTRY.keys())
    picked = st.multiselect("Select algorithms", method_names, default=["IsolationForest", "LOF"])
    
    # Ensemble option
    use_ensemble = False
    if len(picked) >= 2:
        use_ensemble = st.checkbox(
            f"🔗 Also run Ensemble (combines {len(picked)} selected methods)", 
            value=False,
            help="Creates a MeanPercentileEnsemble that combines the scores of all selected methods for more robust predictions."
        )
    
    params = {}
    if picked:
        with st.expander("Advanced Algorithm Parameters"):
            for m in picked:
                st.caption(f"**{m}**")
                defaults = DEFAULT_KWARGS.get(m, {})
                p = {}
                cols = st.columns(len(defaults)) if defaults else [st.empty()]
                for idx, (k, v) in enumerate(defaults.items()):
                    with cols[idx % 3]: # wrap around
                        if isinstance(v, bool):
                            p[k] = st.checkbox(f"{k}", value=v, key=f"{m}_{k}")
                        elif isinstance(v, (int, float)):
                            p[k] = st.number_input(f"{k}", value=v, key=f"{m}_{k}")
                        else:
                            p[k] = app_data._parse_literal(st.text_input(f"{k}", value=str(v), key=f"{m}_{k}"))
                params[m] = p

    # Eval Settings
    st.subheader("📊 Evaluation")
    c1, c2, c3 = st.columns(3)
    run_cv = c1.checkbox("Use CV (requires labels)", value=True)
    n_splits = c2.number_input("KFold splits", 2, 20, 5)
    random_state = c3.number_input("Random state", value=42)

    if cast32:
        X = X.astype(np.float32)

# -----------------------------------------------------------------------------
# TAB 2: Analysis
# -----------------------------------------------------------------------------
with tab_analysis:
    if st.button("🚀 Run Evaluation", type="primary", use_container_width=True):
        progress = st.progress(0)
        status = st.empty()
        results = []
        scores_by_method = {}
        
        total_steps = (n_splits if (run_cv and y is not None) else 1) * max(1, len(picked))
        progress_state = {"done": 0}
        
        def step(n=1):
            progress_state["done"] += n
            progress.progress(min(1.0, progress_state["done"] / max(1, total_steps)))

        import time
        # Feature Engineering Application
        X_processed = X.copy()

        X_df_processed = X_df.copy()
        
        if use_feature_eng and eng_options:
            status.info("🔧 Applying feature engineering...")
            if "Polynomial Features (degree 2)" in eng_options:
                poly = PolynomialFeatures(degree=2, interaction_only=False, include_bias=False)
                X_processed = poly.fit_transform(X_processed)
            if "PCA Dimensionality Reduction" in eng_options and pca_components:
                pca = PCA(n_components=pca_components)
                X_processed = pca.fit_transform(X_processed if len(X_processed.shape) == 2 else X)
            X = X_processed # Update main X reference

        # Loop methods
        for m in picked:
            status.info(f"🔄 Evaluating {m} ...")
            base_factory = (lambda m=m: REGISTRY[m](**params.get(m, {})))
            is_supervised = _method_requires_y(m)
            
            # Construct Factory
            if is_supervised:
                steps = []
                if use_impute: steps.append(("imputer", SimpleImputer(strategy="median")))
                if use_scale: steps.append(("scaler", RobustScaler()))
                steps.append(("cast32", FunctionTransformer(lambda x: np.ascontiguousarray(x, dtype=np.float32), validate=False)))
                factory = lambda m=m, steps=steps: Pipeline(steps + [("model", REGISTRY[m](**params.get(m, {})))])
                X_for_cv = X_processed
            elif use_minimal_mode:
                factory = lambda m=m: base_factory()
                X_for_cv = X_df_processed if not use_feature_eng else X_processed
            elif use_sparse_mode:
                factory = lambda m=m: SparseAutoPreprocessor(
                    base_factory, max_ohe_card=max_ohe_card, use_quantile=qt_opt, 
                    use_power=use_power, svd_components=svd_components, clip_q=(0.001, 0.999), robust_scale=True
                )
                X_for_cv = X_df_processed
            else:
                factory = lambda m=m: AutoPreprocessor(
                    base_factory, impute=use_impute, scale=use_scale, 
                    clip_quantiles=(0.001, 0.999), auto_log1p=True, quantile=None, power=False
                )
                X_for_cv = X_processed.astype(np.float32) if cast32 else X_processed

            # Run Fit/Predict
            try:
                if run_cv and y is not None:
                    t0 = time.perf_counter()
                    mean_ap, aps = cv_average_precision(
                        factory, X_for_cv, y, n_splits=n_splits, 
                        random_state=random_state, progress_cb=lambda inc=1: step(inc)
                    )
                    t_cv = time.perf_counter() - t0
                    
                    # Train final model on all data for scoring/vis
                    t0 = time.perf_counter()
                    model = factory()
                    _fit_model(model, X_for_cv, y, m)
                    s = app_metrics._orient_scores(y, model.score(X_for_cv))
                    t_fit = time.perf_counter() - t0
                    
                    scores_by_method[m] = s
                    
                    metric_payload = app_metrics.compute_all_metrics(y, s)
                    results.append({
                        "method": m, "AP_mean": mean_ap, "AP_std": np.std(aps), 
                        "AP_folds": aps, "Time(s)": t_fit + (t_cv/n_splits), **metric_payload
                    })
                else:
                    t0 = time.perf_counter()
                    model = factory()
                    _fit_model(model, X_for_cv, y, m)
                    s = app_metrics._orient_scores(y, model.score(X_for_cv))
                    t_fit = time.perf_counter() - t0

                    scores_by_method[m] = s
                    
                    metric_payload = app_metrics.compute_all_metrics(y, s) if y is not None else {}
                    results.append({
                        "method": m, 
                        "AP_mean": average_precision(y, s) if y is not None else float("nan"),
                        "AP_std": 0.0, "AP_folds": [], "Time(s)": t_fit, **metric_payload
                    })
                    step(1)
            except Exception as e:
                st.warning(f"⚠️ {m} failed: {e}")
                scores_by_method[m] = None
        
        # Ensemble Evaluation (if enabled and multiple methods selected)
        if use_ensemble and len(picked) >= 2:
            status.info(f"🔗 Evaluating Ensemble ({len(picked)} methods)...")
            ensemble_name = f"Ensemble({'+'.join(picked[:3])}{'...' if len(picked) > 3 else ''})"
            
            try:
                # Create ensemble factory
                ensemble_factory = ENSEMBLE_FACTORY(picked)
                
                # Wrap with preprocessing if needed
                if use_minimal_mode:
                    factory = ensemble_factory
                    X_for_cv = X_df_processed if not use_feature_eng else X_processed
                elif use_sparse_mode:
                    factory = lambda: SparseAutoPreprocessor(
                        ensemble_factory, max_ohe_card=max_ohe_card, use_quantile=qt_opt,
                        use_power=use_power, svd_components=svd_components, clip_q=(0.001, 0.999), robust_scale=True
                    )
                    X_for_cv = X_df_processed
                else:
                    factory = lambda: AutoPreprocessor(
                        ensemble_factory, impute=use_impute, scale=use_scale,
                        clip_quantiles=(0.001, 0.999), auto_log1p=True, quantile=None, power=False
                    )
                    X_for_cv = X_processed.astype(np.float32) if cast32 else X_processed
                
                if run_cv and y is not None:
                    t0 = time.perf_counter()
                    mean_ap, aps = cv_average_precision(
                        factory, X_for_cv, y, n_splits=n_splits,
                        random_state=random_state, progress_cb=lambda inc=1: step(inc)
                    )
                    t_cv = time.perf_counter() - t0
                    
                    # Train final model
                    t0 = time.perf_counter()
                    model = factory()
                    model.fit(X_for_cv, y)
                    s = app_metrics._orient_scores(y, model.score(X_for_cv))
                    t_fit = time.perf_counter() - t0
                    
                    scores_by_method[ensemble_name] = s
                    
                    metric_payload = app_metrics.compute_all_metrics(y, s)
                    results.append({
                        "method": ensemble_name, "AP_mean": mean_ap, "AP_std": np.std(aps),
                        "AP_folds": aps, "Time(s)": t_fit + (t_cv/n_splits), **metric_payload
                    })
                else:
                    t0 = time.perf_counter()
                    model = factory()
                    model.fit(X_for_cv)
                    s = app_metrics._orient_scores(y, model.score(X_for_cv))
                    t_fit = time.perf_counter() - t0
                    
                    scores_by_method[ensemble_name] = s
                    
                    metric_payload = app_metrics.compute_all_metrics(y, s) if y is not None else {}
                    results.append({
                        "method": ensemble_name,
                        "AP_mean": average_precision(y, s) if y is not None else float("nan"),
                        "AP_std": 0.0, "AP_folds": [], "Time(s)": t_fit, **metric_payload
                    })
                    step(1)
                    
            except Exception as e:
                st.warning(f"⚠️ Ensemble failed: {e}")
                scores_by_method[ensemble_name] = None
        
        status.success("✅ Evaluation complete!")
        progress.progress(1.0)
        
        # Store results in session state to persist between tab switches
        st.session_state["results"] = results
        st.session_state["scores"] = scores_by_method
        st.session_state["y_true"] = y
        st.session_state["X_vis"] = X_for_cv # storing the last X used (approximation)
        st.session_state["feature_names"] = X_if.columns if hasattr(X, "columns") else None # Placeholder

    # Render Results if available
    if "results" in st.session_state:
        res_df = pd.DataFrame(st.session_state["results"])
        y_ref = st.session_state.get("y_true")
        
        if not res_df.empty:
            if y_ref is not None:
                st.subheader("🏆 Multi-metric Leaderboard")
                
                # Dynamic column selection
                candidate_cols = [c for c in res_df.columns if c not in {"method", "AP_folds"} and res_df[c].notna().any()]
                # Default Sort: AP_mean > AUC_PR > First Column
                sort_metric = next((c for c in ["AP_mean", "AUC_PR", "AUC_ROC"] if c in candidate_cols), candidate_cols[0])
                
                # Dynamic sort order: Ascending for Time, Descending for others
                ascending = ("Time" in sort_metric)
                res_df_sorted = res_df.sort_values(sort_metric, ascending=ascending)
                
                # Separate metrics for highlighting
                # AP_std should be lower (more stable), Time should be lower (faster)
                lower_is_better = [c for c in candidate_cols if "Time" in c or c == "AP_std"]
                higher_is_better = [c for c in candidate_cols if c != "method" and c not in lower_is_better]

                st.dataframe(
                    res_df_sorted.style
                    .highlight_max(subset=higher_is_better, color="rgba(0, 255, 0, 0.2)")
                    .highlight_min(subset=lower_is_better, color="rgba(0, 255, 0, 0.2)")
                    .format(lambda v: f"{v:.4f}" if isinstance(v, float) else v),
                    width="stretch"
                )
                
                # Metric explanations
                with st.expander("📖 Metric Definitions"):
                    st.markdown("""
| Metric | Range | Description |
|--------|-------|-------------|
| **AP_mean** | 0-1 | **Average Precision** (mean across CV folds). Area under the PR curve. Higher = better ranking of anomalies. |
| **AP_std** | 0+ | Standard deviation of AP across folds. Lower = more stable/consistent model. |
| **AUC_PR** | 0-1 | **Area Under Precision-Recall Curve**. Same as AP but computed on full data. Preferred for imbalanced datasets. |
| **AUC_ROC** | 0-1 | **Area Under ROC Curve**. Measures discrimination ability. 0.5 = random, 1.0 = perfect. |
| **F1** | 0-1 | **F1 Score**. Harmonic mean of Precision and Recall. Balances both metrics. |
| **Precision** | 0-1 | Of all predicted anomalies, what fraction are truly anomalies? High = few false alarms. |
| **Recall** | 0-1 | Of all true anomalies, what fraction did we detect? High = few missed anomalies. |
| **MCC** | -1 to 1 | **Matthews Correlation Coefficient**. Balanced metric even with imbalanced classes. 0 = random, 1 = perfect. |
| **BalancedAcc** | 0-1 | **Balanced Accuracy**. Average of recall for each class. Handles imbalance better than accuracy. |
| **Time(s)** | 0+ | Execution time in seconds. Lower = faster model. |
                    """)
                
                # Recall Plot
                total_anoms = int(np.sum(y_ref))
                st.caption(f"Total Anomalies: {total_anoms} ({total_anoms/len(y_ref):.1%})")
                
                if total_anoms > 0:
                    recall_rows = []
                    for m, s in st.session_state["scores"].items():
                        if s is None: continue
                        rec, found = app_metrics._recall_at_k(y_ref, s, total_anoms)
                        recall_rows.append({"Method": m, "Recall@TotalAnomalies": rec*100})
                    
                    if recall_rows:
                        fig = px.bar(pd.DataFrame(recall_rows), x="Method", y="Recall@TotalAnomalies", 
                                     title="Recall at Top-N (N=Total Anomalies)")
                        st.plotly_chart(fig, width="stretch")
                
                # NEW: Method Correlation Heatmap
                scores_map = st.session_state.get("scores", {})
                if len(scores_map) > 1:
                    st.subheader("🔗 Method Correlation")
                    # Build correlation matrix
                    corr_data = {m: s for m, s in scores_map.items() if s is not None}
                    if corr_data:
                        corr_df = pd.DataFrame(corr_data).corr()
                        fig_corr = px.imshow(corr_df, text_auto=".2f", aspect="auto", color_continuous_scale="RdBu_r")
                        st.plotly_chart(fig_corr, width="stretch")

            else:
                st.subheader("🏆 Leaderboard (No Labels)")
                st.write("Results based on internal scoring (limited metrics available without labels).")
                st.dataframe(res_df)

# -----------------------------------------------------------------------------
# TAB 3: Deep Dives
# -----------------------------------------------------------------------------
with tab_deep_dive:
    if "results" not in st.session_state:
        st.info("Run evaluation first to inspect models.")
    else:
        scores_map = st.session_state["scores"]
        y_ref = st.session_state.get("y_true")
        valid_methods = [m for m, s in scores_map.items() if s is not None]
        
        if not valid_methods:
            st.warning("No valid results to inspect.")
        else:
            selected_method = st.selectbox("Select Method to Inspect", valid_methods)
            scores = scores_map[selected_method]
            
            t_pr, t_cm, t_dist, t_viz, t_shap = st.tabs(["PR Curve", "Confusion Matrix", "Score Dist", "2D Projection", "Explainability (SHAP)"])
            
            with t_pr:
                if y_ref is not None:
                    ap = average_precision(y_ref, scores)
                    prec, rec, _ = precision_recall_points(y_ref, scores)
                    fig = px.area(x=rec, y=prec, labels={"x": "Recall", "y": "Precision"},
                                  title=f"PR Curve (AP={ap:.3f})")
                    fig.update_yaxes(range=[0, 1])
                    st.plotly_chart(fig, width="stretch")
                else:
                    st.info("Requires labels.")
            
            with t_cm:
                if y_ref is not None:
                    percentile = st.slider("Threshold Percentile", 50.0, 99.9, 95.0, 0.1)
                    thr = np.percentile(scores, percentile)
                    cm = app_metrics._confusion_matrix_at_k(y_ref, scores, int(len(scores)*(100-percentile)/100))
                    
                    fig_cm = go.Figure(data=go.Heatmap(
                        z=[[cm["TN"], cm["FP"]], [cm["FN"], cm["TP"]]],
                        x=["Pred Normal", "Pred Anomaly"],
                        y=["Act Normal", "Act Anomaly"],
                        text=[[cm["TN"], cm["FP"]], [cm["FN"], cm["TP"]]],
                        texttemplate="%{text}", colorscale="Blues"
                    ))
                    st.plotly_chart(fig_cm, width="stretch")
                else:
                    st.info("Requires labels.")
            
            with t_dist:
                df_hist = pd.DataFrame({"Score": scores})
                if y_ref is not None:
                    df_hist["Label"] = ["Anomaly" if x==1 else "Normal" for x in y_ref]
                    fig = px.histogram(df_hist, x="Score", color="Label", nbins=50, marginal="box", barmode="overlay")
                else:
                    fig = px.histogram(df_hist, x="Score", nbins=50, marginal="box")
                st.plotly_chart(fig, width="stretch")

            with t_viz:
                proj_type = st.radio("Projection", ["PCA", "t-SNE"], horizontal=True)
                if st.button("Generate Projection"):
                    with st.spinner("Projecting..."):
                        # Use cached X if possible, else we might not have it easily accessible in this scope 
                        # without persisting X in session state relative to the method input.
                        # For now, we warn if X is lost, but we stored X_vis in session state roughly.
                        X_curr = st.session_state.get("X_vis") 
                        if X_curr is None:
                            st.error("Data lost from session. Please Re-run evaluation.")
                        else:
                            # Downsample for vis
                            if len(X_curr) > 2000:
                                idx = np.random.choice(len(X_curr), 2000, replace=False)
                                X_v = X_curr[idx] if isinstance(X_curr, np.ndarray) else X_curr.iloc[idx]
                                s_v = scores[idx]
                                y_v = y_ref[idx] if y_ref is not None else None
                            else:
                                X_v = X_curr
                                s_v = scores
                                y_v = y_ref
                            
                            if proj_type == "PCA":
                                P = PCA(n_components=2).fit_transform(X_v)
                            else:
                                P = TSNE(n_components=2).fit_transform(X_v)
                                
                            df_p = pd.DataFrame(P, columns=["D1", "D2"])
                            df_p["Score"] = s_v
                            if y_v is not None:
                                df_p["Label"] = ["Anomaly" if i==1 else "Normal" for i in y_v]
                                fig = px.scatter(df_p, x="D1", y="D2", color="Score", symbol="Label", title=f"{proj_type} Projection")
                            else:
                                fig = px.scatter(df_p, x="D1", y="D2", color="Score", title=f"{proj_type} Projection")
                            
                            st.plotly_chart(fig, width="stretch")

            with t_shap:
                st.info("SHAP (SHapley Additive exPlanations) explains the output of any machine learning model.")
                if st.button("Values are calculated via re-training. Click to Compute SHAP"):
                    with st.spinner("Training model & computing SHAP values..."):
                        # Re-construct pipeline for selected method
                        # We need to reuse the factory logic. 
                        # Ideally this should be centralized, but duplicating for now to avoid massive refactor
                        m = selected_method
                        
                        # Re-create pipeline steps (simplified logic matching Analysis tab)
                        # Note: We use the *current* X_vis from session state which is pre-processed/subsampled
                        X_shap = st.session_state.get("X_vis")
                        if X_shap is None:
                            st.error("Data context lost. Please run evaluation first.")
                            st.stop()
                            
                        # Downsample for SHAP (it's slow)
                        if len(X_shap) > 500:
                            st.caption("ℹ️ Downsampling to 500 samples for SHAP speed.")
                            X_shap_sub = X_shap.sample(500) if hasattr(X_shap, "sample") else X_shap[:500]
                        else:
                            X_shap_sub = X_shap

                        # Instantiate model
                        model_kwargs = DEFAULT_KWARGS.get(m, {})
                        # Ideally parameters should be retrieved from session state or UI, 
                        # but we might simply respect defaults or what was configured. 
                        # Limitation: We don't have the exact params used in the run stored easily unless we expand session state.
                        # For now, using defaults + UI overrides if active? 
                        # We will just grab from REGISTRY defaults or what is in params dict if accessible.
                        # We don't have access to 'params' variable from Analysis tab scope here.
                        # Fallback: Use defaults.
                        
                        base_model = REGISTRY[m](**model_kwargs) 
                        
                        # We need to wrap it to fit? Or just fit directly if it handles X.
                        # Most wrapper methods in REGISTRY handle 2D numpy or DF.
                        try:
                            base_model.fit(X_shap_sub)
                            
                            # Create Explainer
                            # Use a generic explainer or KernelExplainer. 
                            # Since most are PyOD or custom, we use the decision_function.
                            
                            # Wrapper for prediction to make SHAP happy (expects numpy -> numpy)
                            def predict_wrapper(data):
                                return base_model.score(data)
                                
                            explainer = shap.Explainer(predict_wrapper, X_shap_sub)
                            shap_values = explainer(X_shap_sub)
                            
                            # Plot
                            st.subheader("Feature Importance (Summary Plot)")
                            fig_shap, ax = plt.subplots()
                            shap.summary_plot(shap_values, X_shap_sub, plot_type="bar", show=False)
                            st.pyplot(fig_shap)
                            
                            st.subheader("Detailed Summary Plot")
                            fig_shap2, ax2 = plt.subplots()
                            shap.summary_plot(shap_values, X_shap_sub, show=False)
                            st.pyplot(fig_shap2)
                            
                        except Exception as e:
                            st.error(f"Failed to compute SHAP: {e}")
