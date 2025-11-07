"""
=================================================================================
COMPREHENSIVE STRUCTURAL EQUATION MODELING (SEM) ANALYSIS - CORRECTED VERSION
=================================================================================
A research-grade implementation for advanced SEM analysis using rpy2 and lavaan

FIXES APPLIED:
- Corrected fit.measures parameter syntax for lavaan
- Fixed output directory paths
- Improved error handling
- Enhanced data generation with validation

Requirements:
- R with lavaan package installed
- Python packages: rpy2, numpy, pandas, matplotlib, seaborn

Author: Research Analytics Team
Date: 2025 (Corrected Version)
=================================================================================
"""

import os
import warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
from pathlib import Path

# Define output path in current directory
OUTPUT_DIR = Path(__file__).parent / "sem_outputs"
OUTPUT_DIR.mkdir(exist_ok=True)

print(f"Output directory created/verified: {OUTPUT_DIR.absolute()}")

# Configure plotting style
plt.style.use("seaborn-v0_8-darkgrid")
sns.set_palette("husl")
warnings.filterwarnings("ignore")

# ============================================================================
# R ENVIRONMENT SETUP
# ============================================================================

print("\n" + "=" * 80)
print("SETTING UP R ENVIRONMENT")
print("=" * 80)

try:
    import rpy2.situation

    right_path = rpy2.situation.get_r_home()
    os.environ["R_HOME"] = right_path
    print(f"✓ R Home: {right_path}")
except Exception as e:
    print(f"✗ Error setting R_HOME: {e}")
    raise

import rpy2.robjects as ro
from rpy2.robjects import pandas2ri, default_converter
from rpy2.robjects.packages import importr
from rpy2.robjects.conversion import localconverter
from rpy2.robjects.vectors import FloatVector

pandas2ri.activate()

# Import R packages
print("\nLoading R packages...")
try:
    lavaan = importr("lavaan")
    print("✓ lavaan loaded")
    base = importr("base")
    print("✓ base loaded")
    stats = importr("stats")
    print("✓ stats loaded")
except Exception as e:
    print(f"✗ Error loading R packages: {e}")
    print("Please install lavaan in R: install.packages('lavaan')")
    raise

# Try to import semPlot for visualization (optional)
try:
    semplot = importr("semPlot")
    SEMPLOT_AVAILABLE = True
    print("✓ semPlot loaded (optional)")
except:
    SEMPLOT_AVAILABLE = False
    print("Note: semPlot not available (optional)")

# Try to import psych for reliability (optional)
try:
    psych = importr("psych")
    PSYCH_AVAILABLE = True
    print("✓ psych loaded (optional)")
except:
    PSYCH_AVAILABLE = False
    print("Note: psych not available (optional)")


# ============================================================================
# DATA GENERATION WITH REALISTIC RESEARCH STRUCTURE
# ============================================================================


class SEMDataGenerator:
    """Generate realistic synthetic data for SEM analysis."""

    def __init__(self, n_samples=500, seed=42):
        self.n = n_samples
        self.seed = seed
        np.random.seed(seed)
        print(f"\n{'='*80}")
        print(f"DATA GENERATOR INITIALIZED")
        print(f"{'='*80}")
        print(f"Sample size: {n_samples}")
        print(f"Random seed: {seed}")

    def generate_research_data(self):
        """
        Generate data for a comprehensive research model:
        - Service Quality (4 indicators)
        - Customer Satisfaction (4 indicators)
        - Trust (3 indicators)
        - Loyalty (3 indicators)
        - Word of Mouth (3 indicators)

        Theoretical model:
        Service Quality → Satisfaction → Trust → Loyalty → Word of Mouth
        Service Quality → Trust (direct path)
        Satisfaction → Loyalty (direct path)
        """

        print(f"\nGenerating synthetic data (n={self.n})...")

        # Exogenous latent variable
        service_quality = np.random.normal(0, 1, self.n)

        # Endogenous latent variables with structural paths
        satisfaction = 0.65 * service_quality + np.random.normal(0, 0.6, self.n)
        trust = (
            0.40 * service_quality
            + 0.45 * satisfaction
            + np.random.normal(0, 0.5, self.n)
        )
        loyalty = 0.30 * satisfaction + 0.50 * trust + np.random.normal(0, 0.4, self.n)
        wom = 0.70 * loyalty + np.random.normal(0, 0.4, self.n)

        # Generate observed indicators with different factor loadings
        data = pd.DataFrame(
            {
                # Service Quality indicators (SQ)
                "sq1": service_quality * 0.85 + np.random.normal(0, 0.4, self.n),
                "sq2": service_quality * 0.80 + np.random.normal(0, 0.5, self.n),
                "sq3": service_quality * 0.78 + np.random.normal(0, 0.5, self.n),
                "sq4": service_quality * 0.82 + np.random.normal(0, 0.45, self.n),
                # Customer Satisfaction indicators (SAT)
                "sat1": satisfaction * 0.88 + np.random.normal(0, 0.4, self.n),
                "sat2": satisfaction * 0.85 + np.random.normal(0, 0.45, self.n),
                "sat3": satisfaction * 0.80 + np.random.normal(0, 0.5, self.n),
                "sat4": satisfaction * 0.83 + np.random.normal(0, 0.48, self.n),
                # Trust indicators (TRUST)
                "trust1": trust * 0.86 + np.random.normal(0, 0.42, self.n),
                "trust2": trust * 0.84 + np.random.normal(0, 0.45, self.n),
                "trust3": trust * 0.82 + np.random.normal(0, 0.47, self.n),
                # Loyalty indicators (LOY)
                "loy1": loyalty * 0.87 + np.random.normal(0, 0.4, self.n),
                "loy2": loyalty * 0.85 + np.random.normal(0, 0.43, self.n),
                "loy3": loyalty * 0.83 + np.random.normal(0, 0.45, self.n),
                # Word of Mouth indicators (WOM)
                "wom1": wom * 0.89 + np.random.normal(0, 0.38, self.n),
                "wom2": wom * 0.86 + np.random.normal(0, 0.42, self.n),
                "wom3": wom * 0.84 + np.random.normal(0, 0.44, self.n),
            }
        )

        # Scale to 1-7 Likert scale (common in research)
        data = data.apply(
            lambda x: np.round((x - x.min()) / (x.max() - x.min()) * 6 + 1)
        )
        data = data.clip(1, 7)  # Ensure range

        print(f"✓ Data generated successfully")
        print(f"  Shape: {data.shape}")
        print(f"  Variables: {list(data.columns)}")
        print(f"  Scale range: {data.min().min():.0f} - {data.max().max():.0f}")

        # Print sample statistics
        print(f"\n  Sample Statistics:")
        print(f"  Mean range: {data.mean().min():.2f} - {data.mean().max():.2f}")
        print(f"  Std range: {data.std().min():.2f} - {data.std().max():.2f}")

        return data


# ============================================================================
# SEM MODEL SPECIFICATIONS
# ============================================================================


class SEMModels:
    """Repository of SEM model specifications."""

    @staticmethod
    def measurement_model():
        """CFA model for measurement validation."""
        return """
        # Measurement Model (CFA)
        ServiceQuality =~ sq1 + sq2 + sq3 + sq4
        Satisfaction =~ sat1 + sat2 + sat3 + sat4
        Trust =~ trust1 + trust2 + trust3
        Loyalty =~ loy1 + loy2 + loy3
        WOM =~ wom1 + wom2 + wom3
        """

    @staticmethod
    def full_structural_model():
        """Full structural model with all paths."""
        return """
        # Measurement Model
        ServiceQuality =~ sq1 + sq2 + sq3 + sq4
        Satisfaction =~ sat1 + sat2 + sat3 + sat4
        Trust =~ trust1 + trust2 + trust3
        Loyalty =~ loy1 + loy2 + loy3
        WOM =~ wom1 + wom2 + wom3
        
        # Structural Model
        Satisfaction ~ sq_sat*ServiceQuality
        Trust ~ sq_trust*ServiceQuality + sat_trust*Satisfaction
        Loyalty ~ sat_loy*Satisfaction + trust_loy*Trust
        WOM ~ loy_wom*Loyalty
        
        # Indirect effects (for mediation analysis)
        indirect_sq_wom := sq_sat * sat_trust * trust_loy * loy_wom
        indirect_sat_wom := sat_trust * trust_loy * loy_wom
        total_effect := sq_sat * sat_trust * trust_loy * loy_wom + sq_trust * trust_loy * loy_wom
        """

    @staticmethod
    def alternative_model():
        """Alternative model for comparison (direct paths)."""
        return """
        # Measurement Model
        ServiceQuality =~ sq1 + sq2 + sq3 + sq4
        Satisfaction =~ sat1 + sat2 + sat3 + sat4
        Trust =~ trust1 + trust2 + trust3
        Loyalty =~ loy1 + loy2 + loy3
        WOM =~ wom1 + wom2 + wom3
        
        # Alternative Structural Model (with additional direct paths)
        Satisfaction ~ ServiceQuality
        Trust ~ ServiceQuality + Satisfaction
        Loyalty ~ Satisfaction + Trust + ServiceQuality
        WOM ~ Loyalty + Trust
        """


# ============================================================================
# SEM ANALYZER CLASS
# ============================================================================


class SEMAnalyzer:
    """Comprehensive SEM analysis with advanced features."""

    def __init__(self, data):
        self.data = data
        self.data_r = pandas2ri.py2rpy(data)
        self.results = {}
        print(f"\n{'='*80}")
        print("SEM ANALYZER INITIALIZED")
        print(f"{'='*80}")
        print(f"Data shape: {data.shape}")
        print(f"Missing values: {data.isnull().sum().sum()}")

    def run_cfa(self):
        """Run Confirmatory Factor Analysis."""
        print("\n" + "=" * 80)
        print("CONFIRMATORY FACTOR ANALYSIS (CFA)")
        print("=" * 80)

        model = SEMModels.measurement_model()

        try:
            fit = lavaan.cfa(
                model,
                data=self.data_r,
                estimator="MLR",  # Robust ML estimation
                missing="fiml",  # Full Information ML for missing data
            )

            self.results["cfa"] = fit

            # Print summary - CORRECTED: use fit.measures not fit_measures
            print("\nCFA Model Summary:")
            summary_result = lavaan.summary(
                fit, standardized=True, **{"fit.measures": True, "rsquare": True}
            )
            print(summary_result)
            print("✓ CFA completed successfully")

            return fit

        except Exception as e:
            print(f"✗ Error in CFA: {e}")
            raise

    def run_sem(self, model_type="full"):
        """Run full structural equation model."""
        print("\n" + "=" * 80)
        print(f"STRUCTURAL EQUATION MODEL ({model_type.upper()})")
        print("=" * 80)

        if model_type == "full":
            model = SEMModels.full_structural_model()
        elif model_type == "alternative":
            model = SEMModels.alternative_model()
        else:
            raise ValueError("model_type must be 'full' or 'alternative'")

        try:
            fit = lavaan.sem(
                model, data=self.data_r, estimator="MLR", missing="fiml", se="robust"
            )

            self.results[f"sem_{model_type}"] = fit

            # Print summary - CORRECTED
            print(f"\n{model_type.upper()} SEM Model Summary:")
            summary_result = lavaan.summary(
                fit, standardized=True, **{"fit.measures": True, "rsquare": True}
            )
            print(summary_result)
            print(f"✓ {model_type.upper()} SEM completed successfully")

            return fit

        except Exception as e:
            print(f"✗ Error in SEM ({model_type}): {e}")
            raise

    def bootstrap_analysis(self, fit, n_bootstrap=1000):
        """Perform bootstrap analysis for robust inference."""
        print("\n" + "=" * 80)
        print(f"BOOTSTRAP ANALYSIS (n={n_bootstrap})")
        print("=" * 80)

        try:
            # Get model syntax from fitted object
            model_syntax = str(lavaan.lavInspect(fit, "model"))

            fit_boot = lavaan.sem(
                model_syntax,
                data=self.data_r,
                estimator="MLR",
                se="bootstrap",
                bootstrap=n_bootstrap,
            )

            print("\nBootstrap Results:")
            print(lavaan.summary(fit_boot, standardized=True))

            # Get bootstrap confidence intervals
            ci = lavaan.parameterEstimates(fit_boot, **{"boot.ci.type": "perc"})
            ci_df = self._r_to_pandas(ci)

            print("✓ Bootstrap analysis completed successfully")

            return fit_boot, ci_df

        except Exception as e:
            print(f"✗ Error in bootstrap analysis: {e}")
            raise

    def get_fit_measures(self, fit):
        """Extract and interpret model fit measures."""
        try:
            with localconverter(default_converter):
                fit_measures_r = lavaan.fitMeasures(fit)

            measure_names = list(fit_measures_r.names)
            measure_values = list(fit_measures_r)

            fit_df = pd.DataFrame(
                {"Measure": measure_names, "Value": measure_values}
            ).set_index("Measure")

            # Key fit indices
            key_indices = [
                "chisq",
                "df",
                "pvalue",
                "cfi",
                "tli",
                "rmsea",
                "rmsea.ci.lower",
                "rmsea.ci.upper",
                "srmr",
                "aic",
                "bic",
            ]

            key_fit = fit_df.loc[[idx for idx in key_indices if idx in fit_df.index], :]

            return key_fit

        except Exception as e:
            print(f"✗ Error extracting fit measures: {e}")
            raise

    def interpret_fit(self, fit_df):
        """Provide interpretation of fit indices."""
        print("\n" + "=" * 80)
        print("MODEL FIT INTERPRETATION")
        print("=" * 80)

        interpretation = []

        # CFI
        if "cfi" in fit_df.index:
            cfi = fit_df.loc["cfi", "Value"]
            if cfi >= 0.95:
                cfi_interp = "Excellent"
            elif cfi >= 0.90:
                cfi_interp = "Acceptable"
            else:
                cfi_interp = "Poor"
            interpretation.append(f"CFI = {cfi:.3f} ({cfi_interp})")

        # TLI
        if "tli" in fit_df.index:
            tli = fit_df.loc["tli", "Value"]
            if tli >= 0.95:
                tli_interp = "Excellent"
            elif tli >= 0.90:
                tli_interp = "Acceptable"
            else:
                tli_interp = "Poor"
            interpretation.append(f"TLI = {tli:.3f} ({tli_interp})")

        # RMSEA
        if "rmsea" in fit_df.index:
            rmsea = fit_df.loc["rmsea", "Value"]
            if rmsea <= 0.05:
                rmsea_interp = "Excellent"
            elif rmsea <= 0.08:
                rmsea_interp = "Acceptable"
            else:
                rmsea_interp = "Poor"
            interpretation.append(f"RMSEA = {rmsea:.3f} ({rmsea_interp})")

        # SRMR
        if "srmr" in fit_df.index:
            srmr = fit_df.loc["srmr", "Value"]
            if srmr <= 0.05:
                srmr_interp = "Excellent"
            elif srmr <= 0.08:
                srmr_interp = "Acceptable"
            else:
                srmr_interp = "Poor"
            interpretation.append(f"SRMR = {srmr:.3f} ({srmr_interp})")

        print("\nFit Index Interpretations:")
        print("-" * 40)
        for interp in interpretation:
            print(f"  {interp}")

        # Overall assessment
        print("\nCutoff Criteria (Hu & Bentler, 1999):")
        print("  CFI/TLI: ≥ 0.95 (excellent), ≥ 0.90 (acceptable)")
        print("  RMSEA: ≤ 0.05 (excellent), ≤ 0.08 (acceptable)")
        print("  SRMR: ≤ 0.05 (excellent), ≤ 0.08 (acceptable)")

        return interpretation

    def reliability_analysis(self, fit):
        """Calculate reliability measures for each construct."""
        print("\n" + "=" * 80)
        print("RELIABILITY ANALYSIS")
        print("=" * 80)

        try:
            # Get standardized estimates
            std_est = lavaan.standardizedSolution(fit)
            std_df = self._r_to_pandas(std_est)

            # Filter measurement model loadings
            loadings_df = std_df[std_df["op"] == "=~"][["lhs", "rhs", "est.std"]]
            loadings_df.columns = ["Construct", "Indicator", "Loading"]

            constructs = loadings_df["Construct"].unique()

            reliability_results = []

            for construct in constructs:
                construct_loadings = loadings_df[loadings_df["Construct"] == construct][
                    "Loading"
                ].values

                # Composite Reliability (CR)
                sum_loadings = np.sum(construct_loadings)
                sum_loadings_sq = np.sum(construct_loadings**2)
                error_var = len(construct_loadings) - sum_loadings_sq
                cr = (sum_loadings**2) / ((sum_loadings**2) + error_var)

                # Average Variance Extracted (AVE)
                ave = sum_loadings_sq / len(construct_loadings)

                # Cronbach's Alpha approximation
                n = len(construct_loadings)
                avg_loading = np.mean(construct_loadings)
                alpha = (n * avg_loading**2) / (
                    (n * avg_loading**2) + (1 - avg_loading**2)
                )

                reliability_results.append(
                    {
                        "Construct": construct,
                        "N_Items": n,
                        "CR": cr,
                        "AVE": ave,
                        "Alpha": alpha,
                        "Avg_Loading": avg_loading,
                    }
                )

            reliability_df = pd.DataFrame(reliability_results)

            print("\nReliability Metrics:")
            print(reliability_df.to_string(index=False))

            print("\nInterpretation Guidelines:")
            print("  CR (Composite Reliability): > 0.70 (acceptable), > 0.80 (good)")
            print("  AVE (Average Variance Extracted): > 0.50 (acceptable)")
            print("  Cronbach's Alpha: > 0.70 (acceptable), > 0.80 (good)")

            return reliability_df

        except Exception as e:
            print(f"✗ Error in reliability analysis: {e}")
            raise

    def validity_analysis(self, reliability_df):
        """Assess convergent and discriminant validity."""
        print("\n" + "=" * 80)
        print("VALIDITY ANALYSIS")
        print("=" * 80)

        print("\nConvergent Validity (Fornell & Larcker, 1981):")
        print("-" * 50)
        print("Criteria: AVE > 0.50 AND CR > 0.70")
        print()

        for _, row in reliability_df.iterrows():
            ave_ok = row["AVE"] > 0.50
            cr_ok = row["CR"] > 0.70
            status = "✓ PASS" if (ave_ok and cr_ok) else "✗ FAIL"
            print(
                f"{row['Construct']:20} AVE={row['AVE']:.3f} CR={row['CR']:.3f} {status}"
            )

        print("\n\nDiscriminant Validity (Fornell & Larcker criterion):")
        print("-" * 50)
        print(
            "Criteria: Square root of AVE should exceed correlations with other constructs"
        )
        print("\nSquare Root of AVE:")
        for _, row in reliability_df.iterrows():
            sqrt_ave = np.sqrt(row["AVE"])
            print(f"  {row['Construct']:20} √AVE = {sqrt_ave:.3f}")

        print(
            "\nNote: Compare these values with construct correlations from the model output."
        )

    def modification_indices(self, fit, threshold=10):
        """Get modification indices for model improvement."""
        print("\n" + "=" * 80)
        print(f"MODIFICATION INDICES (threshold = {threshold})")
        print("=" * 80)

        try:
            mi = lavaan.modificationIndices(
                fit, **{"sort.": True, "minimum.value": threshold}
            )
            mi_df = self._r_to_pandas(mi)

            if len(mi_df) > 0:
                print(f"\nTop modifications (MI > {threshold}):")
                print(mi_df.head(20).to_string(index=False))
                print("\nNote: High MI suggests adding these paths may improve fit.")
                print("However, changes should be theoretically justified.")
            else:
                print(f"\nNo modification indices exceed threshold of {threshold}.")
                print("Model specification appears adequate.")

            return mi_df

        except Exception as e:
            print(f"✗ Error calculating modification indices: {e}")
            return pd.DataFrame()

    def compare_models(self, fit1, fit2, model1_name="Model 1", model2_name="Model 2"):
        """Compare two nested or non-nested models."""
        print("\n" + "=" * 80)
        print(f"MODEL COMPARISON: {model1_name} vs {model2_name}")
        print("=" * 80)

        try:
            # Get fit measures for both models
            fit1_measures = self.get_fit_measures(fit1)
            fit2_measures = self.get_fit_measures(fit2)

            # Compare key indices
            comparison = pd.DataFrame(
                {
                    model1_name: fit1_measures["Value"],
                    model2_name: fit2_measures["Value"],
                }
            )

            key_indices = ["chisq", "df", "cfi", "tli", "rmsea", "srmr", "aic", "bic"]
            comparison_key = comparison.loc[
                [idx for idx in key_indices if idx in comparison.index], :
            ]

            print("\nFit Comparison:")
            print(comparison_key)

            # AIC/BIC comparison
            if "aic" in comparison.index and "bic" in comparison.index:
                aic_diff = (
                    comparison.loc["aic", model2_name]
                    - comparison.loc["aic", model1_name]
                )
                bic_diff = (
                    comparison.loc["bic", model2_name]
                    - comparison.loc["bic", model1_name]
                )

                print("\n" + "-" * 60)
                print(f"AIC difference: {aic_diff:.2f} (negative favors {model2_name})")
                print(f"BIC difference: {bic_diff:.2f} (negative favors {model2_name})")

                if aic_diff < -10:
                    print(f"\nConclusion: {model2_name} is substantially better (AIC)")
                elif aic_diff > 10:
                    print(f"\nConclusion: {model1_name} is substantially better (AIC)")
                else:
                    print("\nConclusion: Models are approximately equivalent (AIC)")

            return comparison_key

        except Exception as e:
            print(f"✗ Error comparing models: {e}")
            raise

    def extract_path_coefficients(self, fit):
        """Extract and format path coefficients."""
        try:
            params = lavaan.parameterEstimates(fit, standardized=True)
            params_df = self._r_to_pandas(params)

            # Filter structural paths
            paths_df = params_df[params_df["op"] == "~"][
                ["lhs", "rhs", "est", "se", "z", "pvalue", "std.all"]
            ]
            paths_df.columns = [
                "DV",
                "IV",
                "Estimate",
                "SE",
                "z",
                "p-value",
                "Std.Estimate",
            ]

            return paths_df

        except Exception as e:
            print(f"✗ Error extracting path coefficients: {e}")
            raise

    def visualize_results(self, fit, filename="sem_results"):
        """Create comprehensive visualization of results."""
        print("\n" + "=" * 80)
        print("GENERATING VISUALIZATIONS")
        print("=" * 80)

        try:
            # Extract estimates
            params = lavaan.parameterEstimates(fit, standardized=True)
            params_df = self._r_to_pandas(params)

            # Create figure with subplots
            fig = plt.figure(figsize=(16, 10))
            gs = fig.add_gridspec(2, 2, hspace=0.3, wspace=0.3)

            # 1. Factor Loadings
            ax1 = fig.add_subplot(gs[0, 0])
            loadings = params_df[params_df["op"] == "=~"][["lhs", "rhs", "std.all"]]
            if len(loadings) > 0:
                loadings_pivot = loadings.pivot(
                    index="rhs", columns="lhs", values="std.all"
                )
                sns.heatmap(
                    loadings_pivot,
                    annot=True,
                    fmt=".3f",
                    cmap="RdYlGn",
                    center=0.5,
                    vmin=0,
                    vmax=1,
                    ax=ax1,
                    cbar_kws={"label": "Std. Loading"},
                )
                ax1.set_title(
                    "Factor Loadings (Standardized)", fontsize=14, fontweight="bold"
                )
                ax1.set_xlabel("Latent Construct")
                ax1.set_ylabel("Indicator")

            # 2. Path Coefficients
            ax2 = fig.add_subplot(gs[0, 1])
            paths = params_df[params_df["op"] == "~"][
                ["lhs", "rhs", "std.all", "pvalue"]
            ]
            if len(paths) > 0:
                paths["sig"] = paths["pvalue"].apply(
                    lambda p: (
                        "***"
                        if p < 0.001
                        else "**" if p < 0.01 else "*" if p < 0.05 else "ns"
                    )
                )
                paths["label"] = paths["std.all"].apply(lambda x: f"{x:.3f}")

                x = range(len(paths))
                colors = ["green" if p < 0.05 else "gray" for p in paths["pvalue"]]
                bars = ax2.barh(x, paths["std.all"], color=colors, alpha=0.7)
                ax2.set_yticks(x)
                ax2.set_yticklabels(
                    [f"{row['rhs']} → {row['lhs']}" for _, row in paths.iterrows()],
                    fontsize=9,
                )
                ax2.set_xlabel("Standardized Coefficient")
                ax2.set_title(
                    "Structural Path Coefficients", fontsize=14, fontweight="bold"
                )
                ax2.axvline(x=0, color="black", linestyle="--", linewidth=0.8)
                ax2.grid(axis="x", alpha=0.3)

                # Add significance stars
                for i, (_, row) in enumerate(paths.iterrows()):
                    ax2.text(
                        row["std.all"] + 0.02,
                        i,
                        row["sig"],
                        va="center",
                        fontsize=10,
                        fontweight="bold",
                    )

            # 3. R-squared values
            ax3 = fig.add_subplot(gs[1, 0])
            rsq = params_df[params_df["op"] == "r2"][["lhs", "est"]]
            if len(rsq) > 0:
                rsq = rsq.sort_values("est", ascending=True)
                bars = ax3.barh(
                    range(len(rsq)), rsq["est"], color="steelblue", alpha=0.7
                )
                ax3.set_yticks(range(len(rsq)))
                ax3.set_yticklabels(rsq["lhs"])
                ax3.set_xlabel("R² (Variance Explained)")
                ax3.set_title(
                    "Explained Variance by Construct", fontsize=14, fontweight="bold"
                )
                ax3.set_xlim(0, 1)

                # Add value labels
                for i, (_, row) in enumerate(rsq.iterrows()):
                    ax3.text(
                        row["est"] + 0.02,
                        i,
                        f"{row['est']:.3f}",
                        va="center",
                        fontsize=10,
                    )

            # 4. Model Fit Summary
            ax4 = fig.add_subplot(gs[1, 1])
            ax4.axis("off")

            fit_measures = self.get_fit_measures(fit)

            fit_text = "MODEL FIT SUMMARY\n" + "=" * 40 + "\n\n"

            key_indices = {
                "chisq": "Chi-square",
                "df": "Degrees of Freedom",
                "pvalue": "p-value",
                "cfi": "CFI",
                "tli": "TLI",
                "rmsea": "RMSEA",
                "srmr": "SRMR",
                "aic": "AIC",
                "bic": "BIC",
            }

            for idx, label in key_indices.items():
                if idx in fit_measures.index:
                    value = fit_measures.loc[idx, "Value"]
                    if idx in ["cfi", "tli", "rmsea", "srmr"]:
                        fit_text += f"{label:25} {value:8.3f}\n"
                    elif idx == "pvalue":
                        fit_text += f"{label:25} {value:8.4f}\n"
                    elif idx in ["aic", "bic"]:
                        fit_text += f"{label:25} {value:10.2f}\n"
                    else:
                        fit_text += f"{label:25} {value:8.2f}\n"

            fit_text += "\n" + "-" * 40 + "\n"
            fit_text += "Interpretation Guidelines:\n"
            fit_text += "  CFI/TLI: ≥ 0.95 (excellent)\n"
            fit_text += "  RMSEA: ≤ 0.05 (excellent)\n"
            fit_text += "  SRMR: ≤ 0.05 (excellent)\n"

            ax4.text(
                0.1,
                0.95,
                fit_text,
                transform=ax4.transAxes,
                fontsize=11,
                verticalalignment="top",
                fontfamily="monospace",
                bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.3),
            )

            plt.suptitle(
                "Comprehensive SEM Analysis Results",
                fontsize=16,
                fontweight="bold",
                y=0.98,
            )

            # Save figure - CORRECTED path
            output_path = Path(OUTPUT_DIR, f"{filename}.png")
            plt.savefig(output_path, dpi=300, bbox_inches="tight")
            print(f"✓ Visualization saved: {output_path}")
            plt.close()

            return fig

        except Exception as e:
            print(f"✗ Error generating visualization: {e}")
            import traceback

            traceback.print_exc()
            return None

    def generate_report(self, filename="sem_analysis_report"):
        """Generate comprehensive analysis report."""
        print("\n" + "=" * 80)
        print("GENERATING COMPREHENSIVE REPORT")
        print("=" * 80)

        try:
            timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

            report = f"""
{'='*80}
COMPREHENSIVE STRUCTURAL EQUATION MODELING (SEM) ANALYSIS REPORT
{'='*80}

Generated: {timestamp}
Sample Size: {len(self.data)}
Analysis Method: Maximum Likelihood with Robust Standard Errors (MLR)

{'='*80}
1. DESCRIPTIVE STATISTICS
{'='*80}

{self.data.describe().to_string()}

{'='*80}
2. MEASUREMENT MODEL (CFA) RESULTS
{'='*80}
"""

            if "cfa" in self.results:
                fit_cfa = self.get_fit_measures(self.results["cfa"])
                report += "\nModel Fit Indices:\n"
                report += fit_cfa.to_string()
                report += "\n"

                reliability = self.reliability_analysis(self.results["cfa"])
                report += "\n\nReliability Metrics:\n"
                report += reliability.to_string(index=False)

            report += f"\n\n{'='*80}\n"
            report += "3. STRUCTURAL MODEL RESULTS\n"
            report += f"{'='*80}\n"

            if "sem_full" in self.results:
                fit_sem = self.get_fit_measures(self.results["sem_full"])
                report += "\nModel Fit Indices:\n"
                report += fit_sem.to_string()
                report += "\n"

                paths = self.extract_path_coefficients(self.results["sem_full"])
                report += "\n\nPath Coefficients:\n"
                report += paths.to_string(index=False)

            report += f"\n\n{'='*80}\n"
            report += "4. CONCLUSIONS AND RECOMMENDATIONS\n"
            report += f"{'='*80}\n\n"

            report += """
Based on the analysis:

1. Model Fit: Evaluate the fit indices against established cutoffs
   - CFI/TLI ≥ 0.95 indicates excellent fit
   - RMSEA ≤ 0.05 and SRMR ≤ 0.05 indicate excellent fit

2. Reliability: All constructs should meet minimum thresholds
   - Composite Reliability (CR) > 0.70
   - Average Variance Extracted (AVE) > 0.50

3. Validity: Check convergent and discriminant validity
   - Convergent: AVE > 0.50 and CR > 0.70
   - Discriminant: √AVE > inter-construct correlations

4. Path Significance: Examine structural paths
   - Significant paths (p < 0.05) support hypothesized relationships
   - Effect sizes indicated by standardized coefficients

5. Model Comparison: If alternative models tested, compare AIC/BIC
   - Lower values indicate better model fit

REFERENCES:
- Hu, L. T., & Bentler, P. M. (1999). Cutoff criteria for fit indexes in 
  covariance structure analysis. Structural Equation Modeling, 6(1), 1-55.
- Fornell, C., & Larcker, D. F. (1981). Evaluating structural equation models 
  with unobservable variables and measurement error. Journal of Marketing 
  Research, 18(1), 39-50.
- Hair, J. F., et al. (2010). Multivariate data analysis (7th ed.). Pearson.

{'='*80}
END OF REPORT
{'='*80}
"""
            # Save report - CORRECTED path with UTF-8 encoding
            output_path = Path(OUTPUT_DIR, f"{filename}.txt")
            with open(output_path, "w", encoding="utf-8") as f:
                f.write(report)

            print(f"✓ Report saved: {output_path}")

            return report

        except Exception as e:
            print(f"✗ Error generating report: {e}")
            raise

    def _r_to_pandas(self, r_dataframe):
        """Convert R dataframe to pandas."""
        with localconverter(default_converter + pandas2ri.converter):
            return ro.conversion.rpy2py(r_dataframe)


# ============================================================================
# MAIN EXECUTION
# ============================================================================


def main():
    """Execute comprehensive SEM analysis."""

    print("\n" + "=" * 80)
    print("COMPREHENSIVE SEM ANALYSIS - RESEARCH PROJECT (CORRECTED)")
    print("=" * 80)
    print("\nThis analysis includes:")
    print("  ✓ Confirmatory Factor Analysis (CFA)")
    print("  ✓ Full Structural Equation Model")
    print("  ✓ Reliability Analysis (CR, AVE, Alpha)")
    print("  ✓ Validity Assessment")
    print("  ✓ Model Fit Evaluation")
    print("  ✓ Modification Indices")
    print("  ✓ Alternative Model Comparison")
    print("  ✓ Comprehensive Visualizations")
    print("  ✓ Publication-Ready Report")

    try:
        # Generate data
        generator = SEMDataGenerator(n_samples=500, seed=42)
        data = generator.generate_research_data()

        # Save raw data
        data_path = Path(OUTPUT_DIR, "research_data.csv")
        data.to_csv(data_path, index=False)
        print(f"\n✓ Raw data saved: {data_path}")

        # Initialize analyzer
        analyzer = SEMAnalyzer(data)

        # Step 1: CFA
        print("\n" + ">" * 80)
        print("STEP 1: CONFIRMATORY FACTOR ANALYSIS")
        print(">" * 80)
        cfa_fit = analyzer.run_cfa()
        cfa_fit_measures = analyzer.get_fit_measures(cfa_fit)
        analyzer.interpret_fit(cfa_fit_measures)
        reliability_df = analyzer.reliability_analysis(cfa_fit)
        analyzer.validity_analysis(reliability_df)

        # Step 2: Full SEM
        print("\n" + ">" * 80)
        print("STEP 2: STRUCTURAL EQUATION MODEL")
        print(">" * 80)
        sem_fit = analyzer.run_sem(model_type="full")
        sem_fit_measures = analyzer.get_fit_measures(sem_fit)
        analyzer.interpret_fit(sem_fit_measures)

        # Path coefficients
        paths = analyzer.extract_path_coefficients(sem_fit)
        print("\n" + "-" * 80)
        print("STRUCTURAL PATH COEFFICIENTS:")
        print("-" * 80)
        print(paths.to_string(index=False))

        # Modification indices
        mi_df = analyzer.modification_indices(sem_fit, threshold=10)

        # Step 3: Alternative Model
        print("\n" + ">" * 80)
        print("STEP 3: ALTERNATIVE MODEL COMPARISON")
        print(">" * 80)
        alt_fit = analyzer.run_sem(model_type="alternative")
        comparison = analyzer.compare_models(
            sem_fit, alt_fit, "Theoretical Model", "Alternative Model"
        )

        # Step 4: Visualizations
        print("\n" + ">" * 80)
        print("STEP 4: CREATING VISUALIZATIONS")
        print(">" * 80)
        fig = analyzer.visualize_results(sem_fit, filename="sem_full_model")

        # Step 5: Generate Report
        print("\n" + ">" * 80)
        print("STEP 5: GENERATING COMPREHENSIVE REPORT")
        print(">" * 80)
        report = analyzer.generate_report(filename="sem_analysis_report")

        # Final summary
        print("\n" + "=" * 80)
        print("ANALYSIS COMPLETE!")
        print("=" * 80)
        print("\nGenerated Files:")
        print("  1. research_data.csv - Raw data")
        print("  2. sem_full_model.png - Visualization")
        print("  3. sem_analysis_report.txt - Comprehensive report")
        print(f"\nAll files saved to: {OUTPUT_DIR.absolute()}")
        print("\nNext steps:")
        print("  • Review model fit indices and interpretation")
        print("  • Examine path coefficients for hypothesis testing")
        print("  • Consider modification indices for model refinement")
        print("  • Compare alternative models if theoretically justified")
        print("  • Report results with appropriate tables and figures")
        print("\n" + "=" * 80)

        return True

    except Exception as e:
        print("\n" + "=" * 80)
        print("ERROR OCCURRED")
        print("=" * 80)
        print(f"Error: {e}")
        import traceback

        traceback.print_exc()
        print("\n" + "=" * 80)
        return False


if __name__ == "__main__":
    success = main()
    if success:
        print("\n🎉 SUCCESS! All analyses completed without errors.")
    else:
        print("\n❌ FAILED! Please check the error messages above.")
