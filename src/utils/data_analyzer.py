import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import io
import base64
import plotly.express as px
import plotly.graph_objects as go
import plotly.figure_factory as ff
from plotly.subplots import make_subplots
from typing import Dict, List, Tuple, Optional, Any
import json

def get_data_profile(df: pd.DataFrame, detailed: bool = False) -> Dict:
    """Generate a comprehensive profile of the dataframe with statistics and insights.
    
    Args:
        df: The pandas DataFrame to analyze
        detailed: Whether to generate detailed stats (more token-intensive)
    
    Returns:
        A dictionary with profile information
    """
    if df is None or df.empty:
        return {"error": "No data available"}
    
    # Sample the dataframe if it's very large to save on computation
    sample_size = 10000
    if len(df) > sample_size:
        df_sample = df.sample(sample_size, random_state=42)
    else:
        df_sample = df
    
    profile = {
        "basic_info": {
            "rows": df.shape[0],
            "columns": df.shape[1],
            "memory_usage": f"{df.memory_usage(deep=True).sum() / (1024*1024):.2f} MB"
        },
        "missing_data": {
            "total_missing": df.isna().sum().sum(),
            "missing_percentage": f"{(df.isna().sum().sum() / (df.shape[0] * df.shape[1]) * 100):.2f}%",
            "columns_with_missing": df.isna().sum()[df.isna().sum() > 0].to_dict()
        },
        "column_stats": {},
        "correlations": None,
        "insights": []
    }
    
    # Limit analysis to most important columns to save tokens
    # Get numeric and categorical columns
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    categorical_cols = df.select_dtypes(include=['object', 'category']).columns
    datetime_cols = df.select_dtypes(include=['datetime']).columns
    
    # For very wide dataframes, focus on the most informative columns
    if len(df.columns) > 30:
        # Choose most important numeric columns (highest variance)
        if len(numeric_cols) > 10:
            numeric_col_variances = df[numeric_cols].var()
            numeric_cols = numeric_col_variances.nlargest(10).index.tolist()
        
        # Choose most important categorical columns (highest cardinality)
        if len(categorical_cols) > 10:
            cat_cardinality = {col: df[col].nunique() for col in categorical_cols}
            cat_cardinality_series = pd.Series(cat_cardinality)
            categorical_cols = cat_cardinality_series.nlargest(10).index.tolist()
    
    # Column statistics (only for the selected columns)
    important_cols = list(numeric_cols) + list(categorical_cols) + list(datetime_cols)
    for col in important_cols:
        col_stats = {"type": str(df[col].dtype)}
        
        if pd.api.types.is_numeric_dtype(df[col]):
            try:
                # Basic stats for all numeric columns
                col_stats.update({
                    "min": df[col].min(),
                    "max": df[col].max(),
                    "mean": df[col].mean(),
                    "median": df[col].median(),
                    "std": df[col].std(),
                    "unique_values": df[col].nunique()
                })
                
                # Add more detailed stats only if requested
                if detailed:
                    col_stats.update({
                        "skew": df[col].skew(),
                        "zeros": (df[col] == 0).sum(),
                        "negative": (df[col] < 0).sum() if df[col].dtype != 'uint' else 0,
                        "percentiles": {
                            "25%": df[col].quantile(0.25),
                            "75%": df[col].quantile(0.75),
                            "95%": df[col].quantile(0.95)
                        }
                    })
            except:
                pass
        elif pd.api.types.is_string_dtype(df[col]) or pd.api.types.is_categorical_dtype(df[col]):
            try:
                # Basic stats for categorical columns
                col_stats.update({
                    "unique_values": df[col].nunique(),
                    "most_common": df[col].value_counts().head(3).to_dict(),
                })
                
                # More detailed stats if requested
                if detailed:
                    col_stats.update({
                        "empty_strings": (df[col] == "").sum(),
                        "avg_length": df[col].str.len().mean() if hasattr(df[col], 'str') else None
                    })
            except:
                pass
        elif pd.api.types.is_datetime64_dtype(df[col]):
            try:
                col_stats.update({
                    "min_date": df[col].min(),
                    "max_date": df[col].max(),
                    "range_days": (df[col].max() - df[col].min()).days
                })
            except:
                pass
        
        profile["column_stats"][col] = col_stats
    
    # Correlations for numeric columns (limited to conserve tokens)
    if len(numeric_cols) > 1:
        # If there are many numeric columns, limit to most important ones
        cols_for_corr = numeric_cols[:min(10, len(numeric_cols))]
        
        try:
            corr_matrix = df[cols_for_corr].corr()
            # Get top correlations (excluding self-correlations)
            top_corrs = []
            for i, col1 in enumerate(corr_matrix.columns):
                for j, col2 in enumerate(corr_matrix.columns):
                    if i < j:  # Upper triangle only (avoid duplicates)
                        # Only include strong correlations to save on tokens
                        corr_value = corr_matrix.loc[col1, col2]
                        if abs(corr_value) > 0.5:  # Only correlations stronger than 0.5
                            top_corrs.append({
                                "col1": col1,
                                "col2": col2,
                                "correlation": corr_value
                            })
            
            # Sort by absolute correlation values
            top_corrs = sorted(top_corrs, key=lambda x: abs(x["correlation"]), reverse=True)
            profile["correlations"] = top_corrs[:5]  # Limit to top 5 correlations to save tokens
        except:
            pass
    
    # Generate insights (being selective to save tokens)
    insights = []
    
    # Check for highly correlated features
    if profile["correlations"]:
        for corr in profile["correlations"]:
            if abs(corr["correlation"]) > 0.8:  # Only report very strong correlations
                insights.append(f"Strong {'positive' if corr['correlation'] > 0 else 'negative'} correlation ({corr['correlation']:.2f}) between {corr['col1']} and {corr['col2']}")
    
    # Check for columns with high missing values (only report significant issues)
    high_missing = {k: v for k, v in profile["missing_data"]["columns_with_missing"].items() 
                   if v / df.shape[0] > 0.3}  # Increased threshold to 30%
    if high_missing:
        for col, count in high_missing.items():
            insights.append(f"Column '{col}' has {count} missing values ({count/df.shape[0]*100:.1f}%)")
    
    # Add more detailed insights only if requested
    if detailed:
        # Check for high skew in numeric columns
        for col, stats in profile["column_stats"].items():
            if "skew" in stats and abs(stats["skew"]) > 2:  # Increased threshold to 2
                insights.append(f"Column '{col}' has strong {'positive' if stats['skew'] > 0 else 'negative'} skew ({stats['skew']:.2f})")
        
        # Check for outliers using IQR (only for key numeric columns)
        key_numeric_cols = list(numeric_cols)[:min(5, len(numeric_cols))]
        for col in key_numeric_cols:
            try:
                Q1 = df[col].quantile(0.25)
                Q3 = df[col].quantile(0.75)
                IQR = Q3 - Q1
                outliers = ((df[col] < (Q1 - 1.5 * IQR)) | (df[col] > (Q3 + 1.5 * IQR))).sum()
                if outliers > 0 and outliers / len(df) > 0.05:  # Only report if > 5% are outliers
                    insights.append(f"Column '{col}' has {outliers} potential outliers ({outliers/len(df)*100:.1f}%)")
            except:
                pass
        
        # Check for data patterns (trends)
        try:
            if 'date' in df.columns or any(col for col in df.columns if 'date' in str(col).lower()):
                date_col = next((col for col in df.columns if 'date' in str(col).lower()), None)
                if date_col:
                    # If we have a date column and numeric columns, check for significant trends
                    if pd.api.types.is_datetime64_dtype(df[date_col]) and len(numeric_cols) > 0:
                        key_numeric = numeric_cols[0]  # Just check the first numeric column
                        
                        # Simple trend check
                        df_sorted = df.sort_values(date_col)
                        if len(df_sorted) > 10:
                            first_half_mean = df_sorted[key_numeric].iloc[:len(df_sorted)//2].mean()
                            second_half_mean = df_sorted[key_numeric].iloc[len(df_sorted)//2:].mean()
                            pct_change = (second_half_mean - first_half_mean) / first_half_mean * 100 if first_half_mean != 0 else 0
                            
                            if abs(pct_change) > 30:  # Only report significant changes (>30%)
                                insights.append(f"Column '{key_numeric}' shows a {abs(pct_change):.1f}% {'increase' if pct_change > 0 else 'decrease'} over time")
        except:
            pass
    
    # Only check for duplicates in smaller datasets
    if len(df) < 100000:
        duplicate_count = df.duplicated().sum()
        if duplicate_count > 0 and duplicate_count / len(df) > 0.01:  # Only report if >1% are duplicates
            insights.append(f"Found {duplicate_count} duplicate rows ({duplicate_count/len(df)*100:.1f}%)")
    
    profile["insights"] = insights[:10]  # Limit to top 10 insights
    
    return profile

def generate_quick_plots(df: pd.DataFrame, max_plots: int = 5) -> Dict[str, str]:
    """Generate a set of informative plots for the dataframe.
    
    Args:
        df: The pandas DataFrame to visualize
        max_plots: Maximum number of plots to generate (to limit token usage)
    
    Returns:
        Dictionary with plot images as base64 strings
    """
    if df is None or df.empty:
        return {}
    
    # Sample the dataframe if it's very large to save on computation
    sample_size = 5000
    if len(df) > sample_size:
        df_sample = df.sample(sample_size, random_state=42)
    else:
        df_sample = df
    
    plots = {}
    plot_count = 0
    
    # Function to convert matplotlib plot to base64
    def fig_to_base64(fig):
        buf = io.BytesIO()
        fig.savefig(buf, format='png', dpi=100, bbox_inches='tight')
        buf.seek(0)
        img_str = base64.b64encode(buf.read()).decode('utf-8')
        plt.close(fig)
        return img_str
    
    # Set lower DPI and figure size to reduce token usage
    plt.rcParams['figure.dpi'] = 80
    
    try:
        # 1. Correlation Heatmap (if we have numeric columns)
        numeric_cols = df_sample.select_dtypes(include=[np.number]).columns
        if len(numeric_cols) > 1 and plot_count < max_plots:
            # Limit to top 10 numeric columns to save on tokens
            if len(numeric_cols) > 10:
                # Select columns with highest variance
                variances = df_sample[numeric_cols].var()
                numeric_cols = variances.nlargest(10).index
            
            # Create a smaller figure for the heatmap
            fig, ax = plt.subplots(figsize=(8, 6))
            corr_matrix = df_sample[numeric_cols].corr()
            mask = np.triu(np.ones_like(corr_matrix, dtype=bool))
            
            # Use a simpler heatmap with fewer annotations
            if len(numeric_cols) <= 6:
                # For smaller matrices, we can afford annotations
                sns.heatmap(corr_matrix, mask=mask, annot=True, fmt=".2f", cmap="coolwarm", 
                            vmin=-1, vmax=1, center=0, linewidths=.5, cbar_kws={"shrink": .5})
            else:
                # For larger matrices, skip annotations to save tokens
                sns.heatmap(corr_matrix, mask=mask, annot=False, cmap="coolwarm", 
                            vmin=-1, vmax=1, center=0, linewidths=.5, cbar_kws={"shrink": .5})
            
            plt.title("Correlation Matrix", fontsize=14)
            plots["correlation"] = fig_to_base64(fig)
            plot_count += 1
    except Exception as e:
        print(f"Error generating correlation plot: {e}")
    
    try:
        # 2. Missing values plot (only if there are missing values)
        if df_sample.isna().sum().sum() > 0 and plot_count < max_plots:
            missing = df_sample.isna().sum()
            missing = missing[missing > 0]
            
            # Only create the plot if there are enough missing values to be meaningful
            if len(missing) > 0 and missing.sum() / (df_sample.shape[0] * df_sample.shape[1]) > 0.01:
                missing_percent = missing / len(df_sample) * 100
                
                # Limit to top 10 columns with missing values
                if len(missing) > 10:
                    missing_percent = missing_percent.nlargest(10)
                
                fig, ax = plt.subplots(figsize=(8, 5))
                missing_percent.plot(kind='bar', ax=ax)
                plt.title('Missing Values by Column (%)', fontsize=14)
                plt.xlabel('Columns', fontsize=12)
                plt.ylabel('Percent Missing', fontsize=12)
                plt.xticks(rotation=45, ha='right')
                plots["missing_values"] = fig_to_base64(fig)
                plot_count += 1
    except Exception as e:
        print(f"Error generating missing values plot: {e}")
    
    try:
        # 3. Distribution of top numeric columns (limited to save tokens)
        if len(numeric_cols) > 0 and plot_count < max_plots:
            # Choose columns with the highest variance for more informative plots
            if len(numeric_cols) > 3:
                variances = df_sample[numeric_cols].var()
                top_numeric_cols = variances.nlargest(min(3, max_plots - plot_count)).index
            else:
                top_numeric_cols = numeric_cols[:min(3, max_plots - plot_count)]
            
            for col in top_numeric_cols:
                if plot_count >= max_plots:
                    break
                    
                # Create smaller figures
                fig, ax = plt.subplots(figsize=(7, 4))
                sns.histplot(df_sample[col].dropna(), kde=True, ax=ax)
                plt.title(f'Distribution of {col}', fontsize=14)
                plt.xlabel(col, fontsize=12)
                plt.ylabel('Frequency', fontsize=12)
                plots[f"dist_{col}"] = fig_to_base64(fig)
                plot_count += 1
    except Exception as e:
        print(f"Error generating distribution plots: {e}")
    
    try:
        # 4. Categorical value counts (only for columns with reasonable cardinality)
        categorical_cols = df_sample.select_dtypes(include=['object', 'category']).columns
        if len(categorical_cols) > 0 and plot_count < max_plots:
            # Find categorical columns with reasonable number of categories
            valid_cat_cols = []
            for col in categorical_cols:
                n_unique = df_sample[col].nunique()
                if 2 <= n_unique <= 8:  # Only visualize columns with 2-8 categories
                    valid_cat_cols.append((col, n_unique))
            
            # Sort by number of categories (prioritize columns with fewer categories)
            valid_cat_cols.sort(key=lambda x: x[1])
            
            # Generate plots for up to 2 categorical columns
            for col, _ in valid_cat_cols[:min(2, max_plots - plot_count)]:
                if plot_count >= max_plots:
                    break
                    
                # Get value counts and limit to top 8 categories
                value_counts = df_sample[col].value_counts().nlargest(8)
                
                fig, ax = plt.subplots(figsize=(7, 4))
                value_counts.plot(kind='bar', ax=ax)
                plt.title(f'Top Values: {col}', fontsize=14)
                plt.xlabel(col, fontsize=12)
                plt.ylabel('Count', fontsize=12)
                plt.xticks(rotation=45, ha='right')
                plots[f"cat_{col}"] = fig_to_base64(fig)
                plot_count += 1
    except Exception as e:
        print(f"Error generating categorical plots: {e}")
    
    # We'll skip time series plots to save on tokens unless specifically requested
    
    return plots

def get_interactive_plots(df: pd.DataFrame, max_plots: int = 3) -> Dict[str, Any]:
    """Generate interactive Plotly plots for the dataframe.
    
    Args:
        df: The pandas DataFrame to visualize
        max_plots: Maximum number of plots to generate (to limit token usage)
    
    Returns:
        Dictionary with Plotly figure JSON strings
    """
    if df is None or df.empty:
        return {}
    
    # Sample the dataframe to save on tokens
    sample_size = 3000
    if len(df) > sample_size:
        df_sample = df.sample(sample_size, random_state=42)
    else:
        df_sample = df
    
    plots = {}
    plot_count = 0
    
    try:
        # 1. Correlation Heatmap (most important interactive plot)
        numeric_cols = df_sample.select_dtypes(include=[np.number]).columns
        if len(numeric_cols) > 1 and plot_count < max_plots:
            # Limit to top 8 numeric columns based on variance
            if len(numeric_cols) > 8:
                variances = df_sample[numeric_cols].var()
                numeric_cols = variances.nlargest(8).index
            
            corr_matrix = df_sample[numeric_cols].corr()
            
            # Create a simplified correlation heatmap
            fig = px.imshow(corr_matrix,
                          color_continuous_scale='RdBu_r',
                          zmin=-1, zmax=1)
            fig.update_layout(
                title='Correlation Matrix',
                width=600,  # Smaller size
                height=500,
                margin=dict(l=40, r=40, t=50, b=40)  # Reduced margins
            )
            
            # Convert to JSON with optimized options
            plots["correlation"] = fig.to_json(
                engine="json", 
                pretty=False,  # No indentation or newlines
                remove_uids=True  # Remove unnecessary IDs
            )
            
            plot_count += 1
    except Exception as e:
        print(f"Error generating interactive correlation plot: {e}")
    
    # Only create additional interactive plots if specifically requested
    # (Interactive plots are more token-intensive)
    
    return plots
