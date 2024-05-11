import streamlit as st
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

def describe(data):
    description = data.describe(include="all")
    na_counts = data.isna().sum()
    na_counts.name = 'Missing'
    return pd.concat([na_counts.to_frame().T, description])

def main():
    st.title("Describing data and its correlation")
    # File uploader
    spectra = st.file_uploader("Upload file", type=["csv", "txt"])

    if spectra is not None:
        spectra_df = pd.read_csv(spectra)
        
        st.subheader("Data Preview:")
        st.write(spectra_df)
        
        # Identify non-numeric columns
        non_numeric_cols = spectra_df.select_dtypes(exclude=['number']).columns
        
        # If there are non-numeric columns
        if not non_numeric_cols.empty:
            st.subheader("Non-Numeric Columns:")
            st.write(non_numeric_cols)

        # Numeric data frame
        spectra_df_numeric = spectra_df.select_dtypes(include=['number'])

        # Describe data
        st.subheader("Data Describe")
        data_describe = describe(spectra_df_numeric)
        st.dataframe(data_describe)

        with st.spinner('Ploting..'):

            # Pairplot
            st.subheader("Pair Plot")
            pairplot_fig = sns.pairplot(spectra_df_numeric)
            st.pyplot(pairplot_fig)
        
            st.subheader("Correlation Plot")    
            # Calculate correlation matrix
            corr = spectra_df_numeric.corr()

            # Generate a mask for the upper triangle
            mask = (corr.values == 1)

            # Plotting the correlation heatmap
            corr_plot_fig, ax = plt.subplots(figsize=(10, 8))
            sns.heatmap(corr, annot=True, fmt=".2f", cmap='coolwarm', mask=mask, square=True, linewidths=.5, ax=ax)
            st.pyplot(corr_plot_fig)


if __name__ == "__main__":
    main()
