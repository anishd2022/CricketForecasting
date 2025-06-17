# Forecasting Scores in T20 Cricket Games

-   Working with Professor Dave Zes at UCLA, I took on a STATS 199 undergraduate research project that aimed to forecast the final score in a T20 inning after an arbitrary intervention point sometime earlier in the inning. I explored various methods, such as linear regression, ARIMA models, KNN simulations, multi-dimensional robust synthetic control (mRSC), and the R [mactivate](https://cran.r-project.org/web/packages/mactivate/mactivate.pdf) library developed by Prof. Zes. I analyzed the accuracy and the pros and cons of each method used.
- My final deliverable `FinalPaper.pdf` is a formal research paper detailing my research and results. This paper can also be found on my website [here](https://anishdeshpande.com/PDFs/ForecastingScores.pdf)

### 📊 Data Collection

- **Source**: All match data was sourced from [Cricsheet](https://cricsheet.org), an open-access ball-by-ball archive of professional cricket matches.

- **Format**: JSON files containing structured event-level data for each delivery in T20 games.

- **Storage**: Parsed and loaded into an AWS-hosted MySQL database for efficient querying and aggregation.

- **Tools Used**:
  - Custom Python scripts to extract metadata (team, venue, toss, innings outcome) and ball-by-ball events (runs, wickets, overs).
  - SQL queries to generate structured datasets suitable for modeling and time-series analysis.

- **Scope**: ~18,500 matches processed, focusing on first-innings forecasting tasks.

- **Data**: The final data structure used for the rest of my analysis is the `t20_tensor_data.npz` and `t20_tensor_data_with_inning.npz`. The exact details of this data structure can be found in my paper. 
