# UFC-Fight-Winner-Predictor-Machine-learning

🥋 UFC Fight Winner Prediction (Machine Learning)

A data science project that predicts the winner of a UFC fight using real fighter statistics and historical fight data.
This project includes web scraping, feature engineering, ML modeling, and a Streamlit-based UI.

📌 Project Features
✔ Web Scraping

Scraped data from ufcstats.com, including:

Fighter attributes (height, reach, stance, age, SLpM, StrAcc, SApM, StrDef, TDAvg, TDAcc, TDDef, SubAvg)

Full UFC fight history with results and methods

✔ Data Cleaning & Feature Engineering

Converted height/reach to cm

Extracted wins/losses/draws/NC from fighter records

Calculated age, win rate, finish rate

Created delta stats (Fighter1 − Fighter2) used for prediction:

striking deltas

grappling deltas

physical deltas

performance deltas

✔ Machine Learning

Model: RandomForestClassifier

Scaler: StandardScaler

Trained on engineered matchup dataset

Predicts winner + confidence score

✔ Streamlit UI

UFC-themed layout (Red Corner vs Blue Corner)

Fighter selection dropdowns

Fighter stat cards

Delta feature table

Winner prediction with confidence

🛠 Tech Stack

Python · Pandas · NumPy · BeautifulSoup · Scikit-learn · RandomForest · Streamlit

frontend - 
<img width="949" height="711" alt="architectura Diagram_3" src="[https://github.com/user-attachments/assets/c129c432-6736-494d-aeea-d8ea4646cecd](https://github.com/LLWICK/UFC-Fight-Winner-Predictor-Machine-learning/blob/main/frontend_images/Screenshot%202026-01-19%20123243.png)" />

