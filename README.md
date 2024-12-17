# Multiple Logistic Regression for Predicting Social Media Usage

This project was created for the completion of university honors credit. Using data from the [2021 Core Trends Survey](https://www.pewresearch.org/dataset/2021-core-trends-survey/?loggedIn=true), the goal
was to use logisitc regression to predict the social media usage of an individual given their characterisitcs.

---

##  Features
- Produces Predictions for 11 social media platforms (Twitter, Facebook, Instagram, Snapchat, Youtube, Reddit, WhatsApp, Nextdoor, Tiktok, LinkedIn, Pinterest)
- Draws from 9 different personal attributes (gender, age, education level, employment status, income, race, relationship status, political affiliation, has children)
- Adjustable learning rate and precision
- Current accuracy sits around 73%

---

## How it works
- The program is essential 11 seperate logistic regression models running at the same time. Using data from about 1350 individuals, the program performs gradient descent
- to minmize prediction error for each social media. To test the model, the predictions of 150 extra individuals are evaluated against their actual survey results.
- Note: a small percentage of survey answers are unnacounted for in the model. For example, if an individual answered "Don't know (9)" to a question that takes an answer between 1 and 3.

- # Installation and Usage
- 1. Clone the repository:
   ```bash
   git clone https://github.com/Daniel-Abrams/Multiple-Logistic-Regression
   ```

- 2. There are 4 files in the repository:
     - Jan 25-Feb 8^J 2021 - Core Trends Survey With Calculated Columns.csv: a csv file containing the data from the survey.
     - data.csv: another csv file containing a fabricated set of data for testing purposes.
     - data.py: a file for extracting the data and preparing it for use.
     - LogisitcRegression.py: the file where all the calculations occur

-3 . Run the program:
```bash
   python LogisticRegression.py
 ```
The program can take about a minute to run. It will likely take longer if you decrease epsilon. Once finished, the terminal will display the predictions for the test group, 
which will be arrays of 0s and 1s. 0 means that a person was predicted not to use a social media, 1 means that they were. There will be 11 arrays displayed(one for each social media). Matching indeces in
each array correspond to the same person. The terminal will then display the proportion of correct predictions.

