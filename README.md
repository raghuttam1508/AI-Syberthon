# Fraud Detection 

- This repository contains a Data Analysis Project of **Synthetic Financial Datasets For Fraud Detection**.
- This project was built under **AI-Syberthon**, a hackathon conducted at my college
- Find the Dataset [here](https://www.kaggle.com/datasets/ealaxi/paysim1?resource=download)


## Understanding the Data
- step - maps a unit of time in the real world. Here, 1 step is 1 hour of time. Total steps -744 (30 days simulation).
- type - CASH-IN, CASH-OUT, DEBIT, PAYMENT and TRANSFER.
- amount - local currency.
- nameOrig - customer starting transaction
- oldbalanceOrg** - initial balance before transaction
- newbalanceOrig - new balance after transaction
- nameDest - customer receiving transaction
- oldbalanceDest - initial balance of recipient before transaction.
- newbalanceDest - new balance of recipient after transaction.
- isFraud - transactions made by fraudulent agents, aiming to profit by taking control of customers' accounts.
- isFlaggedFraud - flags illegal attempts (For eg. attempt to transfer more than 200.000 in a single transaction).
