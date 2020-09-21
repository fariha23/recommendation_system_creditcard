Recommendation_System_CreditCard
==============================

Create recommendation system based on credit card transactional data.

Project Organization
------------

    ├── LICENSE
    ├── README.md          <- The top-level README for developers using this project.
    │
    ├── notebooks          <- Jupyter notebooks.
    │
    │
    ├── reports            <- Generated analysis in PDF
    │   └── figures        <- Generated graphics and figures to be used in reporting
    │
    ├── requirements.txt   <- The requirements file for reproducing the analysis environment, e.g.
    │                         generated with `pip freeze > requirements.txt


--------

Goals:

Create recommendation system(s) using credit card transactional records. The data contains transactions from multiple users and contains GPS coordinates. 
The goals of the recommendation system(s) are as follows:

Goal #1.The recommendation system should recommend similar merchants to the customer based on similarities between the merchants

Goal #2.The recommendation system should recommend merchants to the customer based on their current location. 

Notebooks:

Along with EDA.ipyb and pre-work for goal #2 notebooks(Recommendation_System_Location_Based_Pre_Work.ipynb) following notebooks capture the code for two goals. 
----A notebook titled "Recommendation_System_v1.ipynb" captures code for goal #1. 
----A notebook titled "Recommendation_System_Location_Based_HDBSCAN.ipynb" captures code for goal #2.
(Note: A notebook titled "Recommendation_System_Location_Based_Kmeans.ipynb" captures code for goal #2 however it is only for learning purpose)

--Documentation:
----Documentation on the project (project presentation, milestone reports etc) are provided in folder 'reports'. 
This folder also contains figures used to create the reports. 

----Documentation on the code can be found under notebooks folder titled "Documentation_Recommendation_System_v1" and "Documentation_Recommendation_System_Location_Based"

 
