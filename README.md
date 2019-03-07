# Auto drug verification
## Researchers
1. Assistant Professor Piyapong Khumrin, MD, Faculty of Medicine, Chiang Mai University, Chiang Mai, Thailand
2. Assistant Professor Krit Khwanngern, MD, Faculty of Medicine, Chiang Mai University, Chiang Mai, Thailand
3. Associate Professor Arintaya Phrommintikul, MD, Faculty of Medicine, Chiang Mai University, Chiang Mai, Thailand
4. Associate Professor Nipon Theera-Umpon, PhD, Biomedical Engineering Institute, Chiang Mai University
5. Terence Siganakis, CEO, Growing Data Pty Ltd
6. Alexander Dokumentov, Data Scientist, Growing Data Pty Ltd

## Technical support
1. Atcharaporn Angsuratanawech 
2. Sittipong Moraray
3. Pimpaka Chuamsakul
4. Pitupoom Chumpoo
5. Prawinee Mokmoongmuang

## Duration
6 months (March - August 2019)

## Introduction
Drug prescription errors are common in general practice and in hospitals and result in serious complications in patients ([Velo and Minuz](https://bpspubs.onlinelibrary.wiley.com/doi/pdf/10.1111/j.1365-2125.2009.03425.x), [Ridley et al.](https://onlinelibrary.wiley.com/doi/pdf/10.1111/j.1365-2044.2004.03969.x), [Sapkota et al.](https://www.researchgate.net/profile/Sujata_Sapkota/publication/280315716_Drug_prescribing_pattern_and_prescription_error_in_elderly_A_retrospective_study_of_inpatient_record/links/55b2201508aec0e5f4314737.pdf), [Beddy et al.](https://link.springer.com/article/10.1007/s11845-010-0474-6)). Drug verification process is included into the drug prescription process in the Maharaj Nakhon Chiang Mai for reducing prescription errors. Drug verification process starts when doctors enter medication into the electronic health record system. The system sends a notification to phamacists requesting for drug verification. The phamacists review the prescription with patient history and investigation results and confirm the order. After the prescription is verified, the patients are allowed to collect the drugs. If the prescription is rejected, the doctors need to review the prescription and order a new prescription and notify the phamacists to get approve again.
### Problem statement
Because the precription requires a phamacist to review and manually verify case by case, it create a bottle-neck process and delay the process of treatment. Moreover, a large number of verification requests allow less time for phamacist to review (including the prescription itself such as right drug, right route, right amount and frequency, and the appropriate indication of prescription to the patient condition)  which also increase a risk of error.
### Prior work
Prior research showed that the use of computerized and/or expert systems had a potential benefit to reduce drug prescription errors by developing expert systems to automatically identify the errors and/or providing supporting information why the prescription is inappropriate. 

[Maystre et al.](https://academic.oup.com/jamia/article/17/5/559/831789) used natural language process techniques to extract clinical data and trained machine learning and used pattern matching to verify medications and provided reason supporting the prescription. The evaluation of system performance showed that the system performed well on predicting the route information (F1-measure 86%), dosage (F1-measure 82%), frequency (F1-measure 85%), but poor on duration (F1-measure 39%), and reasons for prescription (F1-measure 27%).

[Schiff et all.](https://academic.oup.com/jamia/article/24/2/281/2924796) developed a system to detect medication errors by using the outlier detection system. Machine learning models were trained with clinical data and used to predict an outlier of new prescriptions (assumed that prescriptions in the abnormal range should be an error). 76% of alert was found clinically valid.

The application of machine learning models could be also used to reduce prescription error focusing on the high alert drug groups such as warfarin ([Hu et al.](https://www.sciencedirect.com/science/article/pii/S0933365712000474)), or identifying inappropriate prescription (based on the association between prescirption and diagnosis) ([Nguyen et al.](https://journals.plos.org/plosone/article?id=10.1371/journal.pone.0082401)).

With the scope of this research, we aim to develop machine learning models which are able to detect presction errors in a high alert drug group (such as warfarin, nacrotic drugs).

Factors for considering inappropriate prescriptions are:
1. Drug prescription itself
   * drug name
   * dose
   * administrative route
   * amount
   * frequency
2. Association between clinical information, diagnosis and prescription
3. Indication of drug prescription
4. Drug-drug interaction

## Objectives
1. Develop machine learning models and/or verification protocol to verify drug (high alert drug group) prescription.

## Aims
1. The performance of machine learning model (and/or verification protocol) shows precision, recall, and F-measure greater than 80%.
2. Present one year drug verification and error analysis compared between before and after using machine learning models.

## Time line
### March 2019
  * Write and submit a research proposal and ethic.
  * Setup a new server.
  * Duplicate clinical data to the server.
  * Map and label column name and description.
  * Join the table data and create a single dataset.
### April 2019
  * Apply NLP and standard medical terminologies to preprocess input features.
  * Design and evaluate machine learning model.
### May 2019
  * Close the project either, the model performance is greater than 80% or it is the last week of May.
### June - August 2019
  * Write and submit a paper.
  
## Materials and methods
### Target group
Clinical records of outer-patient visits from 2006 - 2017 (2006 - 2016 for a training set, and 2017 for a test set) are retrospectively retrieved from the Maharaj Nakhon Chiang Mai electronic health records. Approximately one million records are expected to retrieve per year. Only encoded data (number, string) are included in the experiment (excluded images and scanned document).

### Data preprocessing
All identification data such as name, surname, address, national identification, hospital number will be removed according to patient privacy. Data of interest include:
  * Demographic data such as date of birth, gender
  * History taking and physical examination (including discharge summary)
  * Laboratory and investigation reports
  * ICD-10 (coded by a technical coder)
  * Drug prescription
  * Drug verification by phamacist (target class)
  
## Data analysis
Data from 2006 - 2016 are used to train machine learning models and data from 2017 are used to evaluate the models. We use overall accuracy, precision, recall, F-measure, and area under ROC curve to evaluate and compare predictive performance between models.

## How to use
1. Clone the project and change to dev branch
```
git clone https://github.com/u4507075/icd_10.git
cd icd_10
git checkout dev
```
2. Check out and update dev branch
```
git fetch
git checkout dev
git pull
```
## How it works
## Model evaluation
## Limitations

