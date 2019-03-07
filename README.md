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
Because the precription requires a phamacist to review and manually verify case by case, it create a bottle-neck process and delay the process of treatment. Moreover, a large number of verification requests allow less time for phamacist to review which also increase a risk of error. 
### Prior work
Prior research showed that the use of computerized and/or expert systems could help to reduce drug prescription errors. [Maystre et al.](https://academic.oup.com/jamia/article/17/5/559/831789) used natural language process techniques to extract clinical data and trained machine learning and used pattern matching to verify medications and provided reason supporting the prescription.
## Objectives
1. Use machine learning models to verify drug prescription.

## Aims
1. The performance of machine learning model shows precision, recall, and F-measure are greater than 80%.
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

