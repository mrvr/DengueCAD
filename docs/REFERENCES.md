# Dengue data & challenge references

Curated external datasets, portals, and competitions relevant to DengueCAD
research, benchmarking, and future data acquisition. These sources are
**not** the IEEE TITB 2012 paper’s private hospital surveillance sets; they
are openly cited alternatives for clinical, hematological, and
epidemiological dengue modelling.

---

## Global / surveillance case counts

| # | Resource | URL | Notes |
|---|----------|-----|-------|
| 1 | **OpenDengue** — Data Explorer (global download) | https://opendengue.org/data.html#download-global-data | Public national / spatial / temporal dengue case counts; cite Clarke et al., *Sci Data* 2024, and the Figshare version used. |
| 2 | **Figshare** — Dengue dataset 2005–2025 | https://figshare.com/articles/dataset/dataset_dengue_2005-2025/31441663?file=62282755 | Downloadable dengue time-series archive (2005–2025). |
| 3 | **data.gov / CDC NNDSS** — Table 1J Dengue → Severe dengue | https://catalog.data.gov/dataset/nndss-table-1j-dengue-virus-infections-dengue-to-severe-dengue-285fc | U.S. National Notifiable Diseases Surveillance System provisional counts (dengue and severe dengue). |

---

## Mendeley Data (clinical / biomarkers)

| # | Resource | URL | Notes |
|---|----------|-----|-------|
| 4 | **Mendeley Data** (portal) | https://data.mendeley.com/ | Research-data repository; search for additional dengue clinical sets. |
| 5 | Integrated Dengue Prediction Dataset (clinical biomarkers, weather, risk) 2023–2025 | https://data.mendeley.com/datasets/ywp39jnzk3/2 | XLSX; 1500+ records, ~19 features (NS1/IgG/IgM, platelets, weather, engineered risk). DOI: [10.17632/ywp39jnzk3.2](https://doi.org/10.17632/ywp39jnzk3.2). |
| 6 | A Comprehensive Dengue Dataset of Bangladesh | https://data.mendeley.com/datasets/zdtc3n6xv2/3 | Demographics, serology (NS1/IgG/IgM), clinical symptoms, geography/housing; binary dengue outcome. DOI: [10.17632/zdtc3n6xv2.3](https://doi.org/10.17632/zdtc3n6xv2.3). Also on Kaggle (see #13). |
| 7 | Structured Clinical and Hematological Dataset for Early Dengue Diagnosis (Bangladesh) | https://data.mendeley.com/datasets/673swz9tb4/1 | 1,018 records (697 dengue+ / 321 dengue−); symptoms + platelet/WBC. DOI: [10.17632/673swz9tb4.1](https://doi.org/10.17632/673swz9tb4.1). |

---

## Forecasting challenges & modelling sprints

| # | Resource | URL | Notes |
|---|----------|-----|-------|
| 8 | **DrivenData DengAI** — Predicting Disease Spread | https://www.drivendata.org/competitions/44/dengai-predicting-disease-spread/ | Weekly case prediction for San Juan (PR) and Iquitos (Peru) from environmental features. |
| 9 | **InfoDengue–Mosqlimate Dengue Challenge (IMDC)** | https://sprint.mosqlimate.org/ | Brazil outbreak forecasting sprint (epidemiological + climate + demographic data). |

---

## Kaggle clinical / hematology datasets

| # | Resource | URL | Notes |
|---|----------|-----|-------|
| 10 | Dengue Fever Dataset for Prediction | https://www.kaggle.com/datasets/abdullahbarayan1/dengue-fever-dataset-for-prediction | Prediction-oriented dengue feature set. |
| 11 | Dengue Hematology Insights for Diagnosis and Care | https://www.kaggle.com/datasets/jocelyndumlao/dengue-hematology-insights-for-diagnosis-and-care/data | Hematology-focused diagnostic features. |
| 12 | Dengue Detection Dataset (Clinical Data) | https://www.kaggle.com/datasets/aravind3505/dengue-detection-dataset-clinical-data | Clinical dengue detection attributes. |
| 13 | Dengue Dataset Bangladesh (Kawsar Ahmad) | https://www.kaggle.com/datasets/kawsarahmad/dengue-dataset-bangladesh | Kaggle mirror / source link for the Bangladesh comprehensive set (see also Mendeley #6). |

---

## Suggested use with DengueCAD

1. Prefer **clinical + hematology** sets (Mendeley #5–#7, Kaggle #10–#13) when validating `NMPrediction` against paper-style Tables I–III (symptom/lab features closer to Rao & Kumar IEEE TITB 2012).
2. Use **OpenDengue / Figshare / NNDSS / DengAI / IMDC** for surveillance or forecasting experiments (different task than individual CAD diagnosis).
3. Always record licence (often **CC BY 4.0**), DOI/version, and citation in any derived paper or release notes.

### Example citations (OpenDengue)

- Clarke J, Lim A, Gupte P, Pigott DM, van Panhuis WG, Brady OJ. A global dataset of publicly available dengue case count data. *Sci Data*. 2024;11:296.
- Clarke J, et al. OpenDengue: data from the OpenDengue database. Version [as used]. figshare. https://doi.org/10.6084/m9.figshare.24259573

---

## Quick link list (plain URLs)

1. https://opendengue.org/data.html#download-global-data  
2. https://figshare.com/articles/dataset/dataset_dengue_2005-2025/31441663?file=62282755  
3. https://catalog.data.gov/dataset/nndss-table-1j-dengue-virus-infections-dengue-to-severe-dengue-285fc  
4. https://data.mendeley.com/  
5. https://data.mendeley.com/datasets/ywp39jnzk3/2  
6. https://data.mendeley.com/datasets/zdtc3n6xv2/3  
7. https://data.mendeley.com/datasets/673swz9tb4/1  
8. https://www.drivendata.org/competitions/44/dengai-predicting-disease-spread/  
9. https://sprint.mosqlimate.org/  
10. https://www.kaggle.com/datasets/abdullahbarayan1/dengue-fever-dataset-for-prediction  
11. https://www.kaggle.com/datasets/jocelyndumlao/dengue-hematology-insights-for-diagnosis-and-care/data  
12. https://www.kaggle.com/datasets/aravind3505/dengue-detection-dataset-clinical-data  
14. https://archive.ics.uci.edu/dataset/53/iris — **UCI Iris** (canonical **missForest** demo dataset; Stekhoven & Bühlmann / CRAN missForest README)  
15. https://cran.r-project.org/web/packages/missForest/readme/README.html — missForest package documentation  
