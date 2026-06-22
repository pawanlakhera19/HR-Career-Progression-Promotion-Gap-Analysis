# Career Progression and Promotion Gap Analysis for Retention Optimization at Palo Alto Networks

A Power BI analytics project that moves beyond traditional attrition prediction to explain *why* employees disengage — by uncovering promotion stagnation, role stagnation, and career trajectory patterns. Built as part of the Unified Mentor Data Analytics program.

---

## Problem Statement

Many employees leave organizations not because of sudden dissatisfaction, but due to long gaps between promotions, stagnation in the same role, limited skill growth, and weak managerial continuity. Traditional attrition models flag *who* may leave, but they don't explain *what structural career issues* are pushing employees toward that decision.

This project addresses that gap by building a **career trajectory intelligence dashboard** that identifies promotion-stalled employees and retention opportunities — even before disengagement occurs.

---

## Dataset

- **Source:** IBM HR Analytics Employee Attrition dataset (provided via Unified Mentor, framed for Palo Alto Networks)
- **Size:** 1,470 employee records, 31 original fields
- **Attrition Rate:** 16.12%
- **Data Quality:** No missing values; cleaned and validated in Power Query

Key fields used: Department, JobRole, YearsAtCompany, YearsSinceLastPromotion, YearsInCurrentRole, YearsWithCurrManager, TrainingTimesLastYear, JobSatisfaction, EnvironmentSatisfaction, PerformanceRating, Attrition.

---

## Tools Used

| Tool | Purpose |
|---|---|
| Power Query | Data cleaning, type validation, index column |
| Power BI (DAX) | Feature engineering, KPI measures, segmentation logic |
| Power BI Desktop | Report design, visuals, conditional formatting |

---

## Methodology

### 1. Feature Engineering (Calculated Columns)
Twelve calculated columns were built to translate raw tenure data into career-intelligence signals, including:
- **Promotion Gap Ratio** = YearsSinceLastPromotion / YearsAtCompany
- **Role Stagnation Index** = YearsInCurrentRole / YearsAtCompany
- **Training Intensity Score** = TrainingTimesLastYear / YearsAtCompany
- **Manager Stability Indicator** (Stable / Frequent Change / New Joiner)
- **Promotion Gap Score** (Low / Medium / High)

### 2. Career Cluster Segmentation
Rather than a black-box ML clustering model, employees were segmented using transparent, explainable **rule-based DAX logic** into five career trajectory types:

| Career Cluster | Employees | Attrition Rate |
|---|---|---|
| Fast-Track Performer | 656 | 10.7% |
| High-Risk Stagnation | 232 | 18.5% |
| Stable Long-Term Contributor | 228 | 11.4% |
| Early-Career Explorer | 215 | 34.9% |
| Promotion-Stalled | 139 | 16.5% |

This approach was chosen deliberately — it gives HR stakeholders a fully explainable segmentation logic (auditable in DAX), rather than an opaque ML output, while still surfacing the same structural patterns a clustering model would.

### 3. Retention Opportunity Identification
A **Retention Opportunity Flag** identifies employees who are not yet disengaged (Attrition = No, JobSatisfaction ≥ 3) but show high promotion-gap signals — the highest-value group for proactive HR intervention, since generic retention efforts typically miss them.

### 4. Suggested Actions
Each flagged employee is automatically mapped to a recommended intervention: **Training, Lateral Rotation, or Promotion Review**, based on their underlying stagnation signal.

---

## Dashboard Pages

1. **Executive Overview** — Organization-wide KPIs, department-level attrition, cluster distribution
2. **Career Path Clustering Dashboard** — Cluster distribution, scatter analysis, cluster profile matrix
3. **Promotion Gap Monitor** — Department × Role promotion-gap heatmap, high-gap employee list
4. **Retention Opportunity Panel** — Flagged employees and suggested actions by department
5. **Managerial Insight Dashboard** — Manager stability vs. role stagnation, team-level stability overview

All pages are cross-filtered using synced Department and Job Role slicers.

---

## Key Insights

- **Early-Career Explorers have the highest attrition rate (34.9%)** — despite low promotion gaps, indicating early tenure churn is driven by factors other than promotion stagnation.
- **High-Risk Stagnation employees (18.5% attrition)** combine high promotion gaps with low satisfaction — the clearest actionable risk segment.
- **Fast-Track Performers have the lowest attrition (10.7%)**, validating that visible career progression is strongly linked to retention.
- A meaningful share of currently-active, satisfied employees still show high promotion-gap signals — these represent retention opportunities invisible to standard attrition models.

---

## How to Use

1. Clone or download this repository
2. Open `Palo_Alto_Networks_HR_Dashboard.pbix` in Power BI Desktop
3. Use the Department, Job Role, attribution status, promotion gap score slicers to filter any page
4. Explore the Career Path Clustering page to see the scatter-based segmentation

---

## Repository Contents

- `Palo_Alto Dashboard.pbix` — Power BI dashboard file
- `Palo_Alto_Networks.csv` — Source dataset
- `Palo_alto_network Research paper.pdf` — Full EDA, methodology, and recommendations writeup
- `Palo_Alto dashboard.Pdf - Dashboard page previews

---

## Author

**Pawan Kumar Lakhera**
Aspiring - Data Analyst | Power BI · SQL · Python
#https://www.linkedin.com/in/pawan-lakhera-738429174/ 

---

*Part of the Unified Mentor Data Analytics project series.*
