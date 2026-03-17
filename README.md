# Enterprise-Style-Knowledge-Assistant-with-Role-Based-Answers
An enterprise-style knowledge assistant with role-based answers, retrieval using metadata filters, ranking logic for document freshness.

| Document | Level | Role Access | Authority | Department | Date |
|---|---|---|---|---|---|
| EKAA L1 Theory Manual | 1 | student, practitioner, trainer | official_curriculum | training | 2020-03-01 |
| EKAA L2 Modalities Manual | 2 | practitioner, trainer | official_curriculum | training | 2020-03-01 |
| EKAA L3 Advanced Manual | 3 | practitioner, trainer | official_curriculum | clinical | 2020-03-01 |
| EKAA L5 Clinical Manual | 5 | trainer | official_curriculum | clinical | 2020-03-01 |

student → Level 1 only (max_level = 1)
practitioner → Levels 1, 2, 3 (max_level = 3)
trainer → All levels (max_level = 5)

