# Medical Billing Denial Rules

Knowledge base for claims denial resolution — first 3 rules.

| Rule ID | Description | Results |
|---------|-------------|---------|
| rule_001 | This rule applies to Group 1, practice PRC001, insurance: Medicare, plan: MED. It covers CPT code(s) 99454\|99457\|99458 with denial code 151 and remark code All. Category: Remote Physiologic Monitoring. Keywords:99454\|99457\|99458\|151\|Medicare\|RPM | Action: Adjust Claim. When Medicare denies remote physiologic monitoring CPTs, they can be adjusted. |
| rule_002 | This rule applies to Group 1, practice PRC001, insurance: Medicare, plan: MED. It covers CPT code(s) 95251 with denial code 97 and remark code All. Category: Inclusive Denial. Keywords:95251\|97\|inclusive\|Medicare | Action: Adjust Claim. When Medicare pays E&M and denies 95251 as inclusive, need to adjust. |
| rule_003 | This rule applies to Group 2, practice PRC002, insurance: All, plan: All. It covers CPT code(s) 83036\|84443\|82044\|80061\|80053\|85025 with denial code 50 and remark code M25. Category: Diagnosis Validation. Keywords:M25\|50\|I10\|83036\|84443 | Action: Verify Diagnosis. Ensure that diagnosis code I10 is correctly linked to CPT 83036, 84443, 82044, 80061, 80053, and 85025. |

---

## Notes

- Rule IDs follow format rule_NNN
- Description contains group, practice, insurance, CPT codes, denial codes, and keywords
- Results format: `Action: <action>. <instruction>`