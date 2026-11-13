# Perishable inventory heuristic: Streamlit prototype

Prototype decision-support system for

> Gulecyuz, S., O'Sullivan, B. and Tarim, S.A. (2025). *A heuristic method for perishable inventory
> management under non-stationary demand.* Omega 133, 103267. https://doi.org/10.1016/j.omega.2024.103267

The code uses the paper's notation (Table 1) and only the paper's data. The instances are the Section 6.3
example, the 216 instances of Tables 2–3 (numbered as in Appendix Tables 8–13) and the 12 instances of Tables 4–5.

## Files

| File | What it holds |
|---|---|
| `paper_data.py` | Every number taken from the paper: Tables 1–13 and the Section 6.3 values |
| `demand.py` | Poisson and discretised normal demand; F_{i,t} by convolution |
| `heuristic.py` | Φ (Eqs. 3–5), costs (Eq. 1), EWA, S_ij (Eq. 17), c(i,j) (Eq. 11), Algorithm 1 (Eqs. 18–20), Step 2 simulation (Section 6.2) |
| `sdp.py` | SDP benchmark, Eq. (10), for M = 2 and M = 3 |
| `example63.py` | Algorithm 1 replayed with the arc costs of Fig. 2 |
| `app.py` | Streamlit app: 1 Input data, 2 Run model, 3 Results, 4 Paper check |
| `run_experiments.py` | Re-runs Section 7 and writes `results/table7_instances.csv` and `results/table6_instances.csv` |
| `tests/test_paper.py` | 20 test cases built from the paper's data |

## Run locally

```bash
pip install -r requirements.txt
streamlit run app.py
python -m pytest -q             # tests
python run_experiments.py       # optional: about 2 minutes, refreshes results/
```

Open the local app at http://localhost:8501. This address is only available on the computer running Streamlit.

## Deploy on share.streamlit.io

1. Create a public GitHub repository and upload the contents of this directory (keep `results/` and `.streamlit/`).
2. Go to https://share.streamlit.io, choose **Create app → Deploy a public app from GitHub**.
3. Select the repository, branch `main`, main file path `app.py`, then **Deploy**.
4. Share the public app URL shown after deployment. A local `localhost` URL cannot be opened by other people.

## Reproduction summary (this implementation vs. the paper)

| | Paper | This implementation |
|---|---|---|
| Table 7, avg heuristic total cost (216 instances) | 852.92 | 854.10 |
| Table 7, avg SDP cost, M = 2 / M = 3 | 873.85 / 824.86 | 872.99 / 823.71 |
| Table 7, avg gap, M = 2 / M = 3 | 1.07% / 1.88% | 1.47% / 2.02% |
| Table 7, avg cycle length | 2.26 | 2.23 |
| Table 6, avg heuristic / SDP cost | 691.34 / 669.77 | 695.21 / 669.13 |
| Section 6.3, TC(4) with Y_4 = (1,1,1,0) | 116.2 | 115.2 ± 0.4 |

Known differences: the SDP here is solved for M ≤ 3 only (the paper needed 4,954 s per M = 4 instance in C);
S_ij is the smallest whole number satisfying Eq. (17), while some S values printed in Section 6.3 follow other
rounding (the app can replay the printed values for the example).
