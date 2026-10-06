"""
Lagos Rent Estimator
MSc project: Machine Learning Models for Predicting Residential Rental Prices in
Lagos, Nigeria, with a Web Based Rent Estimation Application.
Awofolaju, Oluwatobiloba Jesufolaju - University of Lagos. Supervisor: Prof. F. A. Oladeji.
Run:  streamlit run app.py
"""
import json, joblib, numpy as np, pandas as pd, streamlit as st

st.set_page_config(page_title="Lagos Rent Estimator", page_icon="house", layout="centered")

@st.cache_resource
def load_assets():
    return (joblib.load("lagos_rent_model.joblib"),
            json.load(open("model_meta.json")),
            json.load(open("area_growth.json")))

model, meta, growth = load_assets()
st.title("Lagos Rent Estimator")
st.caption("Data driven estimate of the annual asking rent for a residential property in "
           f"Lagos, trained on {meta['total_rows']:,} rental listings collected in 2026.")

demo = st.query_params.get("demo") == "1"
areas, types = meta["areas"], meta["types"]

with st.form("f"):
    c1, c2 = st.columns(2)
    with c1:
        area = st.selectbox("Area", areas, index=areas.index("Lekki") if "Lekki" in areas else 0)
        bedrooms = st.number_input("Bedrooms", 1, 10, 3)
        serviced = st.checkbox("Serviced"); furnished = st.checkbox("Furnished")
    with c2:
        ptype = st.selectbox("Property type", types)
        bathrooms = st.number_input("Bathrooms", 1, 12, 3)
        newly = st.checkbox("Newly built")
        year = st.selectbox("Estimate for which year?", [2026, 2027, 2028, 2029, 2030], index=0)
    submitted = st.form_submit_button("Estimate rent", use_container_width=True)

if submitted or demo:
    if demo and not submitted:
        area, ptype, bedrooms, bathrooms, year = "Lekki", "Flat / Apartment", 3, 3, 2028
        serviced = newly = furnished = False
    row = pd.DataFrame([{"Area": area, "Type": ptype, "Bedrooms": bedrooms, "Bathrooms": bathrooms,
                         "Serviced": int(serviced), "NewlyBuilt": int(newly), "Furnished": int(furnished)}])
    base = float(np.expm1(model.predict(row))[0])
    st.subheader("Estimated annual rent (2026 market)")
    st.markdown(f"## NGN {base:,.0f}")
    if year > 2026:
        rate = growth["area_yearly_rates"].get(area, growth["overall_yearly_rate"])
        proj = base * (1 + rate) ** (year - 2026)
        st.subheader(f"Projected for {year}")
        st.markdown(f"## NGN {proj:,.0f}")
        st.info(f"This projection extends the growth measured in {area} between 2022 and 2026 "
                f"(about {rate*100:.1f} percent per year) and assumes that pace continues. "
                "It is a planning guide, not a promise about the market.")
    med = meta["area_medians"].get(area)
    if med:
        st.caption(f"For context, the median asking rent across all {area} listings in the data is NGN {med:,.0f} per year.")
    with st.expander("What drives this estimate?"):
        st.write("The model learned from the Lagos listings below. These shares show how much each "
                 "type of information contributes to its decisions overall.")
        for k, v in meta["importance_grouped"].items():
            st.write(f"{k}: {v*100:.1f} percent"); st.progress(min(1.0, v))

st.divider()
st.caption("This tool produces a statistical estimate of the market asking rent from listing data. "
           "It is not a professional valuation. Data source: publicly displayed listings on "
           "PropertyPro.ng, collected for academic research in 2022 and 2026.")
