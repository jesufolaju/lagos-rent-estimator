"""
Lagos Rent Estimator
====================
A web application built for the MSc project:
"Machine Learning Models for Predicting Residential Rental Prices in Lagos,
Nigeria, with a Web Based Rent Estimation Application"
by Awofolaju, Oluwatobiloba Jesufolaju (University of Lagos).

Run locally:      streamlit run app.py
The trained model (XGBoost pipeline) is loaded from lagos_rent_model.joblib.
Growth rates for future year projection come from area_growth.json and were
measured from PropertyPro Lagos snapshots collected in 2022 and 2026.
"""

import json
import joblib
import numpy as np
import pandas as pd
import streamlit as st

st.set_page_config(page_title="Lagos Rent Estimator", page_icon="🏠", layout="centered")


@st.cache_resource
def load_assets():
    model = joblib.load("lagos_rent_model.joblib")
    meta = json.load(open("model_meta.json"))
    growth = json.load(open("area_growth.json"))
    return model, meta, growth


model, meta, growth = load_assets()

st.title("Lagos Rent Estimator")
st.caption(
    "Data driven estimate of the annual asking rent for a residential property "
    "in Lagos, trained on 12,883 listings collected in 2026."
)

# Demo mode fills the form and shows a result immediately (used for screenshots).
demo = st.query_params.get("demo") == "1"

areas = meta["areas"]
types = meta["types"]

with st.form("estimate_form"):
    c1, c2 = st.columns(2)
    with c1:
        area = st.selectbox("Area", areas, index=areas.index("Lekki") if "Lekki" in areas else 0)
        bedrooms = st.number_input("Bedrooms", 1, 10, 3)
        serviced = st.checkbox("Serviced")
        furnished = st.checkbox("Furnished")
    with c2:
        ptype = st.selectbox("Property type", types)
        bathrooms = st.number_input("Bathrooms", 1, 12, 3)
        newly = st.checkbox("Newly built")
        year = st.selectbox("Estimate for which year?", [2026, 2027, 2028, 2029, 2030], index=0)
    submitted = st.form_submit_button("Estimate rent", use_container_width=True)

if submitted or demo:
    if demo and not submitted:
        area, ptype, bedrooms, bathrooms = "Lekki", "Flat / Apartment", 3, 3
        serviced = newly = furnished = False
        year = 2028

    row = pd.DataFrame([{
        "Area": area, "Type": ptype, "Bedrooms": bedrooms, "Bathrooms": bathrooms,
        "Serviced": int(serviced), "NewlyBuilt": int(newly), "Furnished": int(furnished),
    }])
    base = float(np.expm1(model.predict(row))[0])

    st.subheader("Estimated annual rent (2026 market)")
    st.markdown(f"## ₦ {base:,.0f}")

    if year > 2026:
        rate = growth["area_yearly_rates"].get(area, growth["overall_yearly_rate"])
        projected = base * (1 + rate) ** (year - 2026)
        st.subheader(f"Projected for {year}")
        st.markdown(f"## ₦ {projected:,.0f}")
        st.info(
            f"This projection extends the growth measured in {area} between 2022 and 2026 "
            f"(about {rate * 100:.1f} percent per year) and assumes that pace continues. "
            "It is a planning guide, not a promise about the market."
        )

    med = meta["area_medians"].get(area)
    if med:
        st.caption(f"For context, the median asking rent across all {area} listings in the data is ₦ {med:,.0f} per year.")

    with st.expander("What drives this estimate?"):
        st.write(
            "The model learned from 12,883 Lagos listings. The factors below show how much "
            "each type of information contributes to its decisions overall."
        )
        imp = meta["importance_grouped"]
        for k, v in imp.items():
            st.write(f"{k}: {v * 100:.1f} percent")
            st.progress(min(1.0, v))

st.divider()
st.caption(
    "This tool produces a statistical estimate of the market asking rent from listing data. "
    "It is not a professional valuation. Data source: publicly displayed listings on "
    "PropertyPro.ng, collected for academic research in 2022 and 2026."
)
