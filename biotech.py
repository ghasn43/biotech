import streamlit as st
import pandas as pd
import numpy as np
import io
import json
import math
import matplotlib.pyplot as plt

# -----------------------------
# App Config
# -----------------------------
st.set_page_config(page_title="NanoBio Studio — nanotech × biotech", page_icon="🧬", layout="wide")

# -----------------------------
# Helpers
# -----------------------------
@st.cache_data
def default_materials():
    data = [
        {"Material":"Lipid Nanoparticle (LNP)", "Core":"ionizable lipid", "Size_nm":80, "Zeta_mV":-5,
         "PDI":0.15, "Ligand":"GalNAc", "Payload":"mRNA", "Biodegradable":True, "MTD_mgkg":5,
         "BaseCost_USDmg":0.6},
        {"Material":"Gold NP", "Core":"Au", "Size_nm":30, "Zeta_mV":-20, "PDI":0.12,
         "Ligand":"RGD peptide", "Payload":"siRNA", "Biodegradable":False, "MTD_mgkg":1.2,
         "BaseCost_USDmg":1.8},
        {"Material":"Mesoporous Silica (MSN)", "Core":"SiO2", "Size_nm":120, "Zeta_mV":-30, "PDI":0.2,
         "Ligand":"Folate", "Payload":"small molecule", "Biodegradable":"Partial", "MTD_mgkg":3.5,
         "BaseCost_USDmg":0.25},
        {"Material":"Polymeric NP (PLGA)", "Core":"PLGA", "Size_nm":150, "Zeta_mV":-10, "PDI":0.18,
         "Ligand":"Antibody (scFv)", "Payload":"protein", "Biodegradable":True, "MTD_mgkg":10,
         "BaseCost_USDmg":0.15},
        {"Material":"Metal–Organic Framework (MOF-303)", "Core":"Al–fumarate", "Size_nm":250, "Zeta_mV":-15,
         "PDI":0.25, "Ligand":"Transferrin", "Payload":"enzyme", "Biodegradable":"Partial", "MTD_mgkg":2.0,
         "BaseCost_USDmg":0.4},
        {"Material":"DNA Origami", "Core":"DNA", "Size_nm":60, "Zeta_mV":-25, "PDI":0.08,
         "Ligand":"Aptamer", "Payload":"CRISPR RNP", "Biodegradable":True, "MTD_mgkg":4.0,
         "BaseCost_USDmg":2.5},
    ]
    return pd.DataFrame(data)

def _normalize_biodeg(df):
    df = df.copy()
    df["Biodegradable"] = df["Biodegradable"].apply(
        lambda x: x if isinstance(x, bool) else (str(x) if x is not None else "Unknown")
    )
    return df

@st.cache_data
def default_targets():
    data = [
        {"Target":"Hepatocytes", "Marker":"ASGPR", "PreferredLigand":"GalNAc", "Perfusion":"high", "Barrier":"low"},
        {"Target":"Tumor (integrin αvβ3)", "Marker":"αvβ3", "PreferredLigand":"RGD peptide", "Perfusion":"variable", "Barrier":"medium"},
        {"Target":"Ovarian (folate receptor)", "Marker":"FOLR1", "PreferredLigand":"Folate", "Perfusion":"medium", "Barrier":"medium"},
        {"Target":"Endothelium (transferrin)", "Marker":"TfR1", "PreferredLigand":"Transferrin", "Perfusion":"high", "Barrier":"low"},
        {"Target":"B cells", "Marker":"CD19", "PreferredLigand":"Antibody (scFv)", "Perfusion":"medium", "Barrier":"low"},
        {"Target":"Pulmonary epithelium", "Marker":"EGFR", "PreferredLigand":"Antibody (scFv)", "Perfusion":"high", "Barrier":"high"},
    ]
    return pd.DataFrame(data)

materials = _normalize_biodeg(default_materials())
targets = default_targets()

# -----------------------------
# Sidebar
# -----------------------------
st.sidebar.title("🧪 NanoBio Studio")
mode = st.sidebar.radio("Choose a workspace:", [
    "1) Materials & Targets",
    "2) Design Nanoparticle",
    "3) Delivery Simulation (PK/PD-lite)",
    "4) Toxicity & Safety",
    "5) Cost Estimator",
    "6) Protocol Generator",
    "7) Import/Export",
    "📘 Tutorial / Exercise"
])

# -----------------------------
# 1) Materials & Targets
# -----------------------------
if mode.startswith("1"):
    st.title("🧱 Materials & 🎯 Biological Targets")
    colA, colB = st.columns(2)
    with colA:
        st.subheader("Materials Library")
        st.dataframe(materials, use_container_width=True)
    with colB:
        st.subheader("Targets Library")
        st.dataframe(targets, use_container_width=True)

# -----------------------------
# 2) Design Nanoparticle
# -----------------------------
elif mode.startswith("2"):
    st.title("🧩 Design Your Nanoparticle")
    col1, col2 = st.columns(2)
    with col1:
        base = st.selectbox("Start from material", options=materials["Material"].tolist())
        size = st.slider("Hydrodynamic size (nm)", 20, 400, 100)
        zeta = st.slider("Zeta potential (mV)", -50, 50, -5)
        pdi = st.slider("PDI", 0.05, 0.5, 0.15, step=0.01)
        ligand = st.selectbox("Surface ligand", options=targets["PreferredLigand"].unique())
        payload = st.selectbox("Payload type", options=["mRNA","siRNA","CRISPR RNP","protein","enzyme","small molecule"])
    with col2:
        target = st.selectbox("Biological target", options=targets["Target"].tolist())
        dose = st.number_input("Dose (mg/kg)", 0.01, 50.0, 3.0)
        encaps_eff = st.slider("Encapsulation (%)", 1, 99, 70)
        release_t12 = st.slider("Release half-life (h)", 0.5, 72.0, 8.0, step=0.5)
        steric = st.slider("Steric shielding (0–1)", 0.0, 1.0, 0.6, step=0.05)

    design = {
        "Material": base, "Size_nm": size, "Zeta_mV": zeta, "PDI": pdi,
        "Ligand": ligand, "Payload": payload, "Target": target,
        "Dose_mgkg": dose, "Encap_%": encaps_eff,
        "Release_t12_h": release_t12, "Steric": steric,
    }
    st.json(design)
    st.session_state["design"] = design

# -----------------------------
# 3) Delivery Simulation
# -----------------------------
elif mode.startswith("3"):
    st.title("🚚 Delivery Simulation (PK/PD-lite)")

    if "design" not in st.session_state:
        st.info("Go to 'Design Nanoparticle' first. Using defaults.")
        st.session_state["design"] = {
            "Material": "Lipid Nanoparticle (LNP)",
            "Size_nm": 100, "Zeta_mV": -5, "PDI": 0.15,
            "Ligand": "GalNAc", "Payload": "mRNA", "Target": "Hepatocytes",
            "Dose_mgkg": 3.0, "Encap_%": 70, "Release_t12_h": 8.0, "Steric": 0.6
        }

    d = st.session_state["design"]

    def uptake_factor(size_nm, zeta_mV, steric):
        size_term = math.exp(-((size_nm-100)**2)/(2*40**2))
        charge_term = math.exp(-((zeta_mV+5)**2)/(2*15**2))
        steric_term = 1 - 0.4*steric
        return max(0.01, min(1.5, size_term*charge_term*steric_term))

    uf = uptake_factor(d["Size_nm"], d["Zeta_mV"], d["Steric"])
    st.markdown(f"**Uptake efficiency factor:** {uf:.2f}")

    def simulate_pkpd(dose_mgkg, encap_pct, release_t12_h,
                      k_elim=0.3, k12=0.4, k21=0.2,
                      hours=48, dt=0.25, uptake_eff=0.3):
        steps = int(hours/dt) + 1
        t = np.linspace(0, hours, steps)
        Cc, Ct, payload_free = np.zeros(steps), np.zeros(steps), np.zeros(steps)
        Cc[0] = dose_mgkg * (encap_pct/100)
        k_rel = math.log(2)/release_t12_h
        for i in range(1, steps):
            dCc = -(k_elim*Cc[i-1]) - k12*(Cc[i-1]-Ct[i-1])
            dCt = k12*(Cc[i-1]-Ct[i-1]) - k21*(Ct[i-1])
            Cc[i] = max(0, Cc[i-1] + dCc*dt)
            Ct[i] = max(0, Ct[i-1] + dCt*dt)
            released = Ct[i]*uptake_eff*(1 - math.exp(-k_rel*dt))
            payload_free[i] = payload_free[i-1] + released
        return pd.DataFrame({"time_h": t, "Cc": Cc, "Ct": Ct, "Payload_free": payload_free})

    hours = st.slider("Simulate for (hours)", 8, 168, 48, step=4)
    sim = simulate_pkpd(d["Dose_mgkg"], d["Encap_%"], d["Release_t12_h"], hours=hours, dt=0.25, uptake_eff=uf)

    st.markdown("#### Concentration–time profiles")
    fig1, ax1 = plt.subplots()
    ax1.plot(sim["time_h"], sim["Cc"], label="Central (Cc)")
    ax1.plot(sim["time_h"], sim["Ct"], label="Tissue (Ct)")
    ax1.set_xlabel("Time (h)")
    ax1.set_ylabel("Relative concentration")
    ax1.legend()
    st.pyplot(fig1)

    st.markdown("#### Released payload over time")
    fig2, ax2 = plt.subplots()
    ax2.plot(sim["time_h"], sim["Payload_free"], label="Payload released")
    ax2.set_xlabel("Time (h)")
    ax2.set_ylabel("Cumulative units (arb.)")
    ax2.legend()
    st.pyplot(fig2)

    auc_payload = np.trapz(sim["Payload_free"], sim["time_h"])
    st.metric("Exposure proxy (area under payload curve)", f"{auc_payload:.1f}")

    st.download_button("Download simulation CSV", sim.to_csv(index=False).encode(), file_name="nanobio_simulation.csv")

# -----------------------------
# 4) Toxicity & Safety
# -----------------------------
elif mode.startswith("4"):
    st.title("⚠️ Toxicity & Safety (Heuristic)")

    if "design" not in st.session_state:
        st.info("Go to 'Design Nanoparticle' first. Using defaults.")
        st.session_state["design"] = {
            "Material": "Lipid Nanoparticle (LNP)",
            "Size_nm": 100, "Zeta_mV": -5, "PDI": 0.15,
            "Ligand": "GalNAc", "Payload": "mRNA", "Target":"Hepatocytes",
            "Dose_mgkg": 3.0, "Encap_%": 70, "Release_t12_h": 8.0, "Steric": 0.6
        }
    d = st.session_state["design"]

    size_penalty = max(0, (d["Size_nm"]-120)/200)
    charge_penalty = max(0, (abs(d["Zeta_mV"]) - 15)/35)
    dose_penalty = d["Dose_mgkg"]/10
    pdi_penalty = max(0, (d["PDI"]-0.2)*2)

    risk = 3 + 3*size_penalty + 2*charge_penalty + 3*dose_penalty + 2*pdi_penalty - 1.5*d["Steric"]
    risk = float(np.clip(risk, 0, 10))
    st.metric("Composite risk score (0=low, 10=high)", f"{risk:.1f}")

    st.markdown("**Guidance (educational only):**")
    st.markdown("- Keep **PDI ≤ 0.2** for consistent behavior\n- Avoid highly **cationic** zeta unless justified\n- **Dose** conservatively; stay below MTD\n- Balance **steric shielding** with **uptake**")

# -----------------------------
# 5) Cost Estimator
# -----------------------------
elif mode.startswith("5"):
    st.title("💲 Cost Estimator (bench-scale)")

    if "design" not in st.session_state:
        st.info("Go to 'Design Nanoparticle' first. Using defaults.")
        st.session_state["design"] = {"Material": "Lipid Nanoparticle (LNP)", "Dose_mgkg": 3.0}
    d = st.session_state["design"]

    base_cost = st.number_input("Base nanomaterial cost $/mg", min_value=0.01, value=0.5, step=0.01)
    ligand_cost = st.number_input("Ligand cost $/mg", min_value=0.01, value=5.0, step=0.01)
    payload_cost = st.number_input("Payload cost $/mg", min_value=0.01, value=10.0, step=0.01)

    batch_mg = st.slider("Batch size (mg)", 5, 5000, 200)
    ligand_ratio = st.slider("Ligand mass fraction (%)", 0.1, 10.0, 1.0, step=0.1)
    payload_ratio = st.slider("Payload mass fraction (%)", 0.5, 30.0, 5.0, step=0.5)

    cost = (
        base_cost*(batch_mg*(1 - ligand_ratio/100 - payload_ratio/100)) +
        ligand_cost*(batch_mg*ligand_ratio/100) +
        payload_cost*(batch_mg*payload_ratio/100)
    )
    st.metric("Estimated batch material cost", f"${cost:,.2f}")

    subj_mass = st.number_input("Subject mass (kg)", min_value=1.0, value=70.0, step=1.0)
    mg_per_subject = d["Dose_mgkg"]*subj_mass
    doses_per_batch = max(1, math.floor(batch_mg / mg_per_subject))

    st.write(f"**Approx. doses per batch:** {doses_per_batch}")
    st.write(f"**Cost per subject (materials only):** ${cost / max(doses_per_batch,1):,.2f}")

# -----------------------------
# 6) Protocol Generator
# -----------------------------
elif mode.startswith("6"):
    st.title("📋 Protocol Generator")

    if "design" not in st.session_state:
        st.info("Go to 'Design Nanoparticle' first. Using defaults.")
        st.session_state["design"] = {
            "Material": "Lipid Nanoparticle (LNP)", "Size_nm": 100, "PDI": 0.15,
            "Ligand": "GalNAc", "Payload": "mRNA", "Target": "Hepatocytes",
            "Dose_mgkg": 3.0, "Encap_%": 70, "Release_t12_h": 8.0, "Steric": 0.6, "Zeta_mV": -5
        }
    d = st.session_state["design"]

    steps = [
        f"Formulate {d['Material']} at ~{d['Size_nm']} nm (target PDI {d['PDI']:.2f})",
        f"Conjugate ligand {d['Ligand']} and confirm activity",
        f"Load payload {d['Payload']} at {d['Encap_%']}% encapsulation",
        f"Adjust shielding (steric={d['Steric']:.2f}); confirm zeta {d['Zeta_mV']} mV",
        f"In vitro uptake in target {d['Target']} cells",
        f"Release kinetics assay; fit t½ ≈ {d['Release_t12_h']} h",
        f"In vivo PK: dose {d['Dose_mgkg']} mg/kg and collect samples",
        "Toxicology panel and histopathology",
        "Iterate and optimize"
    ]
    proto = "\n".join([f"- {s}" for s in steps])
    st.text_area("Protocol", proto, height=240)
    st.download_button("Download protocol.txt", proto.encode(), file_name="nanobio_protocol.txt")

# -----------------------------
# 7) Import/Export
# -----------------------------
elif mode.startswith("7"):
    st.title("🔄 Import / Export")

    if "design" in st.session_state:
        design_json = json.dumps(st.session_state["design"], indent=2)
        st.code(design_json, language="json")
        st.download_button("Download design.json", design_json, file_name="design.json")

    st.download_button("materials.csv", default_materials().to_csv(index=False).encode(), file_name="materials.csv")
    st.download_button("targets.csv", default_targets().to_csv(index=False).encode(), file_name="targets.csv")

# -----------------------------
# 📘 Tutorial / Exercise
# -----------------------------
elif mode.startswith("📘"):
    st.title("📘 Classroom Tutorial: Designing Nanoparticles")

    st.markdown("""
    ## 🎯 Learning Goals
    - Understand how nanoparticle parameters affect biological delivery.
    - Explore trade-offs between efficacy, toxicity, and cost.
    - Practice generating a protocol outline.

    ## 📝 Instructions

    ### Step 1: Materials & Targets
    - Browse the nanoparticle and target tables.
    - **Question:** Which pair is most promising for mRNA delivery?

    ### Step 2: Design Nanoparticle
    - Adjust size, charge, ligand, payload, dose, etc.
    - Save your design JSON.
    - **Task:** Compare a small neutral LNP (~80 nm, –5 mV) vs a large cationic NP (~200 nm, +20 mV).

    ### Step 3: Delivery Simulation
    - Run a 48-hour simulation.
    - Observe the central vs tissue concentration curves.
    - **Question:** Which design gives better tissue delivery and why?

    ### Step 4: Toxicity & Safety
    - Check the risk scores.
    - **Discussion:** How do size, charge, and dose affect toxicity?

    ### Step 5: Cost Estimator
    - Estimate costs for a 200 mg batch.
    - **Question:** Which input (ligand, payload, base material) drives the cost up most?

    ### Step 6: Protocol Generator
    - Generate a protocol for your design.
    - **Discussion:** Which step is most critical for success?

    ## 📚 Wrap-Up Discussion
    1. Which nanoparticle design would you select for liver targeting and why?
    2. How does steric shielding (PEGylation) affect circulation vs uptake?
    3. What trade-offs did you observe between efficacy, toxicity, and cost?

    ## 🎓 Suggested Assignment
    - Export your design as JSON.
    - Include simulation graphs.
    - Submit a 1-page summary covering:
      - Design rationale
      - Predicted benefits and risks
      - Cost per subject
      - Draft protocol steps
    """)
