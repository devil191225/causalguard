# Commercialization: Data Strategy and Path to Product

This note frames how to think about **real-world evidence** for an ICS/OT anomaly product, how public datasets rank for **commercial usefulness** (not only research merit), and a short **path to productize** this repository.

---

## 1. Four buckets of data

If the goal is **data you can actually use** in R&D and eventually in sales conversations, sources fall into four buckets:

1. **Public ICS benchmark datasets** — Accessible, reproducible, good for papers and baselines; rarely sufficient alone to prove value on a customer’s plant.
2. **Public OT network traces** — Strong for protocol- and flow-centric detection; limited for deep **process causal** modeling unless paired with semantics.
3. **Partner or customer telemetry** — Historian/SCADA exports under contract; strongest for proof buyers respect.
4. **Your own testbed-generated data** — Full control of rights and scenarios; essential for demos and regression tests.

**Uncomfortable truth:** truly representative plant telemetry with full fidelity is **rarely public**. Serious teams usually use public benchmarks for early R&D, then pivot to **partner or owned data** for productization.

---

## 2. Public benchmarks (good starting points)

Useful references for process- or cyber-physical anomaly work:

| Resource | Role |
| -------- | ---- |
| **HAI** | Hardware-in-the-loop ICS testbed (turbine, boiler, water-treatment style components); widely used for process-aware anomaly detection. [HAI on GitHub](https://github.com/icsdataset/hai) · [USENIX CSET20 paper](https://www.usenix.org/conference/cset20/presentation/shin) |
| **BATADAL / BATADAL 2.0** | Water distribution process data; 2.0 adds network-oriented material—useful when you want process + communications storylines. [Cornell / Galelli data](https://galelli.cee.cornell.edu/research/data/data-analytics/) |
| **ICS-NAD** | Large corpus (network PCAP + labeled CSV features, multiple vendors and attack types) for network-centric validation and scale. [Scientific Data article](https://www.nature.com/articles/s41597-026-06738-x) |
| **Tommy Morris / UAH ICS datasets** | Classic gas pipeline, water tank, power-style scenarios; older but still useful for methods work. [UAH ICS data sets](https://sites.google.com/a/uah.edu/tommy-morris-uah/ics-data-sets) |
| **Cyber4OT** | OT packet traces with normal and attack traffic (Zenodo); strong for network analytics side validation. [Zenodo record](https://zenodo.org/records/13935639) |

These are strong for **benchmarking, ablations, and demos**. They are usually **not**, by themselves, proof that the product works in a given customer’s OT environment (lab constraints, licensing, and distribution shift).

---

## 3. “Real” data for a sellable product

The sources that most often produce evidence **buyers** care about:

- **Design partners**: utilities, manufacturers, campus plants, integrators—historian or SCADA exports under NDA.
- **MSSPs / OT monitoring vendors** willing to co-pilot on sanitized or joint-use data.
- **System integrators and OEMs** with demo lines, training plants, or managed sites.
- **Your internal testbed** with PLCs or soft PLCs, historian logging, and scripted scenarios.

Partner and owned data usually capture **site-specific behavior, drift, maintenance windows, and nuisance alarms** that public sets miss. Public benchmarks can show you **built** something; partner data is what shows it **works** in conditions closer to production.

---

## 4. What to collect (especially for a causal product)

The highest-value inputs are not “packets only”:

- **Process telemetry**: sensors, actuators, setpoints, controller modes, alarms.
- **Context**: tag naming, asset inventory, one-line or P&ID-style process knowledge, operating modes, scheduled maintenance.
- **Security telemetry** (if in scope): OT flows, protocol fields, engineering-station activity.
- **Labels or weak labels**: incidents, operator actions, known upsets, maintenance events.

A causal anomaly story gets stronger when deviations link to **operational context**, not only to unlabeled vectors.

---

## 5. Best acquisition path (realistic sequence)

1. Use **HAI + BATADAL (+ ICS-NAD where network claims matter)** as a **public benchmark suite** for credibility and repeatable comparisons.
2. Build a **small demo testbed** (Modbus/OPC UA, historian, scripted attacks) so you own **commercially clear** evaluation data.
3. Engage **2–3 design partners**; ask first for **historian exports** or narrow slices—not full raw production—because legal and ops friction is lower.
4. Contract for **de-identified telemetry** and rights to improve **derived** models and evaluation artifacts.

If you wait for perfect public “real plant” dumps, you may wait indefinitely. The usual winning pattern is **hybrid**: public benchmarks early, **owned** data for rights and demos, **partner** data for validation.

---

## 6. Ranked by commercial usefulness

Scoring question: *How much does this help a paying OT customer believe the product?*

| Rank | Source | Commercial usefulness | Notes |
| ---: | ------ | ---------------------- | ----- |
| 1 | **Partner / customer telemetry (NDA)** | Very high | Closest to product truth: real tags, modes, drift, maintenance. |
| 2 | **Your own testbed data** | High | You control IP and scenario design (spoofing, ramps, mimicry). |
| 3 | **ICS-NAD** | High (network-heavy products) | Very large, multi-vendor, attack-typed network corpus; weaker for pure process causality without extra semantics. |
| 4 | **BATADAL 2.0** | Medium–high | Process + network angle useful for cross-layer story. |
| 5 | **HAI** | Medium–high | Solid process-aware public benchmark from a realistic lab setup. |
| 6 | **Cyber4OT** | Medium | Packet-focused; limited for sensor/actuator causal graphs alone. |
| 7 | **UAH / Morris-style datasets** | Medium | Classic but dated vs. buyer expectations. |
| 8 | **SWaT** (this repo’s default benchmark) | Medium for research; **lower for direct commercial claims** | Famous for benchmarks; risk of overfitting narratives and **licensing/terms** for commercial use must be verified independently. |

**Best fit for this project’s angle (“explainable anomaly on industrial telemetry”):** prioritize **process-rich** sets (HAI, BATADAL) and **partner historian** data; add **ICS-NAD / Cyber4OT** when the product narrative includes **OT network** behavior.

---

## 7. Practical scorecard (blunt)

| Source | Score (/10) | Reality |
| ------ | ------------- | ------- |
| Partner telemetry | 10 | Hardest to obtain; strongest sales engineering evidence. |
| Own testbed | 9 | Best control; use for demos and regression. |
| ICS-NAD | ~8 | Network product validation; not a substitute for plant semantics. |
| BATADAL 2.0 | ~8 | Strong bridge from process research to product story. |
| HAI | ~7.5 | Strong public process-aware benchmark. |
| Cyber4OT | ~6.5 | Packet analytics. |
| UAH / Morris | ~6 | Legacy but usable. |
| SWaT | ~5 for *commercial* positioning | Still high for academic comparability. |

---

## 8. Recommended stack for this codebase

Aligned with an **evidence ladder**:

- **Phase A — Public:** HAI + BATADAL 2.0 (+ ICS-NAD if you sell network-layer robustness).
- **Phase B — Owned:** Testbed telemetry for demos, licensing clarity, and repeatable attack scripts.
- **Phase C — Partner:** De-identified historian + metadata for tuning and customer-specific validation.

That story beats “we only tested on one public benchmark.”

**First segments to approach** (often easier pilots than national critical infrastructure core): **water utilities, smaller municipal operators, industrial integrators, campus facilities.**

---

## 9. Steps to productize this project (concise)

1. **Freeze a product boundary** — e.g. “process historian anomaly + causal explanation + assurance report” *versus* full OT IDS; avoid scope that requires ICS-NAD-scale PCAP ingestion unless committed.
2. **Harden the pipeline** — turn scripts into a **versioned service or job runner**: config, logging, secrets, deterministic builds, reproducible evaluation notebooks or CI jobs.
3. **Ingest reality** — one **OPC UA / Modbus / CSV** path and a minimal **historian adapter**; normalize timestamps, tags, and train/infer separation.
4. **Operationalize models** — scheduled **retrain**, **drift checks**, model/graph versioning, rollback; document what “verification” means in customer language (empirical checks, not physical proofs).
5. **UX and workflow** — alert triage, graph + narrative view, export for SOC/incident response; RBAC and audit logs for OT buyers.
6. **Legal and packaging** — data-processing terms, on-prem or private cloud deployment, support tiers; avoid overselling “formal verification” as mathematical certification of the plant.
7. **Evidence** — run the **Phase A → B → C** data ladder and capture **before/after** metrics and partner testimonials for sales.

---

## 10. Reference links (consolidated)

- [HAI dataset (GitHub)](https://github.com/icsdataset/hai)  
- [BATADAL data (Cornell)](https://galelli.cee.cornell.edu/research/data/data-analytics/)  
- [ICS-NAD (Scientific Data / Nature)](https://www.nature.com/articles/s41597-026-06738-x)  
- [UAH / Tommy Morris ICS datasets](https://sites.google.com/a/uah.edu/tommy-morris-uah/ics-data-sets)  
- [Cyber4OT (Zenodo)](https://zenodo.org/records/13935639)  
- [HAI / CSET20 (USENIX)](https://www.usenix.org/conference/cset20/presentation/shin)

---

*This document reflects a pragmatic go-to-data and go-to-market framing; licensing, terms of use, and segment-specific compliance must be validated with counsel and each dataset’s license before commercial use.*
