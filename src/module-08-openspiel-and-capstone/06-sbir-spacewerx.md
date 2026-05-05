# Lesson 6: From Research to Revenue — Government Contracting for SDA AI

**Module:** ML and Game Theory for Space Power — M08: OpenSpiel and Capstone
**Topic:** DoD innovation ecosystem, SBIR/STTR mechanics, SpaceWERX, OTAs, commercial strategy, ITAR, clearance path

---

> **Disclaimer:** This lesson provides orientation-level context about the government contracting landscape. Funding amounts, eligibility requirements, and regulations change frequently. Verify all details directly from official solicitations at sbir.gov, sam.gov, and SpaceWERX. Consult a licensed government contracts attorney for export control and legal compliance guidance. Nothing in this lesson constitutes legal or regulatory advice.

---


<!-- toc -->

## Where this fits

You now have working ML models for SSA. This lesson answers the question that follows: how do you get paid for them? The DoD innovation pipeline — SBIR, SpaceWERX, OTAs — is purpose-built for exactly your situation: a small, technically capable entity with a novel capability that a large prime contractor has no incentive to build. But the pipeline has gates, timelines, and eligibility traps that have surprised many first-time applicants. This lesson maps the terrain honestly.

---

## 1. The DoD innovation ecosystem: why it exists

The DoD spends enormous resources on R&D, but most of it flows through large defense primes with existing program relationships. Congress recognized decades ago that small businesses produce disproportionate innovation relative to their size. The result was a legislated set-aside structure designed to direct early-stage funding to small companies.

Two mechanisms dominate the early-stage landscape:

- **SBIR (Small Business Innovation Research)**: awards from a federal agency directly to a small business
- **STTR (Small Business Technology Transfer)**: awards to a small business with a formal research-institution partner

Both programs are funded by congressionally mandated set-asides from each participating agency's extramural R&D budget above $100M. The set-aside percentage has been phased in over time; verify the current rate at sbir.gov as it changes by statute. DoD is by far the largest SBIR/STTR funding agency. Within DoD, the Air Force (through AFWERX and SpaceWERX) has become one of the most active innovation conduits for space and emerging tech.

Understanding the ecosystem means understanding that it is not a grant program in the academic sense. It is a procurement mechanism. The government is buying the early-stage development of a capability it expects to eventually use or transition. Your proposal is a bid, not a research application.

---

## 2. SBIR eligibility: the gate before everything else

This section is non-negotiable. If any of these conditions is not met at time of award, the contract cannot be executed. Read every item carefully.

### Entity requirements

You must apply and receive the award as a **for-profit US small business entity** — not as an individual, sole proprietor under your personal name, or partnership. You need a registered legal entity (LLC or corporation) formed in a US state. Form the entity before you apply. A Phase I cannot be awarded to an individual.

### Ownership and control

More than 50% of the company must be owned and controlled by:
- US citizens, or
- Permanent residents (green card holders)

Foreign ownership above 50% is disqualifying, regardless of where the founders live.

### Employee count

Fewer than 500 employees for the applicant entity. This is rarely a constraint at the founding stage, but be aware that acquisition or affiliation with a larger company can change your size classification.

### Principal Investigator (PI) employment requirement

This is the most commonly misunderstood eligibility gate. **The PI must be primarily employed by the small business at the time of award.** "Primarily employed" means more than 50% of their total work effort goes to the company.

A graduate student who remains primarily enrolled full-time at a university does **not** meet this requirement unless they have formally transitioned their primary employment to the company. Being enrolled at UND while designating yourself as PI of a Phase I SBIR creates an eligibility problem at time of award. You have options:

- Transition to part-time enrollment or leave of absence from the university, with the company as your primary employer
- Hire or designate a different PI who is already primarily employed by the company
- Delay application until after graduation and full-time transition to the company

Talk to a government contracts attorney about your specific situation before submitting.

### SAM.gov registration

Your company must have an active SAM.gov registration at time of award. SAM.gov registration requires a Unique Entity Identifier (UEI). Allow **7–14 business days** for new entity registration, and note that SAM.gov registrations must be renewed annually. If your registration lapses, you cannot receive an award. Register early — before you submit your Phase I proposal — so a lapse or processing delay cannot block an award.

### Summary eligibility table

| Requirement | What it means | Common failure mode |
|---|---|---|
| For-profit US small business entity | LLC or corp, registered in a US state | Applying as an individual |
| >50% US citizen/permanent resident ownership | Applies at time of award | Foreign co-founder above 50% |
| PI primarily employed by company (>51% effort) | At time of award, not at time of submission | Full-time student designated as PI |
| <500 employees | Applies to the applicant entity | Affiliation with a larger company |
| Active SAM.gov registration | Renewed annually, UEI required | Lapsed registration at award time |

---

## 3. SBIR Phase I and Phase II mechanics

### Phase I

Phase I is a feasibility study. The question you are answering is: does the technical approach work in principle? You are not required to have a finished product — you are required to demonstrate that your approach is credible, that you understand the problem, and that you can execute.

Phase I awards are typically in the $150K–$300K range for 6–9 months. Confirm current limits from the specific solicitation — SpaceWERX sets its own ceilings within DoD guidance, and these change annually.

**Success rates**: Competitive DoD SBIR Phase I acceptance rates on open topics at SpaceWERX/AFWERX are typically 15–25%. First-time applicants without prior DoD relationships may be lower. Plan for multiple submission cycles. The gap between solicitation open, submission, review, and award notification is often 3–6 months per cycle. SBIR is a valid path to non-dilutive government funding, but it is not a quick or guaranteed one. Budget at least 12–18 months from first submission to first Phase I dollar received.

### Phase II

Phase II is full prototype development. It builds directly on Phase I results and requires a completed Phase I (from any SBIR agency) unless you qualify for Direct-to-Phase-II (see below). Phase II awards are typically $1.5M–$2M for 2 years. Confirm current limits from the specific solicitation.

Phase II proposals are submitted by Phase I awardees and are evaluated on Phase I results, prototype feasibility, and transition potential. Not all Phase I awards receive Phase II follow-on — transition rates vary by topic and program office. DoD expects a clear commercialization and transition plan.

### Direct-to-Phase-II (DP2)

DP2 allows skipping Phase I for companies with a demonstrated track record. This requires prior SBIR/STTR Phase I experience from **any** federal agency — it cannot be the first award a company has ever received. If you have no prior SBIR history at any agency, you are not eligible for DP2.

### Comparison table

| Vehicle | Typical amount | Typical timeline | Key eligibility gate | Primary requirement |
|---|---|---|---|---|
| Phase I | $150K–$300K | 6–9 months | All SBIR eligibility criteria | Feasibility study |
| Phase II | $1.5M–$2M | 24 months | Completed Phase I | Prototype development |
| DP2 | $1.5M–$2M | 24 months | Prior SBIR Phase I (any agency) | Prior award history |
| Pitch Day award | Varies (see §5) | Conditional, +30–90 days to execute | SpaceWERX application gate | Competitive pitch |
| OTA | Varies, typically $1M+ | 6–18 months | Non-traditional contractor on team | Agreement, not a contract |

*Verify all amounts against current solicitations. These figures change annually.*

---

## 4. SpaceWERX and AFWERX specifically

SpaceWERX is the United States Space Force's innovation arm, operating under the broader AFWERX umbrella. It was stood up specifically to accelerate transition of commercial space capabilities into the Space Force, and Space Domain Awareness is one of its explicit focus areas.

SpaceWERX publishes SBIR solicitations on its own portal (ussf.mil/spacewerx) and through AFWERX channels, in addition to the government-wide SBIR solicitations at sbir.gov. Topic areas relevant to SDA AI have included SSA data fusion, conjunction assessment automation, and on-orbit situational awareness.

Key distinctions from other DoD SBIR programs:

- SpaceWERX accepts proposals through a rolling or periodic solicitation model as well as the omnibus DoD SBIR solicitation cycles
- It emphasizes transition early — the expectation is that Phase II will lead to a Space Force program of record or a commercial product with dual-use value
- The program office relationships matter; attending SpaceWERX industry days and technical exchange meetings is how you learn which topics are genuinely funded versus pro forma

---

## 5. Pitch Days

SpaceWERX and AFWERX run competitive pitch events that are sometimes described as "award in the afternoon" events. The marketing language can be misleading. The actual outcome of a successful Pitch Day pitch is a **conditional letter of intent or conditional award** — a signal that the program office intends to award, subject to contract execution. Contract execution typically takes an additional **30–90 days** beyond the event.

Pitch Days are application-gated. You must apply through the SpaceWERX Accelerator or AFWERX SBIR portal in advance, typically months before the event. Not all applicants are selected to pitch. Pitch Day is not a walk-in event.

If you are invited to pitch: prepare a crisp 5-minute technical and commercialization story. Program office evaluators are looking for a credible capability, a clear user problem, and a realistic path to transition. Slides matter less than whether the technical lead can answer hard questions in the Q&A.

---

## 6. Other Transaction Authorities (OTAs): a later-stage vehicle

OTAs are cooperative agreements (not contracts) that allow the government to move faster than the Federal Acquisition Regulation (FAR) normally permits. They are used for prototype agreements and can transition to production without a re-compete if the prototype succeeds.

**Critical eligibility point**: OTA agreements for prototypes require that **at least one non-traditional defense contractor** be meaningfully involved in the work. A non-traditional defense contractor is a company that has not received more than $1 million in DoD contracts in the prior fiscal year (under the FAR cost accounting standards). Solo founders can qualify as non-traditional, but:

- OTAs are typically managed by larger consortia (Consortium Management Organizations like NSTXL, AFWERX OTA consortium) that require membership fees and established relationships
- Solo founders typically cannot receive OTAs as prime without a team that includes the required non-traditional contractor involvement
- OTAs are generally a **later-stage vehicle** — after you have demonstrated a prototype through Phase I or Phase II SBIR, and have a program office interested in a faster path to a production contract

File OTAs under "vehicles to qualify for at year 2–3," not "path to first revenue."

---

## 7. Commercial-first strategy: the Slingshot/Kayhan model

Many of the SDA AI companies that have successfully received SBIR and government contracts did not start with SBIR. Slingshot Aerospace, Kayhan Space, and similar companies built commercial products and revenue first — satellite operator contracts, conjunction alert services, insurance risk scoring for satellite insurers — and used that commercial traction as the credibility basis for later government work.

**Why this works**:

- A Phase I proposal is significantly stronger when you can write "our system has processed X months of real SSA data for Y operators" instead of "we propose to build such a system"
- Commercial contracts with satellite operators are more accessible for a first-time founder than winning a competitive SBIR: the procurement cycle is shorter, the relationship is direct, and there is no SAM.gov eligibility gate
- Commercial revenue while you wait for SBIR cycles (each 3–6 months) is how you stay solvent

**Realistic commercial customers for early-stage SDA AI**:

- Small satellite operators (sub-GEO LEO constellations) who cannot afford dedicated conjunction analysis staff
- Satellite insurers and underwriters who need risk scoring for coverage pricing
- Space traffic management research consortia (academic and non-profit)
- Allied-nation space agencies with less mature SSA infrastructure than the US (note: any international engagement touches ITAR — get export control review first)

**Honest tradeoff table: SBIR-first vs. commercial-first**

| Dimension | SBIR-first | Commercial-first |
|---|---|---|
| Funding type | Non-dilutive government contract | Equity/revenue from commercial sales |
| Timeline to first dollar | 12–18 months from first submission | 3–9 months if you have a paying customer |
| Success probability | 15–25% per Phase I cycle | Depends on sales, no fixed rate |
| Eligibility gate | Entity, SAM.gov, PI employment | None (export control still applies) |
| Proposal writing load | High — government proposals are extensive | Low — commercial sales process |
| Government credibility | Strong after Phase I award | Requires commercials to be compelling to program offices |

---

## 8. Hybrid strategy: commercial proof-of-concept then SBIR

The hybrid approach is arguably the most realistic path for a solo founder with technical depth:

1. **Months 0–6**: Build a minimal viable SDA AI product (conjunction risk scoring, RSO characterization, whatever your strongest capability is). Deploy against public catalog data (Space-Track, Celestrak). Get one paying or LOI-holding commercial customer, even at a nominal contract value.

2. **Months 3–9** (overlapping): Form your legal entity. Register on SAM.gov. Identify the SpaceWERX SBIR topic areas that match your capability. Attend at least one industry day or technical exchange.

3. **Months 6–12**: Submit a Phase I proposal. Your "prior work" section now points to a deployed system and a commercial customer. This is worth more in the proposal than any academic publication.

4. **Months 12–18**: Wait for Phase I results. If selected, execute Phase I work while continuing to grow commercial business. If not selected, revise and resubmit. Use the reviewer feedback.

The commercial proof-of-concept serves double duty: it generates revenue during the SBIR waiting period, and it makes your Phase I proposal materially stronger.

---

## 9. STTR: when it works and when it doesn't

STTR (Small Business Technology Transfer) differs from SBIR in one critical way: it requires a formal subcontract with a US research institution, and that institution must perform **at least 30% of the work** under a genuine subcontract with its own deliverables. This is not a name-on-the-proposal arrangement — the institution must do real work and receive real payment.

### When STTR makes sense

STTR is designed for situations where a university or Federally Funded Research and Development Center (FFRDC) has unique technical capabilities (equipment, IP, datasets) that the small business genuinely needs and cannot replicate. A partnership with a university that has an active SSA research lab, tracking radar, or unique orbital data access is a real STTR case.

### The UND situation: be honest with yourself

UND's Space Studies program is a distance-learning policy and history graduate program. It does not have active SSA/ML research labs, radar infrastructure, or a track record of SBIR-relevant technical subcontract work. Simply having a UND affiliation does not create a viable STTR partnership.

Using UND as an STTR research partner requires:

- Identifying a specific faculty member with genuine technical expertise relevant to your topic (ML, SSA, astrodynamics — not policy or history)
- Confirming that faculty member has time and institutional support to serve as the PI on the university side
- Executing an IP agreement between your company and the university before submission — this is negotiated through UND's tech transfer office and typically takes **4–8 weeks**; universities often assert IP rights over work performed by graduate students using university resources
- Having the university genuinely perform 30% of the technical work with its own subcontract deliverables

If no such faculty member exists at UND, alternative institutions with active SSA/space research programs include:

| Institution | Relevant program |
|---|---|
| Colorado School of Mines | Space Resources Program; space traffic management research |
| University of Colorado Boulder (LASP) | Laboratory for Atmospheric and Space Physics; active space data systems research |
| Embry-Riddle Aeronautical University | Space Physics research; orbital mechanics faculty |
| Purdue School of Aeronautics and Astronautics | Astrodynamics, SSA |
| MIT Lincoln Laboratory (FFRDC) | Active SSA research; FFRDC structure makes subcontracting complex but possible |

STTR is also significantly more administratively complex than SBIR. Between IP negotiation, subcontract execution, and joint reporting, plan for substantially more overhead. For a first award, SBIR is almost always simpler. Pursue STTR when you have a specific, genuine technical need that a named institution can uniquely fill.

---

## 10. ITAR and export control: do not assume

The International Traffic in Arms Regulations (ITAR) control the export of defense articles and defense services on the United States Munitions List (USML). Space systems and their components — including SSA-related technology — appear on the USML. This has direct implications for any AI model you build for SDA applications.

**Whether your trained model is an ITAR-controlled technical item depends on factors including what the model can do, what data it was trained on, and its intended use.** Assumptions are dangerous here. You cannot simply decide "I trained this on public data, so it must be fine." The technical capability of the model — not just its training data provenance — is relevant to the ITAR analysis.

Get a formal export control review from a licensed export control attorney before:

- Commercializing any model intended for space or defense applications
- Sharing model weights, architectures, or technical documentation with any foreign person (including foreign nationals at US universities)
- Accepting investment from any entity with foreign ownership
- Selling or licensing the model to any non-US customer

**What your Blue Origin work history gives you**: Employment at an ITAR-registered company gives you demonstrated familiarity with ITAR compliance procedures and the ability to speak credibly about export control awareness in a proposal context. This is useful resume context. It is not a credential that appears in any federal system, and it is not equivalent to a clearance or a formal export control determination. It does not substitute for a legal review of your specific product.

Key terms to know before talking to a contracts attorney:

| Term | Meaning |
|---|---|
| ITAR | International Traffic in Arms Regulations; administered by State Dept. Directorate of Defense Trade Controls (DDTC) |
| EAR | Export Administration Regulations; administered by Commerce Dept. BIS; covers dual-use items |
| USML | US Munitions List; items on this list are ITAR-controlled |
| CCL | Commerce Control List; items subject to EAR |
| Technical data | Data (including software, models, parameters) that can be used to design, produce, or operate a USML item |
| Foreign person | Any person who is not a US citizen, lawful permanent resident, or protected individual under 8 U.S.C. §1324b(a)(3) |

---

## 11. Clearance path: how it actually works

A common misconception: that a small business owner can initiate the process of getting a facility clearance (FCL). This is not how it works.

**How clearances are actually granted**:

1. A government program office or prime contractor determines they need a vendor to have access to classified information in order to perform work on a specific contract
2. That program office or prime *sponsors* the company for an FCL through the Defense Counterintelligence and Security Agency (DCSA)
3. DCSA conducts the investigation and grants (or denies) the FCL
4. Timeline from initiation of sponsorship to granted FCL: typically **12–24 months**

**The company cannot initiate this process on its own.** There is no form to file with DCSA to start the clock. You need a government entity or cleared prime to sponsor you because they need you to do classified work.

**What triggers clearance sponsorship**:

- A government program office awards you a contract with classified performance requirements
- A cleared prime contractor brings you in as a subcontractor on a classified task
- In the SSA context: Phase II SBIR contracts in unclassified SSA work typically do **not** trigger clearance sponsorship. Most early-stage SDA AI work is unclassified

**The realistic path**:

- Establish yourself in unclassified SSA work via SBIR Phase I/II, commercial contracts, or think tank partnerships
- Build relationships with cleared primes (Leidos, Booz Allen, Peraton, Palantir, etc.) who work on classified SSA programs
- When a cleared prime has a classified subcontract opportunity that matches your capability, they can begin the sponsorship process
- You can also build toward a cleared facility by hiring a key employee who already holds a clearance — that can accelerate some elements of the process

The clearance path is a government-driven process, not a founder-initiated one. Plan your product roadmap around unclassified work for the first 2–3 years.

---

## 12. Product roadmap: realistic 3-year arc

The following arc reflects honest timelines, not optimistic projections.

| Period | Priority actions | Expected outcomes |
|---|---|---|
| **Months 0–3** | Form legal entity. Register SAM.gov (do this first — it takes 7–14 days). Identify initial commercial customer. Begin export control review. | Entity exists. SAM.gov active. One paying or LOI-holding customer. |
| **Months 3–9** | Build and deploy minimal commercial product. Attend SpaceWERX industry day. Identify best-fit Phase I topic. Draft proposal with PI employment compliant. | Working product. Proposal in draft. |
| **Months 6–12** | Submit Phase I proposal. Maintain commercial product. Seek second commercial customer. | First SBIR submission. Product growing. |
| **Months 12–18** | Await Phase I decision (15–25% acceptance). If selected: begin Phase I work. If not: revise and resubmit next cycle. | Phase I award or second submission cycle. |
| **Months 18–30** | Execute Phase I deliverables. Build Phase II proposal. Grow commercial revenue. Build prime contractor relationships. | Phase I deliverables. Phase II submission. |
| **Months 30–36** | Phase II award (if Phase I successful). Hire first employee. Begin cleared prime teaming conversations. | First $1.5M+ government contract. Path to facility clearance via prime sponsorship begins. |

This is not the only path, and not every company follows it linearly. Commercial traction can compress timelines; SBIR rejection cycles can extend them. What matters is treating each phase as a real milestone with measurable criteria, not a hoped-for outcome.

---

## Key Takeaways

- **SBIR eligibility has hard gates**: for-profit entity, SAM.gov registration, PI primarily employed by company. Verify every condition before submitting. A Phase I cannot be awarded to an individual.
- **Phase I acceptance rates are 15–25%** for competitive DoD topics. Budget for multiple cycles. SBIR is non-dilutive and valuable, but slow.
- **Commercial-first is a legitimate strategy**: Slingshot, Kayhan, and others built commercial revenue before SBIR. Direct commercial contracts with satellite operators are more accessible as a first-revenue path.
- **Hybrid works best**: commercial proof-of-concept in months 0–6, then use that as prior work in your Phase I proposal.
- **STTR with UND requires honest self-assessment**: UND Space Studies is not an SSA/ML research lab. STTR requires a specific faculty member, genuine subcontract work (≥30%), and IP negotiation before submission.
- **ITAR analysis is not optional and is not simple**: get a licensed export control attorney before commercializing any model for space or defense use. Do not assume public training data means the model is uncontrolled.
- **Clearance is government-initiated, not founder-initiated**: build unclassified credibility first, then pursue cleared prime relationships that can lead to sponsorship.

---

## Quiz

{{#quiz 06-sbir-spacewerx.toml}}
