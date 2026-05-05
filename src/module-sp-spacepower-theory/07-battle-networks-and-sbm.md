# Lesson 7: Battle Networks, Space Battle Management, and the AI-Enabled Decision Loop

**Module:** Spacepower Theory and Strategic Context — Module SP
**Source:** Todd Harrison, "Battle Networks and the Future Force" (CSIS, 2020); Todd Harrison, "Space Threat Assessment 2020" (CSIS); Alan T. Dugger, "Space as a Gray Zone: The Future of Orbital Warfare" (2024); USSF Space Capstone Publication (2020); Andrew Krepinevich, "Protracted Great-Power War: A Preliminary Assessment"; Bowen & Johnson, "From SSA to SDA: Operational Intelligence in the Space Domain" (2024); Space News, "Kronos Program Overview" (2024); RAND, "Resilience of the U.S. Defense Information Infrastructure" (2023)

---

<!-- toc -->

---

## Where this fits

The previous six lessons built the strategic theory foundation: Dolman's high ground argument, counterspace taxonomy, historical cases, Chinese doctrine, the escalation ladder, and the wargame design rationale. Every lesson has implicitly assumed a context for the tools it describes — but that context, the *operational architecture* those tools live inside, has never been stated directly.

That architecture is the battle network.

This lesson states it explicitly. A battle network is the integrated system of sensors, communications, computing, and decision authorities that connects what military forces can *see* to what they can *do*. Modern battle networks are space-dependent at every layer. SDA/SSA is the space-facing sensing component of those networks, evolving from cataloging objects to producing real-time operational intelligence. AI is the mechanism that makes this integration fast enough to matter. And orbital dominance — the strategic goal Dolman theorized and China and the United States are competing for — is, operationally, the ability to build and preserve a battle network that outperforms the adversary's.

The ML tools this curriculum builds are not research prototypes. They are components of this network. This lesson establishes the frame that makes that claim defensible.

---

## The battle network: sensing to action

Harrison's "Battle Networks and the Future Force" provides the foundational framework. A battle network consists of:

1. **Sensors** — the distributed systems that observe the operational environment (radar, optical, signals intelligence, space-based ISR, orbital tracking)
2. **Networks** — the communications infrastructure that moves sensor data to processors and processed intelligence to decision-makers (satellite communications, tactical data links, ground networks)
3. **Processing** — the systems that turn raw sensor data into actionable intelligence (fusion algorithms, anomaly detectors, intent classifiers, track managers)
4. **Decision authorities** — the humans and automated systems that act on processed intelligence (operators, command authorities, engagement systems)
5. **Effects** — the actions taken based on decisions (maneuvering satellites, executing counterspace operations, diplomatic signaling, kinetic or non-kinetic strikes)

The **OODA loop** (Observe, Orient, Decide, Act) is the battle network's operating cycle. The side that can execute this loop faster and more accurately than its adversary wins engagements — not because it has more assets, but because it can see what the adversary is doing, understand its implications, and act on that understanding before the adversary can respond.

This is the operational frame for Harrison's "force exponent" argument. Adding sensors, communications nodes, or processing capacity to a battle network does not just add capability additively — AI at each stage multiplies the effectiveness of every other node. An additional radar is a force increment. An AI-enabled data fusion system that integrates that radar with ten other sensors and delivers actionable anomaly alerts in seconds rather than hours is a force exponent: it makes the entire network more effective, not just one component.

---

## Space as the decisive substrate

Modern battle networks are not merely *supported* by space — they *run through* space at every critical layer.

**Navigation and timing**: GPS provides the precision navigation and timing that enables precision-guided munitions, coordinated joint operations, and network synchronization. Degrading GPS timing does not just affect satellite navigation — it desynchronizes the entire network, preventing communication protocols from maintaining timing relationships and causing data fusion to fail.

**Intelligence, Surveillance, and Reconnaissance**: Satellite ISR provides broad-area coverage no ground-based system can replicate. Synthetic aperture radar satellites image through clouds; optical satellites provide high-resolution imagery of denied areas; SIGINT satellites collect electronic emissions. The ISR layer of the battle network is overwhelmingly space-dependent for global peer competition.

**Communications**: Satellite communications provide the backbone for long-range, mobile, and maritime communications. AEHF (Advanced Extremely High Frequency) provides nuclear C2 communications; MUOS (Mobile User Objective System) provides UHF mobile communications; commercial SATCOM (including Starlink) provides high-bandwidth broadband. Severing satellite communications severs the network itself.

**Space Battle Management**: The USSF SCP identifies Space Battle Management (SBM) as one of seven core spacepower disciplines. SBM is the C2 layer specifically for the space domain — the systems and processes that enable operators to maintain awareness of the space environment, coordinate responses, and execute space operations as part of a joint campaign. This is the layer that connects SDA outputs to operational decisions.

The implication: an adversary who can degrade the space layer degrades the entire battle network simultaneously. This is why Harrison frames adversary ASAT systems as tools designed to "disrupt U.S. and allied battle networks that depend upon or transit through space" — the target is not individual satellites, it is the network coherence that makes those satellites militarily effective.

---

## From SSA to SDA to operational intelligence

Space Situational Awareness was the original framework: catalog objects in orbit, maintain their tracks, provide conjunction warnings. This is a cataloging function — it tells you what is in orbit and where.

Space Domain Awareness expanded this: not just "where is the object" but "what is it doing, what is its purpose, and what does its behavior imply?" The SSA-to-SDA transition is conceptually equivalent to the intelligence community's distinction between collection and analysis: SSA collects, SDA analyzes.

The operational intelligence evolution takes this a step further. In this framing, SDA outputs are not just intelligence products that an analyst reviews — they are real-time inputs to a Space Battle Management system that uses them to maintain the common operating picture, support planning and deconfliction, and enable fast decisions about orbital operations. The "Planetary Neural Network" concept — proposed as an integration vision for global SDA — describes this explicitly: a system that fuses telemetry, ground sensor data, electromagnetic spectrum data, and publicly available information into a continuously updated orbital operational picture. That is not an intelligence archive. That is a battle network node.

**Kronos** is the current programmatic embodiment of this vision for U.S. and allied operations. Described as "a modernized suite for space battle management and intelligence," Kronos fuses data in real time, supports planning and deconfliction, and provides shared awareness for U.S. and allied operators. The Space Force has also opened access to classified tracking data for commercial firms explicitly described as supporting "battle management, command and control... to see what is happening in orbit." The commercial SDA ecosystem — LeoLabs, Slingshot, ExoAnalytic, and eventually the products this curriculum is building toward — feeds into this same architecture.

The ML maneuver detection, fleet tracking, and intent inference tools built in Module 9 are not independent products. They are the sensing and processing layers of a larger battle network architecture whose C2 layer is Kronos, whose human decision layer is Space Command, and whose effects layer is both diplomatic (attributing gray zone behavior) and operational (maneuvering protected assets, executing space battle management decisions).

---

## The USSF doctrine: orbital warfare and space battle management

The USSF SCP distinguishes two disciplines directly relevant to orbital dominance:

**Orbital Warfare** is defined as "the military operations conducted to seize, retain, and exploit freedom of action in the space domain and to deny the same to an adversary." It encompasses offensive and defensive fires in the orbital environment — the capability to maneuver, protect, and if necessary, degrade adversary orbital systems. Orbital warfare is the kinetic and maneuvering component of space control.

**Space Battle Management** is the C2 complement to orbital warfare: "the art and science of assigning and directing Space Force assets to accomplish operations, missions, and tasks." SBM provides the situational awareness, planning, and coordination that enables orbital warfare to be executed coherently. Without SBM, orbital warfare is uncoordinated asset employment. With SBM, it is a synchronized campaign.

The distinction matters for understanding where the curriculum's tools fit. Maneuver detection, intent inference, and adversarial game theory are SBM functions — they inform the decision layer that coordinates the battle network. Orbital warfare (maneuvering to defend an asset, executing a proximity operations response) is the effects layer that SBM directs.

**Space Domain Awareness** is the third relevant discipline, positioned as the sensing foundation that both orbital warfare and SBM depend on. The SCP defines SDA as "the identification, characterization, and understanding of factors associated with the space domain that could affect space operations." The curriculum's entire ML pipeline — from Module 0's TLE processing through Module 9's intent inference — is an SDA capability.

---

## AI as the force exponent in space battle networks

Harrison's force exponent argument is most important at the intersection of SDA and SBM, where AI accelerates the sensing-to-decision loop:

**Sensing layer**: AI reduces the latency between an event occurring and an alert being generated. A human analyst reviewing TLE data manually might detect a significant maneuver in hours or days. An LSTM maneuver detector with a daily TLE batch update generates the alert within minutes of the batch arriving. For a fleet of 200 watched objects, no human analyst team can maintain this cadence. AI can.

**Processing layer**: AI enables inference that no human analyst can perform at scale. Intent classification — "is this approach trajectory consistent with RPO toward AEHF-6?" — requires simultaneous analysis of conjunction geometry, orbital history, approach rate, and game-theoretic context. The Module 9 intent inference pipeline does this for every watched object on every update cycle. A human analyst team might do it for three high-priority objects.

**Decision layer**: AI decision support systems compress the time from alert to decision. When the SBM system flags a high-probability RPO approach with a 77% intent confidence toward a specific asset, the Space Battle Manager has a structured alert with supporting evidence rather than a raw TLE batch requiring manual analysis. The decision cycle shortens from "analyst reviews data, writes report, briefs commander" to "system presents structured alert with confidence and recommended action, commander acts."

The force exponent effect: as AI nodes are added at each layer — better sensing, faster processing, structured decision support — the entire battle network becomes more effective, not just its individual components. This is the thesis-level claim for why SDA ML investment by the United States matters strategically: it is not building a detection product, it is accelerating the battle network cycle time relative to adversaries.

**The latency requirement**: AI's force exponent effect is conditional on having timely access to sensor data. Harrison is explicit: "AI/ML algorithms depend on having timely access to large volumes of sensor data, as well as reliable communications links to move that data." A maneuver detector that runs on 24-hour-delayed TLE data cannot support decisions that need to be made in hours. The commercial SDA integration argument — why Space Force opened classified tracking to commercial firms — is partly about sensor latency: higher-cadence commercial sensors fill the detection gaps that TLE cadence creates.

---

## Resilience: the battle network that survives first strike

A battle network that cannot absorb adversary counterspace attacks is brittle regardless of its AI capability. The resilience of the sensing and C2 architecture is a prerequisite for the force exponent to function under adversarial conditions.

**Graceful degradation** is the design principle: the network should degrade predictably under attack rather than collapsing at a single point of failure. A ground-based command network with a single fusion node is not graceful — sever the node and the network fails. A mesh architecture with distributed processing and multiple redundant paths degrades gracefully: losing one node reduces capacity but preserves function.

**Disaggregation and diversification**: The PWSA (Proliferated Warfighter Space Architecture) and SDA Tranche architecture are both responses to the fragility of the traditional model of a few exquisite satellites. Hundreds of lower-cost satellites on diverse orbital planes make it prohibitively expensive for an adversary to take down the full constellation. Disaggregation trades per-asset capability for network resilience.

**Dynamic space operations**: An adversary executing a co-orbital approach against a defended asset changes the tactical calculation if the defended asset can maneuver — becoming a moving target rather than a fixed one. The concept of "sustained space maneuver" (satellites that move frequently and unpredictably to enable evasion, deception, and responsive actions) is the orbital equivalent of dispersal and hardening. It connects directly to the game-theoretic framing of Module 8: if the Defender can move its assets, the Adversary's conjunction-masking strategy becomes much harder to execute against a target whose position is uncertain.

**Commercial backup paths**: Starlink in Ukraine demonstrated that commercial satellite communications can substitute for dedicated military communications under adversarial conditions. The resilience value of commercial space is not just cost — it is the proliferation of communication pathways that no adversary can target comprehensively. The CASR (Commercial Augmentation Space Reserve) framework is the institutional mechanism for formalizing this backup path.

---

## Adversary approaches to space battle management

**Chinese approach**: The PLA frames AI as the backbone of its space operations architecture. AI manages the PLA's satellite constellation networks in real time, automates threat analysis, and accelerates orbital decision-making cycles. Chinese doctrine treats information dominance as the first objective of conflict — "seizing command of space network dominance" — and AI-enabled space battle management as the enabling mechanism for information dominance. This is not a distant capability goal; the PLA has been fielding AI-enabled ground control systems for its satellite constellation across LEO, MEO, and GEO.

**Russian approach**: Russia's strategy is asymmetric degradation rather than parity-building. Rather than constructing a space battle management architecture comparable to U.S. and allied systems, Russia invests in capabilities specifically designed to degrade the U.S. battle network's space layer: Peresvet to blind optical ISR, Tirada-2 to jam GEO communications, Krasukha-4 for ground-based electronic warfare against satellite downlinks, Nudol for direct physical destruction. The objective is not to win a battle network competition but to prevent the U.S. battle network from functioning. This is the asymmetric degradation strategy: if you cannot match the adversary's battle network, degrade it to the point where you are competing on more level terms.

**The implications for resilience architecture**: The adversary's strategy determines what resilience must protect against. Against China, which is building toward battle network parity, resilience requires that the U.S. network maintains sufficient capability advantage that China cannot overtake it before conflict — the "race to the top" problem. Against Russia, which is executing asymmetric degradation, resilience requires that the network survives the specific attack vectors Russia has developed: blinding, jamming, and targeted kinetic destruction. These require different architectural responses — disaggregation for survivability against targeted strikes, spectrum diversity for resilience against jamming, alternative ISR paths for resilience against optical blinding.

---

## The cislunar gap in existing battle networks

The entire battle network architecture described above — PWSA, SDA constellation, Kronos, commercial SDA integration — is oriented toward LEO/MEO/GEO. Existing tracking infrastructure has poor coverage above GEO. Existing SSA systems cannot provide adequate situational awareness in the Earth-Moon system at current sensor densities.

As Module SP Lesson 1 notes, the Earth-Moon Lagrange points (EML1, EML2) provide persistent surveillance positions in cislunar space that Earth-based sensors cannot monitor continuously. The lunar south pole represents a strategic logistics node. The Chinese ILRS program and U.S. Artemis architecture are competing to establish the first operational presence at these positions.

For battle network planners, the cislunar gap means: if conflict extends to cislunar space, the SDA infrastructure that enables Space Battle Management in near-Earth orbit does not exist for the cislunar theater. There is no cislunar equivalent of Space-Track, no cislunar Kronos, and no AI-enabled maneuver detection pipeline for objects transiting Earth-Moon space. The strategic gap is not just physical presence — it is the sensing and C2 architecture for the expanded battlespace.

This is the forward research frontier that the Module 9 pipeline points toward: extending the sensing and decision loop from LEO/MEO/GEO to the cislunar theater.

---

## The curriculum as a battle network component

Assembling this into the thesis argument:

The curriculum's ML tools occupy specific layers of the space battle network:

- **Module 0** (TLEs, SGP4, conjunction analysis): the data infrastructure layer — the raw material the sensing layer consumes
- **Module 9, Lessons 1–2** (LSTM/transformer maneuver detection): the sensing layer — converting raw orbital data into maneuver alerts
- **Module 9, Lesson 3** (fleet tracking, anomaly scoring): the track management layer — maintaining situational awareness across the full watched catalog
- **Module 9, Lesson 4** (intent inference): the processing/intelligence layer — converting "this object maneuvered" into "this approach is consistent with RPO toward this asset at this confidence level"
- **Modules 5–8** (CFR, PSRO, OpenSpiel, Rust capstone): the adversary modeling layer — game-theoretic reasoning about what adversary strategies look like, enabling both better intent inference and wargame-based assessment of adversary options
- **Module 8, Lesson 6** (SBIR and government contracting): the integration path — how this capability reaches Kronos, Space Command, and allied operators

The deterrence-by-detection thesis from Module SP Lesson 5 can now be stated in battle network terms: **SDA ML capabilities that feed into Space Battle Management shorten the Defender's OODA loop, increasing the cost of gray zone operations that depend on ambiguity and slow attribution.** An adversary who knows that an AI-enabled battle network will detect, classify, and attribute their orbital behavior within the day-scale latency of Space-Track updates faces a higher-cost operating environment than one who believes their maneuvers will be attributed only after the fact, if at all.

This is the operational claim behind the strategic thesis. It is testable, it is bounded (honest about what TLE latency can and cannot support), and it is the argument that connects this curriculum to the DoD customers, government contracts, and research funding mechanisms described in Module 8.

---

## Key Takeaways

- **A battle network is the integrated sensing-processing-decision-action system that connects what forces can see to what they can do.** Modern battle networks are space-dependent at every critical layer: navigation/timing, ISR, communications, and space battle management itself. Adversary counterspace capabilities are designed to degrade the network, not just individual satellites.
- **AI provides a force exponent, not a force multiplier.** Adding an AI-enabled processing node to a battle network multiplies the effectiveness of every other node — accelerating sensing, enabling inference at scale, and compressing the decision cycle. Harrison's framework frames AI-enabled battle network superiority as the decisive asymmetric advantage in peer competition.
- **Space Battle Management (SBM) and Orbital Warfare are distinct but complementary USSF disciplines.** SBM is the C2 layer that connects SDA outputs to operational decisions; Orbital Warfare is the effects layer that SBM coordinates. The curriculum's ML tools are SDA and SBM functions, not Orbital Warfare functions.
- **The SSA → SDA → operational intelligence evolution positions SDA as a battle network node, not an intelligence archive.** Kronos is the programmatic embodiment: real-time data fusion, planning and deconfliction support, shared awareness for U.S. and allied operators. Products that can feed into Kronos have a direct commercial and operational path.
- **Resilience architecture is a prerequisite for AI force exponent effects to survive adversarial conditions.** Graceful degradation (mesh architecture), disaggregation (PWSA/SDA Tranche), dynamic space operations (maneuvering satellites), and commercial backup paths (CASR, Starlink) are the defensive countermeasures against adversary strategies designed to degrade the battle network.
- **Adversary strategies diverge: China is building toward battle network parity; Russia is executing asymmetric degradation.** These require different resilience responses — parity competition requires sustained investment in capability superiority; asymmetric degradation requires survivability against specific attack vectors (blinding, jamming, kinetic).
- **The cislunar theater is a battle network gap.** Existing SDA architecture is oriented toward LEO/MEO/GEO. There is no cislunar equivalent of Space-Track, no cislunar Kronos, and no AI-enabled decision loop for the expanded battlespace that Artemis and ILRS competition is opening.
- **The curriculum builds the sensing and processing layers of the space battle network.** Maneuver detection, fleet tracking, and intent inference are SDA functions that feed SBM. The game-theoretic tools model the adversary strategies that SBM must counter. The connection to Kronos and government contracting is the path from research tool to operational battle network component.

---

## Quiz

{{#quiz 07-battle-networks-and-sbm.toml}}
