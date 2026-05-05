# Lesson 2: Counterspace Operations and the New RMA

---


<!-- toc -->

## Start with the taxonomy

The Secure World Foundation publishes an annual *Global Counterspace Capabilities* assessment — an open-source analysis of every state's demonstrated ability to attack, degrade, or destroy space systems. It is the closest thing to a public intelligence assessment of the counterspace landscape. The April 2025 edition opens with a quote from the USSF Space Force Delta Doctrine 1 (SFDD-1):

> "Space is a warfighting domain. This is not aspirational — it is an acknowledgment of the operational reality."

That statement anchors everything in this lesson. Counterspace is not a hypothetical future capability — it is an operational present. The taxonomy below describes capabilities that have been tested, demonstrated, or operationally deployed by multiple state actors.

---

## The counterspace taxonomy

The standard way to organize counterspace capabilities is along two axes: **kinetic vs. non-kinetic** and **reversible vs. irreversible**. This produces four quadrants.

```
                    REVERSIBLE          IRREVERSIBLE
                ┌──────────────────┬──────────────────────┐
                │ Electronic attack │ Kinetic ASAT         │
  KINETIC       │ (jamming,        │ (direct-ascent,      │
                │ spoofing RF)     │ co-orbital KE)       │
                ├──────────────────┼──────────────────────┤
                │ Cyber attack     │ High-power laser      │
  NON-KINETIC   │ Dazzling (laser) │ (permanent sensor     │
                │ Spoofing signals │ damage)              │
                └──────────────────┴──────────────────────┘
```

This taxonomy requires some clarification, because "kinetic" in this context means physically interacting with a spacecraft, not necessarily violent:

**Kinetic reversible**: Electronic jamming of uplink or downlink — the satellite is unaffected, only the signal is disrupted. Spoofing GPS signals is kinetic in the RF sense, reversible when the spoofer stops transmitting.

**Kinetic irreversible**: Direct-ascent ASAT (launching a missile that physically impacts a satellite, creating debris). Co-orbital kinetic energy weapons (a maneuvering satellite that positions near a target and impacts it). These create debris fields that persist for years to decades — the irreversibility is not just to the target satellite but to the orbital environment.

**Non-kinetic reversible**: Cyber attacks on satellite command and control — can be remediated with software updates. Laser dazzling of optical sensors — temporary blindness when the laser is off.

**Non-kinetic irreversible**: High-power laser (HPL) attacks on satellite sensors — permanent physical damage without generating debris. Cyber attacks that brick satellite firmware — irreversible without hardware replacement.

A fourth category applies across all quadrants: **attributability**. Kinetic ASAT impacts are attributable — you know what hit your satellite. GPS jamming over a theater of operations is attributable to the jamming platform if you can geolocate it. Cyber attacks on satellite command links are much harder to attribute, especially if the attacker uses third-country infrastructure. High-power laser attacks may be unattributable if the satellite's failure can be made to look like a component failure.

**Wargame relevance**: When you design your capstone game's action space, the counterspace taxonomy tells you which actions are available. The attacker in the conjunction-masking game is using a non-kinetic reversible capability (maneuvering a satellite and using RF deception or orbital geometry to obscure the maneuver from SSA sensors). The specific capability determines the cost, detectability, and escalation implications of each action.

---

## Deterrence stability in space

Deterrence in space is harder than deterrence in the nuclear domain for three reasons.

**Attribution uncertainty**: Nuclear weapons have clear signatures — yield, yield-to-weight ratio, delivery vehicle. Counterspace attacks frequently do not. A satellite that stops functioning may have been attacked or may have suffered a component failure. GPS jamming over a region may originate from military jammers or from commercial interference. Attribution uncertainty means the threat of retaliation — the foundation of deterrence — is weaker. An adversary who believes its attack will not be attributed has reduced incentive to restrain.

**The stability-instability paradox**: This concept originated in nuclear deterrence theory and applies in modified form to space. The paradox: robust strategic deterrence (neither side can eliminate the other's retaliatory capability) may *increase* instability at lower levels of conflict by making leaders confident that escalation will not reach the strategic level. In space terms: if neither side can completely blind the other's space architecture, each side may believe it can conduct limited counterspace operations without triggering strategic retaliation. The result is an environment where limited counterspace conflict is more likely, not less.

**Debris as a shared tragedy**: Kinetic counterspace creates debris that endangers all orbital users, including the attacker's own satellites. This creates a partial deterrent — but only partial. The 2007 Chinese ASAT test and the 2021 Russian NUDOL test both created debris despite international criticism. The deterrent effect of debris creation on the attacker depends on the attacker's own orbital dependence and time horizon. An attacker willing to accept debris in LEO for tactical advantage in a short conflict has lower inhibitions than one dependent on LEO for long-term commercial or military operations.

**The first-strike problem**: If counterspace attacks are reversible and non-attributable, the defender faces a peculiar first-strike problem. An adversary may conduct preparatory counterspace operations — degrading ISR and communication satellites — before initiating terrestrial conflict. By the time the degradation is noticed and attributed, the ground conflict may already be underway. This creates incentives for preemptive counterspace action ("use them or lose them" applied to space-enabled ISR), which is destabilizing.

Krepinevich captures the implication in *The Origins of Victory* (2023):

> "The next great-power war will be the first 'space war,' fought as much over the ability to see, communicate, and navigate as over control of territory."

---

## Domain expansion: how military competition reaches new domains

Andrew Krepinevich's domain expansion theory, developed in *The Origins of Victory*, provides a historical framework for understanding why space is now a contested military domain.

The argument: Military competition drives the expansion of conflict into new operational domains whenever:
1. A new domain provides a decisive military advantage to whoever controls it
2. The technology to exploit that domain becomes accessible to major powers
3. The advantage is large enough to justify the investment in new doctrine, organization, and equipment

Krepinevich traces domain expansion from land to sea (naval power projection), to air (strategic bombing, close air support), to the electromagnetic spectrum (SIGINT, electronic warfare), to cyberspace, and now to space. Each domain expansion followed the same pattern: an early-mover advantage followed by rapid diffusion of the capability and, eventually, competitive balance with doctrines for contesting the new domain.

The implication for space: the early-mover phase is ending. In the 1990s, U.S. space capabilities were so dominant that adversaries had no effective counterspace options — "sanctuary by default" rather than sanctuary by design. The 2007 Chinese ASAT test marked the end of that phase. By 2025, Russia, China, and several other states have demonstrated a range of counterspace capabilities across all four quadrants of the taxonomy above.

**The MTR vs. RMA distinction**: Krepinevich distinguishes between a **Military Technical Revolution (MTR)** and a **Revolution in Military Affairs (RMA)**. An MTR is a new technology that changes what is possible. An RMA is when a state successfully integrates new technology with new doctrine, organization, training, and leadership to create a qualitatively superior fighting force. The Soviet Union pioneered the MTR concept; it was the United States that converted it into an RMA with AirLand Battle and Precision Strike.

Christian Brose's *The Kill Chain* (2020) argues that the U.S. military has had the MTR but has largely failed to execute the RMA — it keeps buying platforms (aircraft carriers, F-35s) rather than investing in the sensor-shooter kill chains that actually win modern warfare. The argument has direct implications for space: the MTR (space-based ISR, GPS) is deployed, but the RMA — integrating space capabilities into joint operations with the speed and precision that space enables — is incomplete.

**Why this matters for SDA ML**: Maneuver detection and behavioral attribution from TLE history is a small piece of the SDA RMA. The MTR is the sensor architecture and data feeds. The RMA requires the decision-support tools — including the ML models you are building — that convert raw orbital data into actionable intelligence fast enough to matter.

---

## PLA space doctrine: what China's strategists say

The primary source for Chinese military space doctrine in English translation is the *Science of Military Strategy* published by the PLA Academy of Military Science. The 2013 edition includes the most detailed public treatment of Chinese space strategy.

Two concepts from *Science of Military Strategy 2013* are directly relevant to wargame design:

**Space deterrence**: The PLA treats space deterrence as a form of strategic deterrence distinct from nuclear deterrence. Chinese strategists argue that demonstrating counterspace capabilities — through testing, exercises, and deployment — deters adversaries from relying on space assets in conflict. This is deterrence by denial (making the adversary uncertain its space assets will function) rather than deterrence by punishment (threatening retaliation). The distinction matters: deterrence by denial requires operationally credible counterspace capabilities, which is why China continued ASAT testing despite international criticism.

**Counter-preemption**: Chinese doctrine explicitly addresses the scenario where the United States strikes China's space assets early in a conflict to degrade its ISR and communications. The Chinese response concept is counter-preemption — taking action to preserve China's space capabilities or to preemptively degrade U.S. space-enabled capabilities before the U.S. can do so. This creates a first-strike incentive on both sides — the side that strikes first degrades the adversary's ability to retaliate effectively in space — which is the stability-instability problem described above in its most acute form.

The *Unrestricted Warfare* framework (Qiao Liang and Wang Xiangsui, 1999) adds a different dimension: Chinese strategists have long argued that modern warfare is not limited to kinetic military operations. "Unrestricted warfare" includes economic warfare, legal warfare, information warfare, and technological warfare. In space terms: a state can compete for space advantage without deploying ASATs — by exporting launch services, building international coalitions around space norms, developing commercial space infrastructure that creates economic dependencies, and deploying dual-use space capabilities that are ambiguously military.

---

## The current counterspace landscape

The Secure World Foundation's *Global Counterspace Capabilities* assessment tracks publicly available evidence of counterspace capabilities across the following state actors as of 2025:

**United States**: Primarily focused on resilience (disaggregation, proliferated LEO constellations, hosted payloads) rather than offensive counterspace. The Space Force's Space Control mission includes both defensive counterspace (protecting U.S. assets) and offensive counterspace (degrading adversary space capabilities). The U.S. has demonstrated direct-ascent ASAT capability historically (1985 MIRACL test; the classified ASAT-on-F-15 program) but has not conducted a debris-generating ASAT test since 2008 and has effectively adopted a self-imposed moratorium on such testing.

**China**: Has demonstrated the full range of counterspace capabilities — direct-ascent kinetic ASAT (2007 test; subsequent tests at non-debris-generating altitudes), co-orbital maneuvering (Shijian series), jamming, spoofing, and cyber capabilities against space systems. The 2013 Dong Neng-3 test reached GEO altitude — the first kinetic ASAT test targeting the GPS/PNT and early warning satellite belt.

**Russia**: Demonstrated a debris-generating direct-ascent ASAT test in 2021 (NUDOL against a defunct Soviet satellite, creating 1,500+ tracked debris fragments). Also operates the Tirada-2 GEO communications jamming satellite, the Luch co-orbital inspection platform (positioned near Intelsat satellites at GEO), and the Peresvet ground-based laser system.

**Other actors**: Iran has demonstrated GPS spoofing (used in the seizure of a U.S. drone in 2011). North Korea has demonstrated GPS jamming in the ROK theater. India conducted a direct-ascent ASAT test in 2019 at low altitude (deliberately minimizing debris). Several other states have purchased or developed jamming capabilities.

**The gray zone**: A significant portion of counterspace activity happens below the threshold of clearly attributable, clearly kinetic attacks. Satellite jamming during exercises. Co-orbital maneuvers near adversary satellites. Cyber probes of satellite command infrastructure. GPS spoofing in civilian airspace. These activities are designed to be operationally useful while remaining below the threshold that would trigger a clear response — the "space as a gray zone" framing from several recent strategic analysis pieces.

For wargame design: gray zone activities are more common than kinetic attacks and harder to model. The action space in a realistic orbital conflict game includes ambiguous actions whose effect on the adversary is probabilistic, whose attribution is uncertain, and whose escalation potential is bounded but real. The conjunction-masking game in the capstone captures one specific gray zone activity — maneuvering to create plausible deniability about intent — in a stylized but analytically useful form.

---

## Russian space doctrine: asymmetric degradation

China's space doctrine is about building parity from behind — constructing the capabilities that allow China to contest U.S. space superiority over decades. Russia's space doctrine is about something different: an aging space power with a deteriorated industrial base that cannot match U.S. capabilities quantity-for-quantity, pursuing asymmetric strategies to degrade U.S. advantage without requiring symmetric investment.

Russia has been a space power since Sputnik (1957). The Soviet space program was a peer competitor to NASA for two decades. Post-Soviet Russia has seen that industrial base atrophy — Russian launch vehicles are competitive, but the satellite manufacturing sector has not kept pace with U.S. or increasingly Chinese capabilities. The strategic implication: Russia cannot win a symmetric space competition. Russian space doctrine therefore emphasizes targeted, reversible, high-leverage capabilities that impose disproportionate costs on U.S. space-dependent military operations.

**The Russian counterspace toolkit**:

*Peresvet*: Revealed by Putin in 2018 as one of six advanced weapons systems, Peresvet is a mobile ground-based laser system. The U.S. assessment: designed to dazzle or permanently damage optical sensors on reconnaissance satellites. Not confirmed operationally deployed but repeatedly tested. Ground-based directed energy allows Russia to threaten LEO reconnaissance satellites without generating debris and without providing an attributable kinetic act.

*Nudol (PL-19)*: Russia's direct-ascent ASAT. Tested multiple times since 2014, culminating in the November 2021 live-fire test that destroyed the defunct Cosmos 1408 satellite at approximately 480 km altitude, generating 1,500+ tracked debris fragments and endangering the ISS. Unlike the 2007 Chinese test, the 2021 Russian test occurred after the U.S. had proposed a moratorium on such testing — a deliberate signal of Russian indifference to the emerging norm.

*Tirada-2*: A GEO-based electronic attack satellite designed to suppress enemy satellite communications over a theater. The concept: a jammer positioned at GEO can deny broadband satellite communications across a large geographic area without generating debris, without kinetic action, and with reversibility when the mission ends.

*Krasukha-4*: Ground-based electronic warfare system that jams space-based radar ISR and drone control links. Extensively deployed in Syria and Ukraine. Demonstrates that effective counterspace does not require reaching orbit — jamming from the ground can deny the operational value of a satellite without touching it.

**Russia's operational record**: Russia has not merely tested these capabilities — it has used them. GPS jamming in the Baltic region is documented since at least 2016, affecting civilian aviation in Norway, Finland, and Estonia. Jamming in Syria has been extensive. In Ukraine, Russia has conducted GNSS spoofing in Kyiv (causing navigation errors for civilian aircraft), jamming of Ukrainian drone control links (significantly limiting Ukrainian UAV effectiveness in certain theaters), and the KA-SAT cyber attack. Russia is the only country to have used its counterspace capabilities at operational scale in a peer-adjacent conflict.

**The doctrinal framework**: Russian Military Doctrine 2014 explicitly identifies degradation of adversary space-based C2 and ISR as a priority task within the "non-nuclear deterrence" posture — alongside precision conventional strikes. Space operations are not a separate domain in Russian military thinking; they are integrated into the combined-arms campaign to establish information dominance before kinetic operations begin. This parallels the Chinese PLA informationized warfare concept (Lesson 4) but with a narrower, more operationally immediate focus.

**The key distinction from Chinese doctrine**: China is building toward parity; Russia is exploiting current U.S. vulnerabilities with what it has. A wargame that treats "Russian adversary" and "Chinese adversary" as equivalent is wrong on the strategy, wrong on the capabilities, and will produce wrong equilibria. The Russian player in an adversarial space game has a different action space (more ground-based, more reversible, more immediately available) and a different objective function (degrading specific operational capabilities rather than establishing long-term positional advantage) than the Chinese player.

---

## Commercial space as military infrastructure

The 2022 Russian invasion of Ukraine changed how the U.S. defense community thinks about commercial space. Before February 24, 2022, commercial satellite operators were understood as dual-use in potential — they could support military operations. After that date, commercial satellite operators became dual-use in practice, in ways that exposed vulnerabilities no existing doctrine had addressed.

**The Viasat KA-SAT hack**: John Klein's *Fight for the Final Frontier* describes the opening move of the Ukraine war in blunt terms: "An hour before Russian troops crossed the border, Russian government hackers conducted cyberattacks against the American satellite company Viasat... resulted in an immediate and significant loss of communication in the early days of the war for the Ukrainian military." The attack was not against a military satellite — it was against a commercial communications satellite used by Ukrainian military and government customers because it was the available option.

The ripple effects were wider than the intended target. The hack disabled satellite modems across Europe, including — improbably — the remote communications systems of 5,800 wind turbines in Germany, rendering them unable to communicate because of their satellite link. A cyberattack on a Ukrainian military communications platform created operational effects in German commercial infrastructure. The boundary between military and civilian space systems does not function the way either legal doctrine or operational planning assumes.

**Starlink in Ukraine**: SpaceX provided Starlink terminals to Ukraine starting in the opening days of the invasion. The operational impact was described by Ukrainian commanders as transformative — Starlink provided resilient, tactically usable communications that Russian jamming efforts could not consistently defeat. One Ukrainian officer put it directly: "fighting without Starlink is like fighting without a gun." SpaceX was not operating as a defense contractor — Elon Musk made repeated public statements about not wanting Starlink used for offensive operations and at one point declined to extend coverage to Crimea for a specific Ukrainian operation. A commercial company's CEO was making real-time tactical decisions affecting an active military operation.

**Maxar and commercial imagery**: Maxar Technologies' commercial satellite imagery was widely credited with enabling real-time attribution of Russian military buildups before the invasion and Russian troop movements during it. Intelligence that previously would have required a classified satellite with restricted distribution was published commercially, shared by open-source analysts, and used to build the international diplomatic coalition against Russia. Commercial imagery changed the information environment for the conflict.

**The CASR framework**: The Pentagon's response to the Ukraine lessons was the **Commercial Augmentation Space Reserve (CASR)** — modeled loosely on the Civil Reserve Air Fleet (CRAF). The concept: the DoD creates contractual frameworks and exercises wargame scenarios with commercial space providers (communications, imagery, SDA) so that in a crisis, commercial capacity can be integrated into military operations with established protocols. The CASR held its first wargaming event as a "major milestone" in 2024 — it is still a framework, not a fully integrated operational capability.

**Strategic implications for SDA products**: Commercial satellite operators are now de facto combatants in great-power competition, whether they intend to be or not. This creates several implications:

- Commercial SDA providers — including products built on the architecture this curriculum teaches — become intelligence infrastructure with strategic value. A commercial maneuver detection product that identifies Chinese orbital positioning before the DoD's classified sensors do has obvious value; it also has obvious targeting implications.
- The Viasat model means commercial space infrastructure is a high-value target in conflict. SDA products that provide indispensable situational awareness inherit the targeting profile of the assets they protect.
- CASR-type frameworks create a market: DoD is willing to pay for commercial space capabilities that can be surged in a crisis. The SBIR and SpaceWERX pathway (Module 8, Lesson 6) is the entry point for a small company building toward CASR-integration.

---

## The space-cyber nexus

The Viasat KA-SAT hack is the most visible case of cyber attack on space infrastructure — but it is a specific instance of a much broader structural vulnerability. Space systems and cyber systems are converging at the operational level in ways that dissolve the conceptual boundary between the two domains.

**Software-defined satellites and the update attack surface**: Modern satellites are increasingly controlled, reconfigured, and improved via software pushed over network links. Starlink's ability to rapidly update its terminals and satellite software to counter Russian jamming in Ukraine is the clearest operational demonstration: SpaceX deployed new anti-jamming firmware over-the-air within weeks of documented Russian jamming campaigns, restoring service that adversarial electronic warfare had degraded. This is a genuine military advantage. It is also an attack surface. The same over-the-air update mechanism that enables rapid capability improvement allows a nation-state adversary with access to the update infrastructure — or the ability to impersonate it — to push malicious firmware to the satellite or terminal fleet. What Viasat's attackers did by targeting the modem provisioning system is the template; future attacks need only find the equivalent mechanism in any sufficiently software-defined space system.

**Supply chain attacks applied to space**: The SolarWinds intrusion (discovered 2020) demonstrated that nation-state actors can compromise widely-used commercial software through the build process — inserting backdoors that survive deployment into secure environments without triggering detection for months or years. Satellite command and control software runs on commercial operating systems, uses commercial networking libraries, and integrates commercial ground hardware. Any component in that supply chain is a potential compromise vector. A satellite ground system with a SolarWinds-style supply chain backdoor could be commanded to alter satellite behavior, suppress anomaly reporting, or inject false telemetry — all while appearing to function normally to operators.

**TLE data integrity as an attack vector**: Space-Track TLE data is publicly available, widely used by commercial and government operators for collision avoidance, and is not cryptographically authenticated. An adversary with access to Space-Track's data pipeline — or the ability to perform man-in-the-middle attacks on operators who ingest that data — could inject false TLE entries. The effects: fictitious conjunction warnings forcing unnecessary maneuvers (operational disruption without kinetic action), false orbital data causing incorrect collision avoidance decisions, or masked real maneuvers that appear as normal station-keeping in the data record.

This has a direct implication for the ML pipeline in Module 9: an LSTM maneuver detector trained on TLE history will produce false positives if adversarial TLE data is injected upstream, and will fail to detect real maneuvers if those maneuvers are masked by corrupted TLE entries. Building data-provenance verification into the SDA pipeline is a security engineering requirement as much as an ML modeling requirement. A product that cannot reason about the integrity of its input data is operationally fragile in exactly the environments where it matters most.

**GPS spoofing as cyber-adjacent attack**: The sophisticated GPS spoofing documented in the Black Sea and Eastern Mediterranean is not simply an RF jamming problem — it is an exploit of the receiver software stack. Spoofing systems generate authentic-looking GPS signals that cause receivers to report a false, consistent position while appearing to function normally, with no receiver-side indication that the navigation solution is corrupted. Ships have logged positions placing them inland; aircraft have displayed incorrect locations. The mechanism is RF; the effect propagates through software. Distinguishing spoofing from jamming from legitimate signal degradation is an attribution and characterization problem — the same behavioral analysis problem the thesis addresses.

**Ground station targeting**: The KA-SAT hack targeted the ground segment — the modem provisioning system, not the satellite itself. Every satellite system has a ground control segment connecting to commercial internet infrastructure, often located across multiple countries with varying security postures. Targeting the ground station is frequently easier and achieves the same operational effect as targeting the space segment. The most hardened satellite in orbit can be effectively disabled by compromising the ground systems that task it, receive its data, or update its software.

**The cyber-kinetic substitution logic**: For an adversary practicing calibrated escalation or gray zone operations, cyber attacks on space infrastructure offer the same operational effect as kinetic counterspace (disabling specific satellite functions) with lower escalation risk, lower cost, and higher deniability. The Viasat hack disabled Ukrainian military communications as effectively as a kinetic ASAT strike would have — without generating debris, without providing a clearly attributable military act, and without crossing the threshold that would trigger a kinetic response. As satellite systems become more software-defined, the cyber substitution becomes more complete. This trend favors adversaries who are willing to conduct sustained cyber operations below kinetic thresholds, which describes both Russia and China.

---

## Deterrence by resilience

Offensive counterspace capabilities are one side of the deterrence equation. The other side is making your own assets hard enough to attack that the calculus turns against the attacker. The U.S. Space Force's approach has shifted from point-defense of a small number of exquisite satellites toward **deterrence by resilience**: making the space architecture so distributed, redundant, and rapidly replenishable that attacking it becomes too costly to be worth doing.

The USSF Space Capstone Publication defines the passive defense approach explicitly: "Passive defense measures include spacecraft maneuverability; self-protection; disaggregation; orbit diversification; large-scale proliferation; communication, transmission, and emissions security..."

In practice, resilience strategy has taken three forms:

**Proliferated LEO (PWSA / SDA Tranche architecture)**: The Space Development Agency's Proliferated Warfighter Space Architecture (PWSA) aims to deploy hundreds of small satellites in LEO providing transport-layer communications and missile warning — capabilities previously provided by a small number of large, exquisite GEO satellites. The logic: attacking one satellite in a 200-satellite transport layer degrades the capability by less than 1%. Attacking the system meaningfully requires attacking many satellites simultaneously, which creates massive debris and triggers Kessler Syndrome consequences the attacker shares. The SDA Tranche 0 and Tranche 1 satellites began launching in 2023. Tranche 2 and beyond will build out the full constellation.

**Starshield**: Starshield is SpaceX's version of Starlink purpose-built for national security applications — including encryption, government payload hosting, and survivable communications for nuclear command-and-control. Where Starlink provides commercial broadband at scale, Starshield provides the same proliferation-based resilience for classified and military communications. The program represents the CASR logic applied to communications: leverage the commercial megaconstellation architecture for military resilience.

**Disaggregation across orbits and operators**: Rather than hosting all capability on government-owned satellites, the USSF is increasingly distributing capability across commercial hosts, foreign partners, and classified assets. An adversary targeting U.S. space capabilities must now identify and attack assets across LEO, MEO, GEO, and HEO — operated by government, commercial, and allied entities — rather than attacking a defined set of government satellites.

The Salter formulation captures the strategic logic: "America remains the world's premier space power, but that dominance is also a source of vulnerability... One response is proliferation: deploying so many commercial satellites and space assets that it becomes prohibitively expensive for adversaries to target our entire space infrastructure."

**Implication for SDA products**: Resilience by proliferation creates a growth market for SDA. Managing a 200-satellite government constellation plus 4,000 Starshield satellites plus allied assets requires automation that human operators cannot provide. Pattern-of-life analysis, anomaly detection, collision avoidance — the LSTM maneuver detection pipeline in Module 9 is exactly the kind of capability that resilient megaconstellations require. The market exists because the strategy demands it.

---

## Allied and partner dimensions

Space competition is not bilateral. U.S. space power depends on allied infrastructure, allied data sharing, and allied diplomatic support — and Chinese gray zone strategy explicitly targets the U.S.-led coalition.

**Five Eyes SSA sharing**: The Five Eyes intelligence partnership (U.S., UK, Canada, Australia, New Zealand) extends to space domain awareness. U.S. Space Command has acknowledged that SSA data sharing with Five Eyes partners significantly extends coverage for tracking adversary satellite behavior. As noted in strategy discussions: "The U.S. already does this as an arrangement with the Five Eyes." The practical effect: orbital events over certain geographic regions are tracked with better fidelity because allied ground-based sensors contribute to the picture.

**NATO Space COE**: NATO declared space an operational domain in 2019. The NATO Space Centre of Excellence (Space COE) in Ramstein Air Base, Germany coordinates allied space doctrine development, wargaming, and capability interoperability. Ally space capabilities — UK, France, Germany, Japan, Australia — are increasingly integrated into U.S. Space Command operations rather than operating in parallel. This matters for wargaming: a game that models only U.S. assets understates the coalition's actual SSA capability.

**EU SST and Galileo**: The European Union Space Surveillance and Tracking (EU SST) network is an independent SSA capability serving European civil and commercial satellite operators. Galileo, the EU's GPS equivalent, provides independent PNT that reduces European dependence on U.S. GPS in a conflict scenario. These are not interoperable with U.S. government systems by default, but they create a broader allied information environment.

**JAXA and Indo-Pacific partnerships**: Japan's JAXA has deep SSA cooperation with NASA and the U.S. Space Force, including data sharing and joint exercises. The Quad (U.S., Japan, Australia, India) space cooperation initiatives are expanding to include SSA and space traffic management. India conducted its own ASAT test in 2019 — demonstrating capability and, by extension, signaling that it will not be a passive observer in space competition.

**Kronos**: The Kronos program, described by Space News, "aims to deliver a modernized suite for space battle management and intelligence... fuse data in real time, support planning and deconfliction, and provide shared awareness for U.S. and allied operators." This is the operational system through which allied SSA data is expected to be integrated with U.S. Space Command. Products that can feed into Kronos — providing maneuver detection, behavioral attribution, or anomaly characterization — have a clear path to allied operator markets.

**Brands' coalition argument applied to SDA**: Chinese orbital behavior that damages allied satellites or denies allied access drives allied investment in SSA and space resilience. Every Chinese co-orbital inspection event near a UK or Australian satellite is an argument for allied SSA spending. A commercial SDA product positioned as allied-operator-ready has a market that exists precisely because China's behavior created it.

---

## What you need to be able to do

After this lesson, you should be able to:

- Classify any specific counterspace capability using the kinetic/non-kinetic, reversible/irreversible taxonomy and note its attributability
- Explain the stability-instability paradox and apply it to space deterrence
- Describe the first-strike problem in space conflict and why it creates incentives for preemptive counterspace action
- Explain Krepinevich's MTR vs. RMA distinction and apply it to the current state of U.S. space power
- Describe the two key concepts from PLA *Science of Military Strategy 2013* (space deterrence by denial; counter-preemption) and their wargame design implications
- Name the primary counterspace actors and characterize each one's demonstrated capability set
- Explain why gray zone space activities are strategically significant and why they are harder to model than kinetic attacks
- Describe the Viasat KA-SAT hack and its strategic implications for how commercial space infrastructure is targeted in conflict
- Explain the PWSA/SDA Tranche architecture and the logic of deterrence by resilience through proliferation
- Describe the CASR framework and its market implications for commercial SDA products
- Name at least three allied/partner SSA frameworks and explain why the coalition dimension matters for SDA product positioning
- Describe Russia's four primary counterspace capabilities (Peresvet, Nudol, Tirada-2, Krasukha-4) and explain how Russia's strategic approach differs from China's
- Explain why Russian counterspace doctrine focuses on asymmetric degradation rather than parity-building, and what that implies for wargame action space design
- Explain why software-defined satellites create a cyber attack surface, and describe the supply chain attack model applied to satellite ground systems
- Explain the TLE data integrity problem and why it matters for ML-based maneuver detection pipelines operating in contested environments
- Describe the cyber-kinetic substitution logic: why cyber attacks on space infrastructure are attractive alternatives to kinetic counterspace for adversaries practicing calibrated escalation

---

{{#quiz 02-counterspace-and-competition.toml}}
