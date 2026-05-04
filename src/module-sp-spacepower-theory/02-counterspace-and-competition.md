# Lesson 2: Counterspace Operations and the New RMA

---

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

## What you need to be able to do

After this lesson, you should be able to:

- Classify any specific counterspace capability using the kinetic/non-kinetic, reversible/irreversible taxonomy and note its attributability
- Explain the stability-instability paradox and apply it to space deterrence
- Describe the first-strike problem in space conflict and why it creates incentives for preemptive counterspace action
- Explain Krepinevich's MTR vs. RMA distinction and apply it to the current state of U.S. space power
- Describe the two key concepts from PLA *Science of Military Strategy 2013* (space deterrence by denial; counter-preemption) and their wargame design implications
- Name the primary counterspace actors and characterize each one's demonstrated capability set
- Explain why gray zone space activities are strategically significant and why they are harder to model than kinetic attacks

---

{{#quiz 02-counterspace-and-competition.toml}}
