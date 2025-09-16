# Quantum Gate-Level Watermarking

A proof-of-concept Qiskit extension to embed and extract a secret bit string by inserting identity subcircuits.

METRICS

depth_base
• What: Logical circuit depth of the BASELINE circuit AFTER transpilation to the chosen backend.
• How: Max number of gate layers the backend will execute (accounts for mapping, scheduling, and optimizations).
• Why it matters: Larger depth ⇒ more time on hardware ⇒ more decoherence/opportunity for errors.

depth_wm
• What: Depth of the WATERMARKED circuit AFTER transpilation to the same backend.
• Overhead: (depth_wm − depth_base) / depth_base.
• Read your numbers: depth_base=39, depth_wm=36 ⇒ overhead −7.69% (the watermark+optimizations slightly reduced depth here).

twoq_base
• What: Count of 2-qubit gates (e.g., CX, CZ) in the transpiled BASELINE circuit.
• Why it matters: 2-qubit gates dominate error on today’s hardware.

twoq_wm
• What: 2-qubit gate count for the WATERMARKED circuit (post-transpile).
• Overhead: (twoq_wm − twoq_base) / twoq_base.
• Read your numbers: twoq_base=9, twoq_wm=11 ⇒ +22.22% (watermark increased entangling cost, common trade-off).

pst_base
• “Probability of Successful Trials” for the BASELINE circuit.
• Definition: PST = Pr[ output == 0…0 ] on the measured (functional) qubits.
• How we compute: counts["0"*n] / shots, where n is the number of functional qubits actually measured after transpile.
• Read your number: 0.0003 ≈ 0; for random circuits this is typical (all-zeros is rarely produced).

pst_wm
• PST for the WATERMARKED circuit, but computed on FUNCTIONAL qubits only.
• How we compute: First drop the ancilla bits from the watermarked counts, then PST = counts["0"*n] / shots using the same n as baseline.
• “Δ” is pst_wm − pst_base; negative means watermark further reduces the chance of all-zeros.
• Read your numbers: pst_wm=0.0000, Δ=−0.0003 ⇒ essentially no mass at all-zeros after watermarking (expected for random circuits).

tvd_base_vs_wm
• Total Variation Distance between BASELINE and WATERMARKED output distributions (on functional qubits).
• Definition: TVD(P,Q) = ½ \* Σ_x |P(x) − Q(x)|, ranges in [0, 1].
– 0 ⇒ distributions are identical.
– 1 ⇒ distributions are completely separated (no overlapping support in the empirical samples).
• How we compute: 1) Convert counts→probabilities for baseline, P(x). 2) Drop ancillas from watermark counts, convert to probabilities Q(x) on the same bit width. 3) Apply the formula above.
• Read your number: 1.0000 ⇒ near-complete separability in the observed samples.
Sanity checks if this looks “too good”: verify (i) ancilla bits were dropped, (ii) both circuits were transpiled to the SAME backend with the SAME seed/opt level,
(iii) bit ordering is consistent, and (iv) shots are sufficient. If all hold, it means your watermark (with that θ and compilation) pushed the distribution far from baseline.

GENERAL NOTES
• All metrics are computed after transpilation to ensure fair, hardware-realistic comparison (as in the paper).
• Depth & 2Q are structural (post-transpile). PST and TVD are statistical (from measured counts).
• Extremely low PST values are normal for random circuits; PST is mainly useful for checking you’re measuring the intended bit width and for detecting strong “easy-state” biases.
• Very high TVD means strong separability; if it’s unexpectedly 1, double-check ancilla dropping and that both distributions live on the exact same bitstring space.
