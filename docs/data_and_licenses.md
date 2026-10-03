# Daten, Lizenzen, Zitierpflichten

Rohdaten werden **nicht** ins Repository eingecheckt (`data/` steht in `.gitignore`).
`flylab download` lädt sie aus den Originalquellen.

| Datensatz | Dateien | Quelle | Lizenz | Zitieren |
|---|---|---|---|---|
| **MaleCNS v1.0** | `body-annotations-male-cns-v1.0-minconf-0.5.feather` (14,5 MB), `body-neurotransmitters-male-cns-v1.0.feather` (43 MB), `connectome-weights-male-cns-v1.0-minconf-0.5-traced-only.feather` (508 MB) | `https://storage.googleapis.com/flyem-male-cns/v1.0/connectome-data/flat-connectome/` | CC BY 4.0 | Berg, Beckett et al., *Sexual dimorphism in the complete Drosophila male central nervous system connectome*, Cell (2026), doi:10.1016/j.cell.2026.08.015 |
| **FlyWire v783** | `Completeness_783.csv`, `Connectivity_783.parquet` | `github.com/philshiu/Drosophila_brain_model` | Daten CC BY-NC 4.0 (FlyWire), Code MIT | Dorkenwald et al., Nature 2024 (doi:10.1038/s41586-024-07558-y); Schlegel et al., Nature 2024 (doi:10.1038/s41586-024-07686-5) |
| **FlyWire-Annotationen** | `Supplemental_file1_neuron_annotations.tsv` | `github.com/flyconnectome/flywire_annotations` | CC BY 4.0 (Repository-Angabe vor Veröffentlichung prüfen) | Schlegel et al., Nature 2024 |
| **LIF-Modell** | Gleichungen und Konstanten | `github.com/philshiu/Drosophila_brain_model` | MIT | Shiu et al., *A Drosophila computational brain model reveals sensorimotor processing*, Nature 2024 |
| **flybody** (optional) | MJCF + Meshes | `github.com/TuragaLab/flybody` | Apache-2.0 | Vaxenburg et al., *Whole-body physics simulation of fruit fly locomotion*, Nature 2025 |
| MuJoCo | Python-Paket | pypi `mujoco` | Apache-2.0 | Todorov et al., IROS 2012 |
| Brian2 | Python-Paket (nur Validierung) | pypi `brian2` | CeCILL 2.1 | Stimberg et al., eLife 2019 |

## Vor einer Veröffentlichung prüfen

- **FlyWire ist nicht-kommerziell (CC BY-NC).** Ergebnisse auf FlyWire-Basis dürfen nicht
  kommerziell genutzt werden. MaleCNS (CC BY) hat diese Einschränkung nicht.
- Die Zitierangaben oben stammen aus den Repositories bzw. der awesome-fruit-fly-Liste. Die
  MaleCNS-Release-Seite (`male-cns.janelia.org`) war aus dem Container nicht erreichbar.
  **Genaue Zitierweise und DOI dort nachprüfen.**
- Abbildungen, die Neuronen-Morphologien zeigen, unterliegen ggf. eigenen Bedingungen
  (hier werden keine Morphologien genutzt).

## Format-Notizen

- MaleCNS-Gewichte: `body_pre, body_post, weight (Synapsenzahl), type_pre, type_post`. Die
  Variante `traced-only` enthält nur Bodies mit Status "Traced"; die volle Datei (1,05 GB)
  enthält auch Fragmente. Neurotransmitter: eine Zeile pro Body (`consensus_nt`, `predicted_nt`,
  Konfidenz).
- FlyWire im Shiu-Format: Kantenliste mit 0-basierten Indizes passend zur Zeilenfolge von
  `Completeness_783.csv`; `Excitatory` ist ±1 je Kante.
- Der Verarbeitungs-Cache (`data/processed/<dataset>/`) enthält `neurons.parquet`,
  `weights.npz` (CSR, post × pre, vorzeichenbehaftete Synapsenzahlen) und `meta.json`.
