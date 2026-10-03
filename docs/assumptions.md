# Annahmen und Modellentscheidungen

Alles, was nicht direkt aus den Konnektom-Daten kommt, steht hier. Jede Annahme hat eine ID,
auf die Code-Kommentare und Berichte verweisen. Stand: 2026-10-03.

## Daten

| ID | Annahme | Begründung / Prüfung |
|---|---|---|
| A-1 | **MaleCNS v1.0 "Neuron" = annotierter Body mit `superclass`.** | Ergibt exakt 166.700, die Zahl der Release-Ankündigung. Geprüft in `flylab sanity`. |
| A-2 | **MaleCNS-Verbindungen aus `connectome-weights-…-traced-only.feather`** (minconf 0.5). Kanten zu Bodies ohne superclass werden verworfen (4.526 von 25,56 Mio. Kanten). | 124,03 Mio. Synapsen im traced-only-Graph vs. "~125 Mio." im Release (0,8 % Abweichung). |
| A-3 | **Vorzeichen MaleCNS:** `consensus_nt` (sonst `predicted_nt`) des präsynaptischen Neurons: ACh +, GABA −, Glutamat −, Histamin −, Dopamin/Serotonin/Octopamin/Tyramin +, unklar/unbekannt +. | GABA/Glu/ACh wie bei Shiu et al. 2024. Histamin hemmend (Photorezeptor→LMC). Modulatorische Transmitter als erregend sind eine Vereinfachung; ihre echte Wirkung ist rezeptorabhängig. |
| A-4 | **FlyWire v783 im Format von Shiu et al.** (`Completeness_783.csv`, `Connectivity_783.parquet`), Vorzeichen pro Kante aus deren `Excitatory`-Spalte. | 138.639 Neuronen, 54.492.922 Synapsen, exakt wie die Referenzdateien. |
| A-5 | **Zelltypen und Seiten:** FlyWire aus `Supplemental_file1_neuron_annotations.tsv` (Schlegel et al. 2024; `cell_type`, sonst `hemibrain_type`). MaleCNS aus `type`, Seite = `somaSide`, sonst `rootSide`. | Sensorische Neuronen (z. B. JO) haben in MaleCNS kein Soma im Datensatz, daher `rootSide`. |
| A-6 | **Shiu-Zuckerneuronen in v783:** Von den 21 IDs aus dem Shiu-Beispiel (v630) existiert `720575940620900446` in v783 nicht mehr und wird übersprungen (20 Neuronen). | `GroupResolver` überspringt fehlende IDs. |
| A-7 | **MaleCNS-Zuckerneuronen** = Typen `LB3b`, `LB3c` (34 Neuronen). | Das sind die FlyWire-Typen der Shiu-Zuckerneuronen (FlyWire-Subklasse "sugar"). |

## Gehirnmodell

| ID | Annahme | Begründung / Prüfung |
|---|---|---|
| B-1 | **LIF-Gleichungen und Konstanten von Shiu et al. 2024** (v₀ = −52 mV, v_th = −45 mV, τ_m = 20 ms, τ_syn = 5 ms, t_ref = 2,2 ms, Delay 1,8 ms, dt = 0,1 ms). | Eigene Engine, Schritt für Schritt äquivalent zu Brian2 (Test `test_brian2_equivalence.py`). |
| B-2 | **Zwei Brian2-Details reproduziert:** (1) das Refraktär-Flag für den State-Update stammt aus dem Threshold-Schritt davor; (2) synaptischer Input an ein refraktäres Neuron bleibt nicht in g. | Ohne (2) lagen die Raten im Mittel ~25 % höher als in Brian2. Gefunden durch Replay identischer Spike-Züge. |
| B-3 | **w_syn pro Datensatz kalibriert:** FlyWire 0,275 mV (Original), MaleCNS 0,15 mV. Kriterium: Zucker-GRNs (150 Hz) rekrutieren ähnlich wenige Neuronen wie die FlyWire-Referenz (FlyWire 340; MaleCNS 314 bei 0,15, 781 bei 0,16, 12.498 bei 0,2). | MaleCNS hat ~2,3× mehr Synapsen. Mit 0,275 mV läuft das Netz weg (17.673 aktiv). `scripts/calibrate_wsyn.py`, Ergebnis in `docs/validation/wsyn_scan_*.json`. w_syn bleibt ein freier Parameter. |
| B-4 | **Poisson-Eingang:** ein Ereignis pro Δt mit P = 1 − exp(−r·Δt), Sprung von w_syn·f_poi = 68,75 mV auf v; Poisson-getriebene Neuronen ohne Refraktärzeit. | Wie `PoissonInput(N=1)` bei Shiu et al. f_poi wird für MaleCNS so skaliert, dass der Sprung gleich bleibt. |
| B-5 | **Lesion** = alle ein- und ausgehenden Synapsen entfernt + Neuron geklemmt (kein Spike, kein Poisson-Input). | Shiu et al. setzen nur ausgehende Gewichte auf 0; das Klemmen macht die Lesion unabhängig von Eingängen. |
| B-6 | **Shuffle-Kontrolle (Standard `targets`):** Pro präsynaptischem Neuron bleiben Zahl, Gewicht und Vorzeichen der Ausgangskanten erhalten; Ziele werden gleichverteilt neu gezogen (keine Selbstkanten). Jeder Trial bekommt einen eigenen Shuffle (Seed = Trial-Seed + 10000). Alternative `degree`: globale Permutation der Kanten-Enden (erhält In- und Out-Grad). | Verschwindet ein Verhalten im Shuffle, hängt es an der echten Verdrahtung. |
| B-7 | **Keine Spontanaktivität, keine Plastizität, keine Neuromodulation, keine Gap Junctions, keine zelltypspezifischen Membranparameter.** | Basismodell. Folge: Hemmung von stillen Neuronen bewirkt nichts (siehe A-11). |
| B-8 | **Pilzkörper-Sparse-Coding fehlt:** Duftreize aktivieren in MaleCNS ~75 % der Kenyon-Zellen (in echten Fliegen ~5–10 %). Das APL-Feedback ist als GABAerge Verbindung vorhanden, wirkt im LIF aber nicht stark genug. | Beobachtet beim ORN-Scan (3.135 von 4.064 KC aktiv). Bekannte Grenze; Ergebnisse zur Futtersuche entsprechend vorsichtig lesen. |

## Sinne (engineered, nicht Konnektom)

| ID | Annahme |
|---|---|
| A-8 | **Looming:** Jedes LPLC2/LC4-Neuron bekommt ein festes Pseudo-Zufalls-Rezeptivfeld auf der Seite seines optischen Lobus (Azimut −15° bis 175° zur eigenen Seite, Elevation −60° bis 80°; LPLC2 30° Radius, LC4 20°). Seed = Neuron-ID, also reproduzierbar. LPLC2-Rate = 150 Hz · (σ(dθ/dt) − σ(0))/(1 − σ(0)) mit σ(x) = Sigmoid((x − 60°/s)/25°/s), also exakt 0 ohne Expansion (eine frühere Version hatte hier ~12 Hz Offset, was zu verfrühten Fluchten führte); LC4-Rate = 120 Hz · min(1, |dθ/dt|/300°/s). Die echte Retinotopie (Spalten-Zuordnung) ist **nicht** verwendet. |
| A-9 | **Optischer Fluss:** HS-Zellen (HSE/HSN/HSS) je Seite kodieren Gierrotation (Rechtsdrehung erregt HS links und umgekehrt) plus Vorwärtsflug; VS-Zellen kodieren Rollen (Rollen nach rechts erregt VS links). 120 Hz bei 8 rad/s. Kein Bildmodell. |
| A-10 | **Johnston-Organ:** Statische Arista-Auslenkung s = 60 · (g_Körper · a)/981 + 0,4 · (Luft_Körper · a) mit Lastachse a = (0,45, ±0,3, −0,84). JO-C feuert bei s > 0 (Hz = s), JO-E bei s < 0. Aufrecht bei 1 g also ~50 Hz auf JO-C; in Schwerelosigkeit nur Fahrtwind. Grob vereinfacht. |
| A-11 | **Photorezeptoren:** R1–R6-Rate = 60 Hz · L/(L+1), L = Summe der Lichtquellen im Rezeptivfeld (Gauß, 45°), Abfall mit Entfernung. Weil R1–R6 histaminerg (hemmend) sind und ein LIF ohne Spontanaktivität Hemmung nicht weiterleiten kann, wird ihr Ausgang **im Szenario Lichtwahl** auf erregend gesetzt (`model_mods`, wird geloggt, gilt auch für die Shuffle-Kontrolle). Ratenkodierte Näherung der Vorzeichenumkehr in der Lamina. |
| A-12 | **Geruch:** Attraktive ORNs (DM1, DM2, DM4, VM2) je Antenne, Rate = 50 Hz · c/(c+0,2). Duftfeld = Gaußsche Wolke (σ = 2,5 cm), keine Turbulenz und keine Advektion. |
| A-13 | **Tonischer Lokomotionsantrieb:** In Futtersuche und Lichtwahl bekommt DNp09 (P9-ähnlich) 60 Hz Poisson-Input, damit die Fliege überhaupt läuft. Ohne Spontanaktivität würde sie stehen bleiben. Lenkung muss aus dem Netz kommen. |

## Motorik und Körper

| ID | Annahme |
|---|---|
| A-14 | **Ausleseschicht (handgesetzt, nicht trainiert):** Raten der DN-Gruppen, exponentiell gefiltert (τ = 20 ms). Abheben, sobald die Giant-Fiber-Rate > 20 Hz ist (≈ 1 Spike). Sprungrichtung = −(L − R) von DNp02/DNp11. Vorwärts = tanh((DNp09 − MDN)/30 Hz). Lenken am Boden = tanh(((DNa02 + DNa01)_L − (…)_R)/30 Hz), Konvention „DNa02 links → Linksdrehung“ (Rayshubskiy et al. 2020). Flugschub = tanh(DNg02/50 Hz). Gieren im Flug = DNa02 L − R. Rollen im Flug = −((DNp20 + DNb06)_L − (…)_R): Das sind die stärksten DN-Ziele der VS-Zellen in MaleCNS (eigene Konnektom-Analyse). **Das Vorzeichen des Roll-Kanals ist eine Designentscheidung (korrigierend gesetzt).** Das Gier-Vorzeichen folgt aus der Literatur-Konvention plus Verdrahtung (HS → DNa02 ipsilateral). Die Sprungrichtung ist handgesetzt und **nicht** validiert. Optional ersetzt eine trainierte Sprungrichtung (Ridge auf alle DNs, `scripts/fit_escape_readout.py`) diesen Kanal. |
| A-15 | **Bewegungsprimitive statt Gelenke:** Die Ausleseschicht erzeugt Kraft und Drehmoment am Thorax (durch den Gesamtschwerpunkt). Laufen = Geschwindigkeitsservo (3 cm/s, τ = 10 ms), Drehen = Drehratenservo (6 rad/s), Sprung = 5 ms Schub auf 60 cm/s unter 0,9 rad Elevation, Flug = Schub entlang der Körperhochachse (Grundschub = Gewicht bei 1 g, ±60 % moduliert) plus Drehmomente; Kommando 1 entspricht 3 rad/s Drehrate. Bei 10 rad/s lag die Schleifenverstärkung über 1, und bei ~30 ms neuronaler Verzögerung schwang die Roll-Regelung auf (gemessen: Korrelation −0,21 statt +0,25). Das steht für VNC, Muskeln und Flügelaerodynamik zusammen. |
| A-16 | **Flapping counter-torque:** Im Flug wirkt eine passive Rotationsdämpfung (τ = 10 ms, Hedrick et al. 2009). Physik, kein Gehirn. Ein optionaler Halteren-Reflex (`haltere_reflex_gain`) ist standardmäßig **aus**. |
| A-17 | **Grundschub hängt nicht von g ab.** In Schwerelosigkeit erzeugt dieselbe Flügelleistung Nettoschub; die Fliege driftet, bis Luftwiderstand und Schub im Gleichgewicht sind (~235 cm/s). Eine Anpassung müsste aus dem Netz kommen (DNg02). |
| A-18 | **Körper:** Standard `simple` (Ellipsoide, ~0,98 mg, Beine als Kufen). Optional `flybody` (Vaxenburg et al., Apache-2.0) als **starre** Figur in Ruhepose (Gelenke entfernt), mit anatomischer Geometrie und Massenverteilung. Bodenreibung niedrig (0,05; Fliegengeometrie 0,02), weil Laufen als Servo modelliert ist. |
| A-19 | **Einheiten CGS** (cm, g, s), wie bei flybody. Luft: ρ = 0,00128 g/cm³, η = 0,000185 Poise, MuJoCo-Ellipsoid-Fluidmodell. Zero-g ändert nur `gravity`, die Luft bleibt. |

## Szenarien

| ID | Annahme |
|---|---|
| A-20 | **Spinne (geskriptet):** startet 5–7 cm entfernt in zufälliger Richtung, schleicht mit 6 cm/s und sprintet mit 30 cm/s, sobald sie < 3 cm entfernt ist. Fang bei Abstand < Radius + 0,25 cm. Einer Fliege über 0,8 cm Höhe folgt sie nicht. Das RL-Spinnenmodell ist noch nicht gebaut. |
| A-21 | **Looming-Stimulus:** dunkle Kugel, r = 0,5 cm, 25 cm/s (r/v = 20 ms), Start in 10 cm Entfernung, zufälliger Azimut, 20° Elevation. |
| A-23 | **Böen in Zero-g:** Ornstein-Uhlenbeck-Drehmoment (τ = 100 ms). Skala: so stark, dass es allein gegen die Flapping-Dämpfung ~3 rad/s erzeugen würde. Ohne andauernde Störung dämpft die Physik jede Startdrehung in ~30 ms weg, und Regelung wäre unnötig. |
| A-22 | **Lichtwahl:** T-Labyrinth, helle Seite pro Trial zufällig (Intensität 5 vs. 0,1). Gewählt ist, sobald \|y\| > 1 cm im Querarm erreicht ist. |

## Daten-URLs (geprüft am 2026-10-03)

- MaleCNS v1.0: `https://storage.googleapis.com/flyem-male-cns/v1.0/connectome-data/flat-connectome/` (öffentlicher Bucket `gs://flyem-male-cns`; `male-cns.janelia.org` war aus diesem Container blockiert, daher Bucket-Listing und `README_RELEASE_BUCKET.md` verwendet).
- FlyWire v783 (Shiu-Format): `github.com/philshiu/Drosophila_brain_model`.
- FlyWire-Annotationen: `github.com/flyconnectome/flywire_annotations`.
- neuPrint (`neuprint.janelia.org`) war aus dem Container nicht erreichbar und wird nicht gebraucht: Die flachen Feather-Dateien enthalten alles Nötige.
