# Durchsicht der Referenzen: was übernommen wurde und was nicht

Stand 2026-10-03. Geprüft wurde, was sich aus dem Container klonen und lesen ließ.

## Shiu et al. 2024: `philshiu/Drosophila_brain_model` (MIT)

**Was es ist:** LIF-Modell des ganzen FlyWire-Gehirns in Brian2. Neuronen werden per Poisson-Input
aktiviert oder per Gewicht 0 stummgeschaltet; ausgegeben werden Spike-Raten. Das Beispiel
(Zuckerneuronen → MN9) ist der Standardbefund, an dem sich Nachbauten messen lassen.

**Übernommen:**
- Gleichungen, Konstanten und Update-Semantik (`brain/lif.py`). Eine Brian2-Referenz steht in
  `brain/brian2_ref.py` und ist fast wörtlich aus `model.py` übernommen.
- Die Datendateien `Completeness_783.csv` und `Connectivity_783.parquet` (FlyWire v783, mit
  Vorzeichen pro Kante).
- Das Validierungsexperiment Zucker-GRNs → MN9 (Neuron-IDs aus `example.ipynb`).

**Nicht übernommen:**
- Der Brian2-Lauf in jedem Trial. Ein Brian2-Trial braucht hier 12–100 s, nur um das Netz
  aufzubauen. Für eine geschlossene Schleife, die jede Millisekunde Daten mit MuJoCo austauscht,
  ist das ungeeignet. Ersatz: eigene schrittweise Engine (numpy/numba). Die Äquivalenz ist
  getestet: identische Spike-Zeiten bei identischem Input
  (`tests/test_brian2_equivalence.py`) und r = 0,999 über alle antwortenden Neuronen im
  Zuckerexperiment (`docs/validation/m1_results.json`).

**Gefundene Feinheit:** Brian2 verwirft synaptischen Input, der ein refraktäres Neuron erreicht.
Die erste Version der Engine akkumulierte ihn und lag dadurch ~25 % zu hoch (MN9 97 statt 78 Hz).
Gefunden durch den Replay-Vergleich, jetzt behoben und durch einen Test abgesichert.

## `townie/awesome-fruit-fly` (Übersichtsliste)

Gute Übersicht über den Trend nach dem MaleCNS-Release im September 2026. Gelernt daraus:
- **Datenzugang:** Die Liste verweist auf `male-cns.janelia.org/download`. Diese Seite war aus
  dem Container blockiert, der dahinterliegende öffentliche Bucket `gs://flyem-male-cns` aber
  erreichbar. Der Loader nutzt die flachen Feather-Dateien aus `v1.0/connectome-data/flat-connectome/`.
- **Methodik, die sich bewährt hat:** Projekte mit Shuffle- oder Rewire-Kontrolle
  (`ClutchMedia775/fly-brain-drone`, `webergithub/fruitfly-lab` mit "real 10.2 vs. scrambled 0.7",
  `dohun1214/malecns-asteroids`) können zeigen, ob ein Verhalten an der Verdrahtung hängt.
  Projekte mit trainierter Ausleseschicht und ohne Kontrolle (`Fly Tic-Tac-Toe`: "rewired graph
  scores about the same") zeigen oft, dass die Ausleseschicht die Arbeit macht. Darum hat hier
  jedes Experiment automatisch eine Shuffle-Kontrolle.
- **Häufige Schwächen** (bewusst vermieden): Gewichte, Sinnes-Mapping oder Motorprogramme werden
  nicht offengelegt; "Lernen" wird ohne Kontrolle behauptet; die Wirkung der Ausleseschicht
  wird nicht vom Konnektom getrennt.
- **Giant-Fiber-Pfad (LPLC2/LC4 → DNp01)** ist in mehreren Demos die robusteste Verbindung von
  Sinnesreiz zu Verhalten. Hier dient er als erstes geschlossenes Verhalten (M3).

**Nicht geklont:** Keines der Community-Repos ist Abhängigkeit. Die meisten sind Demos mit
eigener Spiel- oder Anzeige-Logik und ohne Tests. Wo eine Idee übernommen wurde (Shuffle-Kontrolle,
Giant-Fiber-Auslese), steht sie hier und in `docs/assumptions.md`.

## `TuragaLab/flybody` (Apache-2.0)

MuJoCo-Modell der Fliege (CGS-Einheiten, 0,98 mg, 108 Freiheitsgrade, Flügel-Fluidgeometrie).
Wird optional per `flylab download --flybody` geklont. Die Controller von flybody sind trainierte
MLP-Policies und nicht Teil dieses Projekts. Hier wird flybody als **starre anatomische Figur**
verwendet: Geometrie, Massenverteilung und Fluidformen bleiben, die Gelenke werden entfernt, und
die Bewegung kommt aus denselben Primitiven wie beim einfachen Körper. Gelenksteuerung durch das
Konnektom ist ein offenes Ziel (siehe Roadmap in der README).

## Nicht verwendet (bisher)

| Ressource | Grund |
|---|---|
| FlyVis | Hängt von PyTorch und vortrainierten Gewichten ab. Die Ausgaben enden bei Medulla- und Lobula-Platten-Typen (T4/T5 …); für LPLC2/LC4 bräuchte man einen Adapter. Der analytische Looming-Encoder reicht für M3/M4. Nächster Schritt in der Roadmap. |
| NeuroMechFly / flygym | Ähnliche Rolle wie flybody. Eine zweite Körper-Implementierung bringt im jetzigen Primitive-Ansatz keinen Mehrwert. |
| neuPrint-Python | Braucht ein Token und war aus dem Container nicht erreichbar. Die flachen Release-Dateien reichen aus. |
| FlyGM | Graph-RL-Controller, eine andere Fragestellung (lernt Gewichte). |
